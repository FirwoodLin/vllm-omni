# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Manifest-driven full-duplex MiniCPM-o E2E benchmark.

This driver is intentionally separate from ``humdial_arrival_rate.py``.  The
arrival-rate benchmark measures open-loop serving pressure; this benchmark
adds a reproducible interaction contract: an initial user turn, an explicit
barge-in, a second user turn, and (optionally) a playback-grounded interrupt.
Every action and server event is written to an event ledger so that an SLO
pass is auditable rather than inferred from aggregate token counters.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from benchmarks.minicpmo.humdial_arrival_rate import BrowserPlaybackClock
except ModuleNotFoundError:  # direct ``python benchmarks/minicpmo/humdial_e2e.py``
    from humdial_arrival_rate import BrowserPlaybackClock

from vllm_omni.clients.duplex import (
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    read_pcm16_wav,
    reference_audio_data_url,
    wait_for_condition as wait_for,
)

try:
    from benchmarks.minicpmo.duplex_probe_client import RealtimeSession
except ModuleNotFoundError:  # direct ``python benchmarks/minicpmo/humdial_e2e.py``
    from duplex_probe_client import RealtimeSession

DEFAULT_MODEL = Path("/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/models--OpenBMB--MiniCPM-o-4_5")
MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class E2ECase:
    case_id: str
    initial_audio: Path
    interrupt_audio: Path
    external_interrupt_ms: int
    playback_anchor_ms: int
    operation: str
    expected_text_contains: tuple[str, ...]
    session_window_s: float
    instructions: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "initial_audio": str(self.initial_audio),
            "interrupt_audio": str(self.interrupt_audio),
            "external_interrupt_ms": self.external_interrupt_ms,
            "playback_anchor_ms": self.playback_anchor_ms,
            "operation": self.operation,
            "expected_text_contains": list(self.expected_text_contains),
            "session_window_s": self.session_window_s,
            "instructions": self.instructions,
        }


@dataclass(frozen=True)
class ScheduledE2ERequest:
    index: int
    arrival_s: float
    case: E2ECase

    def as_dict(self) -> dict[str, object]:
        return {"index": self.index, "arrival_s": round(self.arrival_s, 9), **self.case.as_dict()}


def _positive_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")
    if float(value) <= 0:
        raise ValueError(f"{field} must be positive")
    return float(value)


def load_manifest(path: Path) -> list[E2ECase]:
    """Load and validate a case manifest, resolving audio relative to it."""
    manifest_path = path.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version must be {MANIFEST_SCHEMA_VERSION}")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("manifest cases must be a non-empty list")
    cases: list[E2ECase] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"cases[{index}] must be an object")
        case_id = raw.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError(f"cases[{index}].case_id must be non-empty and unique")
        seen.add(case_id)

        def audio_path(field: str) -> Path:
            value = raw.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"cases[{index}].{field} is required")
            resolved = (manifest_path.parent / value).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"{field} does not exist: {resolved}")
            # Validate format before starting a GPU run.  This keeps malformed
            # fixtures from becoming an opaque server-side failure.
            read_pcm16_wav(resolved)
            return resolved

        initial_audio = audio_path("initial_audio")
        interrupt_audio = audio_path("interrupt_audio")
        external_interrupt_ms = raw.get("external_interrupt_ms", 12_000)
        playback_anchor_ms = raw.get("playback_anchor_ms", 5_000)
        if (
            isinstance(external_interrupt_ms, bool)
            or not isinstance(external_interrupt_ms, int)
            or external_interrupt_ms < 0
        ):
            raise ValueError(f"cases[{index}].external_interrupt_ms must be a non-negative integer")
        if isinstance(playback_anchor_ms, bool) or not isinstance(playback_anchor_ms, int) or playback_anchor_ms < 0:
            raise ValueError(f"cases[{index}].playback_anchor_ms must be a non-negative integer")
        operation = raw.get("operation", "")
        if not isinstance(operation, str):
            raise ValueError(f"cases[{index}].operation must be a string")
        expected = raw.get("expected_text_contains", [])
        if not isinstance(expected, list) or not all(isinstance(item, str) and item for item in expected):
            raise ValueError(f"cases[{index}].expected_text_contains must be a list of strings")
        session_window_s = _positive_number(raw.get("session_window_s", 60.0), f"cases[{index}].session_window_s")
        instructions = raw.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise ValueError(f"cases[{index}].instructions must be a string or null")
        cases.append(
            E2ECase(
                case_id=case_id,
                initial_audio=initial_audio,
                interrupt_audio=interrupt_audio,
                external_interrupt_ms=external_interrupt_ms,
                playback_anchor_ms=playback_anchor_ms,
                operation=operation,
                expected_text_contains=tuple(expected),
                session_window_s=session_window_s,
                instructions=instructions,
            )
        )
    return cases


def build_schedule(
    cases: list[E2ECase],
    *,
    request_rate: float,
    duration_s: float,
    seed: int,
) -> list[ScheduledE2ERequest]:
    """Build an exact-count, deterministic open-loop arrival trace.

    Cases are sampled with replacement.  This is deliberate: a 5-minute run
    at a rate larger than the number of manifest rows should increase offered
    load, not silently reduce the request count.
    """
    if not cases:
        raise ValueError("cases must be non-empty")
    request_rate = _positive_number(request_rate, "request_rate")
    duration_s = _positive_number(duration_s, "duration_s")
    count = max(1, round(request_rate * duration_s))
    rng = random.Random(seed)
    arrivals = sorted(rng.random() * duration_s for _ in range(count))
    sampled = [cases[rng.randrange(len(cases))] for _ in range(count)]
    return [ScheduledE2ERequest(index=i, arrival_s=t, case=sampled[i]) for i, t in enumerate(arrivals)]


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def response_text(events: list[dict[str, object]], response_id: str | None) -> str:
    if not response_id:
        return ""
    chunks: list[str] = []
    for event in events:
        event_response_id = event.get("response_id")
        response = event.get("response")
        if event_response_id is None and isinstance(response, dict):
            event_response_id = response.get("id")
        if event_response_id != response_id:
            continue
        if event.get("type") not in {
            "response.audio_transcript.delta",
            "response.output_text.delta",
            "response.text.delta",
        }:
            continue
        delta = event.get("delta")
        if isinstance(delta, str):
            chunks.append(delta)
    return "".join(chunks)


def evaluate_context(text: str, expected: tuple[str, ...]) -> bool | None:
    if not expected:
        return None
    normalized = _normalize_text(text)
    return all(_normalize_text(item) in normalized for item in expected)


def _event_response_id(event: dict[str, object]) -> str | None:
    response_id = event.get("response_id")
    if isinstance(response_id, str) and response_id:
        return response_id
    response = event.get("response")
    if isinstance(response, dict) and isinstance(response.get("id"), str):
        return str(response["id"])
    return None


def _received_at_for_event(
    events: list[dict[str, object]], received_at_s: list[float], target: dict[str, object]
) -> float:
    """Find an event's receive time while scanning the event ledger backwards."""
    return next(
        received_at
        for event, received_at in zip(reversed(events), reversed(received_at_s), strict=True)
        if event is target
    )


def _response_first_output_at(
    events: list[dict[str, object]],
    received_at_s: list[float],
    response_id: str,
) -> float | None:
    """Return the first non-empty text/audio packet for one response."""
    for event, received_at in zip(events, received_at_s, strict=True):
        if _event_response_id(event) != response_id:
            continue
        if event.get("type") == "response.audio.delta":
            delta = event.get("delta") or event.get("audio")
            if isinstance(delta, str) and delta:
                return received_at
        if event.get("type") in {
            "response.audio_transcript.delta",
            "response.output_text.delta",
            "response.text.delta",
        } and isinstance(event.get("delta"), str) and event["delta"]:
            return received_at
    return None


def _usable_input_commit_at(
    events: list[dict[str, object]],
    received_at_s: list[float],
    response_id: str,
    input_committed_at_s: float | None,
) -> float | None:
    """Use commit as a latency origin only when output had not started yet.

    Native duplex auto-response may start while input audio is still being
    uploaded.  In that case the client-side ``input_audio_buffer.commit`` is
    observed after the first response packet, and using it would produce a
    physically impossible negative TTFP/TTFT.  Falling back to the first
    response output preserves a non-negative, turn-local metric while retaining
    commit-based timing whenever the event ordering makes it meaningful.
    """
    if input_committed_at_s is None:
        return None
    first_output_at_s = _response_first_output_at(events, received_at_s, response_id)
    if first_output_at_s is not None and input_committed_at_s > first_output_at_s:
        return None
    return input_committed_at_s


def _timing_measurement_origin(input_commit_used: bool) -> dict[str, str]:
    if input_commit_used:
        return {
            "ttft": "input_audio_buffer.commit client send to first non-empty text delta",
            "ttfp": "input_audio_buffer.commit client send to first audio packet",
            "rtf": "commit-to-last-audio receive time divided by emitted audio duration",
        }
    return {
        "ttft": "response.created client receive to first non-empty text delta; input commit followed output",
        "ttfp": "response.created client receive to first audio packet; input commit followed output",
        "rtf": "response-created-to-last-audio receive time divided by emitted audio duration",
    }


def _audio_duration_ms(event: dict[str, object], sample_rate_hz: int) -> float:
    encoded = event.get("delta") or event.get("audio")
    if not isinstance(encoded, str) or not encoded:
        return 0.0
    try:
        return len(base64.b64decode(encoded)) * 1000.0 / (sample_rate_hz * PCM16_BYTES_PER_SAMPLE)
    except Exception:
        return 0.0


class EventLedger:
    """Append-only actions plus raw server events with stable local sequence."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.rows: list[dict[str, object]] = []

    def action(self, event_name: str, **payload: object) -> float:
        now = time.monotonic()
        self.rows.append(
            {
                "kind": "action",
                "event_name": event_name,
                "event_received_at_s": now,
                "session_id": self.session_id,
                **payload,
            }
        )
        return now

    def server_events(self, client: RealtimeSession) -> None:
        for seq, (event, received_at_s) in enumerate(
            zip(client.events.events, client.events.event_received_at_s, strict=True)
        ):
            row: dict[str, object] = {
                "kind": "server_event",
                "event_seq": seq,
                "event_name": event.get("type"),
                "event_received_at_s": received_at_s,
                "session_id": self.session_id,
                "response_id": _event_response_id(event),
                "event": event,
            }
            metadata = event.get("metadata")
            if not isinstance(metadata, dict) and isinstance(event.get("response"), dict):
                metadata = event["response"].get("metadata")
            if isinstance(metadata, dict):
                duplex_event = metadata.get("duplex_event")
                if isinstance(duplex_event, dict):
                    for key in ("session_id", "incarnation", "epoch", "turn_id"):
                        if key in duplex_event:
                            row[key] = duplex_event[key]
                vllm_omni = metadata.get("vllm_omni")
                if isinstance(vllm_omni, dict):
                    for key in ("session_id", "incarnation", "epoch", "turn_id"):
                        if key in vllm_omni and key not in row:
                            row[key] = vllm_omni[key]
            for key in ("session_id", "incarnation", "epoch", "turn_id"):
                if key in event and key not in row:
                    row[key] = event[key]
            self.rows.append(row)

    def write(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as output:
            for row in self.rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")


class SimulatedPlaybackAdapter:
    """Browser-like ACK loop used for deterministic L2 playback grounding."""

    def __init__(self, *, initial_buffer_ms: int, progress_ms: int) -> None:
        self.clock = BrowserPlaybackClock(initial_buffer_ms=initial_buffer_ms, progress_ms=progress_ms)
        self.stop = asyncio.Event()
        self.task: asyncio.Task[None] | None = None

    async def start(self, client: RealtimeSession) -> None:
        self.task = asyncio.create_task(self.clock.run(client, self.stop))

    def rendered_ms(self, response_id: str) -> float | None:
        return self.clock.rendered_ms(response_id)

    async def wait_until(self, response_id: str, target_ms: int, timeout_s: float) -> float:
        if target_ms == 0:
            return self.rendered_ms(response_id) or 0.0
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            played = self.rendered_ms(response_id)
            if played is not None and played >= target_ms:
                return played
            await asyncio.sleep(0.02)
        raise TimeoutError(f"playback anchor {target_ms}ms was not reached for {response_id}")

    async def close(self) -> None:
        self.stop.set()
        if self.task is not None:
            try:
                await self.task
            except Exception:
                # If the WebSocket context is already unwinding, the final ACK
                # is no longer observable and must not hide the case result.
                pass
            self.task = None


async def _wait_response_created(client: RealtimeSession, *, after_count: int, timeout_s: float) -> str:
    await wait_for(
        lambda: len(client.events.response_ids) > after_count,
        timeout_s=timeout_s,
        label="response.created",
    )
    return client.events.response_ids[after_count]


async def _wait_input_audio_committed(client: RealtimeSession, *, count: int, timeout_s: float) -> None:
    await wait_for(
        lambda: client.events.count("input_audio_buffer.committed") >= count,
        timeout_s=timeout_s,
        label="input_audio_buffer.committed",
    )


async def _run_case(
    request: ScheduledE2ERequest,
    *,
    args: argparse.Namespace,
    ref_audio: str,
    run_id: str,
    output_dir: Path,
) -> dict[str, object]:
    case = request.case
    session_id = f"humdial-e2e-{run_id}-{request.index:05d}"
    result: dict[str, object] = {
        "index": request.index,
        "session_id": session_id,
        "case_id": case.case_id,
        "operation": case.operation,
        "session_window_s": case.session_window_s,
        "scheduled_arrival_s": request.arrival_s,
        "feedback_contract": args.feedback_contract,
        "transport_success": False,
        "interaction_success": False,
        "context_correct": None,
        "task_success": None,
        "session_slo_pass": False,
        "stale_audio_delta_count": 0,
        "stale_audio_ms": 0.0,
    }
    ledger = EventLedger(session_id)
    client: RealtimeSession | None = None
    player: SimulatedPlaybackAdapter | None = None
    stream_started_at_s = time.monotonic()
    commit_at_s: float | None = None
    first_response_id: str | None = None
    first_render_task: asyncio.Task[float] | None = None
    audio_observe_task: asyncio.Task[None] | None = None

    async def first_render_after(response_id: str) -> float:
        while True:
            played = player.rendered_ms(response_id) if player is not None else None
            if played is not None and played > 0 and commit_at_s is not None:
                return (time.monotonic() - commit_at_s) * 1000.0
            await asyncio.sleep(0.02)

    try:
        initial_pcm = read_pcm16_wav(case.initial_audio)
        interrupt_pcm = read_pcm16_wav(case.interrupt_audio)
        client = RealtimeSession(
            args.url,
            model=str(args.model),
            session_id=session_id,
            ref_audio=ref_audio,
            instructions=case.instructions,
            native_duplex=True,
            auto_response=not args.explicit_all_responses,
            temperature=0.0,
            idle_timeout_s=max(args.timeout_s, case.session_window_s + args.tail_drain_s + 30.0),
            handshake_timeout_s=min(args.timeout_s, 60.0),
        )
        await client.__aenter__()
        try:
            player = SimulatedPlaybackAdapter(
                initial_buffer_ms=args.playback_initial_buffer_ms,
                progress_ms=args.playback_progress_ms,
            )
            await player.start(client)
            stream_started_at_s = time.monotonic()
            ledger.action(
                "initial_stream_started",
                audio_ms=len(initial_pcm) * 1000 / (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE),
            )
            await client.stream_pcm16(initial_pcm, chunk_ms=args.chunk_ms, realtime=True)
            commit_at_s = ledger.action("initial_input_commit")
            await client.commit()
            if args.explicit_all_responses:
                await _wait_input_audio_committed(client, count=1, timeout_s=args.timeout_s)
                await client.send({"type": "response.create"})
                ledger.action("initial_response_create_sent")
            first_response_id = await _wait_response_created(client, after_count=0, timeout_s=args.timeout_s)
            first_render_task = asyncio.create_task(first_render_after(first_response_id))
            audio_observe_task = asyncio.create_task(
                wait_for(
                    lambda: any(
                        event.get("type") == "response.audio.delta" and _event_response_id(event) == first_response_id
                        for event in client.events.events
                    ),
                    timeout_s=min(args.timeout_s, 15.0),
                    label="initial response audio",
                )
            )

            if args.feedback_contract == "L2":
                try:
                    await audio_observe_task
                except TimeoutError:
                    # A listen-only first response cannot reach an audio
                    # playback anchor and is recorded as an explicit failure.
                    result["initial_audio_observed"] = False
                else:
                    result["initial_audio_observed"] = True
                anchor_target = case.playback_anchor_ms
                ledger.action("playback_anchor_wait", response_id=first_response_id, target_ms=anchor_target)
                await player.wait_until(first_response_id, anchor_target, args.anchor_timeout_s)
                ledger.action("playback_anchor_reached", response_id=first_response_id, target_ms=anchor_target)
                result["anchor_reached"] = True
            else:
                # L1's interrupt is tied to commit time, not to the arrival of
                # the first output packet.  Do not let a slow/no-audio response
                # shift the offered external event.
                result["anchor_reached"] = None
                elapsed_ms = (time.monotonic() - commit_at_s) * 1000.0 if commit_at_s is not None else 0.0
                await asyncio.sleep(max(0.0, case.external_interrupt_ms / 1000.0 - elapsed_ms))
                ledger.action("external_clock_interrupt_due", target_ms=case.external_interrupt_ms)
                if audio_observe_task.done():
                    result["initial_audio_observed"] = audio_observe_task.exception() is None
                else:
                    result["initial_audio_observed"] = False

            interrupt_at_s = ledger.action("barge_in_requested", response_id=first_response_id)
            await client.send({"type": "barge_in"})
            ledger.action("barge_in_sent", response_id=first_response_id)
            await client.stream_pcm16(interrupt_pcm, chunk_ms=args.chunk_ms, realtime=True)
            second_commit_at_s = ledger.action("interrupt_input_commit")
            await client.commit()
            if args.explicit_followup_response or args.explicit_all_responses:
                await _wait_input_audio_committed(client, count=2, timeout_s=args.timeout_s)
                await client.send({"type": "response.create"})
                ledger.action("followup_response_create_sent")
            result["interrupt_input_audio_ms"] = (
                len(interrupt_pcm) * 1000 / (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)
            )

            await wait_for(
                lambda: any(
                    event.get("type") == "response.done" and _event_response_id(event) == first_response_id
                    for event in client.events.events
                ),
                timeout_s=args.timeout_s,
                label="interrupted response.done",
            )
            old_done = next(
                event
                for event in reversed(client.events.events)
                if event.get("type") == "response.done" and _event_response_id(event) == first_response_id
            )
            old_done_at_s = _received_at_for_event(
                client.events.events, client.events.event_received_at_s, old_done
            )
            result["old_response_status"] = (
                (old_done.get("response") or {}).get("status") if isinstance(old_done.get("response"), dict) else None
            )
            result["old_response_status_reason"] = (
                ((old_done.get("response") or {}).get("status_details") or {}).get("reason")
                if isinstance(old_done.get("response"), dict)
                and isinstance((old_done.get("response") or {}).get("status_details"), dict)
                else None
            )
            result["interrupt_reaction_ms"] = max(0.0, (old_done_at_s - interrupt_at_s) * 1000.0)
            followup_id: str | None = None
            try:
                followup_id = await _wait_response_created(client, after_count=1, timeout_s=args.timeout_s)
                # The response identity is meaningful as soon as the server
                # emits response.created. Completion can arrive later than
                # the request timeout for long native turns.
                result["followup_response_id"] = followup_id
                await wait_for(
                    lambda: any(
                        event.get("type") == "response.done" and _event_response_id(event) == followup_id
                        for event in client.events.events
                    ),
                    timeout_s=args.timeout_s,
                    label="follow-up response.done",
                )
            except TimeoutError:
                if followup_id is None:
                    result["followup_response_id"] = None
            await asyncio.sleep(args.tail_drain_s)
            stale_events = [
                event
                for event, received_at_s in zip(client.events.events, client.events.event_received_at_s, strict=True)
                if received_at_s >= interrupt_at_s
                and event.get("type") == "response.audio.delta"
                and _event_response_id(event) == first_response_id
            ]
            result["stale_audio_delta_count"] = len(stale_events)
            result["stale_audio_ms"] = sum(
                _audio_duration_ms(event, client.events.output_sample_rate_hz) for event in stale_events
            )
            result["old_response_done_after_interrupt_ms"] = max(0.0, (old_done_at_s - interrupt_at_s) * 1000.0)
            followup_text = response_text(client.events.events, result.get("followup_response_id"))
            result["followup_text"] = followup_text
            result["context_correct"] = evaluate_context(followup_text, case.expected_text_contains)
            result["interaction_success"] = bool(
                result.get("old_response_status") == "cancelled"
                and isinstance(result.get("followup_response_id"), str)
                and result["stale_audio_delta_count"] == 0
                and not client.events.errors()
            )
            result["task_success"] = (
                bool(result["interaction_success"] and result["context_correct"] is True)
                if result["context_correct"] is not None
                else None
            )
            result["transport_success"] = not client.events.errors() and client.events.count("session.closed") == 0
            # session.close is sent below; this provisional value is corrected
            # after the close event is observed.
            metrics: list[dict[str, object]] = []
            for response_id in client.events.response_ids:
                commit_at = commit_at_s if response_id == first_response_id else second_commit_at_s
                usable_commit_at = _usable_input_commit_at(
                    client.events.events,
                    client.events.event_received_at_s,
                    response_id,
                    commit_at,
                )
                timing = client.events.timing_summary(
                    after_s=stream_started_at_s,
                    input_committed_at_s=usable_commit_at,
                    response_id=response_id,
                    measurement_origin=_timing_measurement_origin(usable_commit_at is not None),
                )
                row = {"response_id": response_id}
                row["input_commit_used_as_timing_origin"] = usable_commit_at is not None
                if isinstance(timing.get("request_metrics"), dict):
                    row.update(timing["request_metrics"])
                if isinstance(timing.get("stage0_tokens"), dict):
                    row["stage0_tokens"] = timing["stage0_tokens"]
                if isinstance(timing.get("audio_output"), dict):
                    row["audio_output"] = timing["audio_output"]
                metrics.append(row)
            result["request_metrics"] = metrics
            result["playback"] = player.clock.summary()
            if first_render_task.done() and not first_render_task.cancelled() and first_render_task.exception() is None:
                result["first_render_ms"] = first_render_task.result()
            result["transport_success"] = not client.events.errors() and client.events.count("session.closed") == 1
        finally:
            # Close the playback task before closing the session so no ACK is
            # sent on a socket that is already in the context-manager teardown.
            if player is not None:
                await player.close()
            if client.events.count("session.closed") == 0:
                try:
                    ledger.action("session_close_requested")
                    await client.close_session(timeout_s=min(args.timeout_s, 30.0))
                except Exception as close_error:
                    result.setdefault("error", f"{type(close_error).__name__}: {close_error}")
            result["transport_success"] = not client.events.errors() and client.events.count("session.closed") == 1
            await client.__aexit__(None, None, None)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if first_render_task is not None and not first_render_task.done():
            first_render_task.cancel()
            try:
                await first_render_task
            except asyncio.CancelledError:
                pass
        if audio_observe_task is not None and not audio_observe_task.done():
            audio_observe_task.cancel()
            try:
                await audio_observe_task
            except asyncio.CancelledError:
                pass
        if player is not None:
            try:
                await player.close()
            except Exception as error:
                result.setdefault("error", f"{type(error).__name__}: {error}")
        if client is not None:
            ledger.server_events(client)
        ledger.write(output_dir / f"{session_id}.events.jsonl")
    result["session_slo_pass"] = slo_pass(result, args=args)
    result["actual_duration_s"] = max(0.0, time.monotonic() - stream_started_at_s)
    return result


def slo_pass(result: dict[str, object], *, args: argparse.Namespace) -> bool:
    reaction = result.get("interrupt_reaction_ms")
    stale_ms = result.get("stale_audio_ms")
    if result.get("transport_success") is not True or result.get("interaction_success") is not True:
        return False
    if result.get("task_success") is not True:
        return False
    if args.feedback_contract == "L2" and result.get("anchor_reached") is not True:
        return False
    return (
        isinstance(reaction, int | float)
        and float(reaction) <= args.max_interrupt_reaction_ms
        and isinstance(stale_ms, int | float)
        and float(stale_ms) <= args.max_stale_audio_ms
    )


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    clean = sorted(value for value in values if math.isfinite(value))

    def percentile(fraction: float) -> float | None:
        if not clean:
            return None
        position = (len(clean) - 1) * fraction
        lower = math.floor(position)
        upper = min(lower + 1, len(clean) - 1)
        return clean[lower] + (clean[upper] - clean[lower]) * (position - lower)

    return {
        "count": len(clean),
        "mean": sum(clean) / len(clean) if clean else None,
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(clean) if clean else None,
    }


def _numbers(rows: list[dict[str, object]], key: str) -> list[float]:
    return [
        float(value) for row in rows if isinstance((value := row.get(key)), int | float) and math.isfinite(float(value))
    ]


def summarize_results(
    results: list[dict[str, object]],
    *,
    request_rate: float,
    arrival_window_s: float,
    wall_time_s: float,
    feedback_contract: str,
) -> dict[str, object]:
    known_context = [result for result in results if result.get("context_correct") is not None]
    metrics = [metric for result in results for metric in result.get("request_metrics", []) if isinstance(metric, dict)]
    audio_outputs = [metric["audio_output"] for metric in metrics if isinstance(metric.get("audio_output"), dict)]
    playout_intervals = sum(max(0, int(output.get("chunk_count", 0) or 0) - 1) for output in audio_outputs)
    playout_misses = sum(int(output.get("playout_deadline_miss_count", 0) or 0) for output in audio_outputs)
    offered_window_s = sum(float(result.get("session_window_s", 0.0) or 0.0) for result in results)
    good_window_s = sum(
        float(result.get("session_window_s", 0.0) or 0.0)
        for result in results
        if result.get("session_slo_pass") is True
    )
    return {
        "workload_mode": "full_duplex_e2e",
        "feedback_contract": feedback_contract,
        "offered_request_rate_sessions_per_s": request_rate,
        "arrival_window_s": arrival_window_s,
        "request_count": len(results),
        "transport_success_count": sum(result.get("transport_success") is True for result in results),
        "interaction_success_count": sum(result.get("interaction_success") is True for result in results),
        "task_success_count": sum(result.get("task_success") is True for result in results),
        "task_unknown_count": sum(result.get("task_success") is None for result in results),
        "failure_count": sum(result.get("session_slo_pass") is not True for result in results),
        "task_success_rate": (
            sum(result.get("task_success") is True for result in known_context) / len(known_context)
            if known_context
            else None
        ),
        "context_correct_rate": (
            sum(result.get("context_correct") is True for result in known_context) / len(known_context)
            if known_context
            else None
        ),
        "session_slo_pass_count": sum(result.get("session_slo_pass") is True for result in results),
        "session_slo_pass_rate": sum(result.get("session_slo_pass") is True for result in results) / len(results)
        if results
        else 0.0,
        "offered_session_window_s": offered_window_s,
        "good_session_window_s": good_window_s,
        "goodput_session_window_fraction": good_window_s / offered_window_s if offered_window_s else 0.0,
        "completed_goodput_per_arrival_window_s": sum(result.get("session_slo_pass") is True for result in results)
        / arrival_window_s,
        "arrival_schedule_lag_ms": _numeric_summary(_numbers(results, "arrival_schedule_lag_ms")),
        "interrupt_reaction_ms": _numeric_summary(_numbers(results, "interrupt_reaction_ms")),
        "first_render_ms": _numeric_summary(_numbers(results, "first_render_ms")),
        "stale_audio_delta_count": sum(int(result.get("stale_audio_delta_count", 0) or 0) for result in results),
        "stale_audio_ms": sum(float(result.get("stale_audio_ms", 0.0) or 0.0) for result in results),
        "request_ttfp_ms": _numeric_summary(_numbers(metrics, "ttfp_ms")),
        "streaming_audio_rtf": _numeric_summary(_numbers(audio_outputs, "streaming_rtf")),
        "playout_interval_count": playout_intervals,
        "playout_deadline_miss_count": playout_misses,
        "playout_deadline_miss_rate": playout_misses / playout_intervals if playout_intervals else None,
        "playback_underrun_ms": _numeric_summary(
            [
                float(playback["underrun_ms"])
                for result in results
                if isinstance((playback := result.get("playback")), dict)
                and isinstance(playback.get("underrun_ms"), int | float)
            ]
        ),
        "wall_time_s": wall_time_s,
        "non_pd_production_path": True,
    }


async def run_schedule(
    schedule: list[ScheduledE2ERequest], *, args: argparse.Namespace, output_dir: Path
) -> tuple[list[dict[str, object]], float]:
    ref_audio = reference_audio_data_url(str(args.ref_audio))
    if ref_audio is None:
        raise ValueError("ref_audio is required")
    run_id = (
        args.run_id
        or f"seed{args.seed}-rate{round(args.request_rate * 1_000_000)}-duration{round(args.duration_s * 1000)}"
    )
    loop = asyncio.get_running_loop()
    origin = loop.time()

    async def launch(request: ScheduledE2ERequest) -> dict[str, object]:
        scheduled = origin + request.arrival_s
        await asyncio.sleep(max(0.0, scheduled - loop.time()))
        actual_start = loop.time()
        result = await _run_case(request, args=args, ref_audio=ref_audio, run_id=run_id, output_dir=output_dir)
        result["actual_start_s"] = actual_start - origin
        result["actual_finish_s"] = loop.time() - origin
        result["arrival_schedule_lag_ms"] = max(0.0, actual_start - scheduled) * 1000.0
        return result

    results = await asyncio.gather(*(asyncio.create_task(launch(request)) for request in schedule))
    return results, loop.time() - origin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--request-rate", type=float, required=True)
    parser.add_argument("--duration-s", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20_260_901)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--schedule-only", action="store_true")
    parser.add_argument("--feedback-contract", choices=("L1", "L2"), default="L2")
    parser.add_argument("--url", default="ws://127.0.0.1:8113/v1/realtime")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--tail-drain-s", type=float, default=2.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--anchor-timeout-s", type=float, default=30.0)
    parser.add_argument("--playback-initial-buffer-ms", type=int, default=300)
    parser.add_argument("--playback-progress-ms", type=int, default=80)
    parser.add_argument("--explicit-followup-response", action="store_true")
    parser.add_argument("--explicit-all-responses", action="store_true")
    parser.add_argument("--max-interrupt-reaction-ms", type=float, default=2_000.0)
    parser.add_argument("--max-stale-audio-ms", type=float, default=0.0)
    args = parser.parse_args()
    if args.ref_audio is None:
        args.ref_audio = args.model / "assets" / "HT_ref_audio.wav"
    if args.chunk_ms <= 0 or args.chunk_ms > 1000:
        parser.error("--chunk-ms must be in [1, 1000]")
    if args.duration_s <= 0 or args.request_rate <= 0 or args.timeout_s <= 0 or args.anchor_timeout_s <= 0:
        parser.error("rate, duration, timeout, and anchor timeout must be positive")
    if args.tail_drain_s < 0 or args.playback_initial_buffer_ms < 0 or args.playback_progress_ms <= 0:
        parser.error("tail drain/buffer must be non-negative and progress positive")
    if args.max_interrupt_reaction_ms < 0 or args.max_stale_audio_ms < 0:
        parser.error("SLO limits must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to mix artifacts in non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_manifest(args.manifest)
    schedule = build_schedule(cases, request_rate=args.request_rate, duration_s=args.duration_s, seed=args.seed)
    schedule_payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest": str(args.manifest.expanduser().resolve()),
        "seed": args.seed,
        "request_rate_sessions_per_s": args.request_rate,
        "arrival_window_s": args.duration_s,
        "feedback_contract": args.feedback_contract,
        "request_count": len(schedule),
        "requests": [request.as_dict() for request in schedule],
    }
    (args.output_dir / "schedule.json").write_text(
        json.dumps(schedule_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.schedule_only:
        print(json.dumps(schedule_payload, ensure_ascii=False, indent=2))
        return
    if not args.model.exists() or not args.ref_audio.is_file():
        raise FileNotFoundError(f"model/ref audio does not exist: {args.model}, {args.ref_audio}")
    results, wall_time_s = asyncio.run(run_schedule(schedule, args=args, output_dir=args.output_dir))
    with (args.output_dir / "sessions.jsonl").open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = summarize_results(
        results,
        request_rate=args.request_rate,
        arrival_window_s=args.duration_s,
        wall_time_s=wall_time_s,
        feedback_contract=args.feedback_contract,
    )
    summary.update(
        {
            "seed": args.seed,
            "manifest": str(args.manifest.expanduser().resolve()),
            "model": str(args.model),
            "chunk_ms": args.chunk_ms,
            "tail_drain_s": args.tail_drain_s,
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
