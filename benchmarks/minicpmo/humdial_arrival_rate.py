# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Open-arrival MiniCPM-o native-duplex benchmark over HumDial audio.

The offered load is measured in newly opened sessions per second.  Every
session still uploads its source WAV on an absolute 200 ms media clock, so the
benchmark never manufactures load by accelerating user audio.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import random
import time
import wave
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from vllm_omni.experimental.fullduplex.client import (
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    RealtimeDuplexClient,
    build_realtime_url,
    read_pcm16_wav,
    reference_audio_data_url,
    summarize_session_request_metrics,
)
from vllm_omni.experimental.fullduplex.client import (
    chunk_period_ms as negotiated_chunk_period_ms,
)

DEFAULT_DATASET_ROOT = Path(
    "/mnt/nvme1n1/ml_research/linbinbin1/src-omni-modal/Humdial-Track2-Test"
)
DEFAULT_MODEL = Path(
    "/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/"
    "models--OpenBMB--MiniCPM-o-4_5"
)
DEFAULT_SEED = 20_260_901
SCHEDULE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HumDialCase:
    path: Path
    relative_path: str
    language: str
    scenario: str
    duration_s: float

    def as_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "language": self.language,
            "scenario": self.scenario,
            "duration_s": round(self.duration_s, 6),
        }


@dataclass(frozen=True)
class ScheduledRequest:
    index: int
    arrival_s: float
    case: HumDialCase

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "arrival_s": round(self.arrival_s, 9),
            **self.case.as_dict(),
        }


def _stream_seed(seed: int, stream_offset: int) -> int:
    """Keep deterministic random streams separate without hashing state."""
    return seed + stream_offset


def _wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError(f"HumDial WAV must be mono: {path}")
        if wav_file.getsampwidth() != PCM16_BYTES_PER_SAMPLE:
            raise ValueError(f"HumDial WAV must be PCM16: {path}")
        if wav_file.getframerate() != PCM16_SAMPLE_RATE:
            raise ValueError(f"HumDial WAV must be 16 kHz: {path}")
        if wav_file.getcomptype() != "NONE":
            raise ValueError(f"HumDial WAV must be uncompressed PCM: {path}")
        return wav_file.getnframes() / wav_file.getframerate()


def discover_humdial_cases(root: Path) -> list[HumDialCase]:
    """Discover non-clean Track-2 test cases with stable relative identities."""
    test_root = root.expanduser().resolve() / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(f"HumDial test directory does not exist: {test_root}")
    cases: list[HumDialCase] = []
    for path in sorted(test_root.glob("*_test_nondev/*/*.wav")):
        if path.name.startswith("clean_"):
            continue
        language_dir = path.parent.parent.name
        language = language_dir.split("_", 1)[0]
        cases.append(
            HumDialCase(
                path=path,
                relative_path=path.relative_to(root.resolve()).as_posix(),
                language=language,
                scenario=path.parent.name,
                duration_s=_wav_duration_s(path),
            )
        )
    if not cases:
        raise ValueError(f"No non-clean HumDial test WAVs found below {test_root}")
    return cases


def _proportional_allocation(
    sizes: dict[str, int],
    count: int,
    *,
    rng: random.Random,
) -> dict[str, int]:
    total = sum(sizes.values())
    if count < 0 or count > total:
        raise ValueError(f"sample count {count} is outside available population {total}")
    allocation = {key: count * size // total for key, size in sizes.items()}
    remaining = count - sum(allocation.values())
    tie_breakers = {key: rng.random() for key in sizes}
    ranked = sorted(
        sizes,
        key=lambda key: (
            -(count * sizes[key] / total - allocation[key]),
            tie_breakers[key],
            key,
        ),
    )
    for key in ranked:
        if remaining <= 0:
            break
        if allocation[key] < sizes[key]:
            allocation[key] += 1
            remaining -= 1
    if remaining:
        raise AssertionError(f"failed to allocate {remaining} stratified samples")
    return allocation


def _duration_buckets(cases: list[HumDialCase], bucket_count: int = 4) -> dict[str, list[HumDialCase]]:
    ordered = sorted(cases, key=lambda case: (case.duration_s, case.relative_path))
    buckets: dict[str, list[HumDialCase]] = defaultdict(list)
    for index, case in enumerate(ordered):
        bucket = min(bucket_count - 1, index * bucket_count // len(ordered))
        buckets[str(bucket)].append(case)
    return dict(buckets)


def stratified_sample(cases: list[HumDialCase], count: int, *, seed: int) -> list[HumDialCase]:
    """Sample without replacement by language/scenario, then duration quartile."""
    if count < 1:
        raise ValueError("sample count must be positive")
    rng = random.Random(_stream_seed(seed, 0))
    coarse: dict[str, list[HumDialCase]] = defaultdict(list)
    for case in cases:
        coarse[f"{case.language}/{case.scenario}"].append(case)
    coarse_allocation = _proportional_allocation(
        {key: len(group) for key, group in coarse.items()},
        count,
        rng=rng,
    )
    selected: list[HumDialCase] = []
    for coarse_key in sorted(coarse):
        target = coarse_allocation[coarse_key]
        if not target:
            continue
        buckets = _duration_buckets(coarse[coarse_key])
        bucket_allocation = _proportional_allocation(
            {key: len(group) for key, group in buckets.items()},
            target,
            rng=rng,
        )
        for bucket_key in sorted(buckets):
            group = list(buckets[bucket_key])
            rng.shuffle(group)
            selected.extend(group[: bucket_allocation[bucket_key]])
    if len(selected) != count or len({case.relative_path for case in selected}) != count:
        raise AssertionError("stratified HumDial sampling did not produce the requested unique count")
    rng.shuffle(selected)
    return selected


def build_schedule(
    cases: list[HumDialCase],
    *,
    request_rate: float,
    duration_s: float,
    seed: int,
) -> list[ScheduledRequest]:
    """Build an exact-count conditional-Poisson arrival trace."""
    if not math.isfinite(request_rate) or request_rate <= 0:
        raise ValueError("request_rate must be finite and positive")
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    request_count = max(1, round(request_rate * duration_s))
    sampled = stratified_sample(cases, request_count, seed=seed)
    arrival_rng = random.Random(_stream_seed(seed, 1_000_003))
    arrivals = sorted(arrival_rng.random() * duration_s for _ in range(request_count))
    return [
        ScheduledRequest(index=index, arrival_s=arrival_s, case=case)
        for index, (arrival_s, case) in enumerate(zip(arrivals, sampled, strict=True))
    ]


def schedule_payload(
    schedule: list[ScheduledRequest],
    *,
    dataset_root: Path,
    request_rate: float,
    duration_s: float,
    seed: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "dataset_root": str(dataset_root.resolve()),
        "seed": seed,
        "request_rate_sessions_per_s": request_rate,
        "arrival_window_s": duration_s,
        "request_count": len(schedule),
        "realized_request_rate_sessions_per_s": len(schedule) / duration_s,
        "requests": [request.as_dict() for request in schedule],
    }
    return payload


@dataclass
class _PlaybackResponse:
    queued_ms: float = 0.0
    played_ms: float = 0.0
    updated_at_s: float = 0.0
    resume_at_s: float | None = None
    playing: bool = False
    audio_done: bool = False
    cancelled: bool = False
    final_acked: bool = False
    next_progress_ms: int = 80
    underrun_started_at_s: float | None = None
    underrun_ms: float = 0.0


class BrowserPlaybackClock:
    """Approximate the demo AudioWorklet's buffering and ACK media clock."""

    def __init__(self, *, initial_buffer_ms: int = 300, progress_ms: int = 80) -> None:
        if initial_buffer_ms < 0 or progress_ms <= 0:
            raise ValueError("playback buffer must be non-negative and progress interval positive")
        self.initial_buffer_ms = initial_buffer_ms
        self.progress_ms = progress_ms
        self.cursor = 0
        self.responses: dict[str, _PlaybackResponse] = {}

    @staticmethod
    def _is_cancelled(event: dict[str, object]) -> bool:
        response = event.get("response")
        return isinstance(response, dict) and response.get("status") == "cancelled"

    @staticmethod
    def _audio_duration_ms(event: dict[str, object], default_rate: int) -> float:
        encoded = event.get("delta") or event.get("audio")
        if not isinstance(encoded, str) or not encoded:
            return 0.0
        raw = base64.b64decode(encoded)
        rate = event.get("sample_rate_hz")
        sample_rate = rate if isinstance(rate, int) and rate > 0 else default_rate
        return len(raw) * 1000.0 / (sample_rate * PCM16_BYTES_PER_SAMPLE)

    def _advance(self, state: _PlaybackResponse, now_s: float) -> None:
        if state.updated_at_s <= 0:
            state.updated_at_s = now_s
            return
        cursor_s = state.updated_at_s
        if not state.cancelled and not state.playing and state.resume_at_s is not None and now_s >= state.resume_at_s:
            if state.underrun_started_at_s is not None:
                state.underrun_ms += max(0.0, state.resume_at_s - state.underrun_started_at_s) * 1000.0
                state.underrun_started_at_s = None
            cursor_s = max(cursor_s, state.resume_at_s)
            state.resume_at_s = None
            state.playing = True
        if state.playing and now_s > cursor_s:
            available_ms = max(0.0, state.queued_ms - state.played_ms)
            elapsed_ms = (now_s - cursor_s) * 1000.0
            consumed_ms = min(available_ms, elapsed_ms)
            state.played_ms += consumed_ms
            if consumed_ms >= available_ms - 0.001:
                exhausted_at_s = cursor_s + consumed_ms / 1000.0
                state.playing = False
                if not state.audio_done and not state.cancelled:
                    state.underrun_started_at_s = exhausted_at_s
        state.updated_at_s = now_s

    def _ingest(self, client: RealtimeDuplexClient) -> None:
        events = client.events
        while self.cursor < len(events.events):
            index, self.cursor = self.cursor, self.cursor + 1
            event = events.events[index]
            response_id = events.response_id(event)
            if not response_id:
                continue
            received_at_s = events.event_received_at_s[index]
            state = self.responses.setdefault(
                response_id,
                _PlaybackResponse(updated_at_s=received_at_s, next_progress_ms=self.progress_ms),
            )
            self._advance(state, received_at_s)
            event_type = event.get("type")
            if event_type == "response.audio.delta":
                was_empty = state.queued_ms <= state.played_ms + 0.001
                state.queued_ms += self._audio_duration_ms(event, events.output_sample_rate_hz)
                if was_empty and not state.playing and state.resume_at_s is None:
                    state.resume_at_s = received_at_s + self.initial_buffer_ms / 1000.0
            elif event_type == "response.audio.done":
                state.audio_done = True
            elif event_type == "response.done" and self._is_cancelled(event):
                state.cancelled = True
                state.playing = False
                state.resume_at_s = None
            elif event_type == "output_audio_buffer.cleared":
                state.cancelled = True
                state.playing = False
                state.resume_at_s = None

    async def step(self, client: RealtimeDuplexClient) -> None:
        self._ingest(client)
        now_s = time.monotonic()
        for response_id, state in self.responses.items():
            self._advance(state, now_s)
            if state.cancelled or state.queued_ms <= 0:
                continue
            played_ms = min(int(state.played_ms), int(state.queued_ms))
            if played_ms >= state.next_progress_ms and not state.final_acked:
                await client.send_playback_ack(response_id, played_ms, commit=False)
                while state.next_progress_ms <= played_ms:
                    state.next_progress_ms += self.progress_ms
            drained = state.audio_done and state.played_ms >= state.queued_ms - 0.001
            if drained and not state.final_acked:
                await client.send_playback_ack(response_id, int(state.queued_ms), commit=True)
                state.final_acked = True

    async def run(self, client: RealtimeDuplexClient, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self.step(client)
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.02)
            except asyncio.TimeoutError:
                pass
        await self.step(client)

    def rendered_ms(self, response_id: str, *, now_s: float | None = None) -> float | None:
        """Return the simulated render cursor for one response.

        The E2E controller uses this accessor to wait for a playback anchor.
        It deliberately exposes the value as *simulated* rendering progress;
        the benchmark must not label it as a physical speaker/audio-device
        measurement.
        """
        state = self.responses.get(response_id)
        if state is None:
            return None
        self._advance(state, time.monotonic() if now_s is None else now_s)
        return min(state.played_ms, state.queued_ms)

    def queued_ms(self, response_id: str) -> float | None:
        """Return the amount of response audio observed by the simulator."""
        state = self.responses.get(response_id)
        return None if state is None else state.queued_ms

    def summary(self) -> dict[str, object]:
        return {
            "response_count": len(self.responses),
            "final_ack_count": sum(state.final_acked for state in self.responses.values()),
            "cancelled_count": sum(state.cancelled for state in self.responses.values()),
            "queued_audio_ms": round(sum(state.queued_ms for state in self.responses.values()), 3),
            "played_audio_ms": round(sum(state.played_ms for state in self.responses.values()), 3),
            "underrun_ms": round(sum(state.underrun_ms for state in self.responses.values()), 3),
            "initial_buffer_ms": self.initial_buffer_ms,
            "progress_interval_ms": self.progress_ms,
        }


async def _stream_pcm16_absolute(
    client: RealtimeDuplexClient,
    pcm16: bytes,
    *,
    chunk_ms: int,
    model_unit_ms: int,
) -> tuple[dict[str, object], list[float]]:
    chunk_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * chunk_ms // 1000
    if chunk_bytes <= 0:
        raise ValueError("chunk_ms is too small")
    if model_unit_ms <= 0:
        raise ValueError("model_unit_ms must be positive")
    started_at_s = time.monotonic()
    audio_end_ms = 0
    lags_ms: list[float] = []
    send_ms: list[float] = []
    model_unit_sent_at_s: list[float] = []
    chunks = 0
    for offset in range(0, len(pcm16), chunk_bytes):
        scheduled_at_s = started_at_s + offset / (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)
        await asyncio.sleep(max(0.0, scheduled_at_s - time.monotonic()))
        send_started_at_s = time.monotonic()
        chunk = pcm16[offset : offset + chunk_bytes]
        duration_ms = len(chunk) * 1000 // (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)
        audio_end_ms += duration_ms
        await client.send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("ascii"),
                "input_audio_format": "pcm16",
                "sample_rate_hz": PCM16_SAMPLE_RATE,
                "duration_ms": duration_ms,
                "audio_end_ms": audio_end_ms,
            }
        )
        sent_at_s = time.monotonic()
        chunks += 1
        lags_ms.append(max(0.0, send_started_at_s - scheduled_at_s) * 1000.0)
        send_ms.append(max(0.0, sent_at_s - send_started_at_s) * 1000.0)
        while audio_end_ms >= (len(model_unit_sent_at_s) + 1) * model_unit_ms:
            model_unit_sent_at_s.append(sent_at_s)
    target_end_s = started_at_s + len(pcm16) / (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)
    await asyncio.sleep(max(0.0, target_end_s - time.monotonic()))
    return (
        {
            "chunk_count": chunks,
            "input_audio_ms": audio_end_ms,
            "model_unit_ms": model_unit_ms,
            "complete_model_unit_count": len(model_unit_sent_at_s),
            "mean_send_lag_ms": sum(lags_ms) / len(lags_ms) if lags_ms else 0.0,
            "max_send_lag_ms": max(lags_ms, default=0.0),
            "mean_send_duration_ms": sum(send_ms) / len(send_ms) if send_ms else 0.0,
            "max_send_duration_ms": max(send_ms, default=0.0),
        },
        model_unit_sent_at_s,
    )


def _model_unit_decision_metrics(
    collector,
    *,
    model_unit_sent_at_s: list[float],
    model_unit_ms: int,
) -> dict[str, object]:
    """Correlate client input units with decisions by client media-clock end."""
    decisions: dict[int, tuple[str, str | None, float]] = {}
    unpaired_decision_count = 0

    def source_int(metadata: object, name: str) -> int | None:
        if not isinstance(metadata, dict):
            return None
        nested = metadata.get("vllm_omni")
        raw_value = nested.get(name) if isinstance(nested, dict) else None
        if raw_value is None:
            raw_value = metadata.get(name)
        if isinstance(raw_value, int) and not isinstance(raw_value, bool):
            return raw_value
        return None

    for event, received_at_s in zip(
        collector.events,
        collector.event_received_at_s,
        strict=True,
    ):
        event_type = event.get("type")
        if event_type == "response.listen":
            response = event.get("response")
            response_metadata = (
                response.get("metadata") if isinstance(response, dict) else None
            )
            projected = response_metadata if isinstance(response_metadata, dict) else event
            if projected.get("buffering") is True or projected.get("model_listen") is False:
                continue
            decision = ("listen", collector.response_id(event), received_at_s)
            source_audio_end_ms = source_int(projected, "source_audio_end_ms")
        elif event_type == "response.audio.delta":
            delta = event.get("delta") or event.get("audio")
            if not isinstance(delta, str) or not delta:
                continue
            decision = ("audio", collector.response_id(event), received_at_s)
            metadata = event.get("metadata")
            source_audio_end_ms = source_int(event, "source_audio_end_ms")
            if source_audio_end_ms is None:
                source_audio_end_ms = source_int(metadata, "source_audio_end_ms")
        else:
            continue

        # Model-generated silence continuations have a server input sequence
        # but no client media-clock boundary. They advance an active response;
        # they are not new client-input decisions and are intentionally skipped.
        if source_audio_end_ms is None or source_audio_end_ms < 0:
            continue
        if source_audio_end_ms == 0 or source_audio_end_ms % model_unit_ms:
            continue
        unit_index = source_audio_end_ms // model_unit_ms - 1
        if not 0 <= unit_index < len(model_unit_sent_at_s):
            unpaired_decision_count += 1
            continue
        seq = unit_index + 1
        # One model unit can expose more than one packet for the same decision.
        # Latency is defined by the first observable non-buffering decision.
        if seq not in decisions or received_at_s < decisions[seq][2]:
            decisions[seq] = decision

    rows = []
    for seq in sorted(decisions):
        unit_index = seq - 1
        decision_kind, response_id, received_at_s = decisions[seq]
        rows.append(
            {
                "input_unit_index": unit_index,
                "input_audio_end_ms": (unit_index + 1) * model_unit_ms,
                "decision_kind": decision_kind,
                "response_id": response_id,
                "decision_latency_ms": round(
                    (received_at_s - model_unit_sent_at_s[unit_index]) * 1000.0,
                    3,
                ),
            }
        )
    return {
        "measurement_origin": (
            "client send completion for source_audio_end_ms to the correlated "
            "non-buffering listen/audio decision receive"
        ),
        "model_unit_ms": model_unit_ms,
        "input_unit_count": len(model_unit_sent_at_s),
        "decision_count": len(decisions),
        "unpaired_input_unit_count": len(model_unit_sent_at_s) - len(decisions),
        "unpaired_decision_count": unpaired_decision_count,
        "rows": rows,
    }


def _response_metrics(
    client: RealtimeDuplexClient,
    *,
    stream_started_at_s: float,
    session_id: str,
) -> list[dict[str, object]]:
    metrics: list[dict[str, object]] = []
    measurement_origin = {
        "ttft": "response.created client receive to first non-empty text delta",
        "ttfp": "response.created client receive to first audio packet",
        "rtf": "response.created client receive to last audio packet divided by emitted audio duration",
    }
    for request_index, response_id in enumerate(client.events.response_ids):
        timing = client.events.timing_summary(
            after_s=stream_started_at_s,
            response_id=response_id,
            measurement_origin=measurement_origin,
        )
        row: dict[str, object] = {
            "session_id": session_id,
            "request_index": request_index,
            "response_id": response_id,
        }
        raw_request = timing.get("request_metrics")
        if isinstance(raw_request, dict):
            row.update(raw_request)
        stage0 = timing.get("stage0_tokens")
        if isinstance(stage0, dict):
            row["stage0_tokens"] = stage0
        audio_output = timing.get("audio_output")
        if isinstance(audio_output, dict):
            row["audio_output"] = audio_output
        metrics.append(row)
    return metrics


async def run_one_session(
    request: ScheduledRequest,
    *,
    pcm16: bytes,
    url: str,
    model: str,
    ref_audio_data_url: str,
    run_id: str,
    chunk_ms: int,
    tail_drain_s: float,
    timeout_s: float,
    playback_initial_buffer_ms: int,
    playback_progress_ms: int,
) -> dict[str, object]:
    session_id = f"humdial-{run_id}-{request.index:05d}"
    started_at_s = time.monotonic()
    result: dict[str, object] = {
        "index": request.index,
        "session_id": session_id,
        "relative_path": request.case.relative_path,
        "language": request.case.language,
        "scenario": request.case.scenario,
        "input_duration_s": request.case.duration_s,
        "success": False,
    }
    client: RealtimeDuplexClient | None = None
    player_task: asyncio.Task[None] | None = None
    player_stop = asyncio.Event()
    player = BrowserPlaybackClock(
        initial_buffer_ms=playback_initial_buffer_ms,
        progress_ms=playback_progress_ms,
    )
    try:
        realtime_url = build_realtime_url(
            url,
            model,
            autostart=False,
            native_duplex=True,
            session_id=session_id,
        )
        async with RealtimeDuplexClient(realtime_url) as client:
            await client.configure(
                model,
                ref_audio=ref_audio_data_url,
                native_duplex=True,
                auto_response=True,
                temperature=0.0,
                session_id=session_id,
                idle_timeout_s=max(timeout_s, request.case.duration_s + tail_drain_s + 30.0),
                timeout_s=min(timeout_s, 60.0),
            )
            player_task = asyncio.create_task(player.run(client, player_stop))
            stream_started_at_s = time.monotonic()
            result["connection_setup_ms"] = (stream_started_at_s - started_at_s) * 1000.0
            model_unit_ms = negotiated_chunk_period_ms(client.events.events)
            input_summary, model_unit_sent_at_s = await _stream_pcm16_absolute(
                client,
                pcm16,
                chunk_ms=chunk_ms,
                model_unit_ms=model_unit_ms,
            )
            result["input"] = input_summary
            await asyncio.sleep(tail_drain_s)
            player_stop.set()
            await player_task
            player_task = None
            await client.close_session(timeout_s=min(timeout_s, 30.0))
            errors = client.events.errors()
            request_metrics = _response_metrics(
                client,
                stream_started_at_s=stream_started_at_s,
                session_id=session_id,
            )
            model_unit_decisions = _model_unit_decision_metrics(
                client.events,
                model_unit_sent_at_s=model_unit_sent_at_s,
                model_unit_ms=model_unit_ms,
            )
            result.update(
                {
                    "success": not errors and client.events.count("session.closed") == 1,
                    "error_events": errors,
                    "response_count": len(client.events.response_ids),
                    "model_listen_count": client.events.count("response.listen"),
                    "response_cancelled_count": sum(
                        BrowserPlaybackClock._is_cancelled(event)
                        for event in client.events.events
                        if event.get("type") == "response.done"
                    ),
                    "request_metrics": request_metrics,
                    "model_unit_decisions": model_unit_decisions,
                    "session_request_metrics": summarize_session_request_metrics(
                        [row for row in request_metrics if isinstance(row.get("rtf"), int | float)],
                        session_id=session_id,
                    ),
                    "playback": player.summary(),
                }
            )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        player_stop.set()
        if player_task is not None:
            player_task.cancel()
            try:
                await player_task
            except asyncio.CancelledError:
                pass
        result["latency_s"] = time.monotonic() - started_at_s
    return result


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
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(clean) if clean else None,
    }


def _maximum_concurrency(results: list[dict[str, object]]) -> int:
    boundaries: list[tuple[float, int]] = []
    for result in results:
        started = result.get("actual_start_s")
        finished = result.get("actual_finish_s")
        if isinstance(started, int | float) and isinstance(finished, int | float):
            boundaries.extend(((float(started), 1), (float(finished), -1)))
    current = maximum = 0
    for _, delta in sorted(boundaries, key=lambda item: (item[0], item[1])):
        current += delta
        maximum = max(maximum, current)
    return maximum


def summarize_results(
    results: list[dict[str, object]],
    *,
    request_rate: float,
    arrival_window_s: float,
    wall_time_s: float,
) -> dict[str, object]:
    request_metrics = [
        metric
        for result in results
        for metric in (result.get("request_metrics") or [])
        if isinstance(metric, dict)
    ]
    audio_outputs = [
        output
        for metric in request_metrics
        if isinstance((output := metric.get("audio_output")), dict)
    ]
    model_unit_metrics = [
        metrics
        for result in results
        if isinstance((metrics := result.get("model_unit_decisions")), dict)
    ]
    model_unit_rows = [
        row
        for metrics in model_unit_metrics
        for row in (metrics.get("rows") or [])
        if isinstance(row, dict)
    ]
    def numbers(rows: list[dict[str, object]], key: str) -> list[float]:
        return [float(value) for row in rows if isinstance((value := row.get(key)), int | float)]

    playout_interval_count = sum(
        max(0, int(output.get("chunk_count", 0) or 0) - 1)
        for output in audio_outputs
    )
    playout_deadline_miss_count = sum(numbers(audio_outputs, "playout_deadline_miss_count"))

    return {
        "offered_request_rate_sessions_per_s": request_rate,
        "request_count": len(results),
        "success_count": sum(result.get("success") is True for result in results),
        "failure_count": sum(result.get("success") is not True for result in results),
        "success_rate": sum(result.get("success") is True for result in results) / len(results) if results else 0.0,
        "completed_goodput_per_arrival_window_s": (
            sum(result.get("success") is True for result in results) / arrival_window_s
        ),
        "wall_time_s": wall_time_s,
        "maximum_client_concurrency": _maximum_concurrency(results),
        "arrival_schedule_lag_ms": _numeric_summary(numbers(results, "arrival_schedule_lag_ms")),
        "session_latency_s": _numeric_summary(numbers(results, "latency_s")),
        "connection_setup_ms": _numeric_summary(numbers(results, "connection_setup_ms")),
        "request_ttfp_ms": _numeric_summary(numbers(request_metrics, "ttfp_ms")),
        "request_rtf": _numeric_summary(numbers(request_metrics, "rtf")),
        "model_unit_decision_latency_ms": _numeric_summary(
            numbers(model_unit_rows, "decision_latency_ms")
        ),
        "model_unit_audio_decision_latency_ms": _numeric_summary(
            numbers(
                [row for row in model_unit_rows if row.get("decision_kind") == "audio"],
                "decision_latency_ms",
            )
        ),
        "model_unit_listen_decision_latency_ms": _numeric_summary(
            numbers(
                [row for row in model_unit_rows if row.get("decision_kind") == "listen"],
                "decision_latency_ms",
            )
        ),
        "model_unit_input_count": sum(
            int(metrics.get("input_unit_count", 0) or 0) for metrics in model_unit_metrics
        ),
        "model_unit_decision_count": sum(
            int(metrics.get("decision_count", 0) or 0) for metrics in model_unit_metrics
        ),
        "model_unit_unpaired_input_count": sum(
            int(metrics.get("unpaired_input_unit_count", 0) or 0)
            for metrics in model_unit_metrics
        ),
        "model_unit_unpaired_decision_count": sum(
            int(metrics.get("unpaired_decision_count", 0) or 0)
            for metrics in model_unit_metrics
        ),
        "streaming_audio_rtf": _numeric_summary(numbers(audio_outputs, "streaming_rtf")),
        "audio_response_count": len(audio_outputs),
        "playout_interval_count": playout_interval_count,
        "playout_deadline_miss_count": playout_deadline_miss_count,
        "playout_deadline_miss_rate": (
            playout_deadline_miss_count / playout_interval_count if playout_interval_count else None
        ),
        "playback_underrun_ms": _numeric_summary(
            [
                float(playback["underrun_ms"])
                for result in results
                if isinstance((playback := result.get("playback")), dict)
                and isinstance(playback.get("underrun_ms"), int | float)
            ]
        ),
    }


async def run_schedule(
    schedule: list[ScheduledRequest],
    *,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], float]:
    unique_paths = {request.case.path for request in schedule}
    audio_by_path = {path: read_pcm16_wav(path) for path in unique_paths}
    ref_audio = reference_audio_data_url(str(args.ref_audio))
    if ref_audio is None:
        raise ValueError("ref_audio is required")
    run_id = args.run_id or (
        f"seed{args.seed}-rate{round(args.request_rate * 1_000_000)}-duration{round(args.duration_s * 1000)}"
    )
    loop = asyncio.get_running_loop()
    origin_s = loop.time()

    async def launch(request: ScheduledRequest) -> dict[str, object]:
        scheduled_at_s = origin_s + request.arrival_s
        await asyncio.sleep(max(0.0, scheduled_at_s - loop.time()))
        actual_start_s = loop.time()
        result = await run_one_session(
            request,
            pcm16=audio_by_path[request.case.path],
            url=args.url,
            model=str(args.model),
            ref_audio_data_url=ref_audio,
            run_id=run_id,
            chunk_ms=args.chunk_ms,
            tail_drain_s=args.tail_drain_s,
            timeout_s=args.timeout_s,
            playback_initial_buffer_ms=args.playback_initial_buffer_ms,
            playback_progress_ms=args.playback_progress_ms,
        )
        result["scheduled_arrival_s"] = request.arrival_s
        result["actual_start_s"] = actual_start_s - origin_s
        result["actual_finish_s"] = loop.time() - origin_s
        result["arrival_schedule_lag_ms"] = max(0.0, actual_start_s - scheduled_at_s) * 1000.0
        return result

    tasks = [asyncio.create_task(launch(request)) for request in schedule]
    results = await asyncio.gather(*tasks)
    return results, loop.time() - origin_s


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--request-rate", type=float, required=True, help="New duplex sessions per second.")
    parser.add_argument("--duration-s", type=float, default=300.0, help="Arrival window duration.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--schedule-only", action="store_true")
    parser.add_argument("--url", default="ws://127.0.0.1:8113/v1/realtime")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--run-id", help="Session-id namespace; defaults to the seed/rate/duration tuple.")
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--tail-drain-s", type=float, default=2.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--playback-initial-buffer-ms", type=int, default=300)
    parser.add_argument("--playback-progress-ms", type=int, default=80)
    args = parser.parse_args()
    if args.ref_audio is None:
        args.ref_audio = args.model / "assets" / "HT_ref_audio.wav"
    if args.chunk_ms <= 0 or args.chunk_ms > 1000:
        parser.error("--chunk-ms must be in [1, 1000]")
    if args.tail_drain_s < 0 or args.timeout_s <= 0:
        parser.error("--tail-drain-s must be non-negative and --timeout-s positive")
    if args.playback_initial_buffer_ms < 0 or args.playback_progress_ms <= 0:
        parser.error("playback buffer/progress values must be non-negative/positive")
    return args


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to mix artifacts in non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = discover_humdial_cases(args.dataset_root)
    schedule = build_schedule(
        cases,
        request_rate=args.request_rate,
        duration_s=args.duration_s,
        seed=args.seed,
    )
    payload = schedule_payload(
        schedule,
        dataset_root=args.dataset_root,
        request_rate=args.request_rate,
        duration_s=args.duration_s,
        seed=args.seed,
    )
    _write_json(args.output_dir / "schedule.json", payload)
    if args.schedule_only:
        print(json.dumps({key: payload[key] for key in payload if key != "requests"}, ensure_ascii=False, indent=2))
        return
    if not args.model.exists() or not args.ref_audio.is_file():
        raise FileNotFoundError(f"model/ref audio does not exist: {args.model}, {args.ref_audio}")
    results, wall_time_s = asyncio.run(
        run_schedule(
            schedule,
            args=args,
        )
    )
    with (args.output_dir / "sessions.jsonl").open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = summarize_results(
        results,
        request_rate=args.request_rate,
        arrival_window_s=args.duration_s,
        wall_time_s=wall_time_s,
    )
    summary.update(
        {
            "seed": args.seed,
            "dataset_root": str(args.dataset_root.resolve()),
            "model": str(args.model),
            "chunk_ms": args.chunk_ms,
            "tail_drain_s": args.tail_drain_s,
            "non_pd_production_path": True,
        }
    )
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
