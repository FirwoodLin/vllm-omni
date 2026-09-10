# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Concurrent MiniCPM-o Realtime duplex and resumable-session E2E driver."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import statistics
import sys
import time
import uuid
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@lru_cache(maxsize=1)
def _scenario_module():
    """Delay heavyweight vLLM imports so ``--help`` remains directly executable."""
    from tests.e2e.online_serving.helpers import minicpmo_realtime_duplex_scenarios

    return minicpmo_realtime_duplex_scenarios


def _ref_audio_data_url(path: str) -> str:
    return _scenario_module()._ref_audio_data_url(path)


def _url_with_model(*args, **kwargs) -> str:
    return _scenario_module()._url_with_model(*args, **kwargs)


async def run_demo(args):
    return await _scenario_module().run_demo(args)


class _SynchronizedStartGate:
    """One-shot, failure-propagating start gate for Python 3.10+."""

    def __init__(self, parties: int, *, timeout_s: float):
        if parties < 1:
            raise ValueError("synchronized start parties must be positive")
        if timeout_s <= 0:
            raise ValueError("synchronized start timeout must be positive")
        self._parties = parties
        self._timeout_s = timeout_s
        self._arrived = 0
        self._released = False
        self._released_at_s: float | None = None
        self._failure: BaseException | None = None
        self._event = asyncio.Event()
        self._lock = asyncio.Lock()

    async def wait(self) -> float:
        async with self._lock:
            if self._failure is not None:
                raise RuntimeError(f"synchronized start aborted: {self._failure!r}") from self._failure
            if self._released:
                assert self._released_at_s is not None
                return self._released_at_s
            self._arrived += 1
            if self._arrived == self._parties:
                self._released = True
                self._released_at_s = asyncio.get_running_loop().time()
                self._event.set()
                return self._released_at_s

        try:
            await asyncio.wait_for(self._event.wait(), timeout=self._timeout_s)
        except asyncio.TimeoutError as exc:
            failure = TimeoutError(
                f"synchronized start timed out after {self._timeout_s:.1f}s "
                f"({self._arrived}/{self._parties} sessions ready)"
            )
            await self.abort(failure)
            if self._released and self._failure is None:
                assert self._released_at_s is not None
                return self._released_at_s
            raise failure from exc

        if self._failure is not None:
            raise RuntimeError(f"synchronized start aborted: {self._failure!r}") from self._failure
        assert self._released_at_s is not None
        return self._released_at_s

    async def abort(self, failure: BaseException) -> None:
        async with self._lock:
            if self._released or self._failure is not None:
                return
            self._failure = failure
            self._event.set()


async def _run_demo_with_start_gate(args: SimpleNamespace):
    start_gate = getattr(args, "start_barrier", None)
    try:
        connection_delay_s = float(getattr(args, "connection_delay_s", 0.0))
        if connection_delay_s > 0:
            await asyncio.sleep(connection_delay_s)
        return await run_demo(args)
    except BaseException as exc:
        if start_gate is not None:
            await start_gate.abort(exc)
        raise


def _open_loop_unit_pcm16(pcm16: bytes, *, duration_ms: int = 1000) -> bytes:
    """Return one exact-duration, speech-bearing PCM16 benchmark unit."""
    if not pcm16:
        raise ValueError("open-loop input WAV has no audio")
    unit_bytes = 16_000 * 2 * duration_ms // 1000
    active = _scenario_module()._pcm16_active_slice(pcm16, duration_ms)
    if len(active) >= unit_bytes:
        return active[:unit_bytes]
    repeats = math.ceil(unit_bytes / len(active))
    return (active * repeats)[:unit_bytes]


def _open_loop_numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))

    def percentile(quantile: float) -> float | None:
        if not finite:
            return None
        position = (len(finite) - 1) * quantile
        lower = int(position)
        upper = min(lower + 1, len(finite) - 1)
        fraction = position - lower
        return finite[lower] + fraction * (finite[upper] - finite[lower])

    return {
        "count": len(finite),
        "min": min(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(finite) if finite else None,
        "mean": statistics.fmean(finite) if finite else None,
    }


def _open_loop_schedule_summary(ticks: list[dict[str, object]]) -> dict[str, object]:
    send_lag_ms = [float(value) for tick in ticks if isinstance((value := tick.get("send_lag_ms")), int | float)]
    send_duration_ms = [
        float(value) for tick in ticks if isinstance((value := tick.get("send_duration_ms")), int | float)
    ]
    return {
        "scheduled_unit_count": len(ticks),
        "send_lag_ms": _open_loop_numeric_summary(send_lag_ms),
        "send_duration_ms": _open_loop_numeric_summary(send_duration_ms),
        "end_send_lag_ms": send_lag_ms[-1] if send_lag_ms else None,
    }


def _open_loop_append_payload(
    unit_pcm16: bytes,
    *,
    cumulative_audio_ms: int,
    tick_index: int,
    scheduled_ns: int,
    frame_b64: str | None,
    force_listen: bool | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(unit_pcm16).decode("ascii"),
        "input_audio_format": "pcm16",
        "sample_rate_hz": 16_000,
        "duration_ms": 1000,
        "audio_end_ms": cumulative_audio_ms,
        "benchmark_tick_index": tick_index,
        "benchmark_scheduled_monotonic_ns": scheduled_ns,
    }
    if force_listen is not None:
        payload["force_listen"] = bool(force_listen)
    if frame_b64 is not None:
        payload["video_frames"] = [frame_b64]
    return payload


def _fixed_duty_tick_is_speak(tick_index: int, unit_count: int, speak_duty: float) -> bool:
    """Return a deterministic Bresenham-style speak/listen assignment.

    ``speak_duty`` is intentionally required to produce an integral number of
    speak units for the requested run.  This keeps the experiment label exact
    rather than silently rounding a 25/50/100% target.
    """
    if unit_count <= 0:
        raise ValueError("fixed-duty unit_count must be positive")
    if not 0 <= speak_duty <= 1:
        raise ValueError("fixed-duty speak_duty must be between 0 and 1")
    if not 0 <= tick_index < unit_count:
        raise ValueError("fixed-duty tick_index is outside the run")
    duty = Fraction(str(speak_duty))
    target_speak_units = duty * unit_count
    if target_speak_units.denominator != 1:
        raise ValueError(
            f"fixed-duty speak_duty={speak_duty} does not yield an integral speak-unit count for {unit_count} units"
        )
    target = target_speak_units.numerator
    # Use a ceiling accumulator so a non-zero duty starts with a speak unit;
    # subsequent units are spread as evenly as the integer target allows.
    return ((tick_index + 1) * target + unit_count - 1) // unit_count > (
        tick_index * target + unit_count - 1
    ) // unit_count


async def _ack_completed_open_loop_playback(ws, state, stop: asyncio.Event, *, timeout_s: float) -> None:
    while not stop.is_set():
        await _scenario_module()._ack_all_completed_response_playback(
            ws,
            state,
            timeout_s=timeout_s,
        )
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.05)
        except asyncio.TimeoutError:
            pass


async def _cancel_fixed_duty_responses(
    ws,
    state,
    stop: asyncio.Event,
    *,
    cancel_after_audio_ms: int,
) -> None:
    """Close each fixed-duty response after its first generated audio chunk.

    Native MiniCPM-o keeps a response/data-plane stream open across input
    units.  The fixed-duty experiment needs an explicit response boundary so
    later ticks do not become deferred input on the same response.  Cancelling
    after the first audio delta preserves a generated speak chunk; forced-listen
    responses are cancelled as soon as their listen event arrives.  Both paths
    release the stream before the next tick when possible.
    """
    requested: set[str] = set()
    first_audio_at: dict[str, float] = {}
    delay_s = max(0, int(cancel_after_audio_ms)) / 1000.0

    def response_epoch(response_id: str) -> int | None:
        for event in state.events:
            if event.get("type") != "response.created" or state._event_response_id(event) != response_id:
                continue
            response = event.get("response")
            metadata = response.get("metadata") if isinstance(response, dict) else None
            duplex_event = metadata.get("duplex_event") if isinstance(metadata, dict) else None
            epoch = duplex_event.get("epoch") if isinstance(duplex_event, dict) else None
            return epoch if isinstance(epoch, int) else None
        return None

    def response_has_listen(response_id: str) -> bool:
        epoch = response_epoch(response_id)
        if epoch is None:
            return False
        created_index = next(
            (
                index
                for index, event in enumerate(state.events)
                if event.get("type") == "response.created" and state._event_response_id(event) == response_id
            ),
            None,
        )
        if created_index is None:
            return False
        return any(
            event.get("type") == "response.listen" and event.get("epoch") == epoch
            for event in state.events[created_index + 1 :]
        )

    while not stop.is_set():
        now = asyncio.get_running_loop().time()
        for response_id in list(state.response_ids):
            if response_id in requested or state.response_done(response_id):
                continue
            has_audio = state.response_audio_delta_count(response_id) > 0
            has_listen = response_has_listen(response_id)
            if not has_audio and not has_listen:
                continue
            if has_audio:
                first_audio_at.setdefault(response_id, now)
                if now - first_audio_at[response_id] < delay_s:
                    continue
            requested.add(response_id)
            try:
                await ws.send(json.dumps({"type": "response.cancel", "response_id": response_id}))
                # A forced-listen commit retains its payload in the native
                # committed buffer. Clear it after cancellation so the next
                # speak unit cannot inherit force_listen=True.
                if has_listen:
                    await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
            except ConnectionClosed:
                return
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.01)
        except asyncio.TimeoutError:
            pass


async def _run_open_loop_session(
    args: SimpleNamespace,
    *,
    start_barrier: _SynchronizedStartGate | None,
) -> dict[str, object]:
    scenario = _scenario_module()
    source_pcm16 = scenario._read_wav_pcm16(Path(args.input_wav))
    unit_pcm16 = _open_loop_unit_pcm16(source_pcm16)
    frame_b64 = (
        base64.b64encode(Path(args.frame_image).read_bytes()).decode("ascii") if args.frame_image is not None else None
    )
    session_id = args.session_id
    url = _url_with_model(
        args.url,
        args.model,
        autostart=False if args.ref_audio else None,
        session_id=session_id,
    )
    state = scenario.DemoState()
    reader_stop = asyncio.Event()
    ack_stop = asyncio.Event()
    fixed_duty_stop = asyncio.Event()
    ticks: list[dict[str, object]] = []
    reader = None
    acker = None
    fixed_duty_canceller = None
    speak_duty = getattr(args, "open_loop_speak_duty", None)
    cancel_after_audio_ms = int(getattr(args, "open_loop_cancel_after_audio_ms", 0))
    if speak_duty is not None:
        # Validate before opening the socket so a malformed experiment cannot
        # leave a server session behind.
        _fixed_duty_tick_is_speak(0, args.open_loop_units, float(speak_duty))

    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        reader = asyncio.create_task(scenario._reader(ws, state, reader_stop))
        try:
            await ws.send(json.dumps(scenario._session_update_event(args)))
            await scenario._wait_for(
                state,
                lambda: state.count("session.created") > 0,
                timeout_s=min(float(args.timeout_s), 60.0),
                label="open-loop session.created",
            )
            released_at_s = (
                await start_barrier.wait() if start_barrier is not None else asyncio.get_running_loop().time()
            )
            first_tick_s = released_at_s + 0.25
            if speak_duty is None:
                acker = asyncio.create_task(
                    _ack_completed_open_loop_playback(
                        ws,
                        state,
                        ack_stop,
                        timeout_s=min(float(args.timeout_s), 30.0),
                    )
                )
            if speak_duty is not None:
                fixed_duty_canceller = asyncio.create_task(
                    _cancel_fixed_duty_responses(
                        ws,
                        state,
                        fixed_duty_stop,
                        cancel_after_audio_ms=cancel_after_audio_ms,
                    )
                )
            cumulative_audio_ms = 0
            for tick_index in range(args.open_loop_units):
                scheduled_at_s = first_tick_s + tick_index * args.open_loop_period_ms / 1000.0
                await asyncio.sleep(max(0.0, scheduled_at_s - asyncio.get_running_loop().time()))
                send_started_ns = time.monotonic_ns()
                scheduled_ns = int(scheduled_at_s * 1_000_000_000)
                cumulative_audio_ms += 1000
                tick_is_speak = (
                    _fixed_duty_tick_is_speak(tick_index, args.open_loop_units, float(speak_duty))
                    if speak_duty is not None
                    else None
                )
                await ws.send(
                    json.dumps(
                        _open_loop_append_payload(
                            unit_pcm16,
                            cumulative_audio_ms=cumulative_audio_ms,
                            tick_index=tick_index,
                            scheduled_ns=scheduled_ns,
                            frame_b64=frame_b64,
                            force_listen=(not tick_is_speak) if tick_is_speak is not None else None,
                        )
                    )
                )
                if speak_duty is not None:
                    # Turn-mode native input buffers require an explicit
                    # response_create decision for every unit.  force_listen
                    # makes the listen half consume the unit without speech
                    # generation; the speak half follows the same response
                    # path and is bounded by the canceller above.
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.commit",
                                "final": True,
                                "response_create": True,
                            }
                        )
                    )
                send_completed_ns = time.monotonic_ns()
                ticks.append(
                    {
                        "tick_index": tick_index,
                        "scheduled_monotonic_ns": scheduled_ns,
                        "send_started_monotonic_ns": send_started_ns,
                        "send_completed_monotonic_ns": send_completed_ns,
                        "send_lag_ms": (send_started_ns - scheduled_ns) / 1_000_000.0,
                        "send_duration_ms": (send_completed_ns - send_started_ns) / 1_000_000.0,
                        "audio_end_ms": cumulative_audio_ms,
                        **({"speak": tick_is_speak} if tick_is_speak is not None else {}),
                    }
                )

            await asyncio.sleep(max(0.0, float(args.open_loop_drain_s)))
            ack_stop.set()
            if acker is not None:
                await acker
            if speak_duty is None:
                await scenario._ack_all_completed_response_playback(
                    ws,
                    state,
                    timeout_s=min(float(args.timeout_s), 30.0),
                )
            await ws.send(json.dumps({"type": "session.close"}))
            await scenario._wait_for(
                state,
                lambda: state.count("session.closed") > 0,
                timeout_s=min(float(args.timeout_s), 30.0),
                label="open-loop session.closed",
            )
        finally:
            ack_stop.set()
            fixed_duty_stop.set()
            if acker is not None and not acker.done():
                acker.cancel()
                try:
                    await acker
                except asyncio.CancelledError:
                    pass
            if fixed_duty_canceller is not None and not fixed_duty_canceller.done():
                fixed_duty_canceller.cancel()
                try:
                    await fixed_duty_canceller
                except asyncio.CancelledError:
                    pass
            if state.count("session.created") > 0 and state.count("session.closed") == 0:
                try:
                    await ws.send(json.dumps({"type": "session.close"}))
                except ConnectionClosed:
                    pass
            reader_stop.set()
            if reader is not None:
                reader.cancel()
                try:
                    await reader
                except asyncio.CancelledError:
                    pass

    output_dir = Path(args.output_dir)
    scenario._write_demo_artifacts(state, output_dir, output_audio_format=args.output_audio_format)
    timed_events = state.timing_events.events
    (output_dir / "events.jsonl").write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in timed_events),
        encoding="utf-8",
    )
    (output_dir / "input_schedule.json").write_text(
        json.dumps(ticks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    response_timings = state.response_timing_summaries()
    request_metrics = state.session_request_metrics(session_id=session_id)
    completed_response_ids = [response_id for response_id in state.response_ids if state.response_done(response_id)]
    errors = scenario._unexpected_error_events(state)
    audio_delta_count = state.count("response.audio.delta")
    require_audio_ok = not args.open_loop_require_audio or audio_delta_count > 0
    audio_response_count = sum(
        1 for response_id in state.response_ids if state.response_audio_delta_count(response_id) > 0
    )
    expected_speak_unit_count = (
        sum(1 for tick in ticks if tick.get("speak") is True) if speak_duty is not None else None
    )
    fixed_duty_exact_ok = speak_duty is None or audio_response_count == expected_speak_unit_count
    return {
        "ok": bool(
            len(ticks) == args.open_loop_units
            and state.count("session.created") == 1
            and state.count("session.closed") == 1
            and not errors
            and require_audio_ok
            and fixed_duty_exact_ok
        ),
        "session_id": session_id,
        "completed_response_ids": completed_response_ids,
        "response_timings": response_timings,
        "request_metrics": request_metrics,
        "open_loop": {
            **_open_loop_schedule_summary(ticks),
            "period_ms": args.open_loop_period_ms,
            "drain_s": args.open_loop_drain_s,
            "response_created_count": state.count("response.created"),
            "response_done_count": state.count("response.done"),
            "response_listen_count": state.count("response.listen"),
            "audio_delta_count": audio_delta_count,
            "audio_response_count": audio_response_count,
            "require_audio_ok": require_audio_ok,
            "fixed_speak_duty": float(speak_duty) if speak_duty is not None else None,
            "fixed_speak_unit_count": (
                sum(1 for tick in ticks if tick.get("speak") is True) if speak_duty is not None else None
            ),
            "fixed_listen_unit_count": (
                sum(1 for tick in ticks if tick.get("speak") is False) if speak_duty is not None else None
            ),
            "fixed_audio_response_count": audio_response_count if speak_duty is not None else None,
            "fixed_duty_exact_ok": fixed_duty_exact_ok if speak_duty is not None else None,
            "fixed_response_cancel_after_audio_ms": (cancel_after_audio_ms if speak_duty is not None else None),
            "fixed_response_cancelled_count": state.cancelled_count if speak_duty is not None else None,
        },
        "error_count": len(errors),
        "errors": errors,
        "output_dir": str(output_dir),
    }


async def _run_open_loop_with_start_gate(args: SimpleNamespace, start_barrier: _SynchronizedStartGate | None):
    try:
        connection_delay_s = float(getattr(args, "connection_delay_s", 0.0))
        if connection_delay_s > 0:
            await asyncio.sleep(connection_delay_s)
        return await _run_open_loop_session(args, start_barrier=start_barrier)
    except BaseException as exc:
        if start_barrier is not None:
            await start_barrier.abort(exc)
        raise


def _with_resume_mode(url: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["resume"] = "1"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _response_ids(result: dict[str, object]) -> set[str]:
    values = result.get("completed_response_ids")
    return {value for value in values if isinstance(value, str)} if isinstance(values, list) else set()


def _event_session_id(event: dict[str, object]) -> str | None:
    session_id = event.get("session_id")
    if isinstance(session_id, str):
        return session_id
    inner = event.get("event")
    if isinstance(inner, dict):
        nested_session_id = inner.get("session_id")
        if isinstance(nested_session_id, str):
            return nested_session_id
    return None


def _error_code(event: dict[str, object]) -> str | None:
    error = event.get("error")
    if isinstance(error, dict) and isinstance(error.get("code"), str):
        return error["code"]
    code = event.get("code")
    return code if isinstance(code, str) else None


def _validate_identity_isolation(results: list[dict[str, object]]) -> bool:
    seen: set[str] = set()
    for result in results:
        current = _response_ids(result)
        if current & seen:
            return False
        seen.update(current)
    return True


def _validate_semantic_isolation(
    results: list[dict[str, object]],
    *,
    input_wavs: list[str],
    expected_tokens: list[str],
) -> bool:
    if not input_wavs:
        return True
    if len(results) != len(input_wavs):
        return False
    input_hashes = [hashlib.sha256(Path(path).read_bytes()).digest() for path in input_wavs]
    if len(set(input_hashes)) != len(input_hashes):
        return False
    if not expected_tokens:
        return True
    if len(expected_tokens) != len(results):
        return False
    normalized_expected_tokens = [token.strip().casefold() for token in expected_tokens]
    if any(not token for token in normalized_expected_tokens):
        return False
    for result_index, (result, expected_token) in enumerate(zip(results, normalized_expected_tokens, strict=True)):
        details = result.get("transcript_integrity")
        transcripts = (
            [str(item.get("transcript", "")) for item in details if isinstance(item, dict)]
            if isinstance(details, list)
            else []
        )
        joined_transcripts = "".join(transcripts).casefold()
        if expected_token not in joined_transcripts:
            return False
        if any(
            other_token in joined_transcripts
            for other_index, other_token in enumerate(normalized_expected_tokens)
            if other_index != result_index
        ):
            return False
    return True


async def _receive_until(ws, event_type: str, *, timeout_s: float) -> tuple[dict[str, object], list[dict[str, object]]]:
    async def receive() -> tuple[dict[str, object], list[dict[str, object]]]:
        events: list[dict[str, object]] = []
        while True:
            raw = await ws.recv()
            if not isinstance(raw, str):
                continue
            event = json.loads(raw)
            if not isinstance(event, dict):
                continue
            events.append(event)
            if event.get("type") == event_type:
                return event, events

    return await asyncio.wait_for(receive(), timeout=timeout_s)


def _server_event_sequences(events: list[dict[str, object]]) -> list[int]:
    return [sequence for event in events if isinstance(sequence := event.get("server_event_seq"), int)]


async def _open_admission_session(
    args: argparse.Namespace,
    session_id: str,
) -> tuple[ClientConnection, dict[str, object]]:
    url = _url_with_model(
        args.url,
        args.model,
        autostart=False if getattr(args, "ref_audio", None) else None,
        session_id=session_id,
    )
    ws = await websockets.connect(url, max_size=64 * 1024 * 1024)
    await ws.send(
        json.dumps(
            {
                "type": "session.update",
                "session": {
                    "session_id": session_id,
                    "model": args.model,
                    "modalities": ["audio", "text"],
                    "extra_body": {"native_duplex": True},
                    **({"ref_audio": _ref_audio_data_url(args.ref_audio)} if getattr(args, "ref_audio", None) else {}),
                },
            }
        )
    )
    created, _ = await _receive_until(ws, "session.created", timeout_s=args.timeout_s)
    return ws, created


async def _close_admission_session(ws: ClientConnection, *, timeout_s: float) -> None:
    await ws.send(json.dumps({"type": "session.close"}))
    await _receive_until(ws, "session.closed", timeout_s=timeout_s)
    await ws.close()


async def _admission_probe(args: argparse.Namespace, *, limit: int) -> dict[str, object]:
    if limit < 1:
        raise ValueError("admission limit must be positive")
    prefix = f"admission-{uuid.uuid4().hex}"
    accepted: list[tuple[ClientConnection, dict[str, object]]] = []
    replacement: tuple[ClientConnection, dict[str, object]] | None = None
    overflow_code = None
    try:
        for index in range(limit):
            accepted.append(await _open_admission_session(args, f"{prefix}-accepted-{index}"))

        overflow_id = f"{prefix}-overflow"
        overflow_url = _url_with_model(
            args.url,
            args.model,
            autostart=False if getattr(args, "ref_audio", None) else None,
            session_id=overflow_id,
        )
        async with websockets.connect(overflow_url, max_size=64 * 1024 * 1024) as overflow:
            await overflow.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "session_id": overflow_id,
                            "model": args.model,
                            "modalities": ["audio", "text"],
                            "extra_body": {"native_duplex": True},
                            **(
                                {"ref_audio": _ref_audio_data_url(args.ref_audio)}
                                if getattr(args, "ref_audio", None)
                                else {}
                            ),
                        },
                    }
                )
            )
            error, _ = await _receive_until(overflow, "error", timeout_s=args.timeout_s)
            overflow_code = _error_code(error)

        first_ws, _ = accepted.pop(0)
        await _close_admission_session(first_ws, timeout_s=args.timeout_s)
        replacement = await _open_admission_session(args, f"{prefix}-replacement")

        first_capabilities = accepted[0][1].get("session") if accepted else replacement[1].get("session")
        capabilities = first_capabilities.get("capabilities") if isinstance(first_capabilities, dict) else None
        advertised_multi = capabilities.get("supports_multi_session") if isinstance(capabilities, dict) else None
        admission_mode = capabilities.get("session_admission_mode") if isinstance(capabilities, dict) else None
        return {
            "ok": (
                overflow_code == "resource_exhausted"
                and replacement[1].get("type") == "session.created"
                and admission_mode == "engine_managed"
            ),
            "configured_limit": limit,
            "accepted_before_rejection": limit,
            "overflow_error_code": overflow_code,
            "replacement_accepted": True,
            "advertised_multi_session": advertised_multi,
            "session_admission_mode": admission_mode,
        }
    finally:
        cleanup = list(accepted)
        if replacement is not None:
            cleanup.append(replacement)
        for ws, _ in cleanup:
            try:
                await _close_admission_session(ws, timeout_s=args.timeout_s)
            except Exception:
                await ws.close()


async def _resume_probe(
    args: argparse.Namespace,
    *,
    session_id: str,
    expect_expired: bool = False,
) -> dict[str, object]:
    url = _url_with_model(
        args.url,
        args.model,
        autostart=False if getattr(args, "ref_audio", None) else None,
        session_id=session_id,
    )
    async with websockets.connect(url, max_size=64 * 1024 * 1024) as first:
        await first.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "session_id": session_id,
                        "model": args.model,
                        "modalities": ["audio", "text"],
                        "extra_body": {"native_duplex": True},
                        **(
                            {"ref_audio": _ref_audio_data_url(args.ref_audio)}
                            if getattr(args, "ref_audio", None)
                            else {}
                        ),
                    },
                }
            )
        )
        created, first_events = await _receive_until(first, "session.created", timeout_s=args.timeout_s)
        token = created.get("resume_token")
        incarnation = created.get("incarnation")
        generation = created.get("attachment_generation")
        if not isinstance(token, str) or not isinstance(incarnation, int):
            raise RuntimeError("session.created omitted resumable credentials")
        last_seq = max(_server_event_sequences(first_events), default=0)

    delay_s = args.expire_after_s if expect_expired else args.resume_after_ms / 1000
    if delay_s > 0:
        await asyncio.sleep(delay_s)

    resume_url = _with_resume_mode(url)
    async with websockets.connect(resume_url, max_size=64 * 1024 * 1024) as second:
        await second.send(
            json.dumps(
                {
                    "type": "session.resume",
                    "session_id": session_id,
                    "incarnation": incarnation,
                    "resume_token": token,
                    "last_received_server_event_seq": last_seq,
                }
            )
        )
        if expect_expired:
            error, error_events = await _receive_until(second, "error", timeout_s=args.timeout_s)
            error_payload = error.get("error")
            code = error_payload.get("code") if isinstance(error_payload, dict) else error.get("code")
            return {
                "ok": code == "session_resume_expired",
                "session_id": session_id,
                "expired": True,
                "error_code": code,
                "event_count": len(error_events),
            }
        resumed, replay = await _receive_until(second, "session.resumed", timeout_s=args.timeout_s)
        rotated = resumed.get("resume_token")
        if not isinstance(rotated, str) or rotated == token:
            raise RuntimeError("session.resume did not rotate the resume token")
        await second.send(json.dumps({"type": "session.heartbeat"}))
        heartbeat, heartbeat_events = await _receive_until(
            second,
            "session.heartbeat_ack",
            timeout_s=args.timeout_s,
        )
        await second.send(json.dumps({"type": "session.close"}))
        closed, close_events = await _receive_until(second, "session.closed", timeout_s=args.timeout_s)

    replay_sequences = _server_event_sequences(replay)
    return {
        "ok": (
            resumed.get("session_id") == session_id
            and isinstance(generation, int)
            and resumed.get("attachment_generation") == generation + 1
            and _event_session_id(heartbeat) == session_id
            and _event_session_id(closed) == session_id
            and replay_sequences == sorted(replay_sequences)
        ),
        "session_id": session_id,
        "resumed_session_id": resumed.get("session_id"),
        "heartbeat_session_id": _event_session_id(heartbeat),
        "closed_session_id": _event_session_id(closed),
        "initial_attachment_generation": generation,
        "resumed_attachment_generation": resumed.get("attachment_generation"),
        "replayed_event_count": len(replay_sequences),
        "replayed_event_sequences": replay_sequences,
        "heartbeat_event_count": len(heartbeat_events),
        "close_event_count": len(close_events),
        "token_rotated": True,
    }


async def _takeover_probe(
    args: argparse.Namespace,
    *,
    session_id: str,
) -> dict[str, object]:
    url = _url_with_model(
        args.url,
        args.model,
        autostart=False if getattr(args, "ref_audio", None) else None,
        session_id=session_id,
    )
    first = await websockets.connect(url, max_size=64 * 1024 * 1024)
    second = None
    try:
        await first.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "session_id": session_id,
                        "model": args.model,
                        "modalities": ["audio", "text"],
                        "extra_body": {"native_duplex": True},
                        **(
                            {"ref_audio": _ref_audio_data_url(args.ref_audio)}
                            if getattr(args, "ref_audio", None)
                            else {}
                        ),
                    },
                }
            )
        )
        created, first_events = await _receive_until(first, "session.created", timeout_s=args.timeout_s)
        token = created.get("resume_token")
        incarnation = created.get("incarnation")
        generation = created.get("attachment_generation")
        if not isinstance(token, str) or not isinstance(incarnation, int) or not isinstance(generation, int):
            raise RuntimeError("session.created omitted takeover credentials")
        last_seq = max(_server_event_sequences(first_events), default=0)

        second = await websockets.connect(_with_resume_mode(url), max_size=64 * 1024 * 1024)
        await second.send(
            json.dumps(
                {
                    "type": "session.resume",
                    "session_id": session_id,
                    "incarnation": incarnation,
                    "resume_token": token,
                    "last_received_server_event_seq": last_seq,
                }
            )
        )
        resumed, replay_events = await _receive_until(second, "session.resumed", timeout_s=args.timeout_s)
        replaced, replaced_events = await _receive_until(first, "session.replaced", timeout_s=args.timeout_s)
        await asyncio.wait_for(first.wait_closed(), timeout=args.timeout_s)

        rejected_old_writes = 0
        for _ in range(4):
            try:
                await first.send(json.dumps({"type": "session.heartbeat"}))
            except ConnectionClosed:
                rejected_old_writes += 1

        await second.send(json.dumps({"type": "session.heartbeat"}))
        heartbeat, heartbeat_events = await _receive_until(
            second,
            "session.heartbeat_ack",
            timeout_s=args.timeout_s,
        )
        await second.send(json.dumps({"type": "session.close"}))
        closed, close_events = await _receive_until(second, "session.closed", timeout_s=args.timeout_s)
        rotated_token = resumed.get("resume_token")
        return {
            "ok": (
                resumed.get("session_id") == session_id
                and resumed.get("attachment_generation") == generation + 1
                and isinstance(rotated_token, str)
                and rotated_token != token
                and _event_session_id(replaced) == session_id
                and replaced.get("attachment_generation") == generation
                and rejected_old_writes == 4
                and _event_session_id(heartbeat) == session_id
                and _event_session_id(closed) == session_id
            ),
            "session_id": session_id,
            "initial_attachment_generation": generation,
            "resumed_attachment_generation": resumed.get("attachment_generation"),
            "replaced_attachment_generation": replaced.get("attachment_generation"),
            "token_rotated": isinstance(rotated_token, str) and rotated_token != token,
            "old_attachment_closed": True,
            "rejected_old_writes": rejected_old_writes,
            "replay_event_count": len(replay_events),
            "replaced_event_count": len(replaced_events),
            "heartbeat_event_count": len(heartbeat_events),
            "close_event_count": len(close_events),
        }
    finally:
        if second is not None:
            await second.close()
        await first.close()


def _demo_args(
    args: argparse.Namespace, index: int, start_barrier: _SynchronizedStartGate | None = None
) -> SimpleNamespace:
    validation_mode = "response-required" if args.response_required else "model-policy"
    input_wav = args.session_input_wav[index] if args.session_input_wav else args.input_wav
    return SimpleNamespace(
        url=args.url,
        model=args.model,
        session_id=f"multi-{index}-{uuid.uuid4().hex}",
        input_wav=input_wav,
        ref_audio=args.ref_audio,
        frame_image=args.frame_image,
        turn_input_wav=list(args.turn_input_wav),
        output_dir=str(Path(args.output_dir) / f"session_{index:02d}"),
        output_audio_format="pcm16",
        chunk_ms=args.chunk_ms,
        realtime_input=args.realtime_input,
        first_turn_ms=args.first_turn_ms,
        turn_duration_ms=list(args.turn_duration_ms),
        first_turn_transcript=f"session {index} input",
        omit_transcript_hints=True,
        validation_mode=validation_mode,
        temperature=args.temperature,
        scenario=getattr(args, "scenario", "sequential"),
        silence_ms=getattr(args, "silence_ms", 500),
        require_audio=args.response_required,
        require_distinct_inputs=False,
        expect_empty_turn=list(getattr(args, "expect_empty_turn", []) or []),
        short_ack_ms=350,
        turns=args.turns,
        timeout_s=args.timeout_s,
        model_policy_settle_ms=args.model_policy_settle_ms,
        start_barrier=start_barrier,
        connection_delay_s=args.connection_stagger_ms * index / 1000.0,
        open_loop_units=args.open_loop_units,
        open_loop_period_ms=args.open_loop_period_ms,
        open_loop_drain_s=args.open_loop_drain_s,
        open_loop_require_audio=args.open_loop_require_audio,
        open_loop_speak_duty=args.open_loop_speak_duty,
        open_loop_cancel_after_audio_ms=args.open_loop_cancel_after_audio_ms,
        auto_response=args.open_loop_speak_duty is None,
    )


async def run_multi_session(args: argparse.Namespace) -> dict[str, object]:
    if args.sessions < 1:
        raise ValueError("--sessions must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_result = None
    if args.disconnect_session_index is not None:
        if not 0 <= args.disconnect_session_index < args.sessions:
            raise ValueError("--disconnect-session-index is outside the session range")
        resume_result = await _resume_probe(
            args,
            session_id=f"resume-{args.disconnect_session_index}-{uuid.uuid4().hex}",
        )
    takeover_result = None
    if args.takeover_session_index is not None:
        if not 0 <= args.takeover_session_index < args.sessions:
            raise ValueError("--takeover-session-index is outside the session range")
        takeover_result = await _takeover_probe(
            args,
            session_id=f"takeover-{args.takeover_session_index}-{uuid.uuid4().hex}",
        )
    lifecycle_result = await run_lifecycle_probes(args)

    start_barrier = (
        _SynchronizedStartGate(
            args.sessions,
            timeout_s=min(float(args.timeout_s), 30.0),
        )
        if getattr(args, "synchronized_start", False)
        else None
    )
    if args.open_loop_units > 0:
        session_results = await asyncio.gather(
            *(
                _run_open_loop_with_start_gate(
                    _demo_args(args, index, start_barrier),
                    start_barrier,
                )
                for index in range(args.sessions)
            ),
            return_exceptions=True,
        )
    else:
        session_results = await asyncio.gather(
            *(_run_demo_with_start_gate(_demo_args(args, index, start_barrier)) for index in range(args.sessions)),
            return_exceptions=True,
        )
    failures = [repr(result) for result in session_results if isinstance(result, BaseException)]
    completed = [result for result in session_results if isinstance(result, dict)]
    identity_isolation_ok = _validate_identity_isolation(completed)
    semantic_isolation_ok = _validate_semantic_isolation(
        completed,
        input_wavs=list(args.session_input_wav),
        expected_tokens=list(args.session_expected_token),
    )
    result = {
        "ok": (
            not failures
            and len(completed) == args.sessions
            and all(item.get("ok") is True for item in completed)
            and identity_isolation_ok
            and semantic_isolation_ok
            and (resume_result is None or resume_result.get("ok") is True)
            and (takeover_result is None or takeover_result.get("ok") is True)
            and lifecycle_result["ok"] is True
        ),
        "session_count": args.sessions,
        "workload_mode": "open_loop" if args.open_loop_units > 0 else "closed_loop",
        "open_loop_units": args.open_loop_units if args.open_loop_units > 0 else None,
        "open_loop_period_ms": args.open_loop_period_ms if args.open_loop_units > 0 else None,
        "open_loop_speak_duty": (args.open_loop_speak_duty if args.open_loop_units > 0 else None),
        "open_loop_cancel_after_audio_ms": (args.open_loop_cancel_after_audio_ms if args.open_loop_units > 0 else None),
        "identity_isolation_ok": identity_isolation_ok,
        "semantic_isolation_ok": semantic_isolation_ok,
        "resume": resume_result,
        "takeover": takeover_result,
        "expiry": lifecycle_result["expiry"],
        "admission": lifecycle_result["admission"],
        "failures": failures,
        "sessions": completed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


async def run_lifecycle_probes(args: argparse.Namespace) -> dict[str, object]:
    """Run expiry and admission probes without requiring model output."""
    if args.sessions < 1:
        raise ValueError("--sessions must be positive")
    expiry_result = None
    if args.expire_session_index is not None:
        if not 0 <= args.expire_session_index < args.sessions:
            raise ValueError("--expire-session-index is outside the session range")
        expiry_result = await _resume_probe(
            args,
            session_id=f"expire-{args.expire_session_index}-{uuid.uuid4().hex}",
            expect_expired=True,
        )
    admission_result = (
        await _admission_probe(args, limit=args.verify_admission_limit)
        if args.verify_admission_limit is not None
        else None
    )
    return {
        "ok": (
            (expiry_result is None or expiry_result.get("ok") is True)
            and (admission_result is None or admission_result.get("ok") is True)
        ),
        "expiry": expiry_result,
        "admission": admission_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8113/v1/realtime?duplex=1")
    parser.add_argument("--base-url", help="Deprecated alias; /v1/realtime is appended when supplied.")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--ref-audio", help="Optional WAV used as the MiniCPM-o voice prompt for every session.")
    parser.add_argument(
        "--frame-image",
        default=None,
        help="Optional image sent as one camera frame per native 1 s audio unit.",
    )
    parser.add_argument("--session-input-wav", action="append", default=[])
    parser.add_argument("--session-expected-token", action="append", default=[])
    parser.add_argument("--turn-input-wav", action="append", default=[])
    parser.add_argument("--output-dir", default="/tmp/minicpmo_pr3907_multi_session_e2e")
    parser.add_argument("--realtime-input", action="store_true")
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument(
        "--open-loop-units",
        type=int,
        default=0,
        help="Send this many native 1 s input units on an absolute clock without waiting for responses.",
    )
    parser.add_argument("--open-loop-period-ms", type=int, default=1000)
    parser.add_argument("--open-loop-drain-s", type=float, default=5.0)
    parser.add_argument(
        "--open-loop-speak-duty",
        type=float,
        default=None,
        help=(
            "Experimental exact speak-unit duty in [0,1]. Each speak response is "
            "closed after its first audio chunk; requires an integral target count."
        ),
    )
    parser.add_argument(
        "--open-loop-cancel-after-audio-ms",
        type=int,
        default=0,
        help="Delay before cancelling each fixed-duty response after its first audio delta.",
    )
    parser.add_argument(
        "--open-loop-require-audio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require every open-loop session to exercise the speech-output path (off for listen-only load).",
    )
    parser.add_argument("--first-turn-ms", type=int, default=1400)
    parser.add_argument("--turn-duration-ms", type=int, action="append", default=[])
    parser.add_argument("--expect-empty-turn", type=int, action="append", default=[])
    parser.add_argument("--response-required", action="store_true")
    parser.add_argument(
        "--scenario",
        choices=["sequential", "listen-only-overlap"],
        default="sequential",
    )
    parser.add_argument("--silence-ms", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--disconnect-session-index", type=int)
    parser.add_argument("--takeover-session-index", type=int)
    parser.add_argument("--resume-after-ms", type=int, default=1000)
    parser.add_argument("--expire-session-index", type=int)
    parser.add_argument("--expire-after-s", type=float, default=6.0)
    parser.add_argument("--verify-admission-limit", type=int)
    parser.add_argument(
        "--synchronized-start",
        action="store_true",
        help="Wait until every session is created before starting input.",
    )
    parser.add_argument(
        "--connection-stagger-ms",
        type=float,
        default=0.0,
        help="Delay connection i by i times this interval while preserving synchronized input start.",
    )
    parser.add_argument("--model-policy-settle-ms", type=int, default=600)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()
    if args.base_url:
        args.url = args.base_url.rstrip("/") + "/v1/realtime?duplex=1"
    if args.frame_image is not None and not Path(args.frame_image).is_file():
        parser.error(f"--frame-image does not exist: {args.frame_image}")
    if args.session_input_wav and len(args.session_input_wav) != args.sessions:
        parser.error("provide exactly one --session-input-wav per session")
    if args.session_expected_token and len(args.session_expected_token) != args.sessions:
        parser.error("provide exactly one --session-expected-token per session")
    if args.session_expected_token and not args.session_input_wav:
        parser.error("--session-expected-token requires --session-input-wav")
    if args.open_loop_units < 0:
        parser.error("--open-loop-units must be non-negative")
    if args.open_loop_period_ms <= 0:
        parser.error("--open-loop-period-ms must be positive")
    if args.open_loop_drain_s < 0:
        parser.error("--open-loop-drain-s must be non-negative")
    if args.open_loop_speak_duty is not None:
        if not 0 <= args.open_loop_speak_duty <= 1:
            parser.error("--open-loop-speak-duty must be between 0 and 1")
        if args.open_loop_units <= 0:
            parser.error("--open-loop-speak-duty requires --open-loop-units")
        target = Fraction(str(args.open_loop_speak_duty)) * args.open_loop_units
        if target.denominator != 1:
            parser.error("--open-loop-speak-duty must yield an integral speak-unit count")
    if args.open_loop_cancel_after_audio_ms < 0:
        parser.error("--open-loop-cancel-after-audio-ms must be non-negative")
    if args.connection_stagger_ms < 0:
        parser.error("--connection-stagger-ms must be non-negative")
    if args.open_loop_units > 0 and args.turn_input_wav:
        parser.error("--turn-input-wav is not used by the open-loop workload")
    normalized_expected_tokens = [token.strip().casefold() for token in args.session_expected_token]
    for token_index, token in enumerate(normalized_expected_tokens):
        if any(
            token in other_token or other_token in token
            for other_token in normalized_expected_tokens[token_index + 1 :]
        ):
            parser.error("--session-expected-token values must not overlap after normalization")
    return args


def main() -> None:
    result = asyncio.run(run_multi_session(parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
