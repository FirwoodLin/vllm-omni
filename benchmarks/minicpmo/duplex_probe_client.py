# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Own probe client over the stable trunk duplex client (``vllm_omni.clients.duplex``).

The HumDial drivers previously sat on the legacy
``vllm_omni.experimental.fullduplex.client`` probe client. The stable client is
the maintained one, but two gaps against this branch's server contract are
bridged here instead of in trunk files:

* the merged ``playback.ack`` handler requires a response identity fence
  (``session_id``/``incarnation``/``epoch``) plus a strictly increasing
  per-response ``observation_seq``; the stable client's ``ack_playback``
  predates that contract, so ``RealtimeSession.send_playback_ack`` assembles
  the wire event itself (identical to the legacy probe client) and sends it
  through the stable client's raw ``send``;
* the HumDial summaries read derived playout metrics (``playout_slack_ms``,
  ``playout_deadline_miss_count``, ``streaming_rtf``) and the request ``rtf``
  that the stable collector's ``timing_summary`` leaves to its caller;
  ``ProbeEventCollector`` ports that derivation on top of the trunk collector.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import time

from vllm_omni.clients.duplex import (
    DuplexClient,
    EventCollector,
    SessionConfig,
)
from vllm_omni.clients.minicpmo_4_5 import create_duplex_session_config
from vllm_omni.metrics.definitions import compute_audio_rtf

__all__ = ["ProbeEventCollector", "RealtimeSession"]


def _rounded_ms(value: float) -> float:
    return round(float(value), 3)


def _interval_summary(values: list[float]) -> dict[str, float | int]:
    clean = sorted(_rounded_ms(value) for value in values if value >= 0)
    if not clean:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    def nearest_rank(percentile: float) -> float:
        index = max(0, int(percentile * len(clean) + 0.999999) - 1)
        return clean[min(index, len(clean) - 1)]

    return {
        "count": len(clean),
        "mean": _rounded_ms(sum(clean) / len(clean)),
        "p50": nearest_rank(0.50),
        "p95": nearest_rank(0.95),
        "max": clean[-1],
    }


_AUDIO_DELTA_EVENT_TYPES = frozenset({"response.audio.delta", "response.output_audio.delta"})


class ProbeEventCollector(EventCollector):
    """Trunk ``EventCollector`` plus the derived playout/RTF keys the drivers read."""

    def timing_summary(
        self,
        *,
        after_s: float,
        input_committed_at_s: float | None = None,
        response_id: str | None = None,
        measurement_origin: dict[str, str] | None = None,
    ) -> dict[str, object]:
        result = super().timing_summary(
            after_s=after_s,
            input_committed_at_s=input_committed_at_s,
            response_id=response_id,
            measurement_origin=measurement_origin,
        )
        audio_output = result.get("audio_output")
        if isinstance(audio_output, dict):
            (
                intervals_ms,
                chunk_durations_ms,
                playout_slack_ms,
                streaming_audio_ms,
                streaming_generation_ms,
            ) = self._derive_audio_chunks(after_s, response_id)
            audio_output["inter_chunk_intervals_ms"] = [_rounded_ms(value) for value in intervals_ms]
            audio_output["chunk_durations_ms"] = [_rounded_ms(value) for value in chunk_durations_ms]
            audio_output["playout_slack_ms"] = [_rounded_ms(value) for value in playout_slack_ms]
            audio_output["minimum_playout_slack_ms"] = (
                _rounded_ms(min(playout_slack_ms)) if playout_slack_ms else None
            )
            audio_output["required_startup_buffer_ms"] = (
                _rounded_ms(max(0.0, -min(playout_slack_ms))) if playout_slack_ms else 0.0
            )
            # Ignore sub-millisecond floating-point residue from monotonic
            # timestamp subtraction at an exact playback boundary.
            audio_output["playout_deadline_miss_count"] = sum(value < -0.001 for value in playout_slack_ms)
            audio_output["streaming_rtf"] = (
                round(streaming_generation_ms / streaming_audio_ms, 6) if streaming_audio_ms > 0 else None
            )
        metrics = result.get("request_metrics")
        if isinstance(metrics, dict):
            generation_ms = metrics.get("audio_generation_ms")
            duration_ms = metrics.get("audio_duration_ms")
            if (
                isinstance(generation_ms, int | float)
                and isinstance(duration_ms, int | float)
                and duration_ms > 0
            ):
                metrics["rtf"] = round(compute_audio_rtf(generation_ms / 1000.0, duration_ms / 1000.0), 6)
        return result

    def _derive_audio_chunks(
        self,
        after_s: float,
        response_id: str | None,
    ) -> tuple[list[float], list[float], list[float], float, float]:
        """Replay the collector's audio events into the raw per-chunk series."""
        received_at_s: list[float] = []
        cumulative_audio_ms: list[float] = []
        for event, received in zip(self.events, self.event_received_at_s, strict=True):
            if received < after_s:
                continue
            if response_id is not None and EventCollector.response_id(event) != response_id:
                continue
            if event.get("type") not in _AUDIO_DELTA_EVENT_TYPES:
                continue
            delta = event.get("delta") or event.get("audio")
            if not isinstance(delta, str) or not delta:
                continue
            received_at_s.append(received)
            metadata = event.get("metadata")
            duration_ms = metadata.get("audio_duration_ms") if isinstance(metadata, dict) else None
            if isinstance(duration_ms, int | float) and math.isfinite(float(duration_ms)):
                cumulative_audio_ms.append(max(0.0, float(duration_ms)))

        intervals_ms = [(current - previous) * 1000.0 for previous, current in zip(received_at_s, received_at_s[1:])]
        chunk_durations_ms: list[float] = []
        previous_duration_ms = 0.0
        for duration_ms in cumulative_audio_ms:
            chunk_durations_ms.append(
                duration_ms - previous_duration_ms if duration_ms >= previous_duration_ms else duration_ms
            )
            previous_duration_ms = duration_ms
        playout_slack_ms: list[float] = []
        if len(chunk_durations_ms) == len(received_at_s):
            audio_available_ms = 0.0
            first_audio_at_s = received_at_s[0]
            for index in range(1, len(received_at_s)):
                audio_available_ms += max(0.0, chunk_durations_ms[index - 1])
                arrival_elapsed_ms = (received_at_s[index] - first_audio_at_s) * 1000.0
                playout_slack_ms.append(audio_available_ms - arrival_elapsed_ms)
        streaming_audio_ms = sum(max(0.0, duration) for duration in chunk_durations_ms[:-1])
        streaming_generation_ms = (
            max(0.0, (received_at_s[-1] - received_at_s[0]) * 1000.0) if len(received_at_s) > 1 else 0.0
        )
        return intervals_ms, chunk_durations_ms, playout_slack_ms, streaming_audio_ms, streaming_generation_ms


class RealtimeSession:
    """Legacy probe-client-shaped wrapper over one stable ``DuplexClient``.

    Exposes the surface the HumDial drivers use: collector access via
    ``events``, raw ``send``, ``commit``, ``stream_pcm16``,
    ``send_playback_ack`` (identity-fenced), and ``close_session``. The
    session.update handshake happens in ``__aenter__`` from the trunk
    ``SessionConfig``, so every configure-time value must be supplied at
    construction.
    """

    def __init__(
        self,
        url: str,
        *,
        model: str,
        session_id: str | None = None,
        ref_audio: str | None = None,
        instructions: str | None = None,
        initial_user_text: str | None = None,
        native_duplex: bool = True,
        auto_response: bool = True,
        temperature: float | None = None,
        extra_body: dict[str, object] | None = None,
        turn_detection: dict[str, object] | None = None,
        idle_timeout_s: float | None = None,
        handshake_timeout_s: float = 60.0,
    ) -> None:
        session_extra: dict[str, object] = dict(extra_body or {})
        if not native_duplex:
            session_extra["native_duplex"] = False
        if initial_user_text is not None:
            session_extra["duplex_initial_user_text"] = initial_user_text
        config = create_duplex_session_config(
            ref_audio=ref_audio,
            instructions=instructions,
            auto_response=auto_response,
            temperature=temperature,
            turn_detection=dict(turn_detection) if turn_detection is not None else None,
            idle_timeout_s=idle_timeout_s,
            extra_body=session_extra,
        )
        self._client = DuplexClient(
            url,
            model=model,
            config=config,
            session_id=session_id,
            # Benchmark probes never reconnect: a resumed session would
            # silently change the measured workload.
            reconnect=None,
            heartbeat_interval_s=None,
            handshake_timeout_s=handshake_timeout_s,
        )
        self.events = ProbeEventCollector()
        self._consume_task: asyncio.Task[None] | None = None
        self._observation_seq: dict[str, int] = {}

    async def __aenter__(self) -> RealtimeSession:
        await self._client.__aenter__()
        # The handshake consumed session.created before the collector
        # subscribed; seed it so capability/capacity lookups still work.
        self.events.add({"type": "session.created", "session": dict(self._client.session_info)})
        self._consume_task = asyncio.create_task(self.events.consume(self._client))
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        try:
            await self._client.__aexit__(exc_type, exc, traceback)
        finally:
            if self._consume_task is not None:
                self._consume_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._consume_task
                self._consume_task = None

    async def send(self, event: dict[str, object]) -> None:
        await self._client.send(event)

    async def commit(self, *, final: bool = True) -> None:
        await self._client.commit(final=final)

    async def stream_pcm16(
        self,
        pcm: bytes,
        *,
        chunk_ms: int = 200,
        realtime: bool = True,
    ) -> int:
        return await self._client.stream_pcm(pcm, chunk_ms=chunk_ms, realtime=realtime)

    async def close_session(self, *, timeout_s: float = 20.0) -> None:
        await self._client.close(timeout_s=timeout_s)
        if self._consume_task is not None:
            # Let the collector drain the closing events before callers
            # inspect them; omniinteract's probe wrapper uses the same
            # bounded wait.
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(self._consume_task), timeout=5.0)

    async def send_playback_ack(self, response_id: str, played_ms: int, *, commit: bool = True) -> None:
        """Report simulated playback progress for one response.

        ``commit=False`` mirrors the browser worklet's periodic progress
        observations; the terminal drained observation keeps the historical
        default and commits the played prefix.
        """
        event: dict[str, object] = {
            "type": "playback.ack",
            **self._playback_identity(response_id),
            "observation_seq": self._next_observation_seq(response_id),
            "played_ms": int(played_ms),
            "commit": commit,
        }
        if commit:
            event["committed_ms"] = int(played_ms)
        await self._client.send(event)

    def _next_observation_seq(self, response_id: str) -> int:
        observation_seq = self._observation_seq.get(response_id, 0)
        self._observation_seq[response_id] = observation_seq + 1
        return observation_seq

    def _playback_identity(self, response_id: str) -> dict[str, object]:
        session_id: str | None = None
        incarnation: int | None = None
        epoch: int | None = None
        for event in self.events.events:
            if event.get("type") == "session.created":
                session = event.get("session")
                if isinstance(session, dict):
                    raw_session_id = session.get("id") or session.get("session_id")
                    if isinstance(raw_session_id, str) and raw_session_id:
                        session_id = raw_session_id
                    raw_epoch = session.get("epoch")
                    if isinstance(raw_epoch, int) and not isinstance(raw_epoch, bool):
                        epoch = raw_epoch
                raw_incarnation = event.get("incarnation")
                if isinstance(raw_incarnation, int) and not isinstance(raw_incarnation, bool):
                    incarnation = raw_incarnation
            if event.get("type") != "response.created" or self.events.response_id(event) != response_id:
                continue
            raw_session_id = event.get("session_id")
            raw_incarnation = event.get("incarnation")
            raw_epoch = event.get("epoch")
            if isinstance(raw_session_id, str) and raw_session_id:
                session_id = raw_session_id
            if isinstance(raw_incarnation, int) and not isinstance(raw_incarnation, bool):
                incarnation = raw_incarnation
            if isinstance(raw_epoch, int) and not isinstance(raw_epoch, bool):
                epoch = raw_epoch
            response = event.get("response")
            metadata = response.get("metadata") if isinstance(response, dict) else None
            duplex_event = metadata.get("duplex_event") if isinstance(metadata, dict) else None
            if isinstance(duplex_event, dict):
                nested_session_id = duplex_event.get("session_id")
                nested_incarnation = duplex_event.get("incarnation")
                nested_epoch = duplex_event.get("epoch")
                if isinstance(nested_session_id, str) and nested_session_id:
                    session_id = nested_session_id
                if isinstance(nested_incarnation, int) and not isinstance(nested_incarnation, bool):
                    incarnation = nested_incarnation
                if isinstance(nested_epoch, int) and not isinstance(nested_epoch, bool):
                    epoch = nested_epoch
        if session_id is None or incarnation is None or epoch is None:
            raise RuntimeError(f"Missing playback identity for response {response_id}")
        return {
            "session_id": session_id,
            "incarnation": incarnation,
            "epoch": epoch,
            "response_id": response_id,
            "item_id": f"item_{response_id}",
        }
