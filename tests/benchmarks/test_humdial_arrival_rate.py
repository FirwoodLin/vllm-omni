# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import base64
import importlib.util
import sys
import wave
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.experimental.fullduplex.client import RealtimeDuplexClient, RealtimeEventCollector

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SCRIPT = Path(__file__).parents[2] / "benchmarks" / "minicpmo" / "humdial_arrival_rate.py"
LAUNCHER = SCRIPT.with_name("benchmark_humdial_service.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("humdial_arrival_rate_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_launcher():
    sys.path.insert(0, str(LAUNCHER.parent))
    try:
        spec = importlib.util.spec_from_file_location("benchmark_humdial_service_test", LAUNCHER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(LAUNCHER.parent))


def _write_wav(path: Path, duration_ms: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(bytes(16_000 * 2 * duration_ms // 1000))


def test_discovery_excludes_clean_and_sampling_is_stratified_and_reproducible(tmp_path):
    module = _load_module()
    for language in ("cn", "en"):
        for scenario in ("ask", "pause"):
            for index, duration_ms in enumerate((400, 800, 1200, 1600)):
                _write_wav(
                    tmp_path / "test" / f"{language}_test_nondev" / scenario / f"{index:04d}.wav",
                    duration_ms,
                )
            _write_wav(
                tmp_path / "test" / f"{language}_test_nondev" / scenario / "clean_ignored.wav",
                1000,
            )

    cases = module.discover_humdial_cases(tmp_path)
    assert len(cases) == 16
    assert all(not case.path.name.startswith("clean_") for case in cases)

    first = module.stratified_sample(cases, 8, seed=20_260_901)
    repeated = module.stratified_sample(cases, 8, seed=20_260_901)
    assert [case.relative_path for case in first] == [case.relative_path for case in repeated]
    assert len({case.relative_path for case in first}) == 8
    assert Counter((case.language, case.scenario) for case in first) == {
        ("cn", "ask"): 2,
        ("cn", "pause"): 2,
        ("en", "ask"): 2,
        ("en", "pause"): 2,
    }


def test_schedule_has_exact_rate_count_sorted_random_arrivals_and_stable_manifest(tmp_path):
    module = _load_module()
    cases = [
        module.HumDialCase(
            path=tmp_path / f"{index}.wav",
            relative_path=f"test/en_test_nondev/ask/{index}.wav",
            language="en",
            scenario="ask",
            duration_s=1.0 + index / 100,
        )
        for index in range(100)
    ]
    schedule = module.build_schedule(cases, request_rate=0.2, duration_s=300.0, seed=20_260_901)
    repeated = module.build_schedule(cases, request_rate=0.2, duration_s=300.0, seed=20_260_901)
    assert len(schedule) == 60
    assert [request.arrival_s for request in schedule] == sorted(request.arrival_s for request in schedule)
    assert all(0 <= request.arrival_s < 300 for request in schedule)
    assert [request.as_dict() for request in schedule] == [request.as_dict() for request in repeated]

    payload = module.schedule_payload(
        schedule,
        dataset_root=tmp_path,
        request_rate=0.2,
        duration_s=300.0,
        seed=20_260_901,
    )
    repeated_payload = module.schedule_payload(
        repeated,
        dataset_root=tmp_path,
        request_rate=0.2,
        duration_s=300.0,
        seed=20_260_901,
    )
    assert payload["request_count"] == 60
    assert payload["realized_request_rate_sessions_per_s"] == pytest.approx(0.2)
    assert payload == repeated_payload


@pytest.mark.asyncio
async def test_playback_clock_sends_progress_then_terminal_commit():
    module = _load_module()
    received_at = module.time.monotonic() - 1.0
    collector = RealtimeEventCollector()
    collector.add(
        {
            "type": "session.created",
            "session": {"id": "session-0", "epoch": 0},
            "incarnation": 1,
        },
        received_at_s=received_at,
    )
    collector.add(
        {
            "type": "response.created",
            "response": {
                "id": "response-0",
                "metadata": {
                    "duplex_event": {"session_id": "session-0", "incarnation": 1, "epoch": 0}
                },
            },
        },
        received_at_s=received_at,
    )
    collector.add(
        {
            "type": "response.audio.delta",
            "response_id": "response-0",
            "delta": base64.b64encode(bytes(24_000 * 2)).decode(),
            "sample_rate_hz": 24_000,
        },
        received_at_s=received_at,
    )
    collector.add(
        {"type": "response.audio.done", "response_id": "response-0"},
        received_at_s=received_at,
    )
    acknowledgements = []

    async def send_playback_ack(response_id, played_ms, *, commit=True):
        acknowledgements.append((response_id, played_ms, commit))

    fake_client = SimpleNamespace(events=collector, send_playback_ack=send_playback_ack)
    clock = module.BrowserPlaybackClock(initial_buffer_ms=0, progress_ms=80)
    await clock.step(fake_client)

    assert acknowledgements[0][0] == "response-0"
    assert acknowledgements[0][2] is False
    assert acknowledgements[-1] == ("response-0", 1000, True)
    assert clock.summary()["final_ack_count"] == 1


def test_realtime_client_progress_ack_omits_committed_cursor():
    async def exercise():
        client = RealtimeDuplexClient("ws://unused")
        client.events = SimpleNamespace(
            playback_identity=lambda response_id: {
                "session_id": "session-0",
                "incarnation": 1,
                "epoch": 0,
                "response_id": response_id,
                "item_id": f"item_{response_id}",
            }
        )
        sent = []

        async def send(event):
            sent.append(event)

        client.send = send
        await client.send_playback_ack("response-0", 80, commit=False)
        await client.send_playback_ack("response-0", 160)
        return sent

    progress, terminal = asyncio.run(exercise())
    assert progress["observation_seq"] == 0
    assert progress["commit"] is False
    assert "committed_ms" not in progress
    assert terminal["observation_seq"] == 1
    assert terminal["commit"] is True
    assert terminal["committed_ms"] == 160


def test_model_unit_decisions_pair_in_order_and_ignore_buffering_events():
    module = _load_module()
    collector = RealtimeEventCollector()
    collector.add(
        {
            "type": "response.listen",
            "response": {
                "metadata": {
                    "buffering": True,
                    "model_listen": False,
                    "reason": "audio_not_enough",
                    "vllm_omni": {
                        "source_input_seq": 1,
                        "source_audio_end_ms": 1000,
                    },
                }
            },
        },
        received_at_s=1.05,
    )
    collector.add(
        {
            "type": "response.audio.delta",
            "response_id": "response-0",
            "delta": base64.b64encode(b"audio").decode(),
            "vllm_omni": {
                "source_input_seq": 2,
                "source_audio_end_ms": 2000,
            },
        },
        received_at_s=2.40,
    )
    collector.add(
        {
            "type": "response.listen",
            "response": {
                "metadata": {
                    "model_listen": True,
                    "vllm_omni": {
                        "source_input_seq": 1,
                        "source_audio_end_ms": 1000,
                    },
                }
            },
        },
        received_at_s=1.20,
    )
    collector.add(
        {
            "type": "response.listen",
            "response": {
                "metadata": {
                    "model_listen": True,
                    "vllm_omni": {
                        "source_input_seq": 3,
                        "source_audio_end_ms": 3000,
                    },
                }
            },
        },
        received_at_s=3.60,
    )
    collector.add(
        {
            "type": "response.audio.delta",
            "response_id": "response-0",
            "delta": base64.b64encode(b"continuation").decode(),
            "vllm_omni": {
                "source_input_seq": 4,
                "source_audio_end_ms": -1,
            },
        },
        received_at_s=4.20,
    )

    paired = module._model_unit_decision_metrics(
        collector,
        model_unit_sent_at_s=[1.0, 2.0, 3.0],
        model_unit_ms=1000,
    )

    assert paired == {
        "measurement_origin": (
            "client send completion for source_audio_end_ms to the correlated "
            "non-buffering listen/audio decision receive"
        ),
        "model_unit_ms": 1000,
        "input_unit_count": 3,
        "decision_count": 3,
        "unpaired_input_unit_count": 0,
        "unpaired_decision_count": 0,
        "rows": [
            {
                "input_unit_index": 0,
                "input_audio_end_ms": 1000,
                "decision_kind": "listen",
                "response_id": None,
                "decision_latency_ms": pytest.approx(200.0),
            },
            {
                "input_unit_index": 1,
                "input_audio_end_ms": 2000,
                "decision_kind": "audio",
                "response_id": "response-0",
                "decision_latency_ms": pytest.approx(400.0),
            },
            {
                "input_unit_index": 2,
                "input_audio_end_ms": 3000,
                "decision_kind": "listen",
                "response_id": None,
                "decision_latency_ms": pytest.approx(600.0),
            },
        ],
    }


def test_launcher_realtime_slo_requires_deadline_and_protocol_success():
    launcher = _load_launcher()
    passing = {
        "failure_count": 0,
        "audio_response_count": 4,
        "arrival_schedule_lag_ms": {"p99": 12.0},
        "model_unit_decision_latency_ms": {"p99": 900.0},
        "model_unit_unpaired_input_count": 0,
        "model_unit_unpaired_decision_count": 0,
        "streaming_audio_rtf": {"p99": 0.95},
        "playout_deadline_miss_rate": 0.005,
    }
    assert launcher.realtime_slo_reasons(passing) == []

    failing = {
        **passing,
        "failure_count": 1,
        "arrival_schedule_lag_ms": {"p99": 250.0},
        "model_unit_decision_latency_ms": {"p99": 1600.0},
        "model_unit_unpaired_input_count": 2,
        "streaming_audio_rtf": {"p99": 1.1},
        "playout_deadline_miss_rate": 0.02,
    }
    reasons = launcher.realtime_slo_reasons(failing)
    assert len(reasons) == 5
    assert any("sessions failed" in reason for reason in reasons)
    assert any("model-unit decision latency" in reason for reason in reasons)
    assert not any("unpaired input model units" in reason for reason in reasons)
