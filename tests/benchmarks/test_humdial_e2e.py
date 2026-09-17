# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.clients.duplex import EventCollector

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SCRIPT = Path(__file__).parents[2] / "benchmarks" / "minicpmo" / "humdial_e2e.py"
SERVICE = SCRIPT.with_name("benchmark_humdial_service.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("humdial_e2e_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_service():
    sys.path.insert(0, str(SERVICE.parent))
    try:
        spec = importlib.util.spec_from_file_location("humdial_service_e2e_test", SERVICE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SERVICE.parent))


def _write_wav(path: Path, duration_ms: int = 400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(bytes(16_000 * 2 * duration_ms // 1000))


def _manifest(tmp_path: Path, **overrides):
    _write_wav(tmp_path / "initial.wav")
    _write_wav(tmp_path / "interrupt.wav")
    case = {
        "case_id": "case-a",
        "initial_audio": "initial.wav",
        "interrupt_audio": "interrupt.wav",
        "external_interrupt_ms": 1200,
        "playback_anchor_ms": 200,
        "operation": "follow up",
        "expected_text_contains": ["banana"],
        "session_window_s": 10,
    }
    case.update(overrides)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "cases": [case]}), encoding="utf-8")
    return path


def test_manifest_resolves_audio_and_rejects_duplicate_ids(tmp_path):
    module = _load_module()
    cases = module.load_manifest(_manifest(tmp_path))
    assert cases[0].initial_audio == (tmp_path / "initial.wav").resolve()
    assert cases[0].expected_text_contains == ("banana",)

    payload = json.loads((_manifest(tmp_path, case_id="case-a")).read_text())
    payload["cases"].append(payload["cases"][0])
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        module.load_manifest(duplicate)


@pytest.mark.parametrize(
    "field,value,pattern",
    [
        ("external_interrupt_ms", -1, "external_interrupt_ms"),
        ("playback_anchor_ms", -1, "playback_anchor_ms"),
        ("session_window_s", 0, "session_window_s"),
    ],
)
def test_manifest_rejects_invalid_timing(tmp_path, field, value, pattern):
    module = _load_module()
    with pytest.raises(ValueError, match=pattern):
        module.load_manifest(_manifest(tmp_path, **{field: value}))


def test_schedule_exact_count_is_deterministic_and_samples_with_replacement():
    module = _load_module()
    case = module.E2ECase("a", Path("a"), Path("b"), 1000, 200, "op", (), 10.0)
    cases = [case, module.E2ECase("b", Path("c"), Path("d"), 1000, 200, "op", (), 10.0)]
    first = module.build_schedule(cases, request_rate=0.5, duration_s=10, seed=7)
    second = module.build_schedule(cases, request_rate=0.5, duration_s=10, seed=7)
    assert len(first) == 5
    assert [item.as_dict() for item in first] == [item.as_dict() for item in second]
    assert [item.arrival_s for item in first] == sorted(item.arrival_s for item in first)


def test_context_evaluation_is_casefolded_and_unknown_without_labels():
    module = _load_module()
    assert module.evaluate_context("The answer is BANANA.", ("banana",)) is True
    assert module.evaluate_context("The answer is apple.", ("banana",)) is False
    assert module.evaluate_context("anything", ()) is None


def test_received_at_lookup_scans_parallel_event_lists_backwards():
    module = _load_module()
    events = [
        {"type": "response.created", "id": "first"},
        {"type": "response.done", "id": "target"},
        {"type": "session.closed", "id": "last"},
    ]
    assert module._received_at_for_event(events, [1.0, 2.0, 3.0], events[1]) == 2.0


def test_timing_origin_falls_back_when_streaming_output_precedes_commit():
    module = _load_module()
    events = [
        {"type": "response.created", "response": {"id": "response-1"}},
        {
            "type": "response.audio.delta",
            "response_id": "response-1",
            "delta": "audio",
        },
    ]
    received_at = [10.0, 10.2]
    assert module._usable_input_commit_at(events, received_at, "response-1", 10.5) is None
    assert module._timing_measurement_origin(False)["ttfp"].startswith("response.created")
    assert module._usable_input_commit_at(events, received_at, "response-1", 10.1) == 10.1
    collector = EventCollector()
    collector.add(events[0], received_at_s=10.0)
    collector.add(
        {
            **events[1],
            "delta": "YXVkaW8=",
            "metadata": {"audio_duration_ms": 80},
        },
        received_at_s=10.2,
    )
    timing = collector.timing_summary(
        after_s=10.0,
        input_committed_at_s=None,
        response_id="response-1",
        measurement_origin=module._timing_measurement_origin(False),
    )
    assert timing["request_metrics"]["ttfp_ms"] == 200.0
    assert timing["audio_output"]["commit_to_first_audio_ms"] is None


@pytest.mark.asyncio
async def test_followup_response_id_is_available_before_response_done():
    module = _load_module()
    client = SimpleNamespace(events=EventCollector())
    client.events.add({"type": "response.created", "response": {"id": "resp-first"}})
    client.events.add({"type": "response.created", "response": {"id": "resp-followup"}})

    assert await module._wait_response_created(client, after_count=1, timeout_s=0.1) == "resp-followup"


def test_slo_requires_context_correctness_and_l2_anchor():
    module = _load_module()
    args = SimpleNamespace(feedback_contract="L2", max_interrupt_reaction_ms=100, max_stale_audio_ms=0)
    result = {
        "transport_success": True,
        "interaction_success": True,
        "task_success": True,
        "anchor_reached": True,
        "interrupt_reaction_ms": 20,
        "stale_audio_ms": 0,
    }
    assert module.slo_pass(result, args=args)
    result["context_correct"] = None
    result["task_success"] = None
    assert not module.slo_pass(result, args=args)
    result["task_success"] = True
    result["anchor_reached"] = False
    assert not module.slo_pass(result, args=args)


def test_summary_separates_unknown_context_and_counts_window_goodput():
    module = _load_module()
    results = [
        {
            "transport_success": True,
            "interaction_success": True,
            "task_success": True,
            "context_correct": True,
            "session_slo_pass": True,
            "session_window_s": 10,
            "stale_audio_delta_count": 0,
            "stale_audio_ms": 0,
        },
        {
            "transport_success": True,
            "interaction_success": True,
            "task_success": None,
            "context_correct": None,
            "session_slo_pass": False,
            "session_window_s": 20,
            "stale_audio_delta_count": 1,
            "stale_audio_ms": 10,
        },
    ]
    summary = module.summarize_results(
        results, request_rate=0.025, arrival_window_s=300, wall_time_s=30, feedback_contract="L2"
    )
    assert summary["task_success_count"] == 1
    assert summary["task_unknown_count"] == 1
    assert summary["session_slo_pass_rate"] == pytest.approx(0.5)
    assert summary["goodput_session_window_fraction"] == pytest.approx(10 / 30)


@pytest.mark.parametrize(
    ("explicit_followup_response", "explicit_all_responses"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_service_launcher_selects_e2e_client_without_changing_arrival_default(
    explicit_followup_response, explicit_all_responses
):
    service = _load_service()
    args = SimpleNamespace(
        client_mode="e2e",
        e2e_manifest=Path("/tmp/cases.json"),
        feedback_contract="L2",
        dataset_root=Path("/tmp/dataset"),
        host="127.0.0.1",
        port=8113,
        model=Path("/tmp/model"),
        ref_audio=Path("/tmp/ref.wav"),
        chunk_ms=200,
        tail_drain_s=2.0,
        playback_initial_buffer_ms=1000,
        request_timeout_s=180.0,
        explicit_followup_response=explicit_followup_response,
        explicit_all_responses=explicit_all_responses,
    )
    command = service._client_command(
        args,
        request_rate=0.025,
        duration_s=300,
        output_dir=Path("/tmp/out"),
        run_id="run",
        seed=1,
    )
    assert command[1].endswith("humdial_e2e.py")
    assert command[command.index("--manifest") + 1] == "/tmp/cases.json"
    assert command[command.index("--feedback-contract") + 1] == "L2"
    assert command[command.index("--playback-initial-buffer-ms") + 1] == "1000"
    assert ("--explicit-followup-response" in command) is explicit_followup_response
    assert ("--explicit-all-responses" in command) is explicit_all_responses
