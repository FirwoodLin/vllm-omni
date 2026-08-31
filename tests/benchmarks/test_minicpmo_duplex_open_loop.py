# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "benchmarks" / "minicpmo"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_minicpmo_open_loop_append_timing_and_summary(tmp_path):
    summarizer = _load_module(
        "summarize_duplex_saturation",
        BENCHMARK_DIR / "summarize_duplex_saturation.py",
    )
    launcher = _load_module(
        "benchmark_duplex_service_saturation_test",
        BENCHMARK_DIR / "benchmark_duplex_service_saturation.py",
    )
    session_id = "session-test"
    records = [
        {
            "phase": "start",
            "session_id": session_id,
            "kind": "input",
            "operation_id": "op-0",
            "benchmark_tick_index": 0,
            "benchmark_scheduled_monotonic_ns": 1_000_000_000,
            "client_audio_end_ms": 1000,
            "monotonic_ns": 1_010_000_000,
        },
        {
            "phase": "end",
            "session_id": session_id,
            "kind": "input",
            "operation_id": "op-0",
            "benchmark_tick_index": 0,
            "benchmark_scheduled_monotonic_ns": 1_000_000_000,
            "client_audio_end_ms": 1000,
            "monotonic_ns": 1_810_000_000,
            "append_ok": True,
            "emitted_response": True,
        },
        {
            "phase": "start",
            "session_id": session_id,
            "kind": "input",
            "operation_id": "op-1",
            "benchmark_tick_index": 1,
            "benchmark_scheduled_monotonic_ns": 2_000_000_000,
            "client_audio_end_ms": 2000,
            "monotonic_ns": 2_050_000_000,
        },
        {
            "phase": "error",
            "session_id": session_id,
            "kind": "input",
            "operation_id": "op-1",
            "benchmark_tick_index": 1,
            "benchmark_scheduled_monotonic_ns": 2_000_000_000,
            "client_audio_end_ms": 2000,
            "monotonic_ns": 3_100_000_000,
        },
    ]
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "".join(f"INFO MiniCPM-o duplex append timing {json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )

    rows = launcher._duplex_append_timing_records(
        server_log,
        session_ids={session_id},
    )
    assert [row["phase"] for row in rows] == ["end", "error"]
    assert rows[0]["start_queue_lag_ms"] == pytest.approx(10.0)
    assert rows[0]["completion_lag_ms"] == pytest.approx(810.0)
    assert rows[1]["completion_lag_ms"] == pytest.approx(1100.0)

    run_dir = tmp_path / "run"
    session_dir = run_dir / "session_00"
    session_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "ok": True,
                "workload_mode": "open_loop",
                "session_count": 1,
                "sessions": [{"session_id": session_id, "request_metrics": []}],
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
    (session_dir / "events.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "type": "response.listen",
                    "_client_received_at_s": received_at_s,
                    "response": {"metadata": {"vllm_omni": {"stage_metrics": {"0": {}}}}},
                }
            )
            + "\n"
            for received_at_s in (1.1, 2.3)
        ),
        encoding="utf-8",
    )
    (session_dir / "input_schedule.json").write_text(
        json.dumps(
            [
                {"tick_index": 0, "scheduled_monotonic_ns": 1_000_000_000, "send_lag_ms": 1.0},
                {"tick_index": 1, "scheduled_monotonic_ns": 2_000_000_000, "send_lag_ms": 2.0},
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "server_append_timing.json").write_text(
        json.dumps({"session_ids": [session_id], "rows": rows}),
        encoding="utf-8",
    )

    summary = summarizer.summarize_run(run_dir)
    open_loop = summary["open_loop"]
    assert open_loop["scheduled_unit_count"] == 2
    assert open_loop["completed_real_append_count"] == 2
    assert open_loop["unfinished_real_append_count"] == 0
    assert open_loop["completion_deadline_miss_count"] == 1
    assert open_loop["append_error_count"] == 1
    assert open_loop["stage0_completed_unit_count"] == 2
    assert open_loop["stage0_unfinished_unit_count"] == 0
    assert open_loop["stage0_completion_lag_ms"]["median"] == pytest.approx(200.0)
    assert open_loop["stage0_completion_lag_slope_ms_per_tick"]["median"] == pytest.approx(200.0)
    assert open_loop["stage0_deadline_miss_count"] == 0
    assert open_loop["start_queue_lag_slope_ms_per_tick"]["median"] == pytest.approx(40.0)
    assert open_loop["completion_lag_slope_ms_per_tick"]["median"] == pytest.approx(290.0)


def test_minicpmo_duplex_launcher_selects_ordered_unique_gpu_pool():
    _load_module(
        "summarize_duplex_saturation",
        BENCHMARK_DIR / "summarize_duplex_saturation.py",
    )
    launcher = _load_module(
        "benchmark_duplex_service_saturation_gpu_pool_test",
        BENCHMARK_DIR / "benchmark_duplex_service_saturation.py",
    )

    assert launcher._selected_gpus(SimpleNamespace(gpu=4, gpus=None)) == [4]
    assert launcher._selected_gpus(SimpleNamespace(gpu=4, gpus=[6, 2, 6, 7])) == [6, 2, 7]


def test_minicpmo_open_loop_append_payload_includes_one_video_frame():
    driver = _load_module(
        "run_minicpmo_realtime_duplex_multi_session_video_test",
        ROOT / "tests" / "e2e" / "online_serving" / "run_minicpmo_realtime_duplex_multi_session.py",
    )

    payload = driver._open_loop_append_payload(
        b"\x01\x00" * 16_000,
        cumulative_audio_ms=1000,
        tick_index=0,
        scheduled_ns=123,
        frame_b64="encoded-frame",
    )

    assert payload["video_frames"] == ["encoded-frame"]
    assert payload["duration_ms"] == 1000
    assert payload["benchmark_tick_index"] == 0


def test_minicpmo_fixed_duty_schedule_is_exact_and_evenly_spread():
    driver = _load_module(
        "run_minicpmo_realtime_duplex_multi_session_fixed_duty_test",
        ROOT / "tests" / "e2e" / "online_serving" / "run_minicpmo_realtime_duplex_multi_session.py",
    )

    for units, duty, expected in ((4, 0.25, 1), (8, 0.5, 4), (8, 1.0, 8), (8, 0.0, 0)):
        schedule = [driver._fixed_duty_tick_is_speak(i, units, duty) for i in range(units)]
        assert sum(schedule) == expected
    assert [driver._fixed_duty_tick_is_speak(i, 8, 0.5) for i in range(8)] == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    with pytest.raises(ValueError, match="integral speak-unit"):
        driver._fixed_duty_tick_is_speak(0, 8, 0.3)


def test_minicpmo_fixed_duty_listen_payload_is_explicit():
    driver = _load_module(
        "run_minicpmo_realtime_duplex_multi_session_fixed_payload_test",
        ROOT / "tests" / "e2e" / "online_serving" / "run_minicpmo_realtime_duplex_multi_session.py",
    )

    payload = driver._open_loop_append_payload(
        b"\x01\x00" * 16_000,
        cumulative_audio_ms=1000,
        tick_index=0,
        scheduled_ns=123,
        frame_b64=None,
        force_listen=True,
    )

    assert payload["force_listen"] is True


def test_minicpmo_session_connection_delay_precedes_demo(monkeypatch):
    driver = _load_module(
        "run_minicpmo_realtime_duplex_multi_session_stagger_test",
        ROOT / "tests" / "e2e" / "online_serving" / "run_minicpmo_realtime_duplex_multi_session.py",
    )
    sleep = AsyncMock()
    run_demo = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(driver.asyncio, "sleep", sleep)
    monkeypatch.setattr(driver, "run_demo", run_demo)
    args = SimpleNamespace(connection_delay_s=0.075, start_barrier=None)

    result = asyncio.run(driver._run_demo_with_start_gate(args))

    assert result == {"ok": True}
    sleep.assert_awaited_once_with(0.075)
    run_demo.assert_awaited_once_with(args)


def test_minicpmo_multigpu_telemetry_counts_time_samples_not_gpu_rows():
    _load_module(
        "summarize_duplex_saturation",
        BENCHMARK_DIR / "summarize_duplex_saturation.py",
    )
    launcher = _load_module(
        "benchmark_duplex_service_saturation_telemetry_test",
        BENCHMARK_DIR / "benchmark_duplex_service_saturation.py",
    )
    rows = [
        {
            "gpus": [
                {"index": 0, "memory.used": 10, "utilization.gpu": 20, "power.draw": 100.0},
                {"index": 1, "memory.used": 30, "utilization.gpu": 40, "power.draw": 200.0},
            ]
        },
        {
            "gpus": [
                {"index": 0, "memory.used": 50, "utilization.gpu": 60, "power.draw": 300.0},
                {"index": 1, "memory.used": 70, "utilization.gpu": 80, "power.draw": 400.0},
            ]
        },
    ]

    summary = launcher._telemetry_summary(rows)

    assert summary["sample_count"] == 2
    assert summary["gpu_observation_count"] == 4
    assert summary["peak_memory_used_mib"] == 70
    assert summary["per_gpu"]["0"]["sample_count"] == 2
    assert summary["per_gpu"]["1"]["peak_power_draw_w"] == 400


def test_minicpmo_cfm_stats_are_attributed_to_stage2_replicas(tmp_path):
    _load_module(
        "summarize_duplex_saturation",
        BENCHMARK_DIR / "summarize_duplex_saturation.py",
    )
    launcher = _load_module(
        "benchmark_duplex_service_saturation_cfm_replica_test",
        BENCHMARK_DIR / "benchmark_duplex_service_saturation.py",
    )
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "\n".join(
            [
                '(StageEngineCoreProc_stage2_replica0 pid=1) INFO CFM CUDA Graph stats {"cached_shapes": 10, "total_replays": 100}',
                '(StageEngineCoreProc_stage2_replica1 pid=2) INFO CFM CUDA Graph stats {"cached_shapes": 20, "total_replays": 200}',
                '(StageEngineCoreProc_stage2_replica0 pid=1) INFO CFM CUDA Graph stats {"cached_shapes": 11, "total_replays": 110}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert launcher._latest_cfm_graph_stats_per_replica(server_log) == {
        "0": {"cached_shapes": 11, "total_replays": 110},
        "1": {"cached_shapes": 20, "total_replays": 200},
    }
