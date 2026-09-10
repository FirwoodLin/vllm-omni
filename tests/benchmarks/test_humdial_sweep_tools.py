# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT = Path(__file__).parents[2]
RUNNER = ROOT / "benchmarks" / "minicpmo" / "run_humdial_8gpu_sweeps.py"
SUMMARIZER = ROOT / "benchmarks" / "minicpmo" / "summarize_humdial_sweeps.py"
SERVICE = ROOT / "benchmarks" / "minicpmo" / "benchmark_humdial_service.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_builds_fixed_five_minute_plans_and_requires_fresh_attempt(tmp_path):
    runner = _load_module(RUNNER, "run_humdial_8gpu_sweeps_test")
    args = runner.parse_args(
        [
            "--topology",
            "all",
            "--campaign",
            "campaign_01",
            "--attempt",
            "attempt_001",
            "--output-root",
            str(tmp_path),
        ]
    )

    plans = runner.build_plans(args)

    assert [plan.topology for plan in plans] == [
        "separated_422",
        "downstream_colocated",
        "thinker6_downstream_colocated",
        "thinker6_talker4_downstream_colocated",
        "thinker6_downstream_colocated_fixed_talker_kv",
        "full_colocated",
    ]
    assert all("300.0" == plan.command[plan.command.index("--duration-s") + 1] for plan in plans)
    assert all(
        "0.025,0.05,0.1" == plan.command[plan.command.index("--rates") + 1]
        for plan in plans
    )
    assert all("20260901" == plan.command[plan.command.index("--seed") + 1] for plan in plans)
    assert plans[0].output_dir == (
        tmp_path / "campaign_01" / "separated_422" / "attempt_001"
    )

    plans[1].output_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="fresh attempt"):
        runner.ensure_fresh_outputs(plans)

    runner.reserve_output_dirs(plans[:1])
    assert plans[0].output_dir.is_dir()
    with pytest.raises(FileExistsError, match="claimed concurrently"):
        runner.reserve_output_dirs(plans[:1])


def test_runner_propagates_parallel_stage_launch_flag(tmp_path):
    runner = _load_module(RUNNER, "run_humdial_8gpu_sweeps_parallel_test")
    args = runner.parse_args(
        [
            "--topology",
            "thinker6_downstream_colocated",
            "--campaign",
            "campaign_01",
            "--attempt",
            "attempt_parallel",
            "--output-root",
            str(tmp_path),
            "--parallel-stage-launch",
        ]
    )

    plan = runner.build_plans(args)[0]

    assert "--parallel-stage-launch" in plan.command
    assert plan.command[plan.command.index("--omni-master-port") + 1] == "26000"


def test_summarizer_distinguishes_valid_slo_boundary_from_partial_run(tmp_path):
    summarizer = _load_module(SUMMARIZER, "summarize_humdial_sweeps_test")
    valid = {
        "deploy_config": "separated_422.yaml",
        "run_error": None,
        "stopped_at": {"rate": 0.025, "reasons": ["streaming RTF"]},
        "gpu_telemetry": {
            "peak_memory_used_mib": 103971,
            "peak_utilization_gpu_percent": 100,
            "peak_power_draw_w": 670.11,
            "per_gpu": {"0": {"peak_memory_used_mib": 101799}},
        },
        "runs": [
            {
                "rate": 0.025,
                "repeat": 1,
                "client_exit_code": 0,
                "client_monitor_error": None,
                "realtime_slo_pass": False,
                "realtime_slo_reasons": ["streaming RTF"],
                "summary": {
                    "request_count": 8,
                    "success_count": 8,
                    "failure_count": 0,
                    "maximum_client_concurrency": 3,
                    "model_unit_decision_latency_ms": {"p99": 670.57},
                    "streaming_audio_rtf": {"p50": 1.0166, "p99": 1.4410},
                    "request_rtf": {"p99": 1.1706},
                    "playout_deadline_miss_rate": 0.6989,
                    "playback_underrun_ms": {"mean": 383.02},
                },
            }
        ],
    }
    partial = {
        **valid,
        "deploy_config": "downstream_colocated.yaml",
        "stopped_at": None,
        "runs": [
            {
                **valid["runs"][0],
                "client_exit_code": 1,
                "client_monitor_error": "server health failed",
                "summary": {"error": "missing summary.json"},
            }
        ],
    }

    valid_records = summarizer.records_from_payload(
        tmp_path / "valid" / "sweep_summary.json", valid
    )
    partial_records = summarizer.records_from_payload(
        tmp_path / "partial" / "sweep_summary.json", partial
    )

    assert valid_records[0]["sweep_status"] == "complete"
    assert valid_records[0]["run_status"] == "slo_fail"
    assert valid_records[0]["request_count"] == 8
    assert valid_records[0]["streaming_rtf_p99"] == pytest.approx(1.4410)
    assert valid_records[0]["peak_gpu_memory_mib"] == 103971
    assert partial_records[0]["sweep_status"] == "partial"
    assert partial_records[0]["run_status"] == "partial"

    six_thinker = {**valid, "deploy_config": "thinker6_downstream_colocated.yaml"}
    six_thinker_records = summarizer.records_from_payload(
        tmp_path / "six_thinker" / "sweep_summary.json", six_thinker
    )
    assert six_thinker_records[0]["topology"] == "thinker6_downstream_colocated"

    async_stage1 = {
        **valid,
        "deploy_config": "thinker6_downstream_colocated_async_stage1.yaml",
    }
    async_stage1_records = summarizer.records_from_payload(
        tmp_path / "async_stage1" / "sweep_summary.json", async_stage1
    )
    assert async_stage1_records[0]["topology"] == "thinker6_downstream_colocated_async_stage1"


def test_repository_topology_configs_resolve_without_intermediate_dependencies():
    from vllm_omni.config.stage_config import resolve_deploy_yaml

    config_dir = ROOT / "benchmarks" / "minicpmo" / "configs" / "humdial_8gpu"
    expected_devices = {
        "separated_422.yaml": ["0,1,2,3", "4,5", "6,7"],
        "downstream_colocated.yaml": ["0,1,2,3", "4,5,6,7", "4,5,6,7"],
        "thinker6_downstream_colocated.yaml": ["0,1,2,3,4,5", "6,7", "6,7"],
        "full_colocated.yaml": ["0,1,2,3,4,5,6,7"] * 3,
    }
    expected_replicas = {
        "separated_422.yaml": [4, 2, 2],
        "downstream_colocated.yaml": [4, 4, 4],
        "thinker6_downstream_colocated.yaml": [6, 2, 2],
        "full_colocated.yaml": [8, 8, 8],
    }

    base = resolve_deploy_yaml(config_dir / "base_h200_b32_fullgraph256.yaml")
    assert base["stages"][2]["max_model_len"] == 65536

    for filename, devices in expected_devices.items():
        config_path = config_dir / filename
        assert "intermediate/" not in config_path.read_text(encoding="utf-8")
        resolved = resolve_deploy_yaml(config_path)
        assert [stage["devices"] for stage in resolved["stages"]] == devices
        assert [stage["num_replicas"] for stage in resolved["stages"]] == expected_replicas[filename]
        assert resolved["active_stream_window"] == 64
        assert resolved["duplex_session"]["max_sessions"] == 64
        assert resolved["connectors"]["connector_of_shared_memory"]["extra"][
            "cfm_max_graphs"
        ] == 256


def test_parallel_launcher_splits_replicas_into_stage_waves(tmp_path):
    sys.path.insert(0, str(SERVICE.parent))
    service = _load_module(SERVICE, "benchmark_humdial_service_parallel_test")
    config_path = tmp_path / "parallel.yaml"
    config_path.write_text(
        """
stages:
  - stage_id: 0
    devices: "0,1,2,3"
    num_replicas: 4
  - stage_id: 1
    devices: "4,5"
    num_replicas: 2
  - stage_id: 2
    devices: "4,5"
    num_replicas: 2
""".lstrip(),
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {
            "deploy_config": config_path,
            "gpus": [10, 11, 12, 13, 14, 15],
            "model": tmp_path / "model",
            "omni_master_address": "127.0.0.1",
            "omni_master_port": 26000,
            "host": "127.0.0.1",
            "port": 8113,
            "startup_timeout_s": 60.0,
            "omni_lb_policy": "least-queue-length",
        },
    )()

    head, groups = service._parallel_replica_specs(args)

    assert head.stage_id == 0
    assert head.replica_id == 0
    assert head.logical_devices == (0,)
    assert head.physical_devices == (10,)
    assert [[(item.stage_id, item.replica_id) for item in group] for group in groups] == [
        [(0, 1), (0, 2), (0, 3)],
        [(1, 0), (1, 1)],
        [(2, 0), (2, 1)],
    ]
    assert groups[0][0].physical_devices == (11,)
    assert groups[1][1].physical_devices == (15,)

    import json

    head_command = service._parallel_command(args, head, head=True)
    overrides = json.loads(head_command[head_command.index("--stage-overrides") + 1])
    assert overrides == {"0": {"devices": "0", "num_replicas": 1}}
    assert "--stage-id" in head_command


def test_service_metadata_records_cuda_loader_environment():
    service = _load_module(SERVICE, "benchmark_humdial_service_environment_test")
    snapshot = service._environment_snapshot(
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "VLLM_ENABLE_CUDA_COMPATIBILITY": "0",
            "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64",
        }
    )
    assert snapshot["VLLM_ENABLE_CUDA_COMPATIBILITY"] == "0"
    assert snapshot["LD_LIBRARY_PATH"] == "/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64"
    assert snapshot["VLLM_CUDA_COMPATIBILITY_PATH"] is None
