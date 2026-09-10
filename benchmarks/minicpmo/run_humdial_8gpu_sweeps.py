# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Build or execute reproducible HumDial sweeps for the 8-GPU topologies."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv" / "bin" / "python"
BENCHMARK = Path(__file__).with_name("benchmark_humdial_service.py")
CONFIG_DIR = Path(__file__).with_name("configs") / "humdial_8gpu"
DEFAULT_OUTPUT_ROOT = ROOT / "intermediate" / "humdial_minicpmo_serving"
DEFAULT_DATASET_ROOT = Path(
    "/mnt/nvme1n1/ml_research/linbinbin1/src-omni-modal/Humdial-Track2-Test"
)
DEFAULT_MODEL = Path(
    "/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/"
    "models--OpenBMB--MiniCPM-o-4_5"
)
TOPOLOGIES = {
    "separated_422": CONFIG_DIR / "separated_422.yaml",
    "downstream_colocated": CONFIG_DIR / "downstream_colocated.yaml",
    "thinker6_downstream_colocated": CONFIG_DIR / "thinker6_downstream_colocated.yaml",
    "thinker6_talker4_downstream_colocated": CONFIG_DIR / "thinker6_talker4_downstream_colocated.yaml",
    "thinker6_downstream_colocated_fixed_talker_kv": CONFIG_DIR / "thinker6_downstream_colocated_fixed_talker_kv.yaml",
    "full_colocated": CONFIG_DIR / "full_colocated.yaml",
}
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class SweepPlan:
    topology: str
    output_dir: Path
    command: list[str]


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed < float("inf"):
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed < float("inf"):
        raise argparse.ArgumentTypeError("expected a finite non-negative number")
    return parsed


def _csv_positive_floats(value: str) -> str:
    try:
        parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("rates must be comma-separated numbers") from error
    if not parsed or any(not 0 < item < float("inf") for item in parsed):
        raise argparse.ArgumentTypeError("rates must be finite and positive")
    return ",".join(f"{item:g}" for item in parsed)


def _csv_gpus(value: str) -> str:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPUs must be comma-separated integers") from error
    if not parsed or any(item < 0 for item in parsed) or len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError("GPUs must be unique non-negative indices")
    return ",".join(str(item) for item in parsed)


def _label(value: str) -> str:
    if not _SAFE_LABEL.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "labels must start with an alphanumeric character and contain only "
            "letters, digits, dot, underscore, or dash"
        )
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topology",
        choices=["all", *TOPOLOGIES],
        default="all",
        help="topology to run; all executes the topologies sequentially",
    )
    parser.add_argument("--campaign", type=_label, default="humdial_8gpu")
    parser.add_argument(
        "--attempt",
        type=_label,
        required=True,
        help="new attempt label; an existing directory is always rejected",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--gpus", type=_csv_gpus, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--rates", type=_csv_positive_floats, default="0.025,0.05,0.1")
    parser.add_argument("--duration-s", type=_positive_float, default=300.0)
    parser.add_argument("--seed", type=int, default=20_260_901)
    parser.add_argument("--repeats", type=_positive_int, default=1)
    parser.add_argument("--warmup-rate", type=_positive_float, default=0.05)
    parser.add_argument("--warmup-duration-s", type=_nonnegative_float, default=60.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_positive_int, default=8113)
    parser.add_argument("--omni-master-address", default="127.0.0.1")
    parser.add_argument("--omni-master-port", type=_positive_int, default=26000)
    parser.add_argument(
        "--parallel-stage-launch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "launch stage replicas in separate processes and load replicas within "
            "each stage concurrently"
        ),
    )
    parser.add_argument("--startup-timeout-s", type=_positive_float, default=1800.0)
    parser.add_argument("--request-timeout-s", type=_positive_float, default=180.0)
    parser.add_argument("--chunk-ms", type=_positive_int, default=200)
    parser.add_argument("--tail-drain-s", type=_nonnegative_float, default=2.0)
    parser.add_argument(
        "--model-unit-decision-p99-slo-ms", type=_positive_float, default=1500.0
    )
    parser.add_argument("--telemetry-interval-s", type=_positive_float, default=1.0)
    parser.add_argument("--idle-stability-s", type=_nonnegative_float, default=30.0)
    parser.add_argument("--max-preexisting-memory-mib", type=_positive_int, default=1024)
    parser.add_argument(
        "--omni-lb-policy",
        choices=["random", "round-robin", "least-queue-length"],
        default="least-queue-length",
    )
    parser.add_argument(
        "--stop-on-slo-fail", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="launch services and use GPUs; without this flag only print commands",
    )
    return parser.parse_args(argv)


def build_plans(args: argparse.Namespace) -> list[SweepPlan]:
    topology_names = list(TOPOLOGIES) if args.topology == "all" else [args.topology]
    ref_audio = args.ref_audio or args.model / "assets" / "HT_ref_audio.wav"
    plans: list[SweepPlan] = []
    for topology in topology_names:
        output_dir = args.output_root / args.campaign / topology / args.attempt
        command = [
            str(PYTHON),
            str(BENCHMARK),
            "--model",
            str(args.model),
            "--deploy-config",
            str(TOPOLOGIES[topology]),
            "--dataset-root",
            str(args.dataset_root),
            "--ref-audio",
            str(ref_audio),
            "--gpus",
            args.gpus,
            "--rates",
            args.rates,
            "--duration-s",
            str(args.duration_s),
            "--seed",
            str(args.seed),
            "--repeats",
            str(args.repeats),
            "--warmup-rate",
            str(args.warmup_rate),
            "--warmup-duration-s",
            str(args.warmup_duration_s),
            "--output-dir",
            str(output_dir),
            "--run-label",
            f"humdial-{topology}-{args.attempt}",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--omni-master-address",
            args.omni_master_address,
            "--omni-master-port",
            str(args.omni_master_port),
            "--startup-timeout-s",
            str(args.startup_timeout_s),
            "--request-timeout-s",
            str(args.request_timeout_s),
            "--chunk-ms",
            str(args.chunk_ms),
            "--tail-drain-s",
            str(args.tail_drain_s),
            "--model-unit-decision-p99-slo-ms",
            str(args.model_unit_decision_p99_slo_ms),
            "--telemetry-interval-s",
            str(args.telemetry_interval_s),
            "--idle-stability-s",
            str(args.idle_stability_s),
            "--max-preexisting-memory-mib",
            str(args.max_preexisting_memory_mib),
            "--omni-lb-policy",
            args.omni_lb_policy,
            "--stop-on-slo-fail" if args.stop_on_slo_fail else "--no-stop-on-slo-fail",
        ]
        if args.parallel_stage_launch:
            command.append("--parallel-stage-launch")
        plans.append(SweepPlan(topology=topology, output_dir=output_dir, command=command))
    return plans


def ensure_fresh_outputs(plans: Sequence[SweepPlan]) -> None:
    collisions = [plan.output_dir for plan in plans if plan.output_dir.exists()]
    if collisions:
        paths = ", ".join(str(path) for path in collisions)
        raise FileExistsError(f"fresh attempt required; refusing existing output path(s): {paths}")


def reserve_output_dirs(plans: Sequence[SweepPlan]) -> None:
    """Atomically claim each attempt directory immediately before execution."""
    for plan in plans:
        plan.output_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            plan.output_dir.mkdir()
        except FileExistsError as error:
            raise FileExistsError(
                f"fresh attempt required; output was claimed concurrently: {plan.output_dir}"
            ) from error


def _validate_execution_inputs(args: argparse.Namespace, plans: Sequence[SweepPlan]) -> None:
    ref_audio = args.ref_audio or args.model / "assets" / "HT_ref_audio.wav"
    paths = [PYTHON, BENCHMARK, args.model, args.dataset_root, ref_audio]
    paths.extend(TOPOLOGIES[plan.topology] for plan in plans)
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required execution path(s) missing: {missing}")


def _completion_error(summary_path: Path) -> str | None:
    if not summary_path.is_file():
        return f"missing {summary_path}"
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return f"unreadable summary: {error}"
    if payload.get("run_error") is not None:
        return f"benchmark run_error: {payload['run_error']}"
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        return "summary has no completed workload run"
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            return f"run {index} is malformed"
        if run.get("client_exit_code") != 0 or run.get("client_monitor_error"):
            return f"run {index} client did not complete cleanly"
        summary = run.get("summary")
        if not isinstance(summary, dict):
            return f"run {index} has no complete client summary"
        request_count = summary.get("request_count")
        success_count = summary.get("success_count")
        failure_count = summary.get("failure_count")
        if not isinstance(request_count, int) or request_count <= 0:
            return f"run {index} has no positive request count"
        if not isinstance(success_count, int) or not isinstance(failure_count, int):
            return f"run {index} has no success/failure counts"
        if success_count + failure_count != request_count:
            return f"run {index} success/failure counts do not cover every request"
    return None


def execute_plans(args: argparse.Namespace, plans: Sequence[SweepPlan]) -> None:
    _validate_execution_inputs(args, plans)
    reserve_output_dirs(plans)
    for index, plan in enumerate(plans, start=1):
        print(f"[{index}/{len(plans)}] executing {plan.topology}", flush=True)
        completed = subprocess.run(plan.command, cwd=ROOT, check=False)
        if completed.returncode:
            raise RuntimeError(
                f"{plan.topology} benchmark exited with code {completed.returncode}; "
                f"artifacts remain at {plan.output_dir} and must not be reused"
            )
        completion_error = _completion_error(plan.output_dir / "sweep_summary.json")
        if completion_error:
            raise RuntimeError(
                f"{plan.topology} produced a partial attempt: {completion_error}; "
                f"artifacts remain at {plan.output_dir} and must not be reused"
            )
        print(f"[{index}/{len(plans)}] complete: {plan.output_dir}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plans = build_plans(args)
    ensure_fresh_outputs(plans)
    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"mode: {mode}")
    for plan in plans:
        print(f"\n[{plan.topology}]\noutput: {plan.output_dir}\n{shlex.join(plan.command)}")
    if not args.execute:
        print("\nNo service was started. Add --execute only after GPU allocation is approved.")
        return 0
    execute_plans(args, plans)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
