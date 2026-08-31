# SPDX-License-Identifier: Apache-2.0
"""Run MiniCPM-o 4.5 native-duplex service saturation on one GPU.

This launches the real three-stage vLLM-Omni server, drives synchronized
response-required sessions with the repository's E2E client, and summarizes
both client deadlines and engine stage metrics.  The launcher refuses to use a
GPU that already has material memory allocated, and cleanup targets only the
process group created by this invocation.

Example:

    /app/vllm-omni/.venv/bin/python \
      benchmarks/minicpmo/benchmark_duplex_service_saturation.py \
      --gpu 4 --session-counts 2,4,8 \
      --deploy-config intermediate/minicpmo45_encoder_bench/minicpmo_4_5_duplex_b8.yaml \
      --output-dir intermediate/minicpmo45_encoder_bench/duplex_service_b8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import IO, Any

import yaml
from summarize_duplex_saturation import summarize_run

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path("/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/models--OpenBMB--MiniCPM-o-4_5")
DEFAULT_INPUT_WAV = ROOT / "tests" / "assets" / "minicpmo_4_5" / "response_required_16k.wav"
CLIENT = ROOT / "tests" / "e2e" / "online_serving" / "run_minicpmo_realtime_duplex_multi_session.py"


def _csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _csv_gpu_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated non-negative GPU indices")
    return values


def _selected_gpus(args: argparse.Namespace) -> list[int]:
    requested = args.gpus if args.gpus is not None else [args.gpu]
    return list(dict.fromkeys(requested))


def _nvidia_smi_binary() -> str:
    candidates = []
    driver_bin = os.environ.get("NVIDIA_DRIVER_BIN")
    if driver_bin:
        candidates.append(Path(driver_bin) / "nvidia-smi")
    candidates.append(Path("/usr/bin/nvidia-smi"))
    resolved = shutil.which("nvidia-smi")
    if resolved:
        candidates.append(Path(resolved))
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return str(candidate)
    raise FileNotFoundError("could not find a non-empty nvidia-smi executable")


def _gpu_snapshot(gpu: int) -> dict[str, object]:
    fields = (
        "index",
        "uuid",
        "name",
        "driver_version",
        "memory.used",
        "memory.total",
        "utilization.gpu",
        "power.draw",
    )
    result = subprocess.run(
        [
            _nvidia_smi_binary(),
            f"--id={gpu}",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected one nvidia-smi row for GPU {gpu}, got {lines!r}")
    values = [value.strip() for value in lines[0].split(",")]
    if len(values) != len(fields):
        raise RuntimeError(f"unexpected nvidia-smi row for GPU {gpu}: {lines[0]!r}")
    snapshot: dict[str, object] = dict(zip(fields, values))
    for field in ("index", "memory.used", "memory.total", "utilization.gpu"):
        snapshot[field] = int(str(snapshot[field]))
    try:
        snapshot["power.draw"] = float(str(snapshot["power.draw"]))
    except ValueError:
        pass
    return snapshot


def _compute_apps_snapshot(gpu: int) -> list[dict[str, object]]:
    result = subprocess.run(
        [
            _nvidia_smi_binary(),
            f"--id={gpu}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    apps: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",", maxsplit=2)]
        if len(values) != 3:
            continue
        pid, process_name, used_memory = values
        apps.append(
            {
                "pid": int(pid) if pid.isdigit() else pid,
                "process_name": process_name,
                "used_memory_mib": int(used_memory) if used_memory.isdigit() else used_memory,
            }
        )
    return apps


def _gpu_sample(gpu: int, *, include_compute_apps: bool = False) -> dict[str, object]:
    sample: dict[str, object] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "monotonic_ns": time.monotonic_ns(),
        "gpu": _gpu_snapshot(gpu),
    }
    if include_compute_apps:
        sample["compute_apps"] = _compute_apps_snapshot(gpu)
    return sample


def _gpu_pool_sample(gpus: Sequence[int], *, include_compute_apps: bool = False) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for gpu in gpus:
        row = _gpu_snapshot(gpu)
        if include_compute_apps:
            row["compute_apps"] = _compute_apps_snapshot(gpu)
        rows.append(row)
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "monotonic_ns": time.monotonic_ns(),
        "gpus": rows,
    }


def _verify_stably_idle_gpu(
    gpu: int,
    *,
    max_memory_mib: int,
    stability_s: float,
    sample_interval_s: float = 1.0,
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    started = time.monotonic()
    while True:
        sample = _gpu_sample(gpu, include_compute_apps=True)
        samples.append(sample)
        used_mib = int(sample["gpu"]["memory.used"])  # type: ignore[index]
        if used_mib > max_memory_mib:
            raise RuntimeError(
                f"GPU {gpu} uses {used_mib} MiB during idle preflight, above the safety threshold {max_memory_mib} MiB"
            )
        elapsed = time.monotonic() - started
        if elapsed >= stability_s:
            return samples
        time.sleep(min(sample_interval_s, stability_s - elapsed))


def _verify_stably_idle_gpus(
    gpus: Sequence[int],
    *,
    max_memory_mib: int,
    stability_s: float,
    sample_interval_s: float = 1.0,
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    started = time.monotonic()
    while True:
        sample = _gpu_pool_sample(gpus, include_compute_apps=True)
        samples.append(sample)
        for row in sample["gpus"]:  # type: ignore[union-attr]
            used_mib = int(row["memory.used"])
            if used_mib > max_memory_mib:
                raise RuntimeError(
                    f"GPU {row['index']} uses {used_mib} MiB during idle preflight, above the safety "
                    f"threshold {max_memory_mib} MiB"
                )
        elapsed = time.monotonic() - started
        if elapsed >= stability_s:
            return samples
        time.sleep(min(sample_interval_s, stability_s - elapsed))


class _GpuTelemetry:
    def __init__(self, gpu: int | Sequence[int], output_path: Path, interval_s: float) -> None:
        self._gpus = [gpu] if isinstance(gpu, int) else list(gpu)
        self._output_path = output_path
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.rows: list[dict[str, object]] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="gpu-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self._interval_s * 3))
            if self._thread.is_alive():
                raise RuntimeError("GPU telemetry thread did not stop")

    def _run(self) -> None:
        with self._output_path.open("w", encoding="utf-8", buffering=1) as handle:
            sample_index = 0
            while not self._stop_event.is_set():
                try:
                    row = _gpu_pool_sample(self._gpus, include_compute_apps=sample_index % 10 == 0)
                except Exception as error:
                    row = {
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                        "monotonic_ns": time.monotonic_ns(),
                        "error": repr(error),
                    }
                self.rows.append(row)
                print(json.dumps(row, ensure_ascii=False, sort_keys=True), file=handle, flush=True)
                sample_index += 1
                self._stop_event.wait(self._interval_s)


def _telemetry_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    gpu_rows: list[dict[str, object]] = []
    for sample in rows:
        legacy = sample.get("gpu")
        if isinstance(legacy, dict):
            gpu_rows.append(legacy)
        pool = sample.get("gpus")
        if isinstance(pool, list):
            gpu_rows.extend(row for row in pool if isinstance(row, dict))

    def numeric(field: str) -> list[float]:
        return [
            float(row[field]) for row in gpu_rows if isinstance(row, dict) and isinstance(row.get(field), int | float)
        ]

    memory = numeric("memory.used")
    utilization = numeric("utilization.gpu")
    power = numeric("power.draw")
    per_gpu: dict[str, dict[str, float | int | None]] = {}
    for index in sorted({int(row["index"]) for row in gpu_rows if isinstance(row.get("index"), int)}):
        selected = [row for row in gpu_rows if row.get("index") == index]

        def peak(field: str) -> float | None:
            values = [float(row[field]) for row in selected if isinstance(row.get(field), int | float)]
            return max(values) if values else None

        per_gpu[str(index)] = {
            "sample_count": len(selected),
            "peak_memory_used_mib": peak("memory.used"),
            "peak_utilization_gpu_percent": peak("utilization.gpu"),
            "peak_power_draw_w": peak("power.draw"),
        }

    return {
        "sample_count": len(rows),
        "gpu_observation_count": len(gpu_rows),
        "error_count": sum("error" in row for row in rows),
        "peak_memory_used_mib": max(memory) if memory else None,
        "peak_utilization_gpu_percent": max(utilization) if utilization else None,
        "peak_power_draw_w": max(power) if power else None,
        "per_gpu": per_gpu,
    }


def _latest_cfm_graph_stats(server_log_path: Path) -> dict[str, int] | None:
    marker = "CFM CUDA Graph stats "
    latest: dict[str, int] | None = None
    if not server_log_path.is_file():
        return None
    with server_log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if marker not in line:
                continue
            try:
                candidate = json.loads(line.partition(marker)[2])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and all(isinstance(value, int) for value in candidate.values()):
                latest = candidate
    return latest


def _latest_cfm_graph_stats_per_replica(server_log_path: Path) -> dict[str, dict[str, int]]:
    marker = "CFM CUDA Graph stats "
    replica_marker = "StageEngineCoreProc_stage2_replica"
    latest: dict[str, dict[str, int]] = {}
    if not server_log_path.is_file():
        return latest
    with server_log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if marker not in line:
                continue
            replica = "unattributed"
            if replica_marker in line:
                replica = line.partition(replica_marker)[2].split(maxsplit=1)[0]
            try:
                candidate = json.loads(line.partition(marker)[2])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and all(isinstance(value, int) for value in candidate.values()):
                latest[replica] = candidate
    return latest


def _duplex_append_timing_records(
    server_log_path: Path,
    *,
    session_ids: set[str],
) -> list[dict[str, object]]:
    marker = "MiniCPM-o duplex append timing "
    starts: dict[tuple[str, str], dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    if not server_log_path.is_file() or not session_ids:
        return rows
    with server_log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if marker not in line:
                continue
            try:
                record = json.loads(line.partition(marker)[2])
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("session_id") not in session_ids:
                continue
            if record.get("kind") != "input" or not isinstance(record.get("benchmark_tick_index"), int):
                continue
            operation_id = record.get("operation_id")
            session_id = record.get("session_id")
            if not isinstance(operation_id, str) or not isinstance(session_id, str):
                continue
            key = (session_id, operation_id)
            if record.get("phase") == "start":
                starts[key] = record
                continue
            if record.get("phase") not in {"end", "error"}:
                continue
            start = starts.pop(key, None)
            if start is None:
                continue
            scheduled_ns = start.get("benchmark_scheduled_monotonic_ns")
            started_ns = start.get("monotonic_ns")
            finished_ns = record.get("monotonic_ns")
            if not all(isinstance(value, int) for value in (scheduled_ns, started_ns, finished_ns)):
                continue
            rows.append(
                {
                    "session_id": session_id,
                    "operation_id": operation_id,
                    "tick_index": start["benchmark_tick_index"],
                    "audio_end_ms": start.get("client_audio_end_ms"),
                    "scheduled_monotonic_ns": scheduled_ns,
                    "append_started_monotonic_ns": started_ns,
                    "append_finished_monotonic_ns": finished_ns,
                    "start_queue_lag_ms": (started_ns - scheduled_ns) / 1_000_000.0,
                    "completion_lag_ms": (finished_ns - scheduled_ns) / 1_000_000.0,
                    "append_elapsed_ms": (finished_ns - started_ns) / 1_000_000.0,
                    "phase": record["phase"],
                    "append_ok": record.get("append_ok"),
                    "emitted_response": record.get("emitted_response"),
                }
            )
    return sorted(rows, key=lambda row: (str(row["session_id"]), int(row["tick_index"])))


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_server(base_url: str, process: subprocess.Popen[str], timeout_s: float) -> None:
    # Benchmark hosts commonly export an HTTP proxy for model/network access.
    # Health checks are always local and must never be routed through it.
    direct_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout_s
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"server exited during startup with code {return_code}")
        try:
            with direct_opener.open(f"{base_url}/health", timeout=2) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, urllib.error.URLError) as error:
            last_error = repr(error)
        time.sleep(1)
    raise TimeoutError(f"server was not healthy after {timeout_s:.0f}s: {last_error}")


def _local_health_status(base_url: str) -> int | None:
    direct_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with direct_opener.open(f"{base_url}/health", timeout=2) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except (OSError, urllib.error.URLError):
        return None


def _stop_client(process: subprocess.Popen[str]) -> None:
    for sig, timeout_s in ((signal.SIGINT, 5), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        if process.poll() is not None:
            return
        process.send_signal(sig)
        try:
            process.wait(timeout=timeout_s)
            return
        except subprocess.TimeoutExpired:
            continue


def _run_monitored_client(
    command: list[str],
    *,
    env: dict[str, str],
    log: IO[str],
    server: subprocess.Popen[str],
    base_url: str,
) -> tuple[int, str | None]:
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    unhealthy_checks = 0
    monitor_error = None
    while process.poll() is None:
        time.sleep(1)
        server_exit_code = server.poll()
        health_status = _local_health_status(base_url)
        unhealthy_checks = unhealthy_checks + 1 if health_status != 200 else 0
        if server_exit_code is not None:
            monitor_error = f"server exited during client run with code {server_exit_code}"
        elif unhealthy_checks >= 3:
            monitor_error = f"server health failed 3 consecutive checks (last status={health_status})"
        if monitor_error is not None:
            print(monitor_error, file=log, flush=True)
            _stop_client(process)
            break
    return process.wait(), monitor_error


def _stop_process_group(process: subprocess.Popen[str], process_group: int, log: IO[str]) -> None:
    for sig, timeout_s in ((signal.SIGINT, 30), (signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        if not _process_group_exists(process_group):
            process.poll()
            return
        try:
            os.killpg(process_group, sig)
        except ProcessLookupError:
            process.poll()
            return
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            process.poll()
            if not _process_group_exists(process_group):
                return
            time.sleep(0.25)
        print(f"server process group did not stop after {sig.name}", file=log, flush=True)
    raise RuntimeError(f"server process group {process_group} survived SIGINT/SIGTERM/SIGKILL")


def _git_metadata() -> dict[str, object]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.rstrip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "status_short": git("status", "--short"),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": repr(error)}


def _deploy_config_snapshot(config_path: Path) -> list[dict[str, str]]:
    snapshots: list[dict[str, str]] = []
    seen: set[Path] = set()
    current = config_path.resolve()
    while current not in seen:
        seen.add(current)
        text = current.read_text(encoding="utf-8")
        snapshots.append({"path": str(current), "content": text})
        payload = yaml.safe_load(text)
        base = payload.get("base_config") if isinstance(payload, dict) else None
        if not isinstance(base, str) or not base:
            break
        base_path = Path(base)
        current = base_path.resolve() if base_path.is_absolute() else (current.parent / base_path).resolve()
    return snapshots


def _json_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_is_valid(summary: dict[str, object], *, sessions: int, exit_code: int) -> list[str]:
    reasons: list[str] = []
    if exit_code != 0:
        reasons.append(f"client exit code was {exit_code}")
    if summary.get("ok") is not True:
        reasons.append("summary ok was not true")
    if summary.get("configured_session_count") != sessions:
        reasons.append("configured session count did not match")
    if summary.get("completed_session_count") != sessions:
        reasons.append("completed session count did not match")
    if summary.get("failures"):
        reasons.append("summary contained failures")
    return reasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--deploy-config", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument(
        "--gpus",
        type=_csv_gpu_ints,
        default=None,
        help="Comma-separated physical GPU pool. Overrides --gpu and becomes CUDA_VISIBLE_DEVICES in this order.",
    )
    parser.add_argument("--session-counts", type=_csv_ints, default=_csv_ints("2,4,8"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8113)
    parser.add_argument("--startup-timeout-s", type=float, default=900)
    parser.add_argument("--request-timeout-s", type=float, default=240)
    parser.add_argument("--max-preexisting-memory-mib", type=int, default=1024)
    parser.add_argument("--idle-stability-s", type=float, default=30)
    parser.add_argument("--telemetry-interval-s", type=float, default=1)
    parser.add_argument("--input-wav", type=Path, default=DEFAULT_INPUT_WAV)
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--frame-image", type=Path, help="Image sent once per native 1 s audio unit.")
    parser.add_argument("--first-turn-ms", type=int, default=1400)
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--open-loop-units",
        type=int,
        default=0,
        help="Use fixed-period native input instead of response-gated turns.",
    )
    parser.add_argument("--open-loop-period-ms", type=int, default=1000)
    parser.add_argument("--open-loop-drain-s", type=float, default=5.0)
    parser.add_argument(
        "--open-loop-speak-duty",
        type=float,
        default=None,
        help="Experimental exact speak-unit duty in [0,1]; each response is cancelled after its first audio chunk.",
    )
    parser.add_argument(
        "--open-loop-cancel-after-audio-ms",
        type=int,
        default=0,
        help="Delay before cancelling a fixed-duty response after its first audio delta.",
    )
    parser.add_argument(
        "--open-loop-require-audio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require speech output in open-loop runs; disabled for the Stage0 listen-only capacity workload.",
    )
    parser.add_argument("--model-policy-settle-ms", type=int, default=2000)
    parser.add_argument(
        "--connection-stagger-ms",
        type=float,
        default=0.0,
        help="Stagger session handshakes while retaining synchronized workload start.",
    )
    args = parser.parse_args()
    if args.ref_audio is None:
        args.ref_audio = args.model / "assets" / "HT_ref_audio.wav"
    for path_name in ("model", "deploy_config", "input_wav", "ref_audio"):
        if not getattr(args, path_name).exists():
            parser.error(f"--{path_name.replace('_', '-')} does not exist: {getattr(args, path_name)}")
    if args.frame_image is not None and not args.frame_image.is_file():
        parser.error(f"--frame-image does not exist: {args.frame_image}")
    if any(gpu < 0 for gpu in _selected_gpus(args)) or args.port <= 0 or args.max_preexisting_memory_mib < 0:
        parser.error("gpu, port, and memory threshold must be non-negative/positive")
    if args.turns <= 0 or args.repeats <= 0:
        parser.error("--turns and --repeats must be positive")
    if args.open_loop_units < 0 or args.open_loop_period_ms <= 0 or args.open_loop_drain_s < 0:
        parser.error("open-loop units/drain must be non-negative and period must be positive")
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
    if args.idle_stability_s < 0 or args.telemetry_interval_s <= 0:
        parser.error("--idle-stability-s must be non-negative and --telemetry-interval-s must be positive")
    if args.connection_stagger_ms < 0:
        parser.error("--connection-stagger-ms must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    gpus = _selected_gpus(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to mix artifacts in non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"verifying GPUs {gpus} remain below {args.max_preexisting_memory_mib} MiB for {args.idle_stability_s:.0f}s",
        flush=True,
    )
    preflight_samples = _verify_stably_idle_gpus(
        gpus,
        max_memory_mib=args.max_preexisting_memory_mib,
        stability_s=args.idle_stability_s,
    )
    initial_gpus = preflight_samples[-1]["gpus"]
    used_mib = {str(row["index"]): int(row["memory.used"]) for row in initial_gpus}  # type: ignore[union-attr]

    server_log_path = args.output_dir / "server.log"
    metadata_path = args.output_dir / "metadata.json"
    telemetry_path = args.output_dir / "gpu_telemetry.jsonl"
    output_path = args.output_dir / "saturation_summary.json"
    base_url = f"http://{args.host}:{args.port}"
    realtime_url = f"ws://{args.host}:{args.port}/v1/realtime?duplex=1"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)
    python_paths = [str(ROOT)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    if args.open_loop_units > 0:
        env["MINICPMO_DUPLEX_TIMING"] = "1"
    no_proxy_hosts = {args.host, "127.0.0.1", "localhost", "::1"}
    for variable in ("NO_PROXY", "no_proxy"):
        no_proxy_hosts.update(item.strip() for item in env.get(variable, "").split(",") if item.strip())
        env[variable] = ",".join(sorted(no_proxy_hosts))
    server_command = [
        sys.executable,
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        str(args.model),
        "--omni",
        "--deploy-config",
        str(args.deploy_config.resolve()),
        "--trust-remote-code",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--stage-init-timeout",
        str(int(args.startup_timeout_s)),
        "--init-timeout",
        str(int(args.startup_timeout_s)),
        "--log-stats",
    ]

    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "launcher_command": [sys.executable, *sys.argv],
        "server_command": server_command,
        "arguments": _json_args(args),
        "git": _git_metadata(),
        "deploy_configs": _deploy_config_snapshot(args.deploy_config),
        "gpu_preflight": preflight_samples,
        "compute_apps_before_server": {str(gpu): _compute_apps_snapshot(gpu) for gpu in gpus},
        "environment": {
            name: env.get(name)
            for name in (
                "PATH",
                "LD_LIBRARY_PATH",
                "CUDA_HOME",
                "CUDA_VISIBLE_DEVICES",
                "VLLM_ENABLE_CUDA_COMPATIBILITY",
                "VLLM_CUDA_COMPATIBILITY_PATH",
                "MINICPMO_CODE2WAV_TIMING",
                "MINICPMO_DUPLEX_TIMING",
                "PYTHONPATH",
            )
        },
    }
    _write_json(metadata_path, metadata)

    summaries: list[dict[str, object]] = []
    stopped_early: dict[str, object] | None = None
    run_error: BaseException | None = None
    server: subprocess.Popen[str] | None = None
    process_group: int | None = None
    telemetry = _GpuTelemetry(gpus, telemetry_path, args.telemetry_interval_s)
    telemetry.start()
    with server_log_path.open("w", encoding="utf-8", buffering=1) as server_log:
        print(json.dumps({"server_command": server_command, "gpus": gpus}), file=server_log, flush=True)
        try:
            server = subprocess.Popen(
                server_command,
                cwd=ROOT,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            process_group = os.getpgid(server.pid)
            metadata["server_pid"] = server.pid
            metadata["server_process_group"] = process_group
            _write_json(metadata_path, metadata)
            _wait_for_server(base_url, server, args.startup_timeout_s)
            for sessions in args.session_counts:
                for repeat in range(1, args.repeats + 1):
                    run_dir = args.output_dir / f"sessions_{sessions}_rep_{repeat}"
                    run_dir.mkdir(parents=True, exist_ok=False)
                    client_command = [
                        sys.executable,
                        str(CLIENT),
                        "--url",
                        realtime_url,
                        "--model",
                        str(args.model),
                        "--sessions",
                        str(sessions),
                        "--input-wav",
                        str(args.input_wav),
                        "--ref-audio",
                        str(args.ref_audio),
                        "--output-dir",
                        str(run_dir),
                        "--synchronized-start",
                        "--temperature",
                        "0.0",
                        "--timeout-s",
                        str(args.request_timeout_s),
                        "--model-policy-settle-ms",
                        str(args.model_policy_settle_ms),
                        "--connection-stagger-ms",
                        str(args.connection_stagger_ms),
                    ]
                    if args.frame_image is not None:
                        client_command.extend(["--frame-image", str(args.frame_image)])
                    if args.open_loop_units > 0:
                        client_command.extend(
                            [
                                "--open-loop-units",
                                str(args.open_loop_units),
                                "--open-loop-period-ms",
                                str(args.open_loop_period_ms),
                                "--open-loop-drain-s",
                                str(args.open_loop_drain_s),
                                "--open-loop-cancel-after-audio-ms",
                                str(args.open_loop_cancel_after_audio_ms),
                                (
                                    "--open-loop-require-audio"
                                    if args.open_loop_require_audio
                                    else "--no-open-loop-require-audio"
                                ),
                            ]
                        )
                        if args.open_loop_speak_duty is not None:
                            client_command.extend(["--open-loop-speak-duty", str(args.open_loop_speak_duty)])
                    else:
                        client_command.extend(
                            [
                                "--turns",
                                str(args.turns),
                                "--realtime-input",
                                "--response-required",
                                "--chunk-ms",
                                str(args.chunk_ms),
                                "--first-turn-ms",
                                str(args.first_turn_ms),
                            ]
                        )
                    workload = (
                        f"open_loop_units={args.open_loop_units}" if args.open_loop_units > 0 else f"turns={args.turns}"
                    )
                    print(f"running B={sessions} repeat={repeat}/{args.repeats}, {workload}", flush=True)
                    with (run_dir / "client.log").open("w", encoding="utf-8", buffering=1) as client_log:
                        client_exit_code, client_monitor_error = _run_monitored_client(
                            client_command,
                            env=env,
                            log=client_log,
                            server=server,
                            base_url=base_url,
                        )
                    summary_path = run_dir / "summary.json"
                    if summary_path.is_file():
                        try:
                            if args.open_loop_units > 0:
                                server_log.flush()
                                raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                                raw_sessions = raw_summary.get("sessions") if isinstance(raw_summary, dict) else None
                                session_ids = {
                                    str(item["session_id"])
                                    for item in raw_sessions or []
                                    if isinstance(item, dict) and isinstance(item.get("session_id"), str)
                                }
                                _write_json(
                                    run_dir / "server_append_timing.json",
                                    {
                                        "session_ids": sorted(session_ids),
                                        "rows": _duplex_append_timing_records(
                                            server_log_path,
                                            session_ids=session_ids,
                                        ),
                                    },
                                )
                            summary = summarize_run(run_dir)
                        except Exception as error:
                            summary = {
                                "run_dir": str(run_dir),
                                "ok": False,
                                "configured_session_count": sessions,
                                "completed_session_count": None,
                                "failures": [f"summary parsing failed: {error!r}"],
                            }
                    else:
                        summary = {
                            "run_dir": str(run_dir),
                            "ok": False,
                            "configured_session_count": sessions,
                            "completed_session_count": None,
                            "failures": [f"client did not write {summary_path}"],
                        }
                    summary["client_exit_code"] = client_exit_code
                    summary["client_monitor_error"] = client_monitor_error
                    summary["repeat"] = repeat
                    summary["turns"] = args.turns
                    summary["open_loop_units"] = args.open_loop_units if args.open_loop_units > 0 else None
                    summary["compute_apps_after_run"] = {str(gpu): _compute_apps_snapshot(gpu) for gpu in gpus}
                    summaries.append(summary)
                    reasons = _run_is_valid(summary, sessions=sessions, exit_code=client_exit_code)
                    if client_monitor_error is not None:
                        reasons.append(client_monitor_error)
                    print(
                        json.dumps(
                            {
                                "B": sessions,
                                "repeat": repeat,
                                "turns": args.turns,
                                "ok": not reasons,
                                "client_exit_code": client_exit_code,
                                "completed_session_count": summary.get("completed_session_count"),
                                "failure_reasons": reasons,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    if reasons:
                        stopped_early = {
                            "sessions": sessions,
                            "repeat": repeat,
                            "reasons": reasons,
                        }
                        break
                if stopped_early is not None:
                    break
        except BaseException as error:
            run_error = error
        finally:
            if server is not None and process_group is not None:
                try:
                    _stop_process_group(server, process_group, server_log)
                except BaseException as error:
                    if run_error is None:
                        run_error = error
                    else:
                        print(f"cleanup also failed: {error!r}", file=server_log, flush=True)

    try:
        telemetry.stop()
    except BaseException as error:
        if run_error is None:
            run_error = error
    output = {
        "model": str(args.model),
        "deploy_config": str(args.deploy_config),
        "gpu": gpus[0] if len(gpus) == 1 else None,
        "gpus": gpus,
        "preexisting_memory_mib": used_mib,
        "turns": args.turns,
        "repeats": args.repeats,
        "stopped_early": stopped_early,
        "run_error": repr(run_error) if run_error is not None else None,
        "gpu_telemetry": _telemetry_summary(telemetry.rows),
        "cfm_graph_stats": _latest_cfm_graph_stats(server_log_path),
        "cfm_graph_stats_per_replica": _latest_cfm_graph_stats_per_replica(server_log_path),
        "runs": summaries,
    }
    _write_json(output_path, output)
    metadata["finished_at_utc"] = datetime.now(UTC).isoformat()
    metadata["compute_apps_after_cleanup"] = {str(gpu): _compute_apps_snapshot(gpu) for gpu in gpus}
    metadata["gpu_telemetry"] = output["gpu_telemetry"]
    metadata["stopped_early"] = stopped_early
    metadata["run_error"] = output["run_error"]
    _write_json(metadata_path, metadata)
    print(f"wrote {output_path}")
    if run_error is not None:
        raise run_error
    if stopped_early is not None:
        raise RuntimeError(f"stopped after invalid client run: {stopped_early}")


if __name__ == "__main__":
    main()
