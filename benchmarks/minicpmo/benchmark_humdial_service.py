# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Launch MiniCPM-o and sweep fixed HumDial session-arrival rates."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import benchmark_duplex_service_saturation as service

ROOT = Path(__file__).resolve().parents[2]
CLIENT = Path(__file__).with_name("humdial_arrival_rate.py")
E2E_CLIENT = Path(__file__).with_name("humdial_e2e.py")
DEFAULT_DATASET_ROOT = Path("/mnt/nvme1n1/ml_research/linbinbin1/src-omni-modal/Humdial-Track2-Test")
DEFAULT_MODEL = Path("/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/models--OpenBMB--MiniCPM-o-4_5")


@dataclass(frozen=True)
class _ReplicaLaunchSpec:
    """One stage replica and the physical GPUs assigned to its process."""

    stage_id: int
    replica_id: int
    logical_devices: tuple[int, ...]
    physical_devices: tuple[int, ...]


@dataclass
class _ManagedProcess:
    """A benchmark-owned process with an isolated process group and log."""

    name: str
    command: list[str]
    env: dict[str, str]
    process: subprocess.Popen[str]
    process_group: int
    log_path: Path
    log_file: Any


@dataclass
class _ParallelServer:
    """Head process plus headless stage workers used by parallel startup."""

    head: _ManagedProcess
    workers: list[_ManagedProcess]

    @property
    def pid(self) -> int:
        return self.head.process.pid

    @property
    def process_groups(self) -> list[int]:
        return [item.process_group for item in [self.head, *self.workers]]

    @property
    def managed_processes(self) -> list[_ManagedProcess]:
        return [self.head, *self.workers]

    def poll(self) -> int | None:
        """Expose Popen-like liveness for the existing health/client helpers."""
        for item in self.managed_processes:
            return_code = item.process.poll()
            if return_code is not None:
                return return_code
        return None


_ENGINE_READY_RE = re.compile(r"init engine \(profile, create kv cache, warmup model\) took")
_REMOTE_ATTACH_RE = re.compile(
    r"Remote (?:LLM|diffusion) replica attached stage=(?P<stage>\d+) replica=(?P<replica>\d+)"
)
_RECORDED_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "MINICPMO_DUPLEX_TIMING",
    "MINICPMO_CODE2WAV_TIMING",
    "PYTHONPATH",
    "VLLM_ENABLE_CUDA_COMPATIBILITY",
    "VLLM_CUDA_COMPATIBILITY_PATH",
    "CUDA_HOME",
    "NVIDIA_DRIVER_BIN",
    "NVIDIA_DRIVER_LIB",
    "LD_LIBRARY_PATH",
)


def _environment_snapshot(env: dict[str, str]) -> dict[str, str | None]:
    """Record CUDA loader settings needed to reproduce a benchmark attempt."""
    return {key: env.get(key) for key in _RECORDED_ENVIRONMENT_KEYS}


def _csv_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(not 0 < item < float("inf") for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated finite positive rates")
    return values


def _csv_gpus(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 for item in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected unique comma-separated non-negative GPU indices")
    return values


def _json_arguments(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _nested_number(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    return float(value) if isinstance(value, int | float) else None


def _parse_device_ids(value: object, *, stage_id: int) -> list[int]:
    """Parse a deploy-stage device list for the local parallel launcher."""
    if isinstance(value, int):
        values = [value]
    elif isinstance(value, str):
        try:
            values = [int(item.strip()) for item in value.split(",") if item.strip()]
        except ValueError as error:
            raise ValueError(f"stage {stage_id} devices must be integer ids, got {value!r}") from error
    elif isinstance(value, list):
        try:
            values = [int(item) for item in value]
        except (TypeError, ValueError) as error:
            raise ValueError(f"stage {stage_id} devices must be integer ids, got {value!r}") from error
    else:
        raise ValueError(
            f"parallel stage startup requires explicit integer devices for stage {stage_id}; got {value!r}"
        )
    if not values or any(value < 0 for value in values):
        raise ValueError(f"stage {stage_id} devices must be non-negative and non-empty, got {value!r}")
    return values


def _parallel_replica_specs(args: argparse.Namespace) -> tuple[_ReplicaLaunchSpec, list[list[_ReplicaLaunchSpec]]]:
    """Resolve per-replica placement and stage-order launch groups.

    The head owns replica 0 of stage 0. Remaining replicas are launched as
    headless processes, one stage at a time; replicas within a stage are
    launched concurrently. Stage ordering intentionally keeps replicas that
    share a GPU from loading at the same time (the topology may colocate
    stages, while the current benchmark's stage configs enumerate a full
    per-replica device pool).
    """
    from vllm_omni.config.stage_config import resolve_deploy_yaml

    payload = resolve_deploy_yaml(args.deploy_config)
    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError(f"deploy config has no stages: {args.deploy_config}")

    stages: list[tuple[int, int, list[int]]] = []
    for raw_stage in raw_stages:
        if not isinstance(raw_stage, dict):
            raise ValueError(f"invalid stage entry in deploy config: {raw_stage!r}")
        stage_id = int(raw_stage["stage_id"])
        runtime = raw_stage.get("runtime")
        runtime = runtime if isinstance(runtime, dict) else {}
        devices_value = runtime.get("devices", raw_stage.get("devices"))
        replicas_value = runtime.get("num_replicas", raw_stage.get("num_replicas", 1))
        replicas = int(replicas_value)
        if replicas <= 0:
            raise ValueError(f"parallel stage startup requires num_replicas > 0 for stage {stage_id}")
        stages.append((stage_id, replicas, _parse_device_ids(devices_value, stage_id=stage_id)))

    stages.sort(key=lambda item: item[0])
    if stages[0][0] != 0:
        raise ValueError("parallel stage startup requires stage_id=0 for the API head")
    if [stage_id for stage_id, _, _ in stages] != list(range(len(stages))):
        raise ValueError("parallel stage startup requires contiguous zero-based stage ids")

    specs_by_stage: dict[int, list[_ReplicaLaunchSpec]] = {}
    for stage_id, replicas, logical_devices in stages:
        if len(logical_devices) % replicas:
            raise ValueError(
                f"stage {stage_id} devices={logical_devices!r} cannot be split across num_replicas={replicas}"
            )
        devices_per_replica = len(logical_devices) // replicas
        if devices_per_replica <= 0:
            raise ValueError(f"stage {stage_id} has no devices per replica")
        specs: list[_ReplicaLaunchSpec] = []
        for replica_id in range(replicas):
            logical_slice = tuple(
                logical_devices[replica_id * devices_per_replica : (replica_id + 1) * devices_per_replica]
            )
            try:
                physical_slice = tuple(args.gpus[index] for index in logical_slice)
            except IndexError as error:
                raise ValueError(
                    f"stage {stage_id} references logical devices {logical_slice!r}, but only "
                    f"{len(args.gpus)} launcher GPUs were supplied"
                ) from error
            specs.append(
                _ReplicaLaunchSpec(
                    stage_id=stage_id,
                    replica_id=replica_id,
                    logical_devices=logical_slice,
                    physical_devices=physical_slice,
                )
            )
        specs_by_stage[stage_id] = specs

    head = specs_by_stage[0][0]
    worker_groups = [specs_by_stage[0][1:]]
    worker_groups.extend(specs_by_stage[stage_id] for stage_id in range(1, len(stages)))
    return head, [group for group in worker_groups if group]


def _stage_override(spec: _ReplicaLaunchSpec) -> str:
    """Build a stage override using indices local to the child process."""
    local_devices = ",".join(str(index) for index in range(len(spec.logical_devices)))
    return json.dumps(
        {
            str(spec.stage_id): {
                "devices": local_devices,
                "num_replicas": 1,
            }
        },
        separators=(",", ":"),
    )


def _parallel_command(
    args: argparse.Namespace,
    spec: _ReplicaLaunchSpec,
    *,
    head: bool,
) -> list[str]:
    """Build one stage-based CLI command for the head or a worker replica."""
    command = [
        sys.executable,
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        str(args.model),
        "--omni",
        "--deploy-config",
        str(args.deploy_config.resolve()),
        "--trust-remote-code",
        "--omni-master-address",
        args.omni_master_address,
        "--omni-master-port",
        str(args.omni_master_port),
        "--omni-dp-size-local",
        "1",
        "--stage-overrides",
        _stage_override(spec),
        "--log-stats",
    ]
    if head:
        command.extend(
            [
                "--stage-id",
                str(spec.stage_id),
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--stage-init-timeout",
                str(int(args.startup_timeout_s)),
                "--init-timeout",
                str(int(args.startup_timeout_s)),
                "--omni-lb-policy",
                args.omni_lb_policy,
            ]
        )
    else:
        command.extend(["--headless", "--stage-id", str(spec.stage_id)])
    return command


def _parallel_process_env(
    base_env: dict[str, str],
    spec: _ReplicaLaunchSpec,
    *,
    isolate_compile_cache: bool,
) -> dict[str, str]:
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in spec.physical_devices)
    # Headless workers do not pass through StageRuntime's per-replica cache
    # scoping. Give every externally launched replica its own compile cache so
    # concurrent Triton/torch.compile writes cannot corrupt a sibling cache.
    if isolate_compile_cache:
        cache_root = os.path.abspath(os.path.expanduser(env.get("VLLM_CACHE_ROOT", "~/.cache/vllm")))
        env["VLLM_CACHE_ROOT"] = os.path.join(
            cache_root,
            f"omni_parallel_stage{spec.stage_id}_replica{spec.replica_id}",
        )
    return env


def _spawn_managed_process(
    *,
    name: str,
    command: list[str],
    env: dict[str, str],
    log_dir: Path,
) -> _ManagedProcess:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except BaseException:
        log_file.close()
        raise
    return _ManagedProcess(
        name=name,
        command=command,
        env=env,
        process=process,
        process_group=os.getpgid(process.pid),
        log_path=log_path,
        log_file=log_file,
    )


def _wait_for_tcp_listener(address: str, port: int, process: _ParallelServer, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"parallel head exited while starting the master server with code {return_code}")
        try:
            with socket.create_connection((address, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"OmniMasterServer was not listening at {address}:{port} after {timeout_s:.0f}s")


def _process_log_contains(item: _ManagedProcess, pattern: re.Pattern[str]) -> bool:
    try:
        return bool(pattern.search(item.log_path.read_text(encoding="utf-8", errors="replace")))
    except OSError:
        return False


def _remote_attached_replicas(item: _ManagedProcess, stage_id: int) -> set[int]:
    """Return remote replica ids that the head has attached for one stage."""
    try:
        text = item.log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    return {
        int(match.group("replica"))
        for match in _REMOTE_ATTACH_RE.finditer(text)
        if int(match.group("stage")) == stage_id
    }


def _wait_for_remote_attachments(
    head: _ManagedProcess,
    workers: list[_ManagedProcess],
    server: _ParallelServer,
    *,
    stage_id: int,
    expected_replica_ids: set[int] | None,
    expected_count: int,
    timeout_s: float,
) -> None:
    """Wait for head-side handshakes, which are the real worker-ready signal.

    Headless worker stdout only contains the registration process; the engine
    core is a child process whose initialization logs are not forwarded to
    that file.  The head emits an attach log only after its remote handshake
    has completed, so this barrier is both observable and stage-specific.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for item in [head, *workers]:
            return_code = item.process.poll()
            if return_code is not None:
                tail = item.log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                raise RuntimeError(
                    f"parallel process {item.name} exited during stage {stage_id} startup with code "
                    f"{return_code};\n{tail}"
                )
        attached = _remote_attached_replicas(head, stage_id)
        if expected_replica_ids is not None:
            complete = expected_replica_ids.issubset(attached)
        else:
            complete = len(attached) >= expected_count
        if complete:
            return
        time.sleep(1)
    expected = sorted(expected_replica_ids) if expected_replica_ids is not None else expected_count
    raise TimeoutError(
        f"head did not attach stage {stage_id} replicas {expected} after {timeout_s:.0f}s; "
        f"attached={sorted(_remote_attached_replicas(head, stage_id))}"
    )


def _wait_for_processes_ready(processes: list[_ManagedProcess], server: _ParallelServer, timeout_s: float) -> None:
    """Wait for engine-core initialization markers before starting a next stage."""
    pending = list(processes)
    deadline = time.monotonic() + timeout_s
    while pending and time.monotonic() < deadline:
        for item in pending[:]:
            return_code = item.process.poll()
            if return_code is not None:
                tail = item.log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                raise RuntimeError(
                    f"parallel process {item.name} exited during startup with code {return_code};\n{tail}"
                )
            if _process_log_contains(item, _ENGINE_READY_RE):
                pending.remove(item)
        if pending:
            if server.poll() is not None:
                raise RuntimeError("parallel service process exited while waiting for stage readiness")
            time.sleep(1)
    if pending:
        names = ", ".join(item.name for item in pending)
        raise TimeoutError(f"parallel stage processes were not ready after {timeout_s:.0f}s: {names}")


def _start_parallel_server(
    args: argparse.Namespace,
    *,
    base_env: dict[str, str],
    log_dir: Path,
    launcher_log: Any,
) -> tuple[_ParallelServer, dict[str, object]]:
    """Start the API head and stage workers with per-stage parallel waves."""
    head_spec, worker_groups = _parallel_replica_specs(args)
    head_command = _parallel_command(args, head_spec, head=True)
    head_env = _parallel_process_env(base_env, head_spec, isolate_compile_cache=False)
    head = _spawn_managed_process(
        name=f"stage{head_spec.stage_id}_head_replica{head_spec.replica_id}",
        command=head_command,
        env=head_env,
        log_dir=log_dir,
    )
    server = _ParallelServer(head=head, workers=[])
    try:
        _wait_for_tcp_listener(args.omni_master_address, args.omni_master_port, server, args.startup_timeout_s)
        # The head initializes non-local stages before its API becomes
        # healthy.  Launch those workers first so the head's remote handshake
        # barriers can complete.  Stage-0's extra replicas are dynamically
        # attached by the running orchestrator and therefore launch after the
        # downstream stages and API health check.  This ordering also keeps
        # colocated stage-1/stage-2 replicas in separate memory-safe waves.
        groups_by_stage = {specs[0].stage_id: specs for specs in worker_groups}
        total_waves = len(worker_groups)

        def _launch_group(
            specs: list[_ReplicaLaunchSpec],
            *,
            wave_index: int,
            expected_replica_ids: set[int] | None,
        ) -> None:
            workers: list[_ManagedProcess] = []
            for spec in specs:
                command = _parallel_command(args, spec, head=False)
                env = _parallel_process_env(base_env, spec, isolate_compile_cache=True)
                workers.append(
                    _spawn_managed_process(
                        name=f"stage{spec.stage_id}_worker_replica{spec.replica_id}",
                        command=command,
                        env=env,
                        log_dir=log_dir,
                    )
                )
            server.workers.extend(workers)
            print(
                f"parallel startup wave {wave_index}/{total_waves}: "
                f"stage {specs[0].stage_id}, {len(workers)} replica(s)",
                file=launcher_log,
                flush=True,
            )
            _wait_for_remote_attachments(
                head,
                workers,
                server,
                stage_id=specs[0].stage_id,
                expected_replica_ids=expected_replica_ids,
                expected_count=len(specs),
                timeout_s=args.startup_timeout_s,
            )

        wave_index = 0
        for stage_id in sorted(groups_by_stage):
            if stage_id == 0:
                continue
            wave_index += 1
            _launch_group(
                groups_by_stage[stage_id],
                wave_index=wave_index,
                expected_replica_ids={spec.replica_id for spec in groups_by_stage[stage_id]},
            )

        # Downstream remote stages must be attached before the API can serve.
        # This also ensures stage-1/2 colocated memory is fully initialized
        # before the benchmark starts sending requests.
        service._wait_for_server(
            f"http://{args.host}:{args.port}",
            server,
            args.startup_timeout_s,
        )

        # Dynamic stage-0 registrations are handled by the now-running
        # orchestrator.  Launch them only after the API is healthy, then wait
        # for the corresponding head-side attach events before returning.
        if 0 in groups_by_stage:
            wave_index += 1
            _launch_group(
                groups_by_stage[0],
                wave_index=wave_index,
                expected_replica_ids=None,
            )
    except BaseException:
        _stop_parallel_server(server, launcher_log)
        raise

    launch_metadata = {
        "head_command": head_command,
        "head_pid": head.process.pid,
        "head_process_group": head.process_group,
        "worker_commands": [item.command for item in server.workers],
        "worker_pids": [item.process.pid for item in server.workers],
        "worker_process_groups": [item.process_group for item in server.workers],
        "worker_log_paths": [str(item.log_path) for item in server.workers],
        "head_log_path": str(head.log_path),
        "head_spec": head_spec.__dict__,
        "worker_groups": [[spec.__dict__ for spec in group] for group in worker_groups],
    }
    return server, launch_metadata


def _stop_parallel_server(server: _ParallelServer, log: Any) -> None:
    errors: list[BaseException] = []
    for item in reversed(server.managed_processes):
        try:
            service._stop_process_group(item.process, item.process_group, log)
        except BaseException as error:
            errors.append(error)
        finally:
            item.log_file.close()
    if errors:
        raise RuntimeError(f"failed to stop {len(errors)} parallel process group(s): {errors[0]}") from errors[0]


def _merge_parallel_logs(server: _ParallelServer, launcher_log: Any) -> None:
    """Append child logs to server.log so existing metric parsers still work."""
    for item in server.managed_processes:
        launcher_log.write(f"\n===== {item.name} ({item.log_path}) =====\n")
        try:
            launcher_log.write(item.log_path.read_text(encoding="utf-8", errors="replace"))
        except OSError as error:
            launcher_log.write(f"[failed to read child log: {error}]\n")
    launcher_log.flush()


def realtime_slo_reasons(
    summary: dict[str, object],
    *,
    model_unit_decision_p99_slo_ms: float = 1500.0,
) -> list[str]:
    """Return hard realtime failures without treating the boundary as a crash."""
    reasons: list[str] = []
    if summary.get("workload_mode") == "full_duplex_e2e":
        if int(summary.get("failure_count") or 0):
            reasons.append("one or more full-duplex E2E sessions failed")
        lag = summary.get("arrival_schedule_lag_ms")
        lag_p99 = _nested_number(lag, "p99") if isinstance(lag, dict) else None
        if lag_p99 is None or lag_p99 > 200.0:
            reasons.append(f"arrival schedule lag p99 exceeded 200 ms: {lag_p99}")
        if int(summary.get("task_unknown_count") or 0):
            reasons.append("one or more E2E cases have unknown task correctness")
        return reasons
    if int(summary.get("failure_count") or 0):
        reasons.append("one or more sessions failed")
    if int(summary.get("audio_response_count") or 0) <= 0:
        reasons.append("workload produced no audio responses")
    schedule_lag = summary.get("arrival_schedule_lag_ms")
    schedule_p99 = _nested_number(schedule_lag, "p99") if isinstance(schedule_lag, dict) else None
    if schedule_p99 is None or schedule_p99 > 200.0:
        reasons.append(f"arrival schedule lag p99 exceeded 200 ms: {schedule_p99}")
    decision_latency = summary.get("model_unit_decision_latency_ms")
    decision_p99 = _nested_number(decision_latency, "p99") if isinstance(decision_latency, dict) else None
    if decision_p99 is None or decision_p99 > model_unit_decision_p99_slo_ms:
        reasons.append(
            f"model-unit decision latency p99 exceeded {model_unit_decision_p99_slo_ms:g} ms: {decision_p99}"
        )
    unpaired_decisions = int(summary.get("model_unit_unpaired_decision_count") or 0)
    if unpaired_decisions:
        reasons.append(f"unpaired model decisions were observed: {unpaired_decisions}")
    streaming_rtf = summary.get("streaming_audio_rtf")
    streaming_p99 = _nested_number(streaming_rtf, "p99") if isinstance(streaming_rtf, dict) else None
    if streaming_p99 is None or streaming_p99 > 1.0:
        reasons.append(f"streaming audio RTF p99 exceeded 1.0: {streaming_p99}")
    miss_rate = _nested_number(summary, "playout_deadline_miss_rate")
    if miss_rate is None or miss_rate > 0.01:
        reasons.append(f"playout deadline miss rate exceeded 1%: {miss_rate}")
    return reasons


def _client_command(
    args: argparse.Namespace,
    *,
    request_rate: float,
    duration_s: float,
    output_dir: Path,
    run_id: str,
    seed: int,
) -> list[str]:
    client = E2E_CLIENT if args.client_mode == "e2e" else CLIENT
    command = [
        sys.executable,
        str(client),
        "--request-rate",
        str(request_rate),
        "--duration-s",
        str(duration_s),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--url",
        f"ws://{args.host}:{args.port}/v1/realtime",
        "--model",
        str(args.model),
        "--ref-audio",
        str(args.ref_audio),
        "--run-id",
        run_id,
        "--chunk-ms",
        str(args.chunk_ms),
        "--tail-drain-s",
        str(args.tail_drain_s),
        "--playback-initial-buffer-ms",
        str(args.playback_initial_buffer_ms),
        "--timeout-s",
        str(args.request_timeout_s),
    ]
    if args.client_mode == "e2e":
        command[2:2] = ["--manifest", str(args.e2e_manifest), "--feedback-contract", args.feedback_contract]
        if args.explicit_followup_response:
            command.append("--explicit-followup-response")
        if args.explicit_all_responses:
            command.append("--explicit-all-responses")
    else:
        command[2:2] = ["--dataset-root", str(args.dataset_root)]
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--deploy-config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--client-mode", choices=("arrival", "e2e"), default="arrival")
    parser.add_argument("--e2e-manifest", type=Path)
    parser.add_argument("--feedback-contract", choices=("L1", "L2"), default="L2")
    parser.add_argument("--explicit-followup-response", action="store_true")
    parser.add_argument("--explicit-all-responses", action="store_true")
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--gpus", type=_csv_gpus, default=_csv_gpus("0,1,2,3,4,5,6,7"))
    parser.add_argument("--rates", type=_csv_floats, default=_csv_floats("0.1,0.2,0.4,0.8"))
    parser.add_argument("--duration-s", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20_260_901)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-rate", type=float, default=0.1)
    parser.add_argument("--warmup-duration-s", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-label", default="humdial")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8113)
    parser.add_argument(
        "--omni-master-address",
        default="127.0.0.1",
        help="address for the stage-based OmniMasterServer used by parallel startup",
    )
    parser.add_argument(
        "--omni-master-port",
        type=int,
        default=26000,
        help="port for the stage-based OmniMasterServer used by parallel startup",
    )
    parser.add_argument(
        "--parallel-stage-launch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "launch the API head and each stage replica as separate processes; "
            "replicas within a stage start concurrently, while colocated stages "
            "start in waves"
        ),
    )
    parser.add_argument("--startup-timeout-s", type=float, default=1800.0)
    parser.add_argument("--request-timeout-s", type=float, default=180.0)
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--tail-drain-s", type=float, default=2.0)
    parser.add_argument("--playback-initial-buffer-ms", type=int, default=300)
    parser.add_argument("--model-unit-decision-p99-slo-ms", type=float, default=1500.0)
    parser.add_argument("--telemetry-interval-s", type=float, default=1.0)
    parser.add_argument("--idle-stability-s", type=float, default=30.0)
    parser.add_argument("--max-preexisting-memory-mib", type=int, default=1024)
    parser.add_argument(
        "--omni-lb-policy",
        default="least-queue-length",
        choices=["random", "round-robin", "least-queue-length"],
    )
    parser.add_argument("--stop-on-slo-fail", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.ref_audio is None:
        args.ref_audio = args.model / "assets" / "HT_ref_audio.wav"
    paths = [args.model, args.deploy_config, args.ref_audio]
    if args.client_mode == "arrival":
        paths.append(args.dataset_root)
    if any(not path.exists() for path in paths):
        parser.error(f"model/config/dataset/ref path is missing: {paths}")
    if args.client_mode == "e2e":
        if args.e2e_manifest is None or not args.e2e_manifest.is_file():
            parser.error("--client-mode e2e requires an existing --e2e-manifest")
    if args.duration_s <= 0 or args.repeats <= 0 or args.port <= 0 or args.omni_master_port <= 0:
        parser.error("duration/repeats/ports must be positive")
    if args.parallel_stage_launch and args.omni_master_port == args.port:
        parser.error("--omni-master-port must differ from --port for parallel startup")
    if args.warmup_duration_s < 0 or args.warmup_rate <= 0:
        parser.error("warmup duration must be non-negative and warmup rate positive")
    if args.playback_initial_buffer_ms < 0:
        parser.error("playback initial buffer must be non-negative")
    if args.request_timeout_s <= 0 or args.startup_timeout_s <= 0 or args.model_unit_decision_p99_slo_ms <= 0:
        parser.error("timeouts must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to mix artifacts in non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("GPU idle preflight is observational only; starting service directly", flush=True)
    preflight = [service._gpu_pool_sample(args.gpus, include_compute_apps=True)]
    server_log_path = args.output_dir / "server.log"
    metadata_path = args.output_dir / "metadata.json"
    telemetry_path = args.output_dir / "gpu_telemetry.jsonl"
    output_path = args.output_dir / "sweep_summary.json"
    base_url = f"http://{args.host}:{args.port}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in args.gpus)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])])
    env["MINICPMO_DUPLEX_TIMING"] = "1"
    env["MINICPMO_CODE2WAV_TIMING"] = "1"
    no_proxy = {args.host, "127.0.0.1", "localhost", "::1"}
    for variable in ("NO_PROXY", "no_proxy"):
        no_proxy.update(item.strip() for item in env.get(variable, "").split(",") if item.strip())
        env[variable] = ",".join(sorted(no_proxy))
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
        "--omni-lb-policy",
        args.omni_lb_policy,
        "--log-stats",
    ]
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "arguments": _json_arguments(args),
        "server_command": server_command,
        "deploy_configs": service._deploy_config_snapshot(args.deploy_config),
        "gpu_preflight": preflight,
        "gpu_preflight_enforced": False,
        "environment": _environment_snapshot(env),
    }
    service._write_json(metadata_path, metadata)

    telemetry = service._GpuTelemetry(args.gpus, telemetry_path, args.telemetry_interval_s)
    telemetry.start()
    server: Any | None = None
    parallel_server: _ParallelServer | None = None
    process_group: int | None = None
    run_error: BaseException | None = None
    runs: list[dict[str, object]] = []
    stopped_at: dict[str, object] | None = None
    with server_log_path.open("w", encoding="utf-8", buffering=1) as server_log:
        print(json.dumps({"server_command": server_command, "gpus": args.gpus}), file=server_log, flush=True)
        try:
            if args.parallel_stage_launch:
                parallel_server, parallel_metadata = _start_parallel_server(
                    args,
                    base_env=env,
                    log_dir=args.output_dir / "stage_logs",
                    launcher_log=server_log,
                )
                server = parallel_server
                metadata["server_command"] = parallel_metadata["head_command"]
                metadata["parallel_launch"] = parallel_metadata
                metadata["server_pid"] = parallel_server.pid
                metadata["server_process_group"] = parallel_server.head.process_group
                metadata["server_process_groups"] = parallel_server.process_groups
            else:
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
            service._write_json(metadata_path, metadata)
            service._wait_for_server(base_url, server, args.startup_timeout_s)

            if args.warmup_duration_s > 0:
                warmup_dir = args.output_dir / "warmup"
                warmup_command = _client_command(
                    args,
                    request_rate=args.warmup_rate,
                    duration_s=args.warmup_duration_s,
                    output_dir=warmup_dir,
                    run_id=f"{args.run_label}-warmup",
                    seed=args.seed + 1,
                )
                with (args.output_dir / "warmup.log").open("w", encoding="utf-8", buffering=1) as client_log:
                    exit_code, monitor_error = service._run_monitored_client(
                        warmup_command,
                        env=env,
                        log=client_log,
                        server=server,
                        base_url=base_url,
                    )
                if exit_code or monitor_error:
                    raise RuntimeError(f"warmup failed: exit={exit_code}, monitor={monitor_error}")

            for rate in args.rates:
                for repeat in range(1, args.repeats + 1):
                    rate_label = str(rate).replace(".", "p")
                    run_dir = args.output_dir / f"rate_{rate_label}_rep_{repeat}"
                    command = _client_command(
                        args,
                        request_rate=rate,
                        duration_s=args.duration_s,
                        output_dir=run_dir,
                        run_id=f"{args.run_label}-rate{rate_label}-rep{repeat}",
                        seed=args.seed,
                    )
                    print(f"running rate={rate:g} session/s repeat={repeat}/{args.repeats}", flush=True)
                    with (args.output_dir / f"rate_{rate_label}_rep_{repeat}.log").open(
                        "w", encoding="utf-8", buffering=1
                    ) as client_log:
                        exit_code, monitor_error = service._run_monitored_client(
                            command,
                            env=env,
                            log=client_log,
                            server=server,
                            base_url=base_url,
                        )
                    summary_path = run_dir / "summary.json"
                    summary: dict[str, object] = (
                        json.loads(summary_path.read_text(encoding="utf-8"))
                        if summary_path.is_file()
                        else {"failure_count": None, "error": f"missing {summary_path}"}
                    )
                    reasons = realtime_slo_reasons(
                        summary,
                        model_unit_decision_p99_slo_ms=args.model_unit_decision_p99_slo_ms,
                    )
                    if exit_code:
                        reasons.append(f"client exit code {exit_code}")
                    if monitor_error:
                        reasons.append(monitor_error)
                    run = {
                        "rate": rate,
                        "repeat": repeat,
                        "client_exit_code": exit_code,
                        "client_monitor_error": monitor_error,
                        "realtime_slo_pass": not reasons,
                        "realtime_slo_reasons": reasons,
                        "summary": summary,
                        "compute_apps_after_run": {str(gpu): service._compute_apps_snapshot(gpu) for gpu in args.gpus},
                    }
                    runs.append(run)
                    service._write_json(run_dir / "slo.json", run)
                    print(
                        json.dumps(
                            {
                                "rate": rate,
                                "repeat": repeat,
                                "realtime_slo_pass": not reasons,
                                "reasons": reasons,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    if reasons and args.stop_on_slo_fail:
                        stopped_at = {"rate": rate, "repeat": repeat, "reasons": reasons}
                        break
                if stopped_at is not None:
                    break
        except BaseException as error:
            run_error = error
        finally:
            if parallel_server is not None:
                try:
                    _stop_parallel_server(parallel_server, server_log)
                except BaseException as error:
                    run_error = run_error or error
                _merge_parallel_logs(parallel_server, server_log)
            elif server is not None and process_group is not None:
                try:
                    service._stop_process_group(server, process_group, server_log)
                except BaseException as error:
                    run_error = run_error or error
    try:
        telemetry.stop()
    except BaseException as error:
        run_error = run_error or error

    output = {
        "model": str(args.model),
        "deploy_config": str(args.deploy_config.resolve()),
        "gpus": args.gpus,
        "rates": args.rates,
        "duration_s": args.duration_s,
        "seed": args.seed,
        "non_pd_production_path": True,
        "omni_lb_policy": args.omni_lb_policy,
        "parallel_stage_launch": args.parallel_stage_launch,
        "omni_master_address": args.omni_master_address if args.parallel_stage_launch else None,
        "omni_master_port": args.omni_master_port if args.parallel_stage_launch else None,
        "stopped_at": stopped_at,
        "run_error": repr(run_error) if run_error is not None else None,
        "gpu_telemetry": service._telemetry_summary(telemetry.rows),
        "cfm_graph_stats": service._latest_cfm_graph_stats(server_log_path),
        "cfm_graph_stats_per_replica": service._latest_cfm_graph_stats_per_replica(server_log_path),
        "runs": runs,
    }
    service._write_json(output_path, output)
    metadata["finished_at_utc"] = datetime.now(UTC).isoformat()
    metadata["compute_apps_after_cleanup"] = {str(gpu): service._compute_apps_snapshot(gpu) for gpu in args.gpus}
    metadata["gpu_telemetry"] = output["gpu_telemetry"]
    metadata["stopped_at"] = stopped_at
    metadata["run_error"] = output["run_error"]
    service._write_json(metadata_path, metadata)
    print(f"wrote {output_path}")
    if run_error is not None:
        raise run_error


if __name__ == "__main__":
    main()
