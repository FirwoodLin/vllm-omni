# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Summarize HumDial sweep artifacts without modifying experiment directories."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _metric(payload: object, key: str) -> float | int | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return value if isinstance(value, int | float) and not isinstance(value, bool) else None


def _run_complete(run: object) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not isinstance(run, dict):
        return False, ["malformed run entry"]
    if run.get("client_exit_code") != 0:
        reasons.append(f"client_exit_code={run.get('client_exit_code')}")
    if run.get("client_monitor_error"):
        reasons.append(f"client_monitor_error={run['client_monitor_error']}")
    summary = run.get("summary")
    if not isinstance(summary, dict):
        reasons.append("missing client summary")
        return False, reasons
    request_count = summary.get("request_count")
    success_count = summary.get("success_count")
    failure_count = summary.get("failure_count")
    if not isinstance(request_count, int) or request_count <= 0:
        reasons.append("missing positive request_count")
    if not isinstance(success_count, int) or not isinstance(failure_count, int):
        reasons.append("missing success/failure counts")
    elif isinstance(request_count, int) and success_count + failure_count != request_count:
        reasons.append("success/failure counts do not cover every request")
    if summary.get("error"):
        reasons.append(f"client summary error={summary['error']}")
    return not reasons, reasons


def _topology(payload: dict[str, Any], path: Path) -> str:
    config = str(payload.get("deploy_config") or "").lower()
    for name in (
        "thinker6_downstream_colocated_async_stage1",
        "separated_422",
        "thinker6_downstream_colocated",
        "downstream_colocated",
        "full_colocated",
    ):
        if name in config:
            return name
    parent_name = path.parent.parent.name
    return parent_name or path.parent.name


def records_from_payload(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    runs = payload.get("runs")
    run_list = runs if isinstance(runs, list) else []
    completion = [_run_complete(run) for run in run_list]
    sweep_reasons: list[str] = []
    if payload.get("run_error") is not None:
        sweep_reasons.append(f"run_error={payload['run_error']}")
    if not run_list:
        sweep_reasons.append("no workload runs")
    for index, (complete, reasons) in enumerate(completion):
        if not complete:
            sweep_reasons.append(f"run {index + 1}: {'; '.join(reasons)}")
    sweep_status = "partial" if sweep_reasons else "complete"
    telemetry = payload.get("gpu_telemetry")
    telemetry = telemetry if isinstance(telemetry, dict) else {}
    topology = _topology(payload, path)

    if not run_list:
        run_list = [{}]
        completion = [(False, ["no workload run"])]

    records: list[dict[str, Any]] = []
    for run, (complete, run_reasons) in zip(run_list, completion, strict=True):
        run = run if isinstance(run, dict) else {}
        summary = run.get("summary")
        summary = summary if isinstance(summary, dict) else {}
        decision = summary.get("model_unit_decision_latency_ms")
        streaming = summary.get("streaming_audio_rtf")
        request_rtf = summary.get("request_rtf")
        underrun = summary.get("playback_underrun_ms")
        if complete:
            run_status = "pass" if run.get("realtime_slo_pass") is True else "slo_fail"
        else:
            run_status = "partial"
        records.append(
            {
                "source": str(path),
                "topology": topology,
                "attempt": path.parent.name,
                "sweep_status": sweep_status,
                "sweep_status_reasons": sweep_reasons,
                "rate_sessions_per_s": _metric(run, "rate"),
                "repeat": _metric(run, "repeat"),
                "run_status": run_status,
                "run_status_reasons": (
                    run_reasons
                    if run_reasons
                    else list(run.get("realtime_slo_reasons") or [])
                ),
                "request_count": _metric(summary, "request_count"),
                "success_count": _metric(summary, "success_count"),
                "failure_count": _metric(summary, "failure_count"),
                "maximum_client_concurrency": _metric(
                    summary, "maximum_client_concurrency"
                ),
                "decision_latency_p99_ms": _metric(decision, "p99"),
                "streaming_rtf_p50": _metric(streaming, "p50"),
                "streaming_rtf_p99": _metric(streaming, "p99"),
                "request_rtf_p99": _metric(request_rtf, "p99"),
                "playout_deadline_miss_rate": _metric(
                    summary, "playout_deadline_miss_rate"
                ),
                "playback_underrun_mean_ms": _metric(underrun, "mean"),
                "peak_gpu_memory_mib": _metric(telemetry, "peak_memory_used_mib"),
                "peak_gpu_utilization_percent": _metric(
                    telemetry, "peak_utilization_gpu_percent"
                ),
                "peak_gpu_power_w": _metric(telemetry, "peak_power_draw_w"),
                "per_gpu": telemetry.get("per_gpu") or {},
            }
        )
    return records


def _discover(inputs: Sequence[Path]) -> list[Path]:
    paths: set[Path] = set()
    for input_path in inputs:
        if input_path.is_file():
            paths.add(input_path.resolve())
        elif input_path.is_dir():
            direct = input_path / "sweep_summary.json"
            if direct.is_file():
                paths.add(direct.resolve())
            else:
                paths.update(path.resolve() for path in input_path.rglob("sweep_summary.json"))
        else:
            raise FileNotFoundError(input_path)
    return sorted(paths)


def load_records(inputs: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _discover(inputs):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("top-level JSON value is not an object")
        except (OSError, json.JSONDecodeError, TypeError) as error:
            records.append(
                {
                    "source": str(path),
                    "topology": path.parent.parent.name,
                    "attempt": path.parent.name,
                    "sweep_status": "partial",
                    "sweep_status_reasons": [f"unreadable summary: {error}"],
                    "run_status": "partial",
                    "run_status_reasons": [f"unreadable summary: {error}"],
                }
            )
            continue
        records.extend(records_from_payload(path, payload))
    return records


def _display(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(records: Sequence[dict[str, Any]]) -> str:
    headings = [
        "Topology",
        "Attempt",
        "Sweep",
        "Rate",
        "Run",
        "Req/OK",
        "Conc",
        "Decision p99 ms",
        "Stream RTF p50/p99",
        "Miss",
        "Underrun mean ms",
        "GPU peak MiB/util%/W",
    ]
    rows = ["| " + " | ".join(headings) + " |", "| " + " | ".join(["---"] * len(headings)) + " |"]
    for record in records:
        miss = record.get("playout_deadline_miss_rate")
        miss_text = "-" if miss is None else f"{float(miss) * 100:.2f}%"
        values = [
            record.get("topology", "-"),
            record.get("attempt", "-"),
            record.get("sweep_status", "-"),
            _display(record.get("rate_sessions_per_s")),
            record.get("run_status", "-"),
            f"{_display(record.get('request_count'), 0)}/{_display(record.get('success_count'), 0)}",
            _display(record.get("maximum_client_concurrency"), 0),
            _display(record.get("decision_latency_p99_ms"), 2),
            f"{_display(record.get('streaming_rtf_p50'))}/{_display(record.get('streaming_rtf_p99'))}",
            miss_text,
            _display(record.get("playback_underrun_mean_ms"), 2),
            (
                f"{_display(record.get('peak_gpu_memory_mib'), 0)}/"
                f"{_display(record.get('peak_gpu_utilization_percent'), 0)}/"
                f"{_display(record.get('peak_gpu_power_w'), 2)}"
            ),
        ]
        rows.append("| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |")
    rows.extend(
        [
            "",
            "`complete` 表示产物完整；`slo_fail` 是有效的容量边界，不是中断。"
            "`partial` 结果不能用于拓扑容量比较。",
        ]
    )
    return "\n".join(rows) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+", help="summary file or directory")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--output", type=Path, help="optional output file; stdout by default")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = load_records(args.inputs)
    rendered = (
        json.dumps({"records": records}, ensure_ascii=False, indent=2) + "\n"
        if args.format == "json"
        else markdown_table(records)
    )
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
