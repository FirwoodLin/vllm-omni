# SPDX-License-Identifier: Apache-2.0
"""Summarize MiniCPM-o Realtime duplex saturation run artifacts.

The multi-session E2E driver writes ``summary.json`` plus one
``session_XX/events.jsonl`` file per session.  This utility aggregates the
client-observed deadline metrics and the latest engine ``stage_metrics`` for
each response, without requiring a running server.

Example:

    /app/vllm-omni/.venv/bin/python \
      benchmarks/minicpmo/summarize_duplex_saturation.py \
      intermediate/minicpmo45_encoder_bench/duplex_b2 \
      intermediate/minicpmo45_encoder_bench/duplex_b4
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

OPEN_LOOP_WARMUP_UNITS_PER_SESSION = 5


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _summary(values: Iterable[object]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if isinstance(value, int | float) and math.isfinite(float(value))]
    return {
        "count": len(finite),
        "min": min(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
        "p95": _percentile(finite, 0.95),
        "p99": _percentile(finite, 0.99),
        "max": max(finite) if finite else None,
        "mean": statistics.fmean(finite) if finite else None,
    }


def _response_id(event: Mapping[str, Any], fallback: str) -> str:
    for candidate in (
        event.get("response_id"),
        event.get("item_id"),
        (event.get("response") or {}).get("id") if isinstance(event.get("response"), Mapping) else None,
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return fallback


def _stage_metrics_candidates(value: object) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        stage_metrics = value.get("stage_metrics")
        if isinstance(stage_metrics, Mapping):
            yield stage_metrics
        for child in value.values():
            yield from _stage_metrics_candidates(child)
    elif isinstance(value, list):
        for child in value:
            yield from _stage_metrics_candidates(child)


def _latest_stage_metrics(run_dir: Path) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    latest: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for events_path in sorted(run_dir.glob("session_*/events.jsonl")):
        session_name = events_path.parent.name
        with events_path.open(encoding="utf-8") as handle:
            for event_index, line in enumerate(handle):
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, Mapping):
                    continue
                response_id = _response_id(event, f"event-{event_index}")
                for stage_metrics in _stage_metrics_candidates(event):
                    for stage_id, metrics in stage_metrics.items():
                        if isinstance(metrics, Mapping):
                            latest[(session_name, response_id, str(stage_id))] = metrics
    return latest


def _stage_unit_deltas(run_dir: Path) -> dict[str, list[dict[str, float]]]:
    """Recover per-output-unit stage work from cumulative response metrics."""
    delta_fields = ("stage_gen_time_ms", "num_tokens_in", "num_tokens_out")
    previous: dict[tuple[str, str, str], dict[str, float]] = {}
    rows: dict[str, list[dict[str, float]]] = {}
    for events_path in sorted(run_dir.glob("session_*/events.jsonl")):
        session_name = events_path.parent.name
        with events_path.open(encoding="utf-8") as handle:
            for event_index, line in enumerate(handle):
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, Mapping) or event.get("type") != "response.audio.delta":
                    continue
                response_id = _response_id(event, f"event-{event_index}")
                for stage_metrics in _stage_metrics_candidates(event):
                    for stage_id, metrics in stage_metrics.items():
                        if not isinstance(metrics, Mapping):
                            continue
                        key = (session_name, response_id, str(stage_id))
                        prior = previous.get(key, {})
                        row: dict[str, float] = {}
                        current: dict[str, float] = {}
                        for field in delta_fields:
                            value = metrics.get(field)
                            if not isinstance(value, int | float) or not math.isfinite(float(value)):
                                continue
                            numeric = float(value)
                            old = prior.get(field, 0.0)
                            row[field] = numeric - old if numeric >= old else numeric
                            current[field] = numeric
                        if current:
                            previous[key] = current
                        if row:
                            rows.setdefault(str(stage_id), []).append(row)
    return rows


def _client_requests(summary_payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    requests: list[Mapping[str, Any]] = []
    sessions = summary_payload.get("sessions")
    if not isinstance(sessions, list):
        return requests
    for session in sessions:
        if not isinstance(session, Mapping):
            continue
        metrics = session.get("request_metrics")
        if isinstance(metrics, list):
            requests.extend(item for item in metrics if isinstance(item, Mapping))
    return requests


def _client_audio_outputs(summary_payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    outputs: list[Mapping[str, Any]] = []
    sessions = summary_payload.get("sessions")
    if not isinstance(sessions, list):
        return outputs
    for session in sessions:
        if not isinstance(session, Mapping):
            continue
        response_timings = session.get("response_timings")
        if not isinstance(response_timings, Mapping):
            continue
        for timing in response_timings.values():
            if not isinstance(timing, Mapping):
                continue
            audio_output = timing.get("audio_output")
            if not isinstance(audio_output, Mapping):
                continue
            outputs.append(audio_output)
    return outputs


def _finite_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    return [float(item) for item in value if isinstance(item, int | float) and math.isfinite(float(item))]


def _open_loop_input_ticks(run_dir: Path) -> list[Mapping[str, Any]]:
    ticks: list[Mapping[str, Any]] = []
    for path in sorted(run_dir.glob("session_*/input_schedule.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            ticks.extend(item for item in payload if isinstance(item, Mapping))
    return ticks


def _server_append_timing_rows(run_dir: Path) -> list[Mapping[str, Any]]:
    path = run_dir / "server_append_timing.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    return [item for item in rows if isinstance(item, Mapping)] if isinstance(rows, list) else []


def _stage0_completion_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Pair ordered fixed-rate inputs with their client-observed Stage0 results."""
    rows: list[dict[str, Any]] = []
    for schedule_path in sorted(run_dir.glob("session_*/input_schedule.json")):
        ticks_payload = json.loads(schedule_path.read_text(encoding="utf-8"))
        ticks = [item for item in ticks_payload if isinstance(item, Mapping)] if isinstance(ticks_payload, list) else []
        events_path = schedule_path.parent / "events.jsonl"
        if not events_path.is_file():
            continue
        completions: list[tuple[str, float]] = []
        with events_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, Mapping) or event.get("type") not in {
                    "response.listen",
                    "response.audio.delta",
                }:
                    continue
                received_at_s = event.get("_client_received_at_s")
                if not isinstance(received_at_s, int | float) or not math.isfinite(float(received_at_s)):
                    continue
                has_stage0 = any(
                    any(str(stage_id) == "0" for stage_id in stage_metrics)
                    for stage_metrics in _stage_metrics_candidates(event)
                )
                if has_stage0:
                    completions.append((str(event["type"]), float(received_at_s)))
        for tick, (event_type, received_at_s) in zip(ticks, completions):
            scheduled_ns = tick.get("scheduled_monotonic_ns")
            tick_index = tick.get("tick_index")
            if not isinstance(scheduled_ns, int) or not isinstance(tick_index, int):
                continue
            completion_ns = int(received_at_s * 1_000_000_000)
            rows.append(
                {
                    "session_id": schedule_path.parent.name,
                    "tick_index": tick_index,
                    "event_type": event_type,
                    "scheduled_monotonic_ns": scheduled_ns,
                    "completion_monotonic_ns": completion_ns,
                    "stage0_completion_lag_ms": (completion_ns - scheduled_ns) / 1_000_000.0,
                }
            )
    return rows


def _per_session_end_values(rows: list[Mapping[str, Any]], field: str) -> list[float]:
    grouped: dict[str, tuple[int, float]] = {}
    for row in rows:
        session_id = row.get("session_id")
        tick_index = row.get("tick_index")
        value = row.get(field)
        if (
            not isinstance(session_id, str)
            or not isinstance(tick_index, int)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
        ):
            continue
        prior = grouped.get(session_id)
        if prior is None or tick_index > prior[0]:
            grouped[session_id] = (tick_index, float(value))
    return [value for _, value in grouped.values()]


def _per_session_lag_slopes(rows: list[Mapping[str, Any]], field: str) -> list[float]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        session_id = row.get("session_id")
        tick_index = row.get("tick_index")
        value = row.get(field)
        if (
            isinstance(session_id, str)
            and isinstance(tick_index, int)
            and isinstance(value, int | float)
            and math.isfinite(float(value))
        ):
            grouped.setdefault(session_id, []).append((float(tick_index), float(value)))
    slopes: list[float] = []
    for points in grouped.values():
        if len(points) < 2:
            continue
        mean_x = statistics.fmean(x for x, _ in points)
        mean_y = statistics.fmean(y for _, y in points)
        denominator = sum((x - mean_x) ** 2 for x, _ in points)
        if denominator <= 0:
            continue
        slopes.append(sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator)
    return slopes


def _steady_queue_lag_ms(audio_output: Mapping[str, Any]) -> list[float]:
    """Track cumulative lag across consecutive full 1 s MiniCPM units."""
    intervals = _finite_list(audio_output.get("inter_chunk_intervals_ms"))
    durations = _finite_list(audio_output.get("chunk_durations_ms"))
    lag_ms = 0.0
    values: list[float] = []
    for interval_ms, preceding_duration_ms in zip(intervals, durations):
        if preceding_duration_ms < 900.0:
            lag_ms = 0.0
            continue
        lag_ms += interval_ms - 1000.0
        values.append(lag_ms)
    return values


def summarize_run(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary_payload, Mapping):
        raise ValueError(f"{summary_path} must contain a JSON object")

    client_requests = _client_requests(summary_payload)
    client_audio_outputs = _client_audio_outputs(summary_payload)
    client_audio_intervals = [
        interval for output in client_audio_outputs for interval in _finite_list(output.get("inter_chunk_intervals_ms"))
    ]
    steady_audio_intervals = [
        interval
        for output in client_audio_outputs
        for interval, duration in zip(
            _finite_list(output.get("inter_chunk_intervals_ms")),
            _finite_list(output.get("chunk_durations_ms")),
        )
        if duration >= 900.0
    ]
    steady_queue_lag_series = [series for output in client_audio_outputs if (series := _steady_queue_lag_ms(output))]
    steady_queue_lag_ms = [value for series in steady_queue_lag_series for value in series]
    playout_slack_ms = [
        slack for output in client_audio_outputs for slack in _finite_list(output.get("playout_slack_ms"))
    ]
    required_startup_buffer_ms = [
        float(value)
        for output in client_audio_outputs
        if isinstance((value := output.get("required_startup_buffer_ms")), int | float) and math.isfinite(float(value))
    ]
    streaming_rtfs = [
        float(value)
        for output in client_audio_outputs
        if isinstance((value := output.get("streaming_rtf")), int | float) and math.isfinite(float(value))
    ]
    latest = _latest_stage_metrics(run_dir)
    stage_unit_rows = _stage_unit_deltas(run_dir)
    open_loop_ticks = _open_loop_input_ticks(run_dir)
    server_append_rows = _server_append_timing_rows(run_dir)
    stage0_completion_rows = _stage0_completion_rows(run_dir)
    stage_rows: dict[str, list[Mapping[str, Any]]] = {}
    for (_, _, stage_id), metrics in latest.items():
        stage_rows.setdefault(stage_id, []).append(metrics)

    client_fields = ("ttft_ms", "ttfp_ms", "rtf", "audio_generation_ms", "audio_duration_ms")
    stage_fields = (
        "stage_gen_time_ms",
        "serving_time_to_first_output_ms",
        "time_per_output_unit_ms",
        "inter_output_latency_ms",
        "vllm_ttft_ms",
        "vllm_tpot_ms",
        "num_tokens_in",
        "num_tokens_out",
        "audio_duration_s",
    )
    client_send_lag_ms = [
        float(value)
        for row in open_loop_ticks
        if isinstance((value := row.get("send_lag_ms")), int | float) and math.isfinite(float(value))
    ]
    start_queue_lag_ms = [
        float(value)
        for row in server_append_rows
        if isinstance((value := row.get("start_queue_lag_ms")), int | float) and math.isfinite(float(value))
    ]
    completion_lag_ms = [
        float(value)
        for row in server_append_rows
        if isinstance((value := row.get("completion_lag_ms")), int | float) and math.isfinite(float(value))
    ]
    append_elapsed_ms = [
        float(value)
        for row in server_append_rows
        if isinstance((value := row.get("append_elapsed_ms")), int | float) and math.isfinite(float(value))
    ]
    stage0_completion_lag_ms = [
        float(value)
        for row in stage0_completion_rows
        if isinstance((value := row.get("stage0_completion_lag_ms")), int | float) and math.isfinite(float(value))
    ]
    steady_stage0_completion_rows = [
        row
        for row in stage0_completion_rows
        if isinstance(row.get("tick_index"), int) and int(row["tick_index"]) >= OPEN_LOOP_WARMUP_UNITS_PER_SESSION
    ]
    steady_stage0_completion_lag_ms = [
        float(row["stage0_completion_lag_ms"])
        for row in steady_stage0_completion_rows
        if isinstance(row.get("stage0_completion_lag_ms"), int | float)
        and math.isfinite(float(row["stage0_completion_lag_ms"]))
    ]
    return {
        "run_dir": str(run_dir),
        "ok": summary_payload.get("ok"),
        "workload_mode": summary_payload.get("workload_mode", "closed_loop"),
        "configured_session_count": summary_payload.get("session_count"),
        "completed_session_count": len(summary_payload.get("sessions") or []),
        "failures": summary_payload.get("failures") or [],
        "client": {
            **{field: _summary(row.get(field) for row in client_requests) for field in client_fields},
            "inter_chunk_interval_ms": _summary(client_audio_intervals),
            "steady_inter_chunk_interval_ms": _summary(steady_audio_intervals),
            "steady_inter_chunk_deadline_miss_count": sum(value >= 1000.0 for value in steady_audio_intervals),
            "steady_queue_lag_ms": _summary(steady_queue_lag_ms),
            "max_steady_queue_lag_ms": _summary(max(series) for series in steady_queue_lag_series),
            "end_steady_queue_lag_ms": _summary(series[-1] for series in steady_queue_lag_series),
            "streaming_rtf": _summary(streaming_rtfs),
            "playout_slack_ms": _summary(playout_slack_ms),
            "required_startup_buffer_ms": _summary(required_startup_buffer_ms),
            "playout_measured_response_count": len(required_startup_buffer_ms),
            "playout_deadline_miss_count": sum(value < 0.0 for value in playout_slack_ms),
            "rtf_deadline_miss_count": sum(
                1 for row in client_requests if isinstance(row.get("rtf"), int | float) and float(row["rtf"]) >= 1.0
            ),
            "inter_chunk_deadline_miss_count": sum(value >= 1000.0 for value in client_audio_intervals),
        },
        "stages": {
            stage_id: {
                "response_count": len(rows),
                **{field: _summary(row.get(field) for row in rows) for field in stage_fields},
            }
            for stage_id, rows in sorted(stage_rows.items(), key=lambda item: item[0])
        },
        "stage_unit_deltas": {
            stage_id: {
                "unit_count": len(rows),
                **{
                    field: _summary(row.get(field) for row in rows)
                    for field in ("stage_gen_time_ms", "num_tokens_in", "num_tokens_out")
                },
            }
            for stage_id, rows in sorted(stage_unit_rows.items(), key=lambda item: item[0])
        },
        "open_loop": (
            {
                "scheduled_unit_count": len(open_loop_ticks),
                "completed_real_append_count": len(server_append_rows),
                "unfinished_real_append_count": max(0, len(open_loop_ticks) - len(server_append_rows)),
                "stage0_completed_unit_count": len(stage0_completion_rows),
                "stage0_unfinished_unit_count": max(0, len(open_loop_ticks) - len(stage0_completion_rows)),
                "warmup_units_per_session": OPEN_LOOP_WARMUP_UNITS_PER_SESSION,
                "client_send_lag_ms": _summary(client_send_lag_ms),
                "start_queue_lag_ms": _summary(start_queue_lag_ms),
                "completion_lag_ms": _summary(completion_lag_ms),
                "append_elapsed_ms": _summary(append_elapsed_ms),
                "stage0_completion_lag_ms": _summary(stage0_completion_lag_ms),
                "steady_stage0_completion_lag_ms": _summary(steady_stage0_completion_lag_ms),
                "end_stage0_completion_lag_ms": _summary(
                    _per_session_end_values(stage0_completion_rows, "stage0_completion_lag_ms")
                ),
                "stage0_completion_lag_slope_ms_per_tick": _summary(
                    _per_session_lag_slopes(stage0_completion_rows, "stage0_completion_lag_ms")
                ),
                "steady_stage0_completion_lag_slope_ms_per_tick": _summary(
                    _per_session_lag_slopes(steady_stage0_completion_rows, "stage0_completion_lag_ms")
                ),
                "stage0_deadline_miss_count": sum(value >= 1000.0 for value in stage0_completion_lag_ms),
                "steady_stage0_deadline_miss_count": sum(value >= 1000.0 for value in steady_stage0_completion_lag_ms),
                "end_start_queue_lag_ms": _summary(_per_session_end_values(server_append_rows, "start_queue_lag_ms")),
                "end_completion_lag_ms": _summary(_per_session_end_values(server_append_rows, "completion_lag_ms")),
                "start_queue_lag_slope_ms_per_tick": _summary(
                    _per_session_lag_slopes(server_append_rows, "start_queue_lag_ms")
                ),
                "completion_lag_slope_ms_per_tick": _summary(
                    _per_session_lag_slopes(server_append_rows, "completion_lag_ms")
                ),
                "completion_deadline_miss_count": sum(value >= 1000.0 for value in completion_lag_ms),
                "append_error_count": sum(row.get("phase") == "error" for row in server_append_rows),
            }
            if open_loop_ticks
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = {"runs": [summarize_run(path) for path in args.run_dirs]}
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
