# SPDX-License-Identifier: Apache-2.0
"""Benchmark MiniCPM-o 4.5's steady-state Code2Wav speech decoder.

The production stage receives 25 new 25-Hz codec tokens plus three left-context
tokens, runs a 10-step classifier-free-guidance flow decoder, and vocodes the
result to 24-kHz audio.  This benchmark uses the in-tree ``BatchedToken2Wav``
implementation, real flow/HiFT weights, and the checkpoint's reference voice.

Example:

    CUDA_VISIBLE_DEVICES=6 /app/vllm-omni/.venv/bin/python \
        benchmarks/minicpmo/benchmark_code2wav_saturation.py \
        --output-json intermediate/minicpmo45_encoder_bench/code2wav.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
    BatchedToken2Wav,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_token2wav import (
    MiniCPMO45Token2wav,
)

DEFAULT_MODEL = Path("/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/models--OpenBMB--MiniCPM-o-4_5")
SILENCE_TOKEN = 4218


@dataclass
class Code2WavResult:
    batch_size: int
    input_codec_tokens: int
    new_codec_tokens: int
    last_chunk: bool
    logical_audio_seconds_per_item: float
    output_samples_per_item: int
    setup_latency_ms: float
    chunk_latency_ms_median: float
    chunk_latency_ms_p95: float
    chunk_latency_ms_p99: float
    chunk_latency_ms_min: float
    chunk_latency_ms_max: float
    aggregate_audio_seconds_per_second: float
    per_request_rtf: float
    peak_allocated_gib: float
    peak_reserved_gib: float
    cfm_graph_measure_hits: int | None
    cfm_graph_measure_misses: int | None
    cfm_graph_cache_size: int | None


def _csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _event_elapsed(function, device: torch.device):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = function()
    end.record()
    torch.accelerator.synchronize(device)
    return start.elapsed_time(end), output


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _metadata(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "timestamp_unix": time.time(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": props.name,
        "gpu_total_memory_gib": props.total_memory / 2**30,
        "model": str(args.model),
        "prompt_wav": str(args.prompt_wav),
        "float16": args.float16,
        "n_timesteps": args.n_timesteps,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "hift_graph": args.hift_graph,
        "cfm_graph": args.cfm_graph,
        "cfm_max_graphs": args.cfm_max_graphs,
        "batch_sizes": args.batch_sizes,
        "codec_chunk_frames": args.codec_chunk_frames,
        "codec_left_context_frames": args.codec_left_context_frames,
        "last_chunk": args.last_chunk,
    }


def _cfm_graph_cache_info(backend: BatchedToken2Wav) -> dict[str, int] | None:
    wrapper = getattr(backend, "_cfm_graph_wrapper", None)
    stats_snapshot = getattr(wrapper, "stats_snapshot", None)
    if not callable(stats_snapshot):
        return None
    stats = stats_snapshot()
    return {
        "hits": int(stats["hits"]),
        "misses": int(stats["captures"]),
        "size": int(stats["cache_size"]),
        "maxsize": int(wrapper.max_graphs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-wav", type=Path)
    parser.add_argument("--batch-sizes", type=_csv_ints, default=_csv_ints("1,2,4,8,16"))
    parser.add_argument("--codec-chunk-frames", type=int, default=25)
    parser.add_argument("--codec-left-context-frames", type=int, default=3)
    parser.add_argument(
        "--last-chunk",
        action="store_true",
        help="Measure a terminal chunk (for example a short final utterance) instead of a steady non-final chunk",
    )
    parser.add_argument("--n-timesteps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--float16", action="store_true")
    parser.add_argument("--hift-graph", action="store_true")
    parser.add_argument("--cfm-graph", action="store_true")
    parser.add_argument("--cfm-max-graphs", type=int, default=32)
    parser.add_argument(
        "--phase-breakdown",
        action="store_true",
        help="Time flow encoder, 10-step CFM, HiFT, and Python/cache overhead once",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.prompt_wav is None:
        args.prompt_wav = args.model / "assets" / "HT_ref_audio.wav"
    for name in (
        "codec_chunk_frames",
        "codec_left_context_frames",
        "n_timesteps",
        "warmup",
        "repeats",
        "cfm_max_graphs",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; request GPU access before running this benchmark")
    if not args.prompt_wav.is_file():
        raise FileNotFoundError(args.prompt_wav)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    metadata = _metadata(args, device)
    print(json.dumps({"metadata": metadata}, sort_keys=True), flush=True)

    token2wav = MiniCPMO45Token2wav(
        str(args.model / "assets" / "token2wav"),
        float16=args.float16,
        n_timesteps=args.n_timesteps,
        device=device,
    )
    backend = BatchedToken2Wav(
        token2wav,
        connector_config={
            "codec_chunk_frames": args.codec_chunk_frames,
            "codec_left_context_frames": args.codec_left_context_frames,
        },
        hift_graph_config={
            "enabled": args.hift_graph,
            "capture_batch_sizes": args.batch_sizes,
        },
        cfm_graph_config={
            "enabled": args.cfm_graph,
            "max_graphs": args.cfm_max_graphs,
        },
    ).eval()

    torch.accelerator.synchronize(device)
    prompt_wall_start = time.perf_counter()
    features = backend.prepare_prompt("HT_ref_audio", str(args.prompt_wav))
    torch.accelerator.synchronize(device)
    metadata["prompt_preparation_wall_ms"] = (time.perf_counter() - prompt_wall_start) * 1000
    metadata["prompt_codec_tokens"] = int(features.speech_tokens.shape[1])
    metadata["prompt_mel_frames"] = int(features.mels.shape[1])
    metadata["speaker_embedding_shape"] = list(features.speaker_embedding.shape)

    input_frames = args.codec_left_context_frames + args.codec_chunk_frames
    logical_seconds = args.codec_chunk_frames / 25
    results: list[Code2WavResult] = []
    with torch.inference_mode():
        for batch_size in args.batch_sizes:
            setup_ms, states = _event_elapsed(
                lambda: backend.setup_batch(features, batch_size),
                device,
            )
            tokens = torch.randint(
                0,
                6561,
                (batch_size, input_frames),
                device=device,
                dtype=torch.long,
            )
            tokens[:, : args.codec_left_context_frames] = SILENCE_TOKEN

            for _ in range(args.warmup):
                if args.last_chunk:
                    states = backend.setup_batch(features, batch_size)
                _, states = backend.decode_batch(
                    tokens,
                    features,
                    states,
                    last_chunk=args.last_chunk,
                )
            torch.accelerator.synchronize(device)
            torch.accelerator.reset_peak_memory_stats(device)
            graph_info_before = _cfm_graph_cache_info(backend)

            times_ms: list[float] = []
            audios: list[torch.Tensor] = []
            for _ in range(args.repeats):
                if args.last_chunk:
                    states = backend.setup_batch(features, batch_size)
                elapsed_ms, output = _event_elapsed(
                    lambda: backend.decode_batch(
                        tokens,
                        features,
                        states,
                        last_chunk=args.last_chunk,
                    ),
                    device,
                )
                audios, states = output
                times_ms.append(elapsed_ms)

            graph_info_after = _cfm_graph_cache_info(backend)

            median_ms = statistics.median(times_ms)
            result = Code2WavResult(
                batch_size=batch_size,
                input_codec_tokens=input_frames,
                new_codec_tokens=args.codec_chunk_frames,
                last_chunk=args.last_chunk,
                logical_audio_seconds_per_item=logical_seconds,
                output_samples_per_item=int(audios[0].numel()),
                setup_latency_ms=setup_ms,
                chunk_latency_ms_median=median_ms,
                chunk_latency_ms_p95=_percentile(times_ms, 0.95),
                chunk_latency_ms_p99=_percentile(times_ms, 0.99),
                chunk_latency_ms_min=min(times_ms),
                chunk_latency_ms_max=max(times_ms),
                aggregate_audio_seconds_per_second=(batch_size * logical_seconds) / (median_ms / 1000),
                per_request_rtf=(median_ms / 1000) / logical_seconds,
                peak_allocated_gib=torch.accelerator.max_memory_allocated(device) / 2**30,
                peak_reserved_gib=torch.accelerator.max_memory_reserved(device) / 2**30,
                cfm_graph_measure_hits=(
                    graph_info_after["hits"] - graph_info_before["hits"]
                    if graph_info_before is not None and graph_info_after is not None
                    else None
                ),
                cfm_graph_measure_misses=(
                    graph_info_after["misses"] - graph_info_before["misses"]
                    if graph_info_before is not None and graph_info_after is not None
                    else None
                ),
                cfm_graph_cache_size=(graph_info_after["size"] if graph_info_after is not None else None),
            )
            results.append(result)
            print(json.dumps(asdict(result), sort_keys=True), flush=True)
            del states, tokens, audios
            torch.accelerator.empty_cache()

    if args.phase_breakdown:
        batch_size = args.batch_sizes[0]
        with torch.inference_mode():
            states = backend.setup_batch(features, batch_size)
            tokens = torch.randint(
                0,
                6561,
                (batch_size, input_frames),
                device=device,
                dtype=torch.long,
            )
            tokens[:, : args.codec_left_context_frames] = SILENCE_TOKEN
            for _ in range(args.warmup):
                if args.last_chunk:
                    states = backend.setup_batch(features, batch_size)
                _, states = backend.decode_batch(tokens, features, states, last_chunk=args.last_chunk)
            torch.accelerator.synchronize(device)

            events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
            originals = {
                "flow_encoder": backend._encode_chunk,
                "cfm_10_steps": backend._decode_cfm,
                "hift_vocoder": backend._hift_inference,
            }

            def timed(name, function):
                def wrapper(*positional, **keywords):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    output = function(*positional, **keywords)
                    end.record()
                    events[name] = (start, end)
                    return output

                return wrapper

            backend._encode_chunk = timed("flow_encoder", originals["flow_encoder"])
            backend._decode_cfm = timed("cfm_10_steps", originals["cfm_10_steps"])
            backend._hift_inference = timed("hift_vocoder", originals["hift_vocoder"])
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            total_start.record()
            backend.decode_batch(tokens, features, states, last_chunk=args.last_chunk)
            total_end.record()
            torch.accelerator.synchronize(device)
            backend._encode_chunk = originals["flow_encoder"]
            backend._decode_cfm = originals["cfm_10_steps"]
            backend._hift_inference = originals["hift_vocoder"]

            phases = {name: start.elapsed_time(end) for name, (start, end) in events.items()}
            phases["total"] = total_start.elapsed_time(total_end)
            phases["other_cache_and_python"] = phases["total"] - sum(phases[name] for name in originals)
            metadata["phase_breakdown_batch_size"] = batch_size
            metadata["phase_breakdown_ms"] = phases
            print(json.dumps({"phase_breakdown_ms": phases}, sort_keys=True), flush=True)

    metadata["cfm_graph_cache_info"] = _cfm_graph_cache_info(backend)
    hift_wrapper = getattr(backend, "hift_graph_wrapper", None)
    metadata["hift_graph_cache_entries"] = len(getattr(hift_wrapper, "graph", {}))
    metadata["hift_graph_lazy_captures"] = int(getattr(hift_wrapper, "lazy_graph_count", 0))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "metadata": metadata,
                    "results": [asdict(result) for result in results],
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")


if __name__ == "__main__":
    main()
