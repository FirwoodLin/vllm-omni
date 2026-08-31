#!/usr/bin/env python3
"""Measure PersonaPlex's 80 ms tick by component at a fixed batch size.

This benchmark is intentionally performance-only.  It uses the same steady
80 ms PCM frame for every measurement and reports CUDA-event latency for Mimi
encode, input embedding, the temporal backbone, the depformer, Mimi decode,
the eager model core, a captured CUDA-Graph model core, and the complete native
tick.  No transcript, audio quality, or functional-correctness gate is applied.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm_omni.experimental.fullduplex.personaplex.config import (
    FRAME_SIZE,
    PersonaPlexConfig,
)
from vllm_omni.experimental.fullduplex.personaplex.runtime import PersonaPlexEngine

DDL_MS = 80.0


def _percentiles(samples: list[float]) -> dict[str, float | int]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "count": int(values.size),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def _cuda_measure(fn: Callable[[], Any], repeats: int) -> tuple[list[float], Any]:
    samples: list[float] = []
    output: Any = None
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples, output


def _wall_measure(fn: Callable[[], Any], repeats: int) -> list[float]:
    samples: list[float] = []
    for _ in range(repeats):
        torch.accelerator.synchronize()
        start = time.perf_counter()
        fn()
        torch.accelerator.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def _install_local_hf_resolver(model: Path) -> None:
    """Route PersonaPlex's fixed hf_hub_download calls to the supplied mirror."""
    import huggingface_hub

    original = huggingface_hub.hf_hub_download
    model = model.resolve()

    def resolve(repo_id: str, filename: str, *args: Any, **kwargs: Any) -> str:
        candidate = model / filename
        if repo_id in {"nvidia/personaplex-7b-v1", str(model)} and candidate.is_file():
            return str(candidate)
        return original(repo_id, filename, *args, **kwargs)

    huggingface_hub.hf_hub_download = resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--voice-prompt", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.warmup <= 0 or args.repeats <= 0:
        parser.error("batch size, warmup, and repeats must be positive")
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if not args.voice_prompt.is_file():
        parser.error(f"voice prompt does not exist: {args.voice_prompt}")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    _install_local_hf_resolver(args.model)

    config = PersonaPlexConfig(
        hf_repo=str(args.model.resolve()),
        voice_prompt=str(args.voice_prompt.resolve()),
        batch_size=args.batch_size,
    )
    engine = PersonaPlexEngine(config).load()
    engine.open_batch()
    device = torch.device("cuda:0")
    batch = args.batch_size
    pcm_gpu = torch.zeros((batch, FRAME_SIZE), dtype=torch.float32, device=device)
    pcm_cpu = np.zeros((batch, FRAME_SIZE), dtype=np.float32)
    input_tokens = engine._initial.view(1, 17).expand(batch, 17).clone()
    audio_tokens = engine._silence.view(1, 8).expand(batch, 8).clone()
    audio_provided = torch.zeros((batch, 16), dtype=torch.bool, device=device)
    dep_targets = engine._initial[1:].view(1, 16).expand(batch, 16).clone()

    for _ in range(args.warmup):
        engine.step_batch(pcm_cpu)
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()

    results: dict[str, dict[str, float | int | bool | str | None]] = {}

    samples, codes = _cuda_measure(lambda: engine._codec.encode_frame(pcm_gpu), args.repeats)
    results["mimi_encode_eager"] = _percentiles(samples)

    samples, embeddings = _cuda_measure(lambda: engine._emb(input_tokens.unsqueeze(-1)), args.repeats)
    results["input_embedding_eager"] = _percentiles(samples)

    samples, temporal_output = _cuda_measure(lambda: engine._temporal.step(embeddings), args.repeats)
    hidden, text_logits = temporal_output
    results["temporal_eager"] = _percentiles(samples)
    sampled_text = text_logits.float().view(batch, -1).argmax(-1)

    samples, _ = _cuda_measure(
        lambda: engine._dep(
            sampled_text,
            hidden,
            audio_tokens=dep_targets,
            audio_provided=audio_provided,
        ),
        args.repeats,
    )
    results["depformer_eager"] = _percentiles(samples)

    samples, _ = _cuda_measure(lambda: engine._codec.decode_frame(audio_tokens), args.repeats)
    results["mimi_decode_eager"] = _percentiles(samples)

    def eager_core() -> Any:
        core_embeddings = engine._emb(input_tokens.unsqueeze(-1))
        core_hidden, core_logits = engine._temporal.step(core_embeddings)
        core_text = core_logits.float().view(batch, -1).argmax(-1)
        return engine._dep(
            core_text,
            core_hidden,
            audio_tokens=dep_targets,
            audio_provided=audio_provided,
        )

    for _ in range(args.warmup):
        eager_core()
    samples, _ = _cuda_measure(eager_core, args.repeats)
    results["model_core_eager"] = _percentiles(samples)

    # Measure the complete native path before attempting capture.  If a driver
    # rejects capture, the eager tick evidence is already complete and the
    # capture exception can be recorded without losing the primary datapoint.
    samples, _ = _cuda_measure(lambda: engine.step_batch(pcm_cpu), args.repeats)
    results["native_tick_cuda_eager"] = _percentiles(samples)
    wall_samples = _wall_measure(lambda: engine.step_batch(pcm_cpu), args.repeats)
    results["native_tick_wall_eager"] = _percentiles(wall_samples)

    graph_error: str | None = None
    graph: torch.cuda.CUDAGraph | None = None
    try:
        graph = torch.cuda.CUDAGraph()
        torch.accelerator.synchronize()
        with torch.cuda.graph(graph):
            graph_embeddings = engine._emb(input_tokens.unsqueeze(-1))
            graph_hidden, graph_logits = engine._temporal.step(graph_embeddings)
            graph_text = graph_logits.float().view(batch, -1).argmax(-1)
            graph_audio = engine._dep(
                graph_text,
                graph_hidden,
                audio_tokens=dep_targets,
                audio_provided=audio_provided,
            )
        del graph_embeddings, graph_hidden, graph_logits, graph_text, graph_audio
        for _ in range(args.warmup):
            graph.replay()
        samples, _ = _cuda_measure(graph.replay, args.repeats)
        results["model_core_cuda_graph"] = _percentiles(samples)
        results["model_core_cuda_graph"]["captured"] = True
    except Exception as exc:
        graph_error = repr(exc)
        results["model_core_cuda_graph"] = {
            "captured": False,
            "error": graph_error,
            "count": 0,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
        }

    for value in results.values():
        p99 = value.get("p99_ms")
        value["ddl_ms"] = DDL_MS
        value["p99_meets_ddl"] = bool(isinstance(p99, int | float) and p99 < DDL_MS)

    props = torch.cuda.get_device_properties(0)
    payload = {
        "metadata": {
            "timestamp_unix": time.time(),
            "model": str(args.model.resolve()),
            "voice_prompt": str(args.voice_prompt.resolve()),
            "batch_size": batch,
            "frame_samples": FRAME_SIZE,
            "frame_period_ms": DDL_MS,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "gpu_name": props.name,
            "gpu_total_memory_gib": props.total_memory / 2**30,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "performance_only": True,
            "graph_scope": "input embeddings + temporal backbone + greedy text argmax + depformer",
            "codec_graph": False,
            "codec_graph_reason": "streaming convolution state rebinds Python tensor references",
        },
        "results": results,
        "peak_allocated_gib": torch.accelerator.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.accelerator.max_memory_reserved() / 2**30,
        "graph_error": graph_error,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
