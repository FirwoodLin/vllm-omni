#!/usr/bin/env python3
"""Benchmark one PersonaPlex component through CUDA Graph replay only.

The timed region never executes an eager model forward. A small untimed warmup
is used only to initialize CUDA libraries before capture. Each invocation loads
exactly one component so its weights, persistent state, graph-pool overhead, and
latency are isolated from the other PersonaPlex components.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

DDL_MS = 80.0
COMPONENTS = (
    "input_embedding",
    "temporal",
    "depformer",
    "text_argmax",
    "mimi_encoder_transformer",
    "mimi_decoder_transformer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--component", choices=COMPONENTS, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--temporal-context", type=int, default=3000)
    parser.add_argument("--capture-warmup", type=int, default=3)
    parser.add_argument("--replay-warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    if args.temporal_context <= 0:
        parser.error("temporal context must be positive")
    if args.capture_warmup <= 0 or args.replay_warmup <= 0 or args.repeats <= 0:
        parser.error("warmup and repeat counts must be positive")
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    return args


def _memory() -> dict[str, float]:
    return {
        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
    }


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


def _select_tensors(path: Path, predicate: Callable[[str], bool]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if predicate(key):
                tensors[key] = handle.get_tensor(key)
    return tensors


def _main_weights(model: Path, prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    return _select_tensors(model / "model.safetensors", lambda key: key.startswith(prefixes))


def _build_component(
    component: str,
    model_path: Path,
    batch: int,
    temporal_context: int,
) -> tuple[torch.nn.Module | None, Callable[[], Any], dict[str, Any]]:
    device = torch.device("cuda:0")
    bf16 = torch.bfloat16

    if component == "input_embedding":
        from vllm_omni.model_executor.models.personaplex.configuration_personaplex import (
            PersonaPlexConfig,
        )
        from vllm_omni.model_executor.models.personaplex.personaplex_embeddings import (
            PersonaPlexInputEmbeddings,
        )

        module = PersonaPlexInputEmbeddings(PersonaPlexConfig()).to(device, bf16).eval()
        weights = _main_weights(model_path, ("text_emb.", "emb."))
        loaded = module.load_weights(weights)
        expected = len(dict(module.named_parameters()))
        if len(loaded) != expected:
            raise RuntimeError(f"input embedding loaded {len(loaded)}/{expected} parameters")
        sequence = torch.zeros((batch, 17, 1), dtype=torch.long, device=device)
        return (
            module,
            lambda: module(sequence),
            {
                "input_shape": list(sequence.shape),
                "output_shape": [batch, 1, 4096],
            },
        )

    if component == "temporal":
        from vllm_omni.model_executor.models.personaplex.personaplex_temporal import (
            PersonaPlexTemporalStreaming,
        )

        module = PersonaPlexTemporalStreaming(context=temporal_context).to(device, bf16).eval()
        weights = _main_weights(model_path, ("transformer.", "out_norm.", "text_linear."))
        loaded = module.load_weights(weights)
        expected = 32 * 6 + 2
        if loaded != expected:
            raise RuntimeError(f"temporal loaded {loaded}/{expected} tensors")
        module.streaming_init(batch)
        frame = torch.zeros((batch, 1, 4096), dtype=bf16, device=device)
        return (
            module,
            lambda: module.step(frame),
            {
                "input_shape": list(frame.shape),
                "query_length": 1,
                "kv_capacity": temporal_context,
            },
        )

    if component == "depformer":
        from vllm_omni.model_executor.models.personaplex.configuration_personaplex import (
            PersonaPlexConfig,
        )
        from vllm_omni.model_executor.models.personaplex.personaplex_depformer import (
            PersonaPlexDepformer,
        )

        config = PersonaPlexConfig()
        module = (
            PersonaPlexDepformer(
                config.depformer_config,
                temporal_hidden_size=config.temporal_config.hidden_size,
                text_card=config.text_vocab_size,
            )
            .to(device, bf16)
            .eval()
        )
        weights = _main_weights(
            model_path,
            ("depformer.", "depformer_in.", "depformer_emb.", "depformer_text_emb.", "linears."),
        )
        loaded = module.load_weights(weights)
        expected = len(dict(module.named_parameters()))
        if len(loaded) != expected:
            raise RuntimeError(f"depformer loaded {len(loaded)}/{expected} parameters")
        text_token = torch.zeros(batch, dtype=torch.long, device=device)
        hidden = torch.zeros((batch, 1, 4096), dtype=bf16, device=device)
        audio_tokens = torch.zeros((batch, 16), dtype=torch.long, device=device)
        audio_provided = torch.zeros((batch, 16), dtype=torch.bool, device=device)
        return (
            module,
            lambda: module(text_token, hidden, audio_tokens, audio_provided),
            {
                "text_shape": list(text_token.shape),
                "hidden_shape": list(hidden.shape),
                "inner_ar_steps": 16,
            },
        )

    if component == "text_argmax":
        logits = torch.zeros((batch, 1, 1, 32000), dtype=bf16, device=device)
        return None, lambda: logits.float().view(batch, -1).argmax(-1), {"input_shape": list(logits.shape)}

    if component in {"mimi_encoder_transformer", "mimi_decoder_transformer"}:
        from vllm_omni.model_executor.models.personaplex.personaplex_mimi import (
            _MimiStreamingTransformer,
        )

        module = _MimiStreamingTransformer().to(device, torch.float32).eval()
        checkpoint = model_path / "tokenizer-e351c8d8-checkpoint125.safetensors"
        prefix = "encoder_transformer" if component == "mimi_encoder_transformer" else "decoder_transformer"
        weights = _select_tensors(checkpoint, lambda key: key.startswith(f"{prefix}."))
        loaded = module.load_weights(weights, prefix)
        if loaded != 80:
            raise RuntimeError(f"{component} loaded {loaded}/80 tensors")
        module.streaming_init(batch)
        positions = torch.zeros((batch, 2, 512), dtype=torch.float32, device=device)
        return (
            module,
            lambda: module.step(positions),
            {
                "input_shape": list(positions.shape),
                "positions_per_audio_tick": 2,
                "kv_capacity": 250,
            },
        )

    raise AssertionError(component)


def _drop_cpu_weights(fn: Callable[[], Any]) -> None:
    # The selected checkpoint tensors are owned only by the local loader after
    # _build_component returns; a collection here releases their pinned mappings
    # before capture without affecting module parameters.
    import gc

    gc.collect()
    torch.accelerator.empty_cache()
    fn()
    torch.accelerator.synchronize()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    model_path = args.model.resolve()
    module, run, shape_metadata = _build_component(
        args.component,
        model_path,
        args.batch_size,
        args.temporal_context,
    )
    del module  # The closure owns the module when one is present.
    _drop_cpu_weights(run)

    # Untimed initialization only. No eager latency is recorded or reported.
    for _ in range(args.capture_warmup - 1):
        run()
    torch.accelerator.synchronize()
    memory_before_capture = _memory()
    torch.accelerator.reset_peak_memory_stats()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run()
    torch.accelerator.synchronize()
    memory_after_capture = _memory()
    capture_peak = {
        "allocated_gib": torch.accelerator.max_memory_allocated() / 2**30,
        "reserved_gib": torch.accelerator.max_memory_reserved() / 2**30,
    }

    for _ in range(args.replay_warmup):
        graph.replay()
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()

    samples: list[float] = []
    for _ in range(args.repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))

    replay_peak = {
        "allocated_gib": torch.accelerator.max_memory_allocated() / 2**30,
        "reserved_gib": torch.accelerator.max_memory_reserved() / 2**30,
    }
    stats = _percentiles(samples)
    stats["ddl_ms"] = DDL_MS
    stats["p99_meets_ddl"] = bool(stats["p99_ms"] < DDL_MS)

    props = torch.cuda.get_device_properties(0)
    payload: Mapping[str, Any] = {
        "metadata": {
            "timestamp_unix": time.time(),
            "model": str(model_path),
            "component": args.component,
            "batch_size": args.batch_size,
            "temporal_context": args.temporal_context if args.component == "temporal" else None,
            "capture_warmup": args.capture_warmup,
            "replay_warmup": args.replay_warmup,
            "repeats": args.repeats,
            "timed_execution": "cuda_graph_replay_only",
            "eager_latency_measured": False,
            "graph_captured": True,
            "graph_count": 1,
            "graph_replay_count": args.replay_warmup + args.repeats,
            "gpu_name": props.name,
            "gpu_total_memory_gib": props.total_memory / 2**30,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            **shape_metadata,
        },
        "cuda_graph_replay": stats,
        "memory_before_capture": memory_before_capture,
        "memory_after_capture": memory_after_capture,
        "capture_peak": capture_peak,
        "replay_peak": replay_peak,
    }
    # Keep graph outputs alive until all replay measurements are complete.
    assert graph_output is not None
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
