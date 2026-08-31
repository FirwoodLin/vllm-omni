# SPDX-License-Identifier: Apache-2.0
"""Measure the MiniCPM-o 4.5 Thinker (Qwen3) core in isolation.

The production Stage0 combines Vision/APM outputs with the Thinker LLM.  This
microbenchmark deliberately keeps the input as real token embeddings and
measures the Qwen3 transformer only, so its prefill/decode scaling can be
compared with the separate encoder and Talker measurements.  A synthetic KV
prefix represents an already-running conversation; its prefill is outside the
timed unit, matching a scheduler request with history already resident.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import AutoConfig

DEFAULT_MODEL = Path("/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/models--OpenBMB--MiniCPM-o-4_5")
NATIVE_CONTEXTS = "13,79,256"
NATIVE_KV = "0,4096,16384"


@dataclass
class ThinkerResult:
    batch_size: int
    context_length: int
    kv_length: int
    generated_tokens: int
    prefill_latency_ms_p50: float | None
    prefill_latency_ms_p95: float | None
    prefill_latency_ms_p99: float | None
    decode_latency_ms_p50: float | None
    decode_latency_ms_p95: float | None
    decode_latency_ms_p99: float | None
    unit_latency_ms_p50: float | None
    unit_latency_ms_p95: float | None
    unit_latency_ms_p99: float | None
    tokens_per_second: float | None
    aggregate_tokens_per_second: float | None
    peak_allocated_gib: float | None
    peak_reserved_gib: float | None
    repeats_completed: int
    status: str
    oom: bool
    error: str | None = None


def _csv_ints(value: str, *, nonnegative: bool = False) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 if nonnegative else item <= 0 for item in values):
        kind = "non-negative" if nonnegative else "positive"
        raise argparse.ArgumentTypeError(f"expected comma-separated {kind} integers")
    return values


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _load_qwen_class():
    # Transformers keeps Qwen3 in a regular module; the checkpoint's remote
    # MiniCPMO config subclasses Qwen3Config, so the same transformer layout is
    # directly compatible with its ``llm.*`` state-dict prefix.
    module = importlib.import_module("transformers.models.qwen3.modeling_qwen3")
    return module.Qwen3ForCausalLM


def _build_model(model_path: Path, device: torch.device):
    qwen_class = _load_qwen_class()
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    # The checkpoint config leaves this unset, which makes Transformers fall
    # back to eager attention and materializes a quadratic score tensor for a
    # long prefix.  SDPA is the closest available standalone analogue to the
    # fused/paged attention used by the deployed vLLM Thinker.
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = qwen_class(config)
    model.to_empty(device=device)

    index_path = model_path / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    shards = sorted({filename for name, filename in weight_map.items() if name.startswith("llm.")})
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(str(model_path / shard), framework="pt", device=str(device)) as handle:
            for name in handle.keys():
                if name.startswith("llm."):
                    state[name.removeprefix("llm.")] = handle.get_tensor(name)
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Thinker weight loading failed: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    del state
    model.eval()
    return model, config


def _build_embeddings(model, batch_size: int, context_length: int, device: torch.device):
    vocab_size = int(model.config.vocab_size)
    ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device, dtype=torch.long)
    return model.get_input_embeddings()(ids)


def _build_prefix(model, batch_size: int, kv_length: int, device: torch.device):
    if kv_length == 0:
        return None
    return _build_embeddings(model, batch_size, kv_length, device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-sizes", type=lambda x: _csv_ints(x), default=_csv_ints("1,2,4,8"))
    parser.add_argument("--context-lengths", type=lambda x: _csv_ints(x), default=_csv_ints(NATIVE_CONTEXTS))
    parser.add_argument(
        "--kv-lengths", type=lambda x: _csv_ints(x, nonnegative=True), default=_csv_ints(NATIVE_KV, nonnegative=True)
    )
    parser.add_argument("--generated-tokens", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.generated_tokens <= 0 or args.warmup <= 0 or args.repeats <= 0:
        parser.error("generated tokens, warmup, and repeats must be positive")
    if any(k + c + args.generated_tokens > 40960 for k in args.kv_lengths for c in args.context_lengths):
        parser.error("KV + context + generated tokens exceeds Thinker max_position_embeddings=40960")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    print(json.dumps({"event": "loading_thinker", "device": str(device)}, sort_keys=True), flush=True)
    model, config = _build_model(args.model, device)
    props = torch.cuda.get_device_properties(device)
    metadata: dict[str, Any] = {
        "timestamp_unix": time.time(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": props.name,
        "gpu_total_memory_gib": props.total_memory / 2**30,
        "device": str(device),
        "model": str(args.model),
        "weight_prefix": "llm.*",
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.intermediate_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "max_position_embeddings": int(config.max_position_embeddings),
        "generated_tokens": args.generated_tokens,
        "batch_sizes": args.batch_sizes,
        "context_lengths": args.context_lengths,
        "kv_lengths": args.kv_lengths,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "measurement_scope": "Thinker token-embedding prefill + fixed autoregressive decode; prefix KV build excluded",
        "input_scope": "synthetic token embeddings; encoder cost is measured separately",
    }
    metadata["thinker_parameters"] = sum(param.numel() for param in model.parameters())
    print(json.dumps({"metadata": metadata}, sort_keys=True), flush=True)

    results: list[ThinkerResult] = []
    for context_length in args.context_lengths:
        for kv_length in args.kv_lengths:
            for batch_size in args.batch_sizes:
                status, error = "ok", None
                prefill_times: list[float] = []
                decode_times: list[float] = []
                unit_times: list[float] = []
                repeats_completed = 0
                peak_allocated = peak_reserved = None
                try:
                    context = _build_embeddings(model, batch_size, context_length, device)
                    prefix = _build_prefix(model, batch_size, kv_length, device)
                    for _ in range(args.warmup):
                        past = None
                        if prefix is not None:
                            positions = torch.arange(kv_length, device=device).unsqueeze(0).expand(batch_size, -1)
                            past = model.model(
                                inputs_embeds=prefix, position_ids=positions, use_cache=True
                            ).past_key_values
                        positions = (
                            torch.arange(kv_length, kv_length + context_length, device=device)
                            .unsqueeze(0)
                            .expand(batch_size, -1)
                        )
                        output = model.model(
                            inputs_embeds=context, position_ids=positions, past_key_values=past, use_cache=True
                        )
                        hidden, past = output.last_hidden_state[:, -1:], output.past_key_values
                        for step in range(args.generated_tokens):
                            logits = model.lm_head(hidden[:, -1]).float()
                            token = torch.argmax(logits, dim=-1)
                            if step + 1 == args.generated_tokens:
                                break
                            emb = model.get_input_embeddings()(token.unsqueeze(1))
                            pos = torch.full(
                                (batch_size, 1), kv_length + context_length + step, device=device, dtype=torch.long
                            )
                            output = model.model(
                                inputs_embeds=emb, position_ids=pos, past_key_values=past, use_cache=True
                            )
                            hidden, past = output.last_hidden_state, output.past_key_values
                        del past, hidden, output, logits, token
                    torch.accelerator.synchronize(device)
                    torch.accelerator.reset_peak_memory_stats(device)
                    for _ in range(args.repeats):
                        past = None
                        if prefix is not None:
                            positions = (
                                torch.arange(kv_length, device=device, dtype=torch.long)
                                .unsqueeze(0)
                                .expand(batch_size, -1)
                            )
                            prefix_output = model.model(inputs_embeds=prefix, position_ids=positions, use_cache=True)
                            past = prefix_output.past_key_values
                            del prefix_output, positions
                        positions = (
                            torch.arange(kv_length, kv_length + context_length, device=device, dtype=torch.long)
                            .unsqueeze(0)
                            .expand(batch_size, -1)
                        )
                        start = torch.cuda.Event(enable_timing=True)
                        prefill_end = torch.cuda.Event(enable_timing=True)
                        unit_end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        output = model.model(
                            inputs_embeds=context, position_ids=positions, past_key_values=past, use_cache=True
                        )
                        prefill_end.record()
                        hidden, past = output.last_hidden_state[:, -1:], output.past_key_values
                        del output, positions
                        for step in range(args.generated_tokens):
                            logits = model.lm_head(hidden[:, -1]).float()
                            token = torch.argmax(logits, dim=-1)
                            if step + 1 == args.generated_tokens:
                                del logits, token
                                break
                            emb = model.get_input_embeddings()(token.unsqueeze(1))
                            pos = torch.full(
                                (batch_size, 1), kv_length + context_length + step, device=device, dtype=torch.long
                            )
                            output = model.model(
                                inputs_embeds=emb, position_ids=pos, past_key_values=past, use_cache=True
                            )
                            hidden, past = output.last_hidden_state, output.past_key_values
                            del emb, pos, output, logits, token
                        unit_end.record()
                        unit_end.synchronize()
                        prefill_ms = float(start.elapsed_time(prefill_end))
                        unit_ms = float(start.elapsed_time(unit_end))
                        prefill_times.append(prefill_ms)
                        unit_times.append(unit_ms)
                        decode_times.append(unit_ms - prefill_ms)
                        repeats_completed += 1
                        del hidden, past
                    peak_allocated = torch.accelerator.max_memory_allocated(device) / 2**30
                    peak_reserved = torch.accelerator.max_memory_reserved(device) / 2**30
                    del context, prefix
                    torch.accelerator.empty_cache()
                except torch.cuda.OutOfMemoryError as exc:
                    status, error = "oom", repr(exc).splitlines()[0]
                    # A failed attention allocation can leave the local output
                    # (and its KV tensors) live until the next iteration.  Drop
                    # every per-row reference before trying another shape, so
                    # an OOM row cannot contaminate later measurements.
                    context = prefix = past = output = hidden = None
                    logits = token = emb = pos = positions = None
                    gc.collect()
                    torch.accelerator.empty_cache()
                except Exception as exc:
                    status, error = "error", repr(exc).splitlines()[0]
                    # Explicitly release any tensors created before the failing
                    # operation so the next shape starts from a clean baseline.
                    context = prefix = past = output = hidden = None  # noqa: F841
                    logits = token = emb = pos = positions = None  # noqa: F841
                    gc.collect()
                    torch.accelerator.empty_cache()
                result = ThinkerResult(
                    batch_size=batch_size,
                    context_length=context_length,
                    kv_length=kv_length,
                    generated_tokens=args.generated_tokens,
                    prefill_latency_ms_p50=_percentile(prefill_times, 0.50) if prefill_times else None,
                    prefill_latency_ms_p95=_percentile(prefill_times, 0.95) if prefill_times else None,
                    prefill_latency_ms_p99=_percentile(prefill_times, 0.99) if prefill_times else None,
                    decode_latency_ms_p50=_percentile(decode_times, 0.50) if decode_times else None,
                    decode_latency_ms_p95=_percentile(decode_times, 0.95) if decode_times else None,
                    decode_latency_ms_p99=_percentile(decode_times, 0.99) if decode_times else None,
                    unit_latency_ms_p50=_percentile(unit_times, 0.50) if unit_times else None,
                    unit_latency_ms_p95=_percentile(unit_times, 0.95) if unit_times else None,
                    unit_latency_ms_p99=_percentile(unit_times, 0.99) if unit_times else None,
                    tokens_per_second=(batch_size * args.generated_tokens) / (_percentile(decode_times, 0.50) / 1000)
                    if decode_times
                    else None,
                    aggregate_tokens_per_second=(batch_size * args.generated_tokens)
                    / (_percentile(unit_times, 0.50) / 1000)
                    if unit_times
                    else None,
                    peak_allocated_gib=peak_allocated,
                    peak_reserved_gib=peak_reserved,
                    repeats_completed=repeats_completed,
                    status=status,
                    oom=status == "oom",
                    error=error,
                )
                results.append(result)
                print(json.dumps(asdict(result), sort_keys=True), flush=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(
                {"metadata": metadata, "results": [asdict(result) for result in results]},
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")


if __name__ == "__main__":
    main()
