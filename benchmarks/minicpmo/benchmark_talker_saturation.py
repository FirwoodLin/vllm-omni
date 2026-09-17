# SPDX-License-Identifier: Apache-2.0
"""Measure the MiniCPM-o 4.5 Talker AR path in isolation.

The full-duplex service runs Talker as a vLLM LLM stage.  Reproducing the
vLLM scheduler and paged KV manager in a standalone process would hide the
actual model cost behind an otherwise empty Thinker/Code2Wav pipeline, so this
microbenchmark uses the checkpoint's real ``tts.*`` weights and the same
Llama/embedding/head operations as the native Talker.  It deliberately keeps
the KV cache between the condition prefill and the 26-token native duplex
chunk (25 codec frames plus the terminal sample), and uses a batched forward
for B1/B2/B4/B8.

This is a model-compute reference, not a claim about scheduler queueing.  The
JSON records the exact condition and pre-existing KV lengths so the numbers can
be compared with Stage1 unit deltas from the service runs.

Example (GPU access is required):

    CUDA_VISIBLE_DEVICES=0 /app/vllm-omni/.venv/bin/python \
        benchmarks/minicpmo/benchmark_talker_saturation.py \
        --output-json intermediate/minicpmo45_8h200_allocation/\
        component_talker_h200_gpu0_20260829.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import AutoConfig

from vllm_omni.model_executor.models.minicpmo_4_5 import MINICPMO45_DUPLEX_CODEC_TOKENS_PER_CHUNK

DEFAULT_MODEL = Path("/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/models--OpenBMB--MiniCPM-o-4_5")
DEFAULT_BATCH_SIZES = "1,2,4,8"
DEFAULT_CONDITION_LENGTHS = "8,32,128"
DEFAULT_KV_LENGTHS = "0,256,1024"
NATIVE_DUPLEX_TOKENS = MINICPMO45_DUPLEX_CODEC_TOKENS_PER_CHUNK


@dataclass
class TalkerResult:
    batch_size: int
    condition_length: int
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


def _csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _csv_nonnegative_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated non-negative integers")
    return values


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty list")
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _load_remote_tts_class(model_path: Path):
    """Import the checkpoint's HF MiniCPMTTS without importing the full model."""
    # modeling_minicpmo.py uses relative imports (.configuration_minicpmo,
    # .utils, ...).  A small package shim lets Python resolve those files from
    # the local checkpoint while avoiding a network fetch.
    package_name = "minicpmo45_checkpoint"
    package = types.ModuleType(package_name)
    package.__path__ = [str(model_path)]
    sys.modules.setdefault(package_name, package)
    module = importlib.import_module(f"{package_name}.modeling_minicpmo")
    config_module = importlib.import_module(f"{package_name}.configuration_minicpmo")
    return module.MiniCPMTTS, config_module.MiniCPMTTSConfig


def _build_model(model_path: Path, device: torch.device):
    tts_class, tts_config_class = _load_remote_tts_class(model_path)
    hf_config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    tts_config = tts_config_class.from_dict(hf_config.tts_config.to_dict())
    # These generation defaults are read by MiniCPMTTS.__init__, but are not
    # serialized in this checkpoint's tts_config.  They match the deploy YAML.
    tts_config.top_p = 0.85
    tts_config.top_k = 25
    tts_config.repetition_penalty = 1.05
    with torch.device("meta"):
        model = tts_class(tts_config, audio_tokenizer=None)
    # The model has a small non-persistent causal-mask buffer that is not in
    # the safetensors index.  Materialize all meta tensors first, then assign
    # the real checkpoint tensors below; calling ``to`` after assign would
    # still fail on that untouched meta buffer.
    model.to_empty(device=device)

    index_path = model_path / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    shards = {filename for name, filename in weight_map.items() if name.startswith("tts.")}
    if len(shards) != 1:
        raise RuntimeError(f"expected one Talker shard, found {sorted(shards)}")
    shard_path = model_path / shards.pop()
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(shard_path), framework="pt", device=str(device)) as handle:
        for name in handle.keys():
            if name.startswith("tts."):
                state[name.removeprefix("tts.")] = handle.get_tensor(name)
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Talker weight loading failed: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    del state
    model.eval()
    return model, tts_config


def _build_condition(model, tts_config, batch_size: int, condition_length: int, device: torch.device):
    # Native duplex uses condition token ids + projected thinker states, then
    # appends exactly one audio-BOS embedding (see
    # MiniCPMO45OmniTTSForConditionalGeneration._build_condition_embeddings,
    # native_duplex=True).  The offline TTS path appends text-EOS as well, but
    # that is intentionally not benchmarked here.
    if condition_length < 1:
        raise ValueError("condition length must be at least one (audio_bos suffix)")
    dtype = model.emb_text.weight.dtype
    content_length = condition_length - 1
    text_ids = torch.randint(
        0,
        int(tts_config.num_text_tokens),
        (batch_size, content_length),
        device=device,
        dtype=torch.long,
    )
    hidden = torch.randn(
        batch_size,
        content_length,
        int(tts_config.llm_dim),
        device=device,
        dtype=dtype,
    )
    # Match vllm_omni's _build_condition_embeddings.  Normalizing the
    # projected hidden separately is intentional (the production code
    # normalizes before adding text embeddings).
    projected = model.projector_semantic(hidden)
    if bool(getattr(tts_config, "normalize_projected_hidden", False)):
        projected = torch.nn.functional.normalize(projected, p=2, dim=-1)
    content = model.emb_text(text_ids) + projected
    audio_bos = model.emb_text(torch.tensor([int(tts_config.audio_bos_token_id)], device=device, dtype=torch.long))
    boundary = audio_bos.unsqueeze(0).expand(batch_size, -1, -1)
    return torch.cat([content, boundary], dim=1)


def _build_prefix(model, tts_config, batch_size: int, kv_length: int, device: torch.device):
    if kv_length == 0:
        return None
    token_ids = torch.randint(
        0,
        int(tts_config.num_audio_tokens) - 1,
        (batch_size, kv_length),
        device=device,
        dtype=torch.long,
    )
    return model.emb_code[0](token_ids)


@torch.inference_mode()
def _run_unit(model, condition, prefix, kv_length: int, generated_tokens: int, device: torch.device):
    """Run one production-shaped condition prefill plus native duplex chunk."""
    batch_size = int(condition.shape[0])
    condition_length = int(condition.shape[1])
    if prefix is not None:
        prefix_positions = torch.arange(kv_length, device=device, dtype=torch.long).unsqueeze(0)
        prefix_positions = prefix_positions.expand(batch_size, -1)
        prefix_output = model.model(
            inputs_embeds=prefix,
            position_ids=prefix_positions,
            use_cache=True,
        )
        past_key_values = prefix_output.past_key_values
        del prefix_output, prefix_positions
    else:
        past_key_values = None

    condition_positions = (
        torch.arange(
            kv_length,
            kv_length + condition_length,
            device=device,
            dtype=torch.long,
        )
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    condition_output = model.model(
        inputs_embeds=condition,
        position_ids=condition_positions,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = condition_output.past_key_values
    hidden = condition_output.last_hidden_state[:, -1:]
    del condition_output, condition_positions

    generated = []
    for step in range(generated_tokens):
        logits = model.head_code[0](hidden[:, -1]).float()
        # Argmax removes sampling variance while retaining the full codec head
        # and embedding path.  The production sampler's top-p/top-k overhead is
        # tiny relative to the 20-layer Talker forward and is not this test's
        # target.  Exclude EOS from synthetic prefix ids; a fixed step count
        # still measures the terminal sample at step 26.
        token = torch.argmax(logits, dim=-1)
        generated.append(token)
        if step + 1 == generated_tokens:
            break
        token_embedding = model.emb_code[0](token.unsqueeze(1))
        position = torch.full(
            (batch_size, 1),
            kv_length + condition_length + step,
            device=device,
            dtype=torch.long,
        )
        decode_output = model.model(
            inputs_embeds=token_embedding,
            position_ids=position,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = decode_output.past_key_values
        hidden = decode_output.last_hidden_state
        del logits, token_embedding, position, decode_output
    del generated, hidden, past_key_values


def _metadata(args: argparse.Namespace, device: torch.device, tts_config) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "timestamp_unix": time.time(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": props.name,
        "gpu_total_memory_gib": props.total_memory / 2**30,
        "device": str(device),
        "model": str(args.model),
        "weight_prefix": "tts.*",
        "hidden_size": int(tts_config.hidden_size),
        "num_hidden_layers": int(tts_config.num_hidden_layers),
        "num_attention_heads": int(tts_config.num_attention_heads),
        "num_audio_tokens": int(tts_config.num_audio_tokens),
        "max_position_embeddings": int(tts_config.max_position_embeddings),
        "native_duplex_generated_tokens": args.generated_tokens,
        "batch_sizes": args.batch_sizes,
        "condition_lengths": args.condition_lengths,
        "kv_lengths": args.kv_lengths,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "measurement_scope": "condition prefill + 26 Talker decode forwards; existing KV prefix prefill excluded",
        "condition_shape": (
            "native_duplex: N token+hidden pairs followed by one audio_bos embedding (no offline text_eos suffix)"
        ),
        "sampling": "argmax (fixed 26-step loop; EOS excluded from synthetic KV prefix)",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=int, default=0, help="CUDA ordinal inside CUDA_VISIBLE_DEVICES")
    parser.add_argument("--batch-sizes", type=_csv_ints, default=_csv_ints(DEFAULT_BATCH_SIZES))
    parser.add_argument("--condition-lengths", type=_csv_ints, default=_csv_ints(DEFAULT_CONDITION_LENGTHS))
    parser.add_argument(
        "--kv-lengths",
        type=_csv_nonnegative_ints,
        default=_csv_nonnegative_ints(DEFAULT_KV_LENGTHS),
        help="Existing Talker KV positions; include 0 explicitly if desired",
    )
    parser.add_argument("--generated-tokens", type=int, default=NATIVE_DUPLEX_TOKENS)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    for name in ("generated_tokens", "warmup", "repeats"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if any(length < 1 for length in args.condition_lengths):
        parser.error("all --condition-lengths values must be at least 1")
    max_positions = 4096
    if any(k + c + args.generated_tokens > max_positions for k in args.kv_lengths for c in args.condition_lengths):
        parser.error(
            "kv length + condition length + generated tokens must not exceed Talker max_position_embeddings=4096"
        )
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; request GPU access before running this benchmark")
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    print(json.dumps({"event": "loading_talker", "device": str(device)}, sort_keys=True), flush=True)
    model, tts_config = _build_model(args.model, device)
    metadata = _metadata(args, device, tts_config)
    metadata["talker_parameters"] = sum(param.numel() for param in model.parameters())
    print(json.dumps({"metadata": metadata}, sort_keys=True), flush=True)

    results: list[TalkerResult] = []
    for condition_length in args.condition_lengths:
        for kv_length in args.kv_lengths:
            for batch_size in args.batch_sizes:
                status = "ok"
                error = None
                prefill_times: list[float] = []
                decode_times: list[float] = []
                unit_times: list[float] = []
                peak_allocated = peak_reserved = None
                repeats_completed = 0
                try:
                    condition = _build_condition(model, tts_config, batch_size, condition_length, device)
                    prefix = _build_prefix(model, tts_config, batch_size, kv_length, device)
                    for _ in range(args.warmup):
                        _run_unit(model, condition, prefix, kv_length, args.generated_tokens, device)
                    torch.accelerator.synchronize(device)
                    torch.accelerator.reset_peak_memory_stats(device)

                    # The unit is split into a condition prefill and 26 decode
                    # forwards so callers can distinguish condition/KV effects
                    # from steady codec-token throughput.
                    for _ in range(args.repeats):
                        torch.accelerator.synchronize(device)
                        prefix_start = torch.cuda.Event(enable_timing=True)
                        prefill_end = torch.cuda.Event(enable_timing=True)
                        unit_end = torch.cuda.Event(enable_timing=True)
                        # Build the prefix KV outside the timed unit, matching a
                        # steady session whose previous chunks already reside in
                        # the scheduler KV cache.
                        if prefix is not None:
                            prefix_positions = torch.arange(kv_length, device=device, dtype=torch.long).unsqueeze(0)
                            prefix_positions = prefix_positions.expand(batch_size, -1)
                            prefix_start.record()
                            prefix_output = model.model(
                                inputs_embeds=prefix,
                                position_ids=prefix_positions,
                                use_cache=True,
                            )
                            past = prefix_output.past_key_values
                            prefix_start.synchronize()
                            del prefix_output, prefix_positions
                        else:
                            past = None
                        condition_positions = (
                            torch.arange(kv_length, kv_length + condition_length, device=device, dtype=torch.long)
                            .unsqueeze(0)
                            .expand(batch_size, -1)
                        )
                        start = torch.cuda.Event(enable_timing=True)
                        start.record()
                        output = model.model(
                            inputs_embeds=condition,
                            position_ids=condition_positions,
                            past_key_values=past,
                            use_cache=True,
                        )
                        prefill_end.record()
                        past = output.past_key_values
                        hidden = output.last_hidden_state[:, -1:]
                        del output, condition_positions
                        for step in range(args.generated_tokens):
                            logits = model.head_code[0](hidden[:, -1]).float()
                            token = torch.argmax(logits, dim=-1)
                            if step + 1 == args.generated_tokens:
                                del logits, token
                                break
                            token_embedding = model.emb_code[0](token.unsqueeze(1))
                            position = torch.full(
                                (batch_size, 1), kv_length + condition_length + step, device=device, dtype=torch.long
                            )
                            output = model.model(
                                inputs_embeds=token_embedding,
                                position_ids=position,
                                past_key_values=past,
                                use_cache=True,
                            )
                            past = output.past_key_values
                            hidden = output.last_hidden_state
                            del logits, token, token_embedding, position, output
                        unit_end.record()
                        unit_end.synchronize()
                        prefill_times.append(float(start.elapsed_time(prefill_end)))
                        unit_ms = float(start.elapsed_time(unit_end))
                        unit_times.append(unit_ms)
                        decode_times.append(unit_ms - prefill_times[-1])
                        repeats_completed += 1
                        del hidden, past
                    peak_allocated = torch.accelerator.max_memory_allocated(device) / 2**30
                    peak_reserved = torch.accelerator.max_memory_reserved(device) / 2**30
                    del condition, prefix
                    torch.accelerator.empty_cache()
                except torch.cuda.OutOfMemoryError as exc:
                    status = "oom"
                    error = repr(exc).splitlines()[0]
                    try:
                        torch.accelerator.synchronize(device)
                    except Exception:
                        pass
                    torch.accelerator.empty_cache()
                except Exception as exc:  # retain a row so partial sweeps are useful
                    status = "error"
                    error = repr(exc).splitlines()[0]
                    torch.accelerator.empty_cache()

                result = TalkerResult(
                    batch_size=batch_size,
                    condition_length=condition_length,
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
