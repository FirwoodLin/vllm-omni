# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark the MiniCPM-o 4.5 vision and audio encoder stacks.

This script loads only the relevant checkpoint prefixes instead of the 8.2B
Thinker LLM:

* vision: ``vpm`` (SigLIP) + ``resampler``;
* audio: ``apm`` (Whisper encoder) + ``audio_projection_layer`` + pooling.

The synthetic tensors match the shapes emitted by MiniCPM-o's processors.  In
particular, image patches are packed as ``[B, 3, 14, 14 * num_patches]`` and
audio is represented by 80-bin log-Mel frames at 100 frames/second.

GPU execution is required.  Example (use the repository virtualenv):

    CUDA_VISIBLE_DEVICES=4 /app/vllm-omni/.venv/bin/python \
        benchmarks/minicpmo/benchmark_encoder_saturation.py \
        --component all --output-json intermediate/minicpmo45_encoder_bench/run.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import WhisperConfig

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMWhisperEncoder,
    MultiModalProjector,
    Resampler,
    SiglipVisionConfig,
    SiglipVisionTransformer,
)

DEFAULT_MODEL = "/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/models--OpenBMB--MiniCPM-o-4_5"


@dataclass
class BenchResult:
    component: str
    batch_size: int
    input_sequence_length: int
    encoder_sequence_length: int
    output_sequence_length: int
    latency_ms_median: float
    latency_ms_p95: float
    latency_ms_p99: float
    latency_ms_min: float
    latency_ms_max: float
    latency_samples_ms: list[float]
    items_per_second: float
    media_units_per_second: float
    media_unit: str
    approximate_dense_tflops: float
    approximate_flops_per_item_t: float
    peak_allocated_gib: float
    peak_reserved_gib: float
    output_shape: list[int]
    vision_microbatch: int | None = None


def _csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


@contextmanager
def _default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _checkpoint_shard(model_path: Path, prefix: str) -> Path:
    index = _load_json(model_path / "model.safetensors.index.json")["weight_map"]
    shards = {filename for name, filename in index.items() if name.startswith(prefix)}
    if len(shards) != 1:
        raise RuntimeError(f"expected one shard for prefix {prefix!r}, found {sorted(shards)}")
    return model_path / shards.pop()


def _load_prefix(
    module: nn.Module,
    model_path: Path,
    prefix: str,
    device: torch.device,
) -> None:
    """Assign one checkpoint prefix directly into a meta-initialized module."""
    state: dict[str, torch.Tensor] = {}
    shard = _checkpoint_shard(model_path, prefix)
    with safe_open(shard, framework="pt", device=str(device)) as handle:
        for name in handle.keys():
            if name.startswith(prefix):
                state[name.removeprefix(prefix)] = handle.get_tensor(name)
    incompatible = module.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"failed loading {prefix}: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def _meta_module(factory: Callable[[], nn.Module], dtype: torch.dtype) -> nn.Module:
    with _default_dtype(dtype), torch.device("meta"):
        return factory()


class VisionStack(nn.Module):
    """The exact vLLM-Omni VPM batching policy followed by the resampler."""

    def __init__(self, vpm: nn.Module, resampler: nn.Module, microbatch: int):
        super().__init__()
        self.vpm = vpm
        self.resampler = resampler
        self.microbatch = microbatch

    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: torch.Tensor,
        tgt_sizes: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(pixel_values.shape[0])
        chunks: list[torch.Tensor] = []
        for start in range(0, batch_size, self.microbatch):
            end = min(start + self.microbatch, batch_size)
            output = self.vpm(
                pixel_values[start:end],
                patch_attention_mask=patch_attention_mask[start:end],
                tgt_sizes=tgt_sizes[start:end],
            )
            chunks.append(output.last_hidden_state)
        return self.resampler(torch.cat(chunks, dim=0), tgt_sizes)


class AudioStack(nn.Module):
    """Whisper encoder, 1024->4096 projector, and stride-5 average pooling."""

    def __init__(self, apm: nn.Module, projector: nn.Module, pool_step: int):
        super().__init__()
        self.apm = apm
        self.projector = projector
        self.pool = nn.AvgPool1d(pool_step, stride=pool_step)

    def forward(self, mel: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        states = self.apm(
            mel,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
        states = self.projector(states)
        return self.pool(states.transpose(1, 2)).transpose(1, 2)


def _build_vision_stack(
    config: dict[str, Any],
    model_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    microbatch: int,
) -> VisionStack:
    vision_config = SiglipVisionConfig(**config["vision_config"])
    # This is the repository's deployed fallback when FlashAttention 2 is not
    # installed.  The custom SigLIP class has no separate SDPA path.
    vision_config._attn_implementation = "eager"
    vpm = _meta_module(lambda: SiglipVisionTransformer(vision_config), dtype)
    _load_prefix(vpm, model_path, "vpm.", device)

    # Resampler has a non-persistent 2-D positional buffer, so construct it on
    # the real device while assigning checkpoint parameters afterward.
    with _default_dtype(dtype), torch.device(device):
        resampler = Resampler(
            num_queries=int(config["query_num"]),
            embed_dim=int(config["hidden_size"]),
            num_heads=int(config["hidden_size"]) // 128,
            kv_dim=int(config["vision_config"]["hidden_size"]),
            adaptive=True,
        )
    _load_prefix(resampler, model_path, "resampler.", device)
    return VisionStack(vpm, resampler, microbatch).eval()


def _build_audio_stack(
    config: dict[str, Any],
    model_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> AudioStack:
    audio_config = WhisperConfig(**config["audio_config"])
    audio_config._attn_implementation = "sdpa"
    apm = _meta_module(lambda: MiniCPMWhisperEncoder(audio_config), dtype)
    _load_prefix(apm, model_path, "apm.", device)
    projector = _meta_module(
        lambda: MultiModalProjector(
            int(audio_config.encoder_ffn_dim) // 4,
            int(config["hidden_size"]),
        ),
        dtype,
    )
    _load_prefix(projector, model_path, "audio_projection_layer.", device)
    return AudioStack(apm, projector, int(config["audio_pool_step"])).eval()


def _measure(
    function: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> tuple[list[float], torch.Tensor, float, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            output = function()
        torch.accelerator.synchronize(device)
        del output
        torch.accelerator.reset_peak_memory_stats(device)

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        output = torch.empty(0, device=device)
        for start, end in zip(starts, ends, strict=True):
            start.record()
            output = function()
            end.record()
        torch.accelerator.synchronize(device)
        times_ms = [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]
        peak_allocated = torch.accelerator.max_memory_allocated(device) / 2**30
        peak_reserved = torch.accelerator.max_memory_reserved(device) / 2**30
    return times_ms, output, peak_allocated, peak_reserved


def _vision_flops_per_item(config: dict[str, Any], patches: int) -> float:
    vision = config["vision_config"]
    d = int(vision["hidden_size"])
    f = int(vision["intermediate_size"])
    layers = int(vision["num_hidden_layers"])
    patch = int(vision["patch_size"])
    llm_d = int(config["hidden_size"])
    queries = int(config["query_num"])

    patch_embed = 2 * patches * 3 * patch * patch * d
    encoder = layers * (8 * patches * d * d + 4 * patches * d * f + 4 * patches * patches * d)
    # kv projection + MHA (Q/K/V/out) + final output projection.
    resampler = (
        2 * patches * d * llm_d
        + 4 * patches * llm_d * llm_d
        + 6 * queries * llm_d * llm_d
        + 4 * queries * patches * llm_d
    )
    return float(patch_embed + encoder + resampler)


def _audio_flops_per_item(config: dict[str, Any], mel_frames: int) -> float:
    audio = config["audio_config"]
    d = int(audio["d_model"])
    f = int(audio["encoder_ffn_dim"])
    layers = int(audio["encoder_layers"])
    mel_bins = int(audio["num_mel_bins"])
    encoder_frames = (mel_frames - 1) // 2 + 1
    llm_d = int(config["hidden_size"])

    conv1 = 2 * mel_frames * mel_bins * d * 3
    conv2 = 2 * encoder_frames * d * d * 3
    encoder = layers * (
        8 * encoder_frames * d * d + 4 * encoder_frames * d * f + 4 * encoder_frames * encoder_frames * d
    )
    projector = 2 * encoder_frames * (d * llm_d + llm_d * llm_d)
    return float(conv1 + conv2 + encoder + projector)


def _result(
    *,
    component: str,
    batch_size: int,
    input_sequence_length: int,
    encoder_sequence_length: int,
    output_sequence_length: int,
    times_ms: list[float],
    output: torch.Tensor,
    flops_per_item: float,
    media_units_per_item: float,
    media_unit: str,
    peak_allocated: float,
    peak_reserved: float,
    vision_microbatch: int | None = None,
) -> BenchResult:
    median_ms = statistics.median(times_ms)
    seconds = median_ms / 1000
    return BenchResult(
        component=component,
        batch_size=batch_size,
        input_sequence_length=input_sequence_length,
        encoder_sequence_length=encoder_sequence_length,
        output_sequence_length=output_sequence_length,
        latency_ms_median=median_ms,
        latency_ms_p95=_percentile(times_ms, 0.95),
        latency_ms_p99=_percentile(times_ms, 0.99),
        latency_ms_min=min(times_ms),
        latency_ms_max=max(times_ms),
        latency_samples_ms=times_ms,
        items_per_second=batch_size / seconds,
        media_units_per_second=batch_size * media_units_per_item / seconds,
        media_unit=media_unit,
        approximate_dense_tflops=(batch_size * flops_per_item / seconds) / 1e12,
        approximate_flops_per_item_t=flops_per_item / 1e12,
        peak_allocated_gib=peak_allocated,
        peak_reserved_gib=peak_reserved,
        output_shape=list(output.shape),
        vision_microbatch=vision_microbatch,
    )


def _benchmark_vision(
    stack: VisionStack,
    config: dict[str, Any],
    batch_sizes: list[int],
    patch_counts: list[int],
    warmup: int,
    repeats: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    patch_size = int(config["vision_config"]["patch_size"])
    for patches in patch_counts:
        grid_h = math.isqrt(patches)
        while patches % grid_h:
            grid_h -= 1
        grid_w = patches // grid_h
        for batch_size in batch_sizes:
            # This is MiniCPM-o's patch-packed representation, not a conventional
            # square [B, 3, H, W] image tensor.
            pixel_values = torch.randn(
                batch_size,
                3,
                patch_size,
                patch_size * patches,
                device=device,
                dtype=dtype,
            )
            mask = torch.ones(batch_size, 1, patches, device=device, dtype=torch.bool)
            tgt_sizes = torch.tensor(
                [[grid_h, grid_w]] * batch_size,
                device=device,
                dtype=torch.int32,
            )

            function = partial(stack, pixel_values, mask, tgt_sizes)
            times, output, peak_allocated, peak_reserved = _measure(
                function,
                warmup=warmup,
                repeats=repeats,
                device=device,
            )
            result = _result(
                component="vision_vpm_resampler",
                batch_size=batch_size,
                input_sequence_length=patches,
                encoder_sequence_length=patches,
                output_sequence_length=int(config["query_num"]),
                times_ms=times,
                output=output,
                flops_per_item=_vision_flops_per_item(config, patches),
                media_units_per_item=1,
                media_unit="images",
                peak_allocated=peak_allocated,
                peak_reserved=peak_reserved,
                vision_microbatch=stack.microbatch,
            )
            results.append(result)
            print(json.dumps(asdict(result), sort_keys=True), flush=True)
            del pixel_values, mask, tgt_sizes, output
            torch.accelerator.empty_cache()
    return results


def _chunk_attention_mask(
    batch_size: int,
    encoder_frames: int,
    chunk_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    indices = torch.arange(encoder_frames, device=device)
    chunk_end = torch.clamp((indices // chunk_frames + 1) * chunk_frames, max=encoder_frames)
    allowed = indices.unsqueeze(0) < chunk_end.unsqueeze(1)
    mask = torch.zeros((encoder_frames, encoder_frames), device=device, dtype=dtype)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask.view(1, 1, encoder_frames, encoder_frames).expand(batch_size, -1, -1, -1)


def _benchmark_audio(
    stack: AudioStack,
    config: dict[str, Any],
    batch_sizes: list[int],
    mel_frame_counts: list[int],
    warmup: int,
    repeats: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    mel_bins = int(config["audio_config"]["num_mel_bins"])
    pool_step = int(config["audio_pool_step"])
    chunk_frames = int(float(config["audio_chunk_length"]) * 50)
    for mel_frames in mel_frame_counts:
        encoder_frames = (mel_frames - 1) // 2 + 1
        output_frames = (encoder_frames - pool_step) // pool_step + 1
        for batch_size in batch_sizes:
            mel = torch.randn(
                batch_size,
                mel_bins,
                mel_frames,
                device=device,
                dtype=dtype,
            )
            attention_mask = _chunk_attention_mask(
                batch_size,
                encoder_frames,
                chunk_frames,
                device,
                dtype,
            )

            function = partial(stack, mel, attention_mask)
            times, output, peak_allocated, peak_reserved = _measure(
                function,
                warmup=warmup,
                repeats=repeats,
                device=device,
            )
            result = _result(
                component="audio_apm_projector_pool",
                batch_size=batch_size,
                input_sequence_length=mel_frames,
                encoder_sequence_length=encoder_frames,
                output_sequence_length=output_frames,
                times_ms=times,
                output=output,
                flops_per_item=_audio_flops_per_item(config, mel_frames),
                media_units_per_item=mel_frames / 100,
                media_unit="audio_seconds",
                peak_allocated=peak_allocated,
                peak_reserved=peak_reserved,
            )
            results.append(result)
            print(json.dumps(asdict(result), sort_keys=True), flush=True)
            del mel, attention_mask, output
            torch.accelerator.empty_cache()
    return results


def _metadata(device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "timestamp_unix": time.time(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": props.name,
        "gpu_total_memory_gib": props.total_memory / 2**30,
        "gpu_sm_count": props.multi_processor_count,
        "gpu_compute_capability": f"{props.major}.{props.minor}",
        "dtype": "bfloat16",
        "model_path": str(args.model),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "vision_microbatch": args.vision_microbatch,
        "batch_sizes": args.batch_sizes,
        "vision_patches": args.vision_patches,
        "audio_mel_frames": args.audio_mel_frames,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--component", choices=("vision", "audio", "all"), default="all")
    parser.add_argument("--batch-sizes", type=_csv_ints, default=_csv_ints("1,2,4,8,16,32,64"))
    parser.add_argument("--vision-patches", type=_csv_ints, default=_csv_ints("1024"))
    parser.add_argument("--audio-mel-frames", type=_csv_ints, default=_csv_ints("100,500,3000"))
    parser.add_argument("--vision-microbatch", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.vision_microbatch <= 0 or args.warmup < 1 or args.repeats < 1:
        parser.error("vision-microbatch, warmup, and repeats must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; request GPU access before running this benchmark")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = _load_json(args.model / "config.json")
    metadata = _metadata(device, args)
    print(json.dumps({"metadata": metadata}, sort_keys=True), flush=True)

    results: list[BenchResult] = []
    if args.component in ("vision", "all"):
        stack = _build_vision_stack(
            config,
            args.model,
            device,
            dtype,
            args.vision_microbatch,
        )
        results.extend(
            _benchmark_vision(
                stack,
                config,
                args.batch_sizes,
                args.vision_patches,
                args.warmup,
                args.repeats,
                device,
                dtype,
            )
        )
        del stack
        torch.accelerator.empty_cache()

    if args.component in ("audio", "all"):
        stack = _build_audio_stack(config, args.model, device, dtype)
        results.extend(
            _benchmark_audio(
                stack,
                config,
                args.batch_sizes,
                args.audio_mel_frames,
                args.warmup,
                args.repeats,
                device,
                dtype,
            )
        )
        del stack
        torch.accelerator.empty_cache()

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": metadata,
            "results": [asdict(result) for result in results],
        }
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main()
