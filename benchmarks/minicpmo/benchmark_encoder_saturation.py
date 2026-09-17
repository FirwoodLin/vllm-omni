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

Audio streaming mode reproduces the native full-duplex load: 1 s chunks fed
through the checkpoint's ``StreamingMelProcessorExact`` (chunk_ms=1000,
first_chunk_ms=1035, cnn_redundancy_ms=20) into the production
``get_audio_embedding_streaming`` path with a growing per-session audio KV
cache and its 30 s reset.  Concurrency is modeled as independent interleaved
sessions (the production encoder path is batch-1 per session):

    CUDA_VISIBLE_DEVICES=4 /app/vllm-omni/.venv/bin/python \
        benchmarks/minicpmo/benchmark_encoder_saturation.py \
        --component audio --audio-mode streaming \
        --stream-seconds 35 --stream-count 8 \
        --output-json intermediate/minicpmo45_encoder_bench/streaming.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import WhisperConfig

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMForConditionalGeneration,
    MiniCPMWhisperEncoder,
    MultiModalProjector,
    Resampler,
    SiglipVisionConfig,
    SiglipVisionTransformer,
    _get_audio_cache_length,
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


@dataclass
class StreamingBenchResult:
    """One streaming audio-encoder run: stream_count interleaved 1 s-chunk sessions.

    Chunk categories follow the production duplex path: the first chunk of a
    session, steady chunks, and the chunk that triggers the 30 s audio-KV
    reset inside get_audio_embedding_streaming.
    """

    component: str
    stream_count: int
    stream_seconds: int
    chunk_count: int
    mel_shape_first: list[int]
    mel_shape_steady: list[int]
    embeds_per_chunk: int
    first_chunk_ms_median: float
    steady_chunk_ms_p50: float
    steady_chunk_ms_p95: float
    steady_chunk_ms_p99: float
    reset_chunk_ms_p50: float
    reset_chunk_ms_p95: float
    reset_chunk_ms_p99: float
    reset_chunk_count: int
    audio_seconds_per_second: float
    peak_allocated_gib: float
    peak_reserved_gib: float
    steady_chunk_samples_ms: list[float]
    reset_chunk_samples_ms: list[float]


def _csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
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


class StreamingAudioStack(nn.Module):
    """Production duplex streaming audio path bound to a standalone stack.

    Reuses MiniCPMO45OmniLLMForConditionalGeneration.get_audio_embedding_streaming
    unmodified (growing audio_past_key_values, 30 s reset, extra-context
    trimming, projection, stride-5 pooling) with only the apm/projector/pooler
    weights loaded.
    """

    get_audio_embedding_streaming = MiniCPMO45OmniLLMForConditionalGeneration.get_audio_embedding_streaming
    _get_feat_extract_output_lengths = MiniCPMO45OmniLLMForConditionalGeneration._get_feat_extract_output_lengths

    def __init__(self, apm: nn.Module, projector: nn.Module, pool_step: int):
        super().__init__()
        self.apm = apm
        self.audio_projection_layer = projector
        self.audio_avg_pooler = nn.AvgPool1d(pool_step, stride=pool_step)
        self.audio_encoder_layer = -1
        self.audio_past_key_values = None
        self.config = SimpleNamespace(audio_pool_step=pool_step)

    def cache_length(self) -> int:
        if self.audio_past_key_values is None:
            return 0
        return _get_audio_cache_length(self.audio_past_key_values)

    def embed_chunk(
        self,
        batch_feature: dict[str, Any],
        *,
        chunk_idx: int,
    ) -> torch.Tensor:
        embeds = self.get_audio_embedding_streaming(
            batch_feature,
            use_extra_context=True,
            prefix_extra_frames=0 if chunk_idx == 0 else 2,
            suffix_extra_frames=2,
        )
        return torch.cat(embeds[0], dim=0)


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


def _load_streaming_processor(model_path: Path) -> Any:
    """Load the checkpoint processor configured exactly like duplex stage0.

    Mirrors vllm_omni/model_executor/models/minicpmo_4_5/duplex/stage0.py
    (_ensure_streaming_processor): mode="exact", chunk_ms=1000,
    first_chunk_ms=1035, cnn_redundancy_ms=20, sliding window 30 s/10 s.
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    set_streaming = getattr(processor, "set_streaming_mode", None)
    if not callable(set_streaming):
        raise RuntimeError("checkpoint processor does not expose set_streaming_mode")
    set_streaming(
        mode="exact",
        chunk_ms=1000,
        first_chunk_ms=1035,
        cnn_redundancy_ms=20,
        enable_sliding_window=True,
        slide_trigger_seconds=30.0,
        slide_stride_seconds=10.0,
    )
    reset_streaming = getattr(processor, "reset_streaming", None)
    if callable(reset_streaming):
        reset_streaming()
    return processor


class _StreamingSession:
    """One duplex session's audio state: per-session mel buffers and KV cache.

    The processor copy and per-session mel deepcopy follow stage0's session
    setup. Encoder weights are shared through the single StreamingAudioStack;
    each session owns an independent audio_past_key_values that is swapped
    into the stack around every chunk, matching the production save/restore
    in stage0's _stage_audio_embeddings.
    """

    def __init__(self, processor: Any, stack: StreamingAudioStack):
        session_processor = copy.copy(processor)
        shared_mel = getattr(processor, "_streaming_mel_processor", None)
        if shared_mel is not None:
            session_processor._streaming_mel_processor = copy.deepcopy(shared_mel)
        self.processor = session_processor
        self.stack = stack
        self.audio_past_key_values = None

    def feed_chunk(self, waveform: np.ndarray, position: int) -> tuple[dict[str, Any], int]:
        mel_processor = self.processor._streaming_mel_processor
        samples = int(mel_processor.get_chunk_size())
        chunk = waveform[position : position + samples]
        batch_feature = self.processor.process_audio_streaming(
            chunk,
            reset=False,
            return_batch_feature=True,
        )
        if hasattr(batch_feature, "to"):
            batch_feature = batch_feature.to(next(self.stack.parameters()).device)
        return batch_feature, position + samples

    def embed_chunk(self, batch_feature: dict[str, Any], *, chunk_idx: int) -> torch.Tensor:
        previous = self.stack.audio_past_key_values
        self.stack.audio_past_key_values = self.audio_past_key_values
        try:
            embeds = self.stack.embed_chunk(batch_feature, chunk_idx=chunk_idx)
            self.audio_past_key_values = self.stack.audio_past_key_values
            return embeds
        finally:
            self.stack.audio_past_key_values = previous

    def cache_length(self) -> int:
        if self.audio_past_key_values is None:
            return 0
        return _get_audio_cache_length(self.audio_past_key_values)


def _benchmark_audio_streaming(
    apm: nn.Module,
    projector: nn.Module,
    config: dict[str, Any],
    processor: Any,
    stream_count: int,
    stream_seconds: int,
    warmup: int,
    repeats: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[StreamingBenchResult]:
    pool_step = int(config["audio_pool_step"])
    sample_rate = 16000
    total_samples = stream_seconds * sample_rate + sample_rate
    waveform = (
        0.05 * np.sin(2 * math.pi * 440.0 * np.arange(total_samples) / sample_rate)
    ).astype(np.float32)

    # One shared stack (production: one thinker instance, per-session KV).
    stack = StreamingAudioStack(apm, projector, pool_step).eval().to(device=device, dtype=dtype)

    # Warmup: throwaway sessions, a few untimed chunks.
    for _ in range(warmup):
        warm_session = _StreamingSession(processor, stack)
        position = 0
        for chunk_idx in range(min(3, stream_seconds)):
            batch_feature, position = warm_session.feed_chunk(waveform, position)
            warm_session.embed_chunk(batch_feature, chunk_idx=chunk_idx)
        del warm_session
        torch.accelerator.empty_cache()

    first_chunks_ms: list[float] = []
    steady_chunks_ms: list[float] = []
    reset_chunks_ms: list[float] = []
    mel_shape_first: list[int] = []
    mel_shape_steady: list[int] = []
    embeds_per_chunk = 0
    total_audio_seconds = 0.0
    timed_seconds = 0.0

    torch.accelerator.reset_peak_memory_stats(device)
    for _ in range(repeats):
        sessions = [_StreamingSession(processor, stack) for _ in range(stream_count)]
        positions = [0] * stream_count
        chunk_indices = [0] * stream_count
        remaining = [stream_seconds] * stream_count

        loop_start = time.perf_counter()
        while any(remaining):
            for session_idx in range(stream_count):
                if remaining[session_idx] <= 0:
                    continue
                session = sessions[session_idx]
                chunk_idx = chunk_indices[session_idx]
                cache_before = session.cache_length()

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                batch_feature, positions[session_idx] = session.feed_chunk(
                    waveform, positions[session_idx]
                )
                embeds = session.embed_chunk(batch_feature, chunk_idx=chunk_idx)
                end.record()
                torch.accelerator.synchronize(device)
                elapsed_ms = start.elapsed_time(end)

                mel = batch_feature["audio_features"]
                mel_shape = list(mel.shape[-3:]) if hasattr(mel, "shape") else []
                current_embeds = int(embeds.shape[0])
                if embeds_per_chunk == 0:
                    embeds_per_chunk = current_embeds
                elif embeds_per_chunk != current_embeds:
                    raise RuntimeError(
                        f"inconsistent embeds per chunk: {embeds_per_chunk} then {current_embeds}"
                    )

                cache_after = session.cache_length()
                if chunk_idx == 0:
                    first_chunks_ms.append(elapsed_ms)
                    mel_shape_first = mel_shape
                elif cache_after < cache_before:
                    reset_chunks_ms.append(elapsed_ms)
                else:
                    steady_chunks_ms.append(elapsed_ms)
                    if not mel_shape_steady:
                        mel_shape_steady = mel_shape

                chunk_indices[session_idx] += 1
                remaining[session_idx] -= 1
        timed_seconds += time.perf_counter() - loop_start
        total_audio_seconds += stream_count * stream_seconds

        del sessions
        torch.accelerator.empty_cache()

    del stack
    torch.accelerator.empty_cache()

    peak_allocated = torch.accelerator.max_memory_allocated(device) / 2**30
    peak_reserved = torch.accelerator.max_memory_reserved(device) / 2**30
    result = StreamingBenchResult(
        component="audio_streaming_duplex",
        stream_count=stream_count,
        stream_seconds=stream_seconds,
        chunk_count=len(first_chunks_ms) + len(steady_chunks_ms) + len(reset_chunks_ms),
        mel_shape_first=mel_shape_first,
        mel_shape_steady=mel_shape_steady,
        embeds_per_chunk=embeds_per_chunk,
        first_chunk_ms_median=statistics.median(first_chunks_ms) if first_chunks_ms else 0.0,
        steady_chunk_ms_p50=_percentile(steady_chunks_ms, 0.50),
        steady_chunk_ms_p95=_percentile(steady_chunks_ms, 0.95),
        steady_chunk_ms_p99=_percentile(steady_chunks_ms, 0.99),
        reset_chunk_ms_p50=_percentile(reset_chunks_ms, 0.50),
        reset_chunk_ms_p95=_percentile(reset_chunks_ms, 0.95),
        reset_chunk_ms_p99=_percentile(reset_chunks_ms, 0.99),
        reset_chunk_count=len(reset_chunks_ms),
        audio_seconds_per_second=total_audio_seconds / timed_seconds if timed_seconds else 0.0,
        peak_allocated_gib=peak_allocated,
        peak_reserved_gib=peak_reserved,
        steady_chunk_samples_ms=steady_chunks_ms,
        reset_chunk_samples_ms=reset_chunks_ms,
    )
    print(json.dumps(asdict(result), sort_keys=True), flush=True)
    return [result]


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
        "audio_mode": args.audio_mode,
        "stream_seconds": args.stream_seconds,
        "stream_count": args.stream_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--component", choices=("vision", "audio", "all"), default="all")
    parser.add_argument("--audio-mode", choices=("offline", "streaming"), default="offline")
    parser.add_argument("--batch-sizes", type=_csv_ints, default=_csv_ints("1,2,4,8,16,32,64"))
    parser.add_argument("--vision-patches", type=_csv_ints, default=_csv_ints("1024"))
    parser.add_argument("--audio-mel-frames", type=_csv_ints, default=_csv_ints("100,500,3000"))
    parser.add_argument("--stream-seconds", type=int, default=35)
    parser.add_argument("--stream-count", type=int, default=1)
    parser.add_argument("--vision-microbatch", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.vision_microbatch <= 0 or args.warmup < 1 or args.repeats < 1:
        parser.error("vision-microbatch, warmup, and repeats must be positive")
    if args.stream_seconds < 2 or args.stream_count < 1:
        parser.error("stream-seconds must be >= 2 and stream-count must be positive")
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
    streaming_results: list[StreamingBenchResult] = []
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
        if args.audio_mode == "streaming":
            processor = _load_streaming_processor(args.model)
            streaming_results.extend(
                _benchmark_audio_streaming(
                    stack.apm,
                    stack.projector,
                    config,
                    processor,
                    args.stream_count,
                    args.stream_seconds,
                    args.warmup,
                    args.repeats,
                    device,
                    dtype,
                )
            )
        else:
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
        if streaming_results:
            payload["streaming_results"] = [asdict(result) for result in streaming_results]
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main()
