# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from transformers import WhisperConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "benchmarks" / "minicpmo"
DEFAULT_MODEL = Path(
    "/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/models--OpenBMB--MiniCPM-o-4_5"
)

STREAMING_RESULT_FIELDS = {
    "component",
    "stream_count",
    "stream_seconds",
    "chunk_count",
    "mel_shape_first",
    "mel_shape_steady",
    "embeds_per_chunk",
    "first_chunk_ms_median",
    "steady_chunk_ms_p50",
    "steady_chunk_ms_p95",
    "steady_chunk_ms_p99",
    "reset_chunk_ms_p50",
    "reset_chunk_ms_p95",
    "reset_chunk_ms_p99",
    "reset_chunk_count",
    "audio_seconds_per_second",
    "peak_allocated_gib",
    "peak_reserved_gib",
    "steady_chunk_samples_ms",
    "reset_chunk_samples_ms",
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench():
    return _load_module("benchmark_encoder_saturation_test", BENCHMARK_DIR / "benchmark_encoder_saturation.py")


def _small_streaming_stack(bench, max_source_positions: int = 1500):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
        MiniCPMWhisperEncoder,
        MultiModalProjector,
    )

    # encoder_ffn_dim must equal 4 * d_model so the projector in_dim matches
    # the encoder output, mirroring the production config relationship.
    config = WhisperConfig(
        d_model=32,
        encoder_layers=2,
        encoder_attention_heads=2,
        encoder_ffn_dim=128,
        num_mel_bins=80,
        max_source_positions=max_source_positions,
        _attn_implementation="sdpa",
    )
    apm = MiniCPMWhisperEncoder(config).eval()
    projector = MultiModalProjector(32, 48)
    return bench.StreamingAudioStack(apm, projector, pool_step=5)


def test_streaming_bench_module_exports(bench):
    assert hasattr(bench, "StreamingAudioStack")
    assert hasattr(bench, "StreamingBenchResult")
    assert hasattr(bench, "_benchmark_audio_streaming")
    assert hasattr(bench, "_load_streaming_processor")
    assert hasattr(bench, "_StreamingSession")


def test_streaming_bench_result_schema(bench):
    result = bench.StreamingBenchResult(
        component="audio_streaming_duplex",
        stream_count=1,
        stream_seconds=35,
        chunk_count=35,
        mel_shape_first=[1, 80, 102],
        mel_shape_steady=[1, 80, 104],
        embeds_per_chunk=10,
        first_chunk_ms_median=1.0,
        steady_chunk_ms_p50=1.0,
        steady_chunk_ms_p95=1.0,
        steady_chunk_ms_p99=1.0,
        reset_chunk_ms_p50=1.0,
        reset_chunk_ms_p95=1.0,
        reset_chunk_ms_p99=1.0,
        reset_chunk_count=1,
        audio_seconds_per_second=1.0,
        peak_allocated_gib=0.1,
        peak_reserved_gib=0.2,
        steady_chunk_samples_ms=[1.0],
        reset_chunk_samples_ms=[1.0],
    )
    from dataclasses import asdict

    assert set(asdict(result)) == STREAMING_RESULT_FIELDS


def test_parse_args_streaming_defaults(bench, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--audio-mode", "streaming"])
    args = bench.parse_args()
    assert args.audio_mode == "streaming"
    assert args.stream_seconds == 35
    assert args.stream_count == 1


def test_parse_args_streaming_validation(bench, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--audio-mode", "streaming", "--stream-seconds", "1"])
    with pytest.raises(SystemExit):
        bench.parse_args()
    monkeypatch.setattr(sys, "argv", ["prog", "--audio-mode", "streaming", "--stream-count", "0"])
    with pytest.raises(SystemExit):
        bench.parse_args()


def test_streaming_audio_stack_chunk_math(bench):
    stack = _small_streaming_stack(bench)
    # Production duplex mel shapes: first chunk 102 frames, steady 104 frames.
    # With use_extra_context (prefix 2 / suffix 2 from chunk 1 on) each chunk
    # contributes exactly 50 encoder frames -> 10 pooled embeddings.
    mel_frames = [102, 104, 104, 104, 104]
    for chunk_idx, frames in enumerate(mel_frames):
        mel = torch.randn(1, 80, frames)
        batch_feature = {"audio_features": mel, "audio_feature_lens": [torch.tensor([frames])]}
        embeds = stack.embed_chunk(batch_feature, chunk_idx=chunk_idx)
        assert embeds.shape == (10, 48)
        assert stack.cache_length() == 50 * (chunk_idx + 1)


def test_streaming_audio_stack_resets_at_position_limit(bench):
    stack = _small_streaming_stack(bench, max_source_positions=150)
    apm_max_len = stack.apm.embed_positions.weight.shape[0]
    assert apm_max_len == 150
    # Chunks 0/1 fill the cache to 100; chunk 2 would reach 150 >= 150 and
    # must reset the audio KV cache back to a single chunk, matching the
    # production 30 s boundary behavior.
    for chunk_idx in range(3):
        frames = 102 if chunk_idx == 0 else 104
        mel = torch.randn(1, 80, frames)
        batch_feature = {"audio_features": mel, "audio_feature_lens": [torch.tensor([frames])]}
        stack.embed_chunk(batch_feature, chunk_idx=chunk_idx)
        if chunk_idx < 2:
            assert stack.cache_length() == 50 * (chunk_idx + 1)
        else:
            assert stack.cache_length() == 50


@pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="MiniCPM-o 4.5 checkpoint not available")
def test_streaming_mel_processor_chunk_shapes(bench):
    processor = bench._load_streaming_processor(DEFAULT_MODEL)
    mel_processor = processor._streaming_mel_processor
    assert mel_processor.sample_rate == 16000
    assert mel_processor.chunk_samples == 16000

    rng = torch.Generator().manual_seed(0)
    shapes = []
    for _ in range(3):
        samples = mel_processor.get_chunk_size()
        chunk = (torch.randn(samples, generator=rng) * 0.05).numpy().astype("float32")
        batch_feature = processor.process_audio_streaming(
            chunk, reset=False, return_batch_feature=True
        )
        mel = batch_feature["audio_features"]
        shapes.append(tuple(mel.shape))

    assert shapes[0] == (1, 80, 102)
    assert shapes[1] == (1, 80, 104)
    assert shapes[2] == (1, 80, 104)

    mel_processor.reset()
    assert mel_processor.chunk_count == 0
    assert mel_processor.is_first
