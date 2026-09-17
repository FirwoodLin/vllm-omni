# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "benchmarks" / "minicpmo"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_thinker_benchmark_reuses_trunk_speak_token_budget():
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy

    benchmark = _load_module(
        "benchmark_thinker_saturation_test",
        BENCHMARK_DIR / "benchmark_thinker_saturation.py",
    )
    assert benchmark.NATIVE_SPEAK_TOKENS is MiniCPMO45DuplexPolicy.DEFAULT_MAX_NEW_SPEAK_TOKENS_PER_CHUNK


def test_thinker_engine_benchmark_reuses_trunk_speak_token_budget():
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy

    benchmark = _load_module(
        "benchmark_thinker_engine_test",
        BENCHMARK_DIR / "benchmark_thinker_engine.py",
    )
    assert benchmark.NATIVE_SPEAK_TOKENS is MiniCPMO45DuplexPolicy.DEFAULT_MAX_NEW_SPEAK_TOKENS_PER_CHUNK
    assert benchmark.DEFAULT_DEPLOY.name == "minicpmo_4_5_thinker_only.yaml"


def test_talker_benchmark_reuses_trunk_codec_chunk_budget():
    from vllm_omni.model_executor.models.minicpmo_4_5 import MINICPMO45_DUPLEX_CODEC_TOKENS_PER_CHUNK

    benchmark = _load_module(
        "benchmark_talker_saturation_test",
        BENCHMARK_DIR / "benchmark_talker_saturation.py",
    )
    assert benchmark.NATIVE_DUPLEX_TOKENS is MINICPMO45_DUPLEX_CODEC_TOKENS_PER_CHUNK


def test_code2wav_benchmark_reuses_trunk_silence_token():
    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import _SILENCE_TOKEN

    benchmark = _load_module(
        "benchmark_code2wav_saturation_test",
        BENCHMARK_DIR / "benchmark_code2wav_saturation.py",
    )
    assert benchmark.SILENCE_TOKEN is _SILENCE_TOKEN
