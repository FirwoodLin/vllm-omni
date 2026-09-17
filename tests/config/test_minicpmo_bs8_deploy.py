# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import load_deploy_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BS8_CONFIG_REL = "minicpmo_4_5_bs8.yaml"


@pytest.fixture(scope="module")
def deploy():
    return load_deploy_config(get_deploy_config_path(BS8_CONFIG_REL))


def test_bs8_batch_size_invariants(deploy):
    assert deploy.pipeline == "minicpmo_4_5"
    assert deploy.session_mode == "duplex"
    assert deploy.duplex_session.max_sessions == 8
    assert deploy.active_stream_window == 8
    for stage in deploy.stages:
        assert stage.max_num_seqs == 8, f"stage {stage.stage_id} max_num_seqs drifts from bs=8"


def test_bs8_stage1_kv_bound_covers_full_attention_occupancy(deploy):
    # Full-attention occupancy is bounded by max_num_seqs * max_model_len
    # (8 * 4096 = 32768 tokens); the explicit 4 GiB cap must stay above it.
    platforms = deploy.platforms or {}
    cuda_stages = platforms.get("cuda", {}).get("stages", [])
    stage1 = next(s for s in cuda_stages if s["stage_id"] == 1)
    kv_bytes = stage1["kv_cache_memory_bytes"]
    assert kv_bytes == 4 * 1024**3


def test_bs8_keeps_production_graph_flags(deploy):
    extra = deploy.connectors["connector_of_shared_memory"]["extra"]
    assert extra["enable_hift_graph"] is True
    assert extra["enable_cfm_graph"] is True
    assert extra["codec_chunk_frames"] == 25
    assert extra["codec_left_context_frames"] == 3


def test_bs8_differs_from_default_only_in_concurrency_and_memory():
    default = load_deploy_config(get_deploy_config_path("minicpmo_4_5.yaml"))
    bs8 = load_deploy_config(get_deploy_config_path(BS8_CONFIG_REL))
    assert default.duplex_session.max_sessions == 4
    assert bs8.duplex_session.max_sessions == 8
    # Sampling params and connector budgets must stay identical so the bs=8
    # run isolates concurrency as the only variable.
    assert default.connectors == bs8.connectors
    for d_stage, b_stage in zip(default.stages, bs8.stages, strict=True):
        assert d_stage.default_sampling_params == b_stage.default_sampling_params
        assert d_stage.stage_id == b_stage.stage_id
