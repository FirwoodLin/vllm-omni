# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

THINKER_ONLY_CONFIG_REL = "minicpmo_4_5_thinker_only.yaml"


@pytest.fixture(scope="module")
def deploy():
    return load_deploy_config(get_deploy_config_path(THINKER_ONLY_CONFIG_REL))


def test_thinker_only_pipeline_is_registered_single_stage_text():
    pipeline = resolve_pipeline_config("minicpmo_4_5_thinker_only", None)
    assert pipeline is not None
    assert pipeline.model_type == "minicpmo_4_5_thinker_only"
    # Pipeline-level arch override must match the full pipeline: without it
    # vLLM falls back to upstream minicpmv.py for the bare MiniCPMO arch.
    full = resolve_pipeline_config("minicpmo_4_5", None)
    assert pipeline.model_arch == full.model_arch == "MiniCPMO45OmniForConditionalGeneration"
    assert len(pipeline.stages) == 1
    stage = pipeline.stages[0]
    # Stage 0 must mirror the full pipeline's thinker: same model_stage and
    # default arch resolution so the thinker-only load path is byte-identical.
    assert stage.model_stage == full.stages[0].model_stage == "llm"
    assert stage.model_arch == full.stages[0].model_arch is None
    assert stage.final_output is True
    assert stage.final_output_type == "text"
    # No duplex runtime hooks: thinker-only runs the plain turn scheduler.
    assert pipeline.duplex_runtime_extension is None
    assert pipeline.duplex_serving_adapter is None


def test_thinker_only_deploy_single_stage_production_graph_flags(deploy):
    assert deploy.pipeline == "minicpmo_4_5_thinker_only"
    assert deploy.async_chunk is False
    assert deploy.session_mode == "turn"
    assert len(deploy.stages) == 1
    stage = deploy.stages[0]
    assert stage.stage_id == 0
    assert stage.enforce_eager is False
    assert stage.max_num_seqs == 16
    assert stage.gpu_memory_utilization == 0.85
    assert stage.max_num_batched_tokens == 32768


def test_thinker_only_resolves_to_one_stage_with_turn_mode(deploy):
    pipeline = resolve_pipeline_config(deploy.pipeline, None)
    stages = merge_pipeline_deploy(pipeline, deploy, {})
    assert len(stages) == 1
    stage = stages[0]
    assert stage.model_stage == "llm"
    assert stage.final_output_type == "text"
    assert stage.yaml_engine_args["enforce_eager"] is False
    assert stage.yaml_engine_args["session_mode"] == "turn"
    assert stage.yaml_engine_args["max_num_seqs"] == 16
