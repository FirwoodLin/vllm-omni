# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniCPM-o 4.5 pipeline topology (frozen).

Stage 0: Thinker — multimodal understanding + text generation.
Stage 1: Talker  — MiniCPMTTS, emits codec tokens.
Stage 2: Code2Wav — codec tokens to the final audio waveform.

The thinker -> talker bridge uses ``llm2tts``. The talker -> Code2Wav bridge
streams request-routed codec chunks.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni"
MINICPMO45_REFERENCE_AUDIO_KEY = "_minicpmo45_reference_audio"

MINICPMO_4_5_PIPELINE = PipelineConfig(
    model_type="minicpmo_4_5",
    default_deploy_config_name="minicpmo_4_5.yaml",
    model_arch="MiniCPMO45OmniForConditionalGeneration",
    duplex_runtime_extension=(
        "vllm_omni.model_executor.models.minicpmo_4_5.duplex.runtime.MiniCPMO45DuplexRuntimeExtension"
    ),
    duplex_serving_adapter=(
        "vllm_omni.model_executor.models.minicpmo_4_5.duplex.serving_adapter.MiniCPMO45ServingRuntimeAdapter"
    ),
    duplex_control_enabled=True,
    # MiniCPM-o 4.5's HF config.json reports `model_type="minicpmo"` and
    # `architectures=["MiniCPMO"]` — both shared verbatim with older MiniCPM-o
    # 1.0 / 2.6 checkpoints. The only field distinguishing the generations is
    # the top-level ``version`` string, so we register both the shared
    # ``MiniCPMO`` arch (for auto-detection) and the 4.5-specific arch (for
    # repos that opt into the explicit name later), then pin the routing to
    # 4.5 via ``hf_config_predicate``. Without the predicate, loading a 2.6
    # checkpoint would also intersect ``["MiniCPMO"]`` here and get routed
    # into the 4.5 pipeline, which would then fail at load time.
    hf_architectures=("MiniCPMO", "MiniCPMO45OmniForConditionalGeneration"),
    hf_config_predicate=lambda c: str(getattr(c, "version", "")) == "4.5",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="llm",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            hf_config_name="tts_config",
            engine_output_type="latent",
            custom_process_input_func=f"{_PROC}.llm2tts",
            custom_process_next_stage_input_func=f"{_PROC}.tts2code2wav_full_payload",
            async_chunk_process_next_stage_input_func=f"{_PROC}.tts2code2wav_async_chunk",
            sampling_constraints={
                "detokenize": False,
                # MiniCPM-o 4.5 codec EOS is tts_config.num_audio_tokens - 1.
                # Same pattern as Qwen3 talker's stop_token_ids: [2150].
                "stop_token_ids": [6561],
            },
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="MiniCPMO45Code2Wav",
            sync_process_input_func=f"{_PROC}.tts2code2wav_token_only",
            sampling_constraints={"detokenize": True},
            requires_full_payload_input=True,
        ),
    ),
)


# Thinker-only variant: multimodal understanding -> text, no talker/code2wav.
# Modeled on the ming_flash_omni / qwen2_5_omni / audex thinker-only pipelines.
# Stage 0 is a byte-for-byte copy of the full pipeline's stage 0 (same
# model_stage="llm" and default arch resolution); only the duplex handoff
# fields and the latent output type are dropped.
MINICPMO_4_5_THINKER_ONLY_PIPELINE = PipelineConfig(
    model_type="minicpmo_4_5_thinker_only",
    # Pipeline-level arch override: without it vLLM resolves the checkpoint's
    # bare `MiniCPMO` architectures entry to upstream minicpmv.py instead of
    # the 4.5 omni implementation.
    model_arch="MiniCPMO45OmniForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="llm",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
            sampling_constraints={"detokenize": True},
        ),
    ),
)


# Thinker + Talker variant: text (with tts_bos) -> codec tokens, no code2wav.
# Stage 0 is identical to the full pipeline's stage 0 (latent handoff to the
# talker via ``llm2tts``); stage 1 is identical to the full pipeline's stage 1
# except it becomes the final stage so the pipeline terminates after the
# Talker. Used for isolated Talker benchmarks on the production engine path.
MINICPMO_4_5_THINKER_TALKER_PIPELINE = PipelineConfig(
    model_type="minicpmo_4_5_thinker_talker",
    model_arch="MiniCPMO45OmniForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="llm",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=False,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            hf_config_name="tts_config",
            final_output=True,
            final_output_type="latent",
            engine_output_type="latent",
            custom_process_input_func=f"{_PROC}.llm2tts",
            sampling_constraints={
                "detokenize": False,
                # MiniCPM-o 4.5 codec EOS is tts_config.num_audio_tokens - 1.
                "stop_token_ids": [6561],
            },
        ),
    ),
)


# Talker-only variant: bypasses the Thinker entirely and feeds the Talker
# with a fake Thinker hidden-state handoff whose length is controlled by the
# user-supplied token-id list. Used for isolated Talker perf benchmarks on
# the production engine path (e.g. rolling-KV sweep at fixed KV lengths);
# generated codec tokens are not meaningful for ASR quality and are only
# consumed by Talker -> EOS to terminate the request.
MINICPMO_4_5_TALKER_ONLY_PIPELINE = PipelineConfig(
    model_type="minicpmo_4_5_talker_only",
    model_arch="MiniCPMO45OmniForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            final_output=True,
            final_output_type="latent",
            engine_output_type="latent",
            hf_config_name="tts_config",
            # Stage-0 of a single-stage TTS-only pipeline has no upstream
            # source_outputs; the benchmark script stamps the fake Thinker
            # handoff straight onto each ``OmniTokensPrompt.model_intermediate_buffer``
            # and the orchestrator forwards it via ``upgrade_to_omni_request``
            # into ``GPUModelRunner.model_intermediate_buffer[req_id]``.
            sampling_constraints={
                "detokenize": False,
                # Note: do NOT inject ``stop_token_ids`` here. The TTS-only
                # engine has Vocabulary size 0 (no tokenizer is loaded for
                # the codec-only ``tts_config``), and ``SamplingParams.verify``
                # rejects stop_token_ids >= vocab_size at request time.
                # The benchmark drives termination via ``max_tokens`` (the
                # production-engine Talker reaches codec EOS naturally
                # before that cap at every batch size we care about).
            },
        ),
    ),
)
