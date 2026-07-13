# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LingBot-Video pipeline topology (frozen).
Two-stage:
  Stage 0: LLM_AR  — Qwen3.5 rewriter (+ LoRA), generates structured JSON caption
  Stage 1: DIFFUSION — LingBotVideoPipeline, denoising + VAE decode
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

LINGBOT_VIDEO_PIPELINE = PipelineConfig(
    model_type="lingbot_video",
    model_arch="Qwen3_5ForConditionalGeneration",
    hf_architectures=("LingBotVideoPipeline",),
    diffusers_class_name="LingBotVideoPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="rewriter",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=False,
            owns_tokenizer=True,
            model_arch="Qwen3_5ForConditionalGeneration",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="video",
            model_arch="LingBotVideoPipeline",
            custom_process_input_func=("vllm_omni.model_executor.stage_input_processors.lingbot_video.rewriter_to_dit"),
        ),
    ),
)
