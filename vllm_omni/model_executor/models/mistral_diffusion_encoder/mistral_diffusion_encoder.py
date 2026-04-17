# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mistral/Pixtral text encoder for diffusion pipelines, running via vLLM model executor.

Wraps vLLM's Mistral3ForConditionalGeneration (Pixtral) to extract
intermediate layer hidden states and return them via
OmniOutput.multimodal_outputs for consumption by downstream diffusion
stages. Uses the EagleModelMixin.aux_hidden_state_layers mechanism
already built into LlamaModel to collect hidden states at specified
layer indices during the forward pass.

Designed to run as Stage 0 in a multi-stage pipeline where:
  Stage 0: MistralDiffusionEncoder (vLLM model executor) -> prompt_embeds
  Stage 1: Flux2Pipeline (diffusion engine) -> image

The config.json in the text_encoder/ subfolder is a
Mistral3ForConditionalGeneration config with nested text_config and
vision_config. The full model (language + vision tower + projector)
is loaded, enabling multimodal prompt upsampling via embed_multimodal.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.mistral3 import (
    Mistral3DummyInputsBuilder,
    Mistral3ForConditionalGeneration,
    Mistral3MultiModalProcessor,
    Mistral3ProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput


# Default layer indices whose hidden states are collected.
# FLUX.2-dev uses layers 10, 20, 30 (1-indexed) -> 3 x 5120 = 15360
# = joint_attention_dim of Flux2Transformer2DModel.
#
# EagleModelMixin uses 0-indexed layer indices where index k means
# "after k-th layer". LlamaModel.forward yields idx+1 after each layer
# (idx is 0-based loop variable, so idx+1 is 1-indexed layer number).
# Therefore aux_hidden_state_layers=(10, 20, 30) collects after layers
# 10, 20, 30 -- matching the HuggingFace convention.
DEFAULT_TARGET_LAYERS: tuple[int, ...] = (10, 20, 30)


@MULTIMODAL_REGISTRY.register_processor(
    Mistral3MultiModalProcessor,
    info=Mistral3ProcessingInfo,
    dummy_inputs=Mistral3DummyInputsBuilder,
)
class MistralDiffusionEncoder(nn.Module, SupportsMultiModal):
    """Pixtral encoder that outputs intermediate hidden states for diffusion.

    Wraps Mistral3ForConditionalGeneration internally and intercepts the
    forward output to package intermediate layer hidden states into
    OmniOutput. The AR model runner detects have_multimodal_outputs=True
    and calls extract_multimodal_outputs, which unpacks OmniOutput into
    (text_hidden_states, multimodal_outputs) and routes multimodal_outputs
    into the per-request pooler_output dict.

    The downstream stage input processor (encoder2diffusion) extracts
    prompt_embeds from pooler_output and passes them to the diffusion
    pipeline.
    """

    have_multimodal_outputs = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        # Mistral3ForConditionalGeneration requires multimodal_config
        # for vision tower and language model initialization.
        if vllm_config.model_config.multimodal_config is None:
            vllm_config.model_config.multimodal_config = MultiModalConfig()

        self.target_layers = DEFAULT_TARGET_LAYERS

        self.model = Mistral3ForConditionalGeneration(
            vllm_config=vllm_config,
            prefix=prefix,
        )

        # Tell LlamaModel to collect hidden states at target layers.
        self.model.language_model.model.aux_hidden_state_layers = self.target_layers

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.language_model.model.embed_tokens(input_ids)

    def embed_multimodal(self, **kwargs: object):
        return self.model.embed_multimodal(**kwargs)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> OmniOutput:
        num_tokens = 0
        if input_ids is not None:
            num_tokens = input_ids.shape[0]
        elif inputs_embeds is not None:
            num_tokens = inputs_embeds.shape[0]
        is_prefill = num_tokens > 1

        model_output = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )

        if isinstance(model_output, IntermediateTensors):
            return OmniOutput(
                text_hidden_states=model_output,
                multimodal_outputs=None,
            )

        if isinstance(model_output, tuple):
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states = model_output
            aux_hidden_states = []

        if is_prefill and aux_hidden_states:
            prompt_embeds = torch.cat(aux_hidden_states, dim=-1)
            multimodal_outputs = {"prompt_embeds": prompt_embeds}
        else:
            multimodal_outputs = None

        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs=multimodal_outputs,
        )

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput):
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        return self.model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = self.model.load_weights(weights)
        return {f"model.{k}" for k in loaded}
