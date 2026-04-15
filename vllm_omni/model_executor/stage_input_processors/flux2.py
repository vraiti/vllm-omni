# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for FLUX.2-dev: Mistral encoder → Diffusion transition.

Transforms Mistral text encoder hidden states (from vLLM model executor Stage 0)
into prompt_embeds for the Flux2Pipeline diffusion stage (Stage 1).

The encoder outputs hidden states from intermediate layers (e.g., 10, 20, 30)
via pooler_output. These are stacked and reshaped into the joint embedding
format expected by Flux2Transformer2DModel (joint_attention_dim=15360 for
3 layers * 5120 hidden_size).
"""

from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt

# Target sequence length for prompt embeddings fed to the DiT.
# The reference diffusers pipeline pads/truncates to 512 tokens.
# The DiT's text_ids assign RoPE positions based on sequence index,
# so real tokens must be left-padded to appear at the END of the
# sequence (matching the reference tokenizer's padding_side="left").
#
# Padding is done here (on output embeddings) rather than on input
# tokens because vLLM uses causal attention without attention masks.
# Padding input tokens would let real tokens attend to pad tokens,
# contaminating the hidden states. Zero-padding the embeddings after
# encoding keeps hidden states clean while giving the DiT correct
# positional information.
ENCODER_MAX_SEQUENCE_LENGTH = 512


def encoder2diffusion(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | list | None = None,
    requires_multimodal_data: bool = False,
) -> list[dict[str, Any]]:
    """Transform Mistral encoder output into Flux2Pipeline diffusion input.

    Extracts hidden states from the encoder stage's pooler_output and
    packages them as prompt_embeds for the diffusion stage. The diffusion
    pipeline receives pre-computed prompt_embeds and skips its own text
    encoding stage.

    Args:
        stage_list: List of stage clients with engine_outputs.
        engine_input_source: List of source stage IDs (expects [0]).
        prompt: Original prompt(s) from the user request.
        requires_multimodal_data: Whether to forward multimodal data.

    Returns:
        List of diffusion input dicts, one per request.
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    encoder_outputs = stage_list[source_stage_id].engine_outputs
    diffusion_inputs = []

    if not isinstance(prompt, list):
        prompt = [prompt] if prompt is not None else [{}]

    for i, encoder_output in enumerate(encoder_outputs):
        output = encoder_output.outputs[0]

        mm_output = output.multimodal_output
        if not isinstance(mm_output, dict) or "prompt_embeds" not in mm_output:
            raise RuntimeError(
                f"[encoder2diffusion] Request {i}: expected "
                "output.multimodal_output['prompt_embeds'] from encoder."
            )
        prompt_embeds = mm_output["prompt_embeds"]

        # Keep as 2D (seq_len, dim) — the pipeline's torch.stack handles
        # batching across requests. Preserve original dtype (bfloat16) to
        # match the transformer weights.
        if prompt_embeds.dim() == 3 and prompt_embeds.shape[0] == 1:
            prompt_embeds = prompt_embeds.squeeze(0)

        # Left-pad prompt_embeds to ENCODER_MAX_SEQUENCE_LENGTH so that
        # real token embeddings sit at the END of the sequence, matching
        # the reference pipeline's left-padded tokenizer output. The DiT
        # assigns RoPE positions based on sequence index via text_ids.
        seq_len = prompt_embeds.shape[0]
        if seq_len < ENCODER_MAX_SEQUENCE_LENGTH:
            pad_len = ENCODER_MAX_SEQUENCE_LENGTH - seq_len
            padding = torch.zeros(
                pad_len,
                prompt_embeds.shape[-1],
                dtype=prompt_embeds.dtype,
                device=prompt_embeds.device,
            )
            prompt_embeds = torch.cat([padding, prompt_embeds], dim=0)
        elif seq_len > ENCODER_MAX_SEQUENCE_LENGTH:
            # Truncate from the left, keeping real tokens at the end
            prompt_embeds = prompt_embeds[-ENCODER_MAX_SEQUENCE_LENGTH:]

        # Get original prompt info
        original_prompt = prompt[i] if i < len(prompt) else {}
        if isinstance(original_prompt, dict):
            pass
        elif hasattr(original_prompt, "_asdict"):
            original_prompt = original_prompt._asdict()
        elif hasattr(original_prompt, "__dict__"):
            original_prompt = vars(original_prompt)
        else:
            original_prompt = {}

        text_prompt = original_prompt.get("prompt", "")

        diffusion_input: dict[str, Any] = {
            "prompt": text_prompt,
            "prompt_embeds": prompt_embeds,
        }

        # Forward image data for image-to-image mode
        if requires_multimodal_data:
            mm_data = original_prompt.get("multi_modal_data")
            if mm_data:
                pil_image = mm_data.get("image")
                if pil_image is None:
                    images = mm_data.get("images")
                    if images:
                        pil_image = images[0] if isinstance(images, list) else images
                diffusion_input["image"] = pil_image

        # Forward sampling params
        for key in ["seed", "num_inference_steps", "guidance_scale",
                     "negative_prompt", "height", "width"]:
            if key in original_prompt:
                diffusion_input[key] = original_prompt[key]

        diffusion_inputs.append(diffusion_input)

    return diffusion_inputs
