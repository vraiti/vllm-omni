# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processors for LingBot-Video.

Functions:
  format_expand_prompt — User prompt → Stage 0 (EXPAND) input
  expand_to_map        — Stage 0 (EXPAND) → Stage 1 (MAP)
  rewriter_to_dit      — Stage 1 (MAP)   → Stage 2 (DIFFUSION)
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniSamplingParams, OmniTextPrompt, OmniTokensPrompt

logger = init_logger(__name__)

_DURATION_RE = re.compile(r"Video Duration:\s*(\d+)\s*seconds", re.IGNORECASE)


def format_expand_prompt(
    prompt: OmniTextPrompt,
    sampling_params_list: Sequence[OmniSamplingParams],
) -> OmniTextPrompt:
    """Format the user prompt for the EXPAND stage (stage 0).

    Wraps the raw user text with the step-1 EXPAND system prompt and
    Qwen3 chat template (thinking disabled).
    """
    from vllm_omni.diffusion.models.lingbot_video.rewriter_prompts import (
        _step1_text,
    )

    num_frames = 121
    if sampling_params_list:
        num_frames = getattr(sampling_params_list[-1], "num_frames", None) or 121

    fps = 24
    duration = max(1, min(30, num_frames // fps))
    prompt_text = prompt.get("prompt", "")
    step1 = _step1_text("t2v", prompt_text, duration)
    formatted = f"<|im_start|>user\n{step1}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    return OmniTextPrompt(prompt=formatted, modalities=prompt.get("modalities", ["video"]))


def expand_to_map(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | list | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[dict[str, Any]]:
    """Convert EXPAND output to MAP input.

    Stage 0 (EXPAND, no LoRA) produces a detailed prose caption.
    This bridge wraps it with the step-2 MAP system prompt and
    Qwen3 chat template so stage 1 (MAP, with LoRA) can produce
    the structured JSON caption.
    """
    del streaming_context, requires_multimodal_data

    if not source_outputs:
        return []

    expand_output = source_outputs[0]
    detailed_caption = expand_output.outputs[0].text

    if not detailed_caption or not detailed_caption.strip():
        logger.warning("[expand_to_map] EXPAND stage produced empty output")
        return []

    dur = 5
    if isinstance(prompt, dict):
        prompt_text = prompt.get("prompt", "")
        m = _DURATION_RE.search(prompt_text)
        if m:
            dur = int(m.group(1))

    from vllm_omni.diffusion.models.lingbot_video.rewriter_prompts import (
        _step2_text,
    )

    step2 = _step2_text("t2v", detailed_caption, dur)
    formatted = f"<|im_start|>user\n{step2}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    logger.info(
        "[expand_to_map] EXPAND produced %d chars, formatting MAP prompt with duration=%ds",
        len(detailed_caption),
        dur,
    )

    return [{"prompt": formatted}]


def rewriter_to_dit(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | list | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Convert MAP LLM output to diffusion stage input.

    Stage 1 (MAP, Qwen3.5 + rewriter LoRA) generates a structured JSON
    caption.  This bridge passes that caption text as the diffusion prompt.
    """
    del streaming_context, requires_multimodal_data, kwargs

    if not source_outputs:
        return None

    rewriter_output = source_outputs[0]
    generated_text = rewriter_output.outputs[0].text

    if not generated_text or not generated_text.strip():
        logger.warning("[rewriter_to_dit] MAP stage produced empty output")
        return None

    logger.info(
        "[rewriter_to_dit] MAP produced %d chars of caption JSON:\n%s",
        len(generated_text),
        generated_text,
    )

    return {"prompt": generated_text}
