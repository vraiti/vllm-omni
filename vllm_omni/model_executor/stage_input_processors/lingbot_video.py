# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processors for LingBot-Video.

Functions:
  format_expand_prompt — User prompt → Stage 0 (EXPAND) input
  expand_to_map        — Stage 0 (EXPAND) → Stage 1 (MAP)
  rewriter_to_dit      — Stage 1 (MAP)   → Stage 2 (DIFFUSION)
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Sequence
from typing import Any

from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniSamplingParams, OmniTextPrompt, OmniTokensPrompt

logger = init_logger(__name__)

_DURATION_RE = re.compile(r"Video Duration:\s*(\d+)\s*seconds", re.IGNORECASE)

_DARK_SCENE_HINTS = (
    "dark",
    "dim",
    "low light",
    "low-light",
    "night",
    "nighttime",
    "moody",
    "gloomy",
    "ominous",
    "deep shadow",
    "deep shadows",
    "dark shadows",
    "dramatic and moody",
)

_MOTION_BLUR_HINTS = (
    "motion blur",
    "motion-blur",
    "blurred background",
    "background is blurred",
    "background appears blurred",
    "blurred landscape",
    "blurred scenery",
    "blurred surroundings",
    "blurred by",
    "speed blur",
    "long exposure",
)

_PHYSICAL_BLOCK_HINTS = (
    "fantasy",
    "surreal",
    "dreamlike",
    "dream-like",
    "magic",
    "magical",
    "supernatural",
    "physics-bending",
    "physics bending",
    "impossible physics",
    "anti-gravity",
    "antigravity",
    "zero gravity",
    "zero-gravity",
    "weightless",
    "weightlessness",
    "floating in space",
    "outer space",
    "astronaut",
)

_STYLE_BLOCK_HINTS = (
    "painting",
    "illustration",
    "cartoon",
    "drawing",
    "sketch",
    "cgi",
    "3d render",
    "3d-render",
    "digital art",
    "anime",
    "stylized animation",
    "claymation",
    "stop motion",
    "stop-motion",
)


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
        "[expand_to_map] EXPAND produced %d chars, formatting MAP prompt with duration=%ds:\n%s",
        len(detailed_caption),
        dur,
        detailed_caption,
    )

    return [{"prompt": formatted}]


def _has_any_hint(text: str, hints: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in hints)


def _prune_negative(caption_text: str) -> str:
    from vllm_omni.diffusion.models.lingbot_video.pipeline_lingbot_video import (
        DEFAULT_NEGATIVE_VIDEO,
    )

    neg = copy.deepcopy(DEFAULT_NEGATIVE_VIDEO)
    cats = neg["universal_negative"]

    if _has_any_hint(caption_text, _DARK_SCENE_HINTS):
        cats["visual_quality"] = [
            t
            for t in cats["visual_quality"]
            if t not in {"underexposed", "subject hidden in darkness", "crushed blacks"}
        ]

    if _has_any_hint(caption_text, _MOTION_BLUR_HINTS):
        if "temporal_and_motion_stability" in cats:
            cats["temporal_and_motion_stability"] = [
                t for t in cats["temporal_and_motion_stability"] if t != "motion blur"
            ]

    if _has_any_hint(caption_text, _PHYSICAL_BLOCK_HINTS):
        cats.pop("physical_plausibility", None)

    if _has_any_hint(caption_text, _STYLE_BLOCK_HINTS):
        cats.pop("artistic_style", None)

    return json.dumps(neg, ensure_ascii=False)


def rewriter_to_dit(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | list | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Convert MAP LLM output to diffusion stage input.

    Stage 1 (MAP, Qwen3.5 + rewriter LoRA) generates a structured JSON
    caption.  This bridge repairs the JSON, prunes the default negative
    prompt based on caption content, and passes both to the diffusion stage.
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

    from vllm_omni.diffusion.models.lingbot_video.rewriter_prompts import (
        parse_json,
    )

    parsed = parse_json(generated_text)
    if parsed is not None:
        caption = json.dumps(parsed, ensure_ascii=False)
        logger.info("[rewriter_to_dit] JSON repair succeeded")
    else:
        caption = generated_text
        logger.warning("[rewriter_to_dit] JSON repair failed, using raw text")

    negative_prompt = _prune_negative(caption)
    logger.info("[rewriter_to_dit] Pruned negative prompt: %s", negative_prompt)

    return {"prompt": caption, "negative_prompt": negative_prompt}
