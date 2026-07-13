# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for LingBot-Video: Rewriter LLM → Diffusion."""

from typing import Any

from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


def rewriter_to_dit(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | list | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Convert rewriter LLM output to diffusion stage input.

    Stage 0 (Qwen3.5 + rewriter LoRA) generates a structured JSON caption.
    This bridge passes that caption text as the diffusion prompt.
    """
    del streaming_context, requires_multimodal_data, kwargs

    if not source_outputs:
        return None

    rewriter_output = source_outputs[0]
    generated_text = rewriter_output.outputs[0].text

    if not generated_text or not generated_text.strip():
        logger.warning("[rewriter_to_dit] Rewriter produced empty output")
        return None

    logger.info(
        "[rewriter_to_dit] Rewriter produced %d chars of caption JSON",
        len(generated_text),
    )

    return {"prompt": generated_text}
