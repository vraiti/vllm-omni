# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-Omni ``LiveSessionProcessor``: the generic ChatML processor with
Qwen3-Omni's audio budget, voices, and hermes tool calls."""

from __future__ import annotations

from typing import Any, ClassVar

from vllm_omni.entrypoints.openai.live.processor import GenericChatMLProcessor
from vllm_omni.entrypoints.openai.live.protocol import LiveProtocolError
from vllm_omni.entrypoints.openai.live.session import LiveSessionState

# talker_config.speaker_id of the released checkpoints.
_DEFAULT_SPEAKERS = ("chelsie", "ethan", "aiden")


class Qwen3OmniLiveSessionProcessor(GenericChatMLProcessor):
    input_sample_rate: ClassVar[int] = 16_000
    output_sample_rate: ClassVar[int] = 24_000
    default_tool_call_parser: ClassVar[str | None] = "hermes"

    def __init__(self, context) -> None:
        super().__init__(context)
        speakers = context.extra.get("speakers") or _DEFAULT_SPEAKERS
        self._speakers = {str(name).lower() for name in speakers}
        self._voice: str | None = None

    def resolve_voice(self, voice: Any) -> str | None:
        voice_id = super().resolve_voice(voice)
        if voice_id is None:
            return None
        if voice_id.lower() not in self._speakers:
            raise LiveProtocolError(
                "invalid_value",
                f"Unknown voice '{voice_id}'. Available voices: {', '.join(sorted(self._speakers))}.",
                param="audio.output.voice",
            )
        self._voice = voice_id.lower()
        return self._voice

    def additional_information(self, state: LiveSessionState) -> dict[str, Any]:
        return {"speaker": [self._voice]} if self._voice else {}

    def audio_token_count(self, num_samples: int, sample_rate: int) -> int:
        try:
            from vllm.model_executor.models.qwen3_omni_moe_thinker import _get_feat_extract_output_lengths
        except ImportError:
            return super().audio_token_count(num_samples, sample_rate)
        num_samples_16k = round(num_samples * 16_000 / sample_rate)
        # +2 for <|audio_start|> / <|audio_end|>.
        return int(_get_feat_extract_output_lengths(num_samples_16k // 160)) + 2
