# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniCPM-o 4.5 ``LiveSessionProcessor`` (native VAD).

Renders the official duplex layout onto one resumable request::

    <|im_start|>system\\n{prompt}[\\n\\nprevious: {text}]<|im_end|>
    <unit>[audio x10]{model output up to a chunk terminator}</unit>
    <unit>[audio x10]...

Each ``audio_buffer_ms`` (1 s) window is one streaming update: the chunk
terminator the scheduler dropped at the end of the previous segment,
``</unit>``, ``<unit>``, and the window's audio through ``multi_modal_data``
(``live_duplex_unit`` drops the chat format's audio start/end markers).
``MiniCPMODuplexLogitsProcessor`` enforces the listen/speak rules.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from vllm_omni.entrypoints.openai.live.processor import NativeVadProcessor, RenderedPrompt
from vllm_omni.entrypoints.openai.live.protocol import LiveProtocolError
from vllm_omni.entrypoints.openai.live.session import (
    AssistantItem,
    LiveSessionState,
    NativeUnitItem,
    SystemItem,
    UserTextItem,
)

_AUDIO_PLACEHOLDER = "(<audio>./</audio>)"
_DEFAULT_SYSTEM_PROMPT = "Streaming Omni Conversation."
_PREVIOUS_MARKER = "\n\nprevious: "
# Pooled audio embeddings per 1 s unit (MiniCPMO45DuplexPolicy).
_AUDIO_TOKENS_PER_UNIT = 10


class MiniCPMO45LiveSessionProcessor(NativeVadProcessor):
    input_sample_rate: ClassVar[int] = 16_000
    output_sample_rate: ClassVar[int] = 24_000
    # Official streaming_generate defaults.
    max_unit_tokens: ClassVar[int] = 20
    temperature: ClassVar[float] = 0.7
    top_p: ClassVar[float] = 0.8
    top_k: ClassVar[int] = 100
    # limit_mm_per_prompt audio; a resubmission keeps at most this many units.
    max_audio_units: ClassVar[int] = 60

    def __init__(self, context) -> None:
        super().__init__(context)
        tok = context.raw_tokenizer

        def token_id(token: str) -> int:
            value = tok.convert_tokens_to_ids(token)
            if value is None or value == getattr(tok, "unk_token_id", None):
                raise ValueError(f"MiniCPM-o tokenizer has no {token}")
            return int(value)

        self.unit_id = token_id("<unit>")
        self.unit_end_id = token_id("</unit>")
        self.listen_id = token_id("<|listen|>")
        self.tts_bos_id = token_id("<|tts_bos|>")
        self.chunk_eos_id = token_id("<|chunk_eos|>")
        self.chunk_tts_eos_id = token_id("<|chunk_tts_eos|>")
        self.turn_eos_id = token_id("<|turn_eos|>")
        self.forbidden_ids = [token_id("<|tts_pad|>"), *list(getattr(tok, "bad_token_ids", []) or [])]
        self._placeholder_ids = list(tok.encode(_AUDIO_PLACEHOLDER, add_special_tokens=False))
        # Official prepare(): a fresh session starts with the turn ended, so
        # the model may listen.
        self.turn_ended = True
        # Text of units dropped by left-trimming, carried as "previous" context.
        self.previous_text = ""

    # ---- configuration -----------------------------------------------------------

    def resolve_voice(self, voice: Any) -> str | None:
        voice_id = super().resolve_voice(voice)
        if voice_id not in (None, "default"):
            raise LiveProtocolError(
                "invalid_value",
                "MiniCPM-o 4.5 serves its default voice only; use 'default'.",
                param="audio.output.voice",
            )
        return voice_id

    def stage0_sampling_params(self, state: LiveSessionState, base: Any) -> Any:
        params = super().stage0_sampling_params(state, base)
        stops = [self.listen_id, self.chunk_eos_id, self.chunk_tts_eos_id]
        params.stop_token_ids = list(dict.fromkeys([*(params.stop_token_ids or []), *stops]))
        params.max_tokens = self.max_unit_tokens
        params.temperature = self.temperature
        params.top_p = self.top_p
        params.top_k = self.top_k
        params.extra_args = {
            **(params.extra_args or {}),
            "minicpmo_live": {
                "listen": self.listen_id,
                "tts_bos": self.tts_bos_id,
                "chunk_eos": self.chunk_eos_id,
                "chunk_tts_eos": self.chunk_tts_eos_id,
                "turn_eos": self.turn_eos_id,
                "forbidden": self.forbidden_ids,
                "max_unit_tokens": self.max_unit_tokens,
                "turn_ended": self.turn_ended,
            },
        }
        return params

    # ---- rendering -------------------------------------------------------------

    def _system_ids(self, state: LiveSessionState) -> list[int]:
        developer: list[str] = []
        history: list[str] = []
        for item in state.history:
            if isinstance(item, SystemItem):
                developer.append(item.text)
            elif isinstance(item, UserTextItem):
                history.append(f"user: {item.text}")
            elif isinstance(item, AssistantItem):
                history.append(f"assistant: {item.text}")
        prompt = "\n\n".join(developer) or _DEFAULT_SYSTEM_PROMPT
        previous = " ".join(part for part in (*history, self.previous_text) if part)
        if previous:
            prompt += _PREVIOUS_MARKER + previous
        text = f"<|im_start|>system\n{prompt}<|im_end|>"
        return list(self.context.raw_tokenizer.encode(text, add_special_tokens=False))

    def _prompt(self, token_ids: list[int], audio: list[np.ndarray]) -> dict[str, Any]:
        clips = [(clip, self.input_sample_rate) for clip in audio]
        return {
            "prompt_token_ids": token_ids,
            "multi_modal_data": {"audio": clips if len(clips) > 1 else clips[0]},
            "mm_processor_kwargs": {"live_duplex_unit": True},
        }

    def _num_tokens(self, token_ids: list[int], units: int) -> int:
        return len(token_ids) + units * (_AUDIO_TOKENS_PER_UNIT - len(self._placeholder_ids))

    def render_native_turn(
        self,
        state: LiveSessionState,
        unit: NativeUnitItem,
        *,
        first: bool,
        last_sampled_token: int | None,
    ) -> RenderedPrompt:
        if first:
            ids = [*self._system_ids(state), self.unit_id, *self._placeholder_ids]
        else:
            # The previous segment's terminator was sampled but never
            # computed; feed it, close the unit, open the next one.
            head = [last_sampled_token] if last_sampled_token is not None else []
            ids = [*head, self.unit_end_id, self.unit_id, *self._placeholder_ids]
        return RenderedPrompt(self._prompt(ids, [unit.audio]), self._num_tokens(ids, 1))

    def render_native_resume(self, state: LiveSessionState) -> RenderedPrompt:
        units = [item for item in state.history if isinstance(item, NativeUnitItem)]
        ids = self._system_ids(state)
        for index, unit in enumerate(units):
            ids += [self.unit_id, *self._placeholder_ids]
            if index < len(units) - 1:
                ids += [*unit.token_ids, self.unit_end_id]
        return RenderedPrompt(self._prompt(ids, [u.audio for u in units]), self._num_tokens(ids, len(units)))

    def left_trim(self, state: LiveSessionState, keep_units: int) -> bool:
        keep_units = min(keep_units, self.max_audio_units)
        units = [item for item in state.history if isinstance(item, NativeUnitItem)]
        dropped = units[: max(0, len(units) - keep_units)]
        if not dropped:
            return False
        text = " ".join(self.spoken_text(u.token_ids, "") for u in dropped).strip()
        if text:
            self.previous_text = f"{self.previous_text} {text}".strip()
        return super().left_trim(state, keep_units)

    # ---- output ------------------------------------------------------------------

    def observe_stage0_output(self, state: LiveSessionState, token_ids: list[int], output: Any) -> None:
        super().observe_stage0_output(state, token_ids, output)
        for token in token_ids:
            if token == self.turn_eos_id:
                self.turn_ended = True
            elif token not in (self.listen_id, self.chunk_eos_id, self.chunk_tts_eos_id):
                self.turn_ended = False

    def spoken_text(self, token_ids: list[int], text: str) -> str:
        return self.decode(token_ids)
