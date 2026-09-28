# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniCPM-o 4.5 ``LiveSessionProcessor`` (native VAD).

Renders the official duplex layout (``MiniCPMODuplex.prepare`` and
``streaming_prefill``) onto one resumable request::

    <|im_start|>system\\n{prompt}\\n<|audio_start|>[reference voice audio]
    [\\n\\nprevious: {text}]<|audio_end|><|im_end|>
    <unit>[audio x10]{model output up to a chunk terminator}</unit>
    <unit>[audio x10]...

The reference voice is the checkpoint's ``assets/HT_ref_audio.wav``, the clip
the model card's duplex example prepares with; every official duplex entry
point puts one in the system prompt.

Each ``audio_buffer_ms`` (1 s) window is one streaming update: the chunk
terminator the scheduler dropped at the end of the previous segment,
``</unit>``, ``<unit>``, and the window's audio through ``multi_modal_data``
(``live_duplex_unit`` drops the chat format's audio start/end markers).
``MiniCPMODuplexLogitsProcessor`` enforces the listen/speak rules.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import soundfile

from vllm_omni.entrypoints.openai.live.audio import resample
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
_REFERENCE_AUDIO = "assets/HT_ref_audio.wav"
# Pooled audio embeddings per 1 s unit (MiniCPMO45DuplexPolicy).
_AUDIO_TOKENS_PER_UNIT = 10


@functools.cache
def _reference_audio(model_path: str, sample_rate: int) -> np.ndarray:
    """The checkpoint's reference voice clip, mono at ``sample_rate``."""
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        from vllm_omni.transformers_utils.repo_utils import hf_api

        model_dir = Path(hf_api().snapshot_download(model_path, allow_patterns=[_REFERENCE_AUDIO]))
    samples, rate = soundfile.read(model_dir / _REFERENCE_AUDIO, dtype="float32", always_2d=True)
    return resample(samples.mean(axis=1), rate, sample_rate)


class MiniCPMO45LiveSessionProcessor(NativeVadProcessor):
    input_sample_rate: ClassVar[int] = 16_000
    output_sample_rate: ClassVar[int] = 24_000
    # Official streaming_generate defaults.
    max_unit_tokens: ClassVar[int] = 20
    temperature: ClassVar[float] = 0.7
    top_p: ClassVar[float] = 0.8
    top_k: ClassVar[int] = 100
    # Units a resubmission keeps; plus the reference clip, within limit_mm_per_prompt audio (64).
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
        self.reference_audio = _reference_audio(str(context.extra.get("model_path") or ""), self.input_sample_rate)
        # Audio embeddings are pooled to about 10 per second (unit: 1 s -> 10).
        self._reference_tokens = -(-self.reference_audio.size * _AUDIO_TOKENS_PER_UNIT // self.input_sample_rate)

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

    def sampling_params_list(self, base: Any) -> list[Any]:
        params = super().sampling_params_list(base)
        if len(params) > 1 and hasattr(params[1], "clone"):
            # The Talker's per-unit codec budget comes with each handoff
            # (llm2tts); the deploy YAML's offline min_tokens would pad every
            # ~1 s unit.
            params[1] = params[1].clone()
            params[1].min_tokens = 0
        return params

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
        # The reference audio sits between the markers; live_duplex_unit strips
        # the chat format's own markers from every audio item, so write them.
        encode = self.context.raw_tokenizer.encode
        prefix = f"<|im_start|>system\n{prompt}\n<|audio_start|>"
        suffix = (_PREVIOUS_MARKER + previous if previous else "") + "<|audio_end|><|im_end|>"
        return [
            *encode(prefix, add_special_tokens=False),
            *self._placeholder_ids,
            *encode(suffix, add_special_tokens=False),
        ]

    def _prompt(self, token_ids: list[int], audio: list[np.ndarray], *, first: bool = False) -> dict[str, Any]:
        clips = [(clip, self.input_sample_rate) for clip in audio]
        prompt = {
            "prompt_token_ids": token_ids,
            "multi_modal_data": {"audio": clips if len(clips) > 1 else clips[0]},
            "mm_processor_kwargs": {"live_duplex_unit": True},
        }
        if first:
            # The orchestrator hands the request's first prompt to llm2tts on
            # every stage-0 output: this marks the request as Live and carries
            # the unit grammar's token ids. A top-level key, not a
            # model_intermediate_buffer entry, so no engine receives it (the
            # async-chunk prewarm copies the first prompt's buffer downstream).
            prompt["minicpmo_live"] = {
                "listen": self.listen_id,
                "chunk_eos": self.chunk_eos_id,
                "chunk_tts_eos": self.chunk_tts_eos_id,
                "turn_eos": self.turn_eos_id,
            }
        return prompt

    def _num_tokens(self, token_ids: list[int], units: int, *, reference: bool = False) -> int:
        tokens = len(token_ids) + units * (_AUDIO_TOKENS_PER_UNIT - len(self._placeholder_ids))
        if reference:
            tokens += self._reference_tokens - len(self._placeholder_ids)
        return tokens

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
            audio = [self.reference_audio, unit.audio]
        else:
            # The previous segment's terminator was sampled but never
            # computed; feed it, close the unit, open the next one.
            head = [last_sampled_token] if last_sampled_token is not None else []
            ids = [*head, self.unit_end_id, self.unit_id, *self._placeholder_ids]
            audio = [unit.audio]
        return RenderedPrompt(self._prompt(ids, audio, first=first), self._num_tokens(ids, 1, reference=first))

    def render_native_resume(self, state: LiveSessionState) -> RenderedPrompt:
        units = [item for item in state.history if isinstance(item, NativeUnitItem)]
        ids = self._system_ids(state)
        for index, unit in enumerate(units):
            ids += [self.unit_id, *self._placeholder_ids]
            if index < len(units) - 1:
                ids += [*unit.token_ids, self.unit_end_id]
        audio = [self.reference_audio, *(u.audio for u in units)]
        return RenderedPrompt(self._prompt(ids, audio, first=True), self._num_tokens(ids, len(units), reference=True))

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
