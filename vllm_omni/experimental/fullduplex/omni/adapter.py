# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import cached_property
from typing import Any
from uuid import uuid4

import numpy as np
from vllm.engine.protocol import StreamingInput
from vllm.model_executor.model_loader import get_model_cls
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.experimental.fullduplex.core.adapter import (
    AudioChunk,
    ContextLengthError,
    DuplexAdapter,
    DuplexCapability,
    OutputChunk,
)
from vllm_omni.experimental.fullduplex.core.session import DuplexSession
from vllm_omni.experimental.fullduplex.omni.audio_utils import (
    extract_audio_chunks,
)

_INPUT_SAMPLE_RATE = 16000


@dataclass
class _UserTurn:
    audio: np.ndarray


@dataclass
class _AssistantTurn:
    token_ids: list[int]
    complete: bool


class OmniDuplexAdapter(DuplexAdapter):
    """Model-agnostic DuplexAdapter for audio-in / audio+text-out streaming."""

    def __init__(self, engine, serving) -> None:
        self._engine = engine
        self._serving = serving
        self._audio_buffer: list[np.ndarray] = []
        self._realtime_audio_ref: np.ndarray | None = None
        self._active_request_id: str | None = None
        self._history: list[_UserTurn | _AssistantTurn] = []
        self._pending_thinker_tokens: list[int] = []
        self._pending_audio: np.ndarray | None = None
        self._interrupted: bool = False
        self._prompt_token_count: int = 0

    @cached_property
    def _tokenizer(self):
        return cached_tokenizer_from_config(self._serving.model_config)

    @cached_property
    def _audio_placeholder(self) -> str:
        model_cls = get_model_cls(self._serving.model_config)
        placeholder = model_cls.get_placeholder_str("audio", 0)
        if placeholder is None:
            raise RuntimeError(f"Model {model_cls.__name__} does not define an audio placeholder")
        return placeholder

    def capabilities(self) -> DuplexCapability:
        return DuplexCapability(
            input_modalities=frozenset({"audio"}),
            output_modalities=frozenset({"audio", "text"}),
            proactive=False,
        )

    async def on_input(self, session: DuplexSession, modality: str, data: Any) -> None:
        if modality == "audio":
            self._audio_buffer.append(data)

    def should_respond(self, session: DuplexSession) -> bool:
        return len(self._audio_buffer) > 0

    async def respond(self, session: DuplexSession) -> AsyncIterator[OutputChunk]:
        if not self._audio_buffer:
            return

        audio = np.concatenate(self._audio_buffer)
        self._audio_buffer.clear()
        self._realtime_audio_ref = None
        self._pending_audio = audio
        self._pending_thinker_tokens = []
        self._interrupted = False

        request_id = f"dx-{uuid4()}"
        self._active_request_id = request_id

        try:
            streaming_input_gen = self._build_streaming_input(audio)
            sampling_params_list = self._build_sampling_params()

            result_gen = self._engine.generate(
                prompt=streaming_input_gen,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
            )

            stage0_stop_count = 0

            async for output in result_gen:
                stage_id = getattr(output, "stage_id", None)

                if stage_id == 0 and output.outputs:
                    first = output.outputs[0]
                    token_ids = list(first.token_ids)
                    finish_reason = getattr(first, "finish_reason", None)

                    if finish_reason:
                        stage0_stop_count += 1

                    if stage0_stop_count >= 2 and token_ids and not finish_reason:
                        break

                    if token_ids:
                        self._pending_thinker_tokens.extend(token_ids)
                    delta_text = first.text or ""
                    if delta_text:
                        yield OutputChunk("text", delta_text)

                chunks, sr, self._realtime_audio_ref = extract_audio_chunks(output, self._realtime_audio_ref)
                for chunk in chunks:
                    yield OutputChunk(
                        "audio",
                        AudioChunk(pcm_f32=chunk, sample_rate=sr),
                    )
        finally:
            if self._pending_audio is not None:
                self._history.append(_UserTurn(audio=self._pending_audio))
                self._pending_audio = None
            self._history.append(
                _AssistantTurn(
                    token_ids=list(self._pending_thinker_tokens),
                    complete=not self._interrupted and bool(self._pending_thinker_tokens),
                )
            )
            self._pending_thinker_tokens = []
            self._active_request_id = None
            try:
                await self._engine.abort(request_id)
            except Exception:
                pass

    def get_usage(self, session: DuplexSession) -> tuple[int, int]:
        return self._prompt_token_count, len(self._pending_thinker_tokens)

    async def on_barge_in(self, session: DuplexSession) -> None:
        self._audio_buffer.clear()
        self._realtime_audio_ref = None
        self._interrupted = True

    async def _build_streaming_input(
        self,
        audio: np.ndarray,
    ):
        tokenizer = self._tokenizer
        audio_placeholder = self._audio_placeholder

        parts: list[str] = []
        audio_list: list[np.ndarray] = []

        for turn in self._history:
            if isinstance(turn, _UserTurn):
                parts.append(f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n")
                audio_list.append(turn.audio)
            elif isinstance(turn, _AssistantTurn):
                decoded = tokenizer.decode(turn.token_ids, skip_special_tokens=False)
                parts.append(f"<|im_start|>assistant\n{decoded}")
                if not decoded.endswith("<|im_end|>\n"):
                    if decoded.endswith("<|im_end|>"):
                        parts.append("\n")
                    else:
                        parts.append("<|im_end|>\n")

        parts.append(f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n<|im_start|>assistant\n")
        audio_list.append(audio)

        mm_audio: np.ndarray | list[np.ndarray] = audio_list[0] if len(audio_list) == 1 else audio_list

        prompt_text = "".join(parts)
        prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        engine_input = await self._serving.renderer._process_multimodal_async(
            prompt_token_ids,
            {"audio": mm_audio},
            mm_uuids=None,
            mm_processor_kwargs=None,
            tokenization_kwargs=None,
        )
        engine_input["arrival_time"] = time.time()
        self._prompt_token_count = len(engine_input.get("prompt_token_ids", []))
        max_len = self._serving.model_config.max_model_len
        if self._prompt_token_count >= max_len:
            raise ContextLengthError(f"prompt ({self._prompt_token_count} tokens) exceeds max_model_len ({max_len})")
        yield StreamingInput(prompt=engine_input)

    def _build_sampling_params(self):
        from vllm_omni.entrypoints.utils import coerce_param_message_types

        params = list(self._engine.default_sampling_params_list)
        return coerce_param_message_types(params, is_streaming=True)
