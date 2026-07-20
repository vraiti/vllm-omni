# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import cached_property
from typing import Any
from uuid import uuid4

import numpy as np
from vllm.engine.protocol import StreamingInput
from vllm.inputs import TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_cls
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.experimental.fullduplex.core.adapter import (
    AudioChunk,
    DuplexAdapter,
    DuplexCapability,
    OutputChunk,
)
from vllm_omni.experimental.fullduplex.core.session import DuplexSession
from vllm_omni.experimental.fullduplex.omni.audio_utils import (
    extract_audio_chunks,
)

logger = init_logger(__name__)

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

    _MAX_HISTORY_TOKENS = 4096
    _MAX_HISTORY_AUDIO_S = 60.0

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
    def _model_cls(self):
        return get_model_cls(self._serving.model_config)

    @cached_property
    def _tokenizer(self):
        return cached_tokenizer_from_config(self._serving.model_config)

    @cached_property
    def _audio_placeholder(self):
        return self._model_cls.get_placeholder_str("audio", 0)

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
        logger.warning(
            "[duplex-debug] respond() audio shape=%s samples=%d duration=%.2fs",
            audio.shape,
            len(audio),
            len(audio) / _INPUT_SAMPLE_RATE,
        )

        try:
            streaming_input_gen = self._build_streaming_input(audio)
            sampling_params_list = self._build_sampling_params()
            logger.warning(
                "[duplex-debug] sampling_params_list len=%d types=%s",
                len(sampling_params_list),
                [type(p).__name__ for p in sampling_params_list],
            )

            result_gen = self._engine.generate(
                prompt=streaming_input_gen,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
            )
            logger.warning("[duplex-debug] engine.generate() returned, iterating results...")

            stage0_stop_count = 0
            _iter_count = 0

            async for output in result_gen:
                _iter_count += 1
                stage_id = getattr(output, "stage_id", None)
                if _iter_count <= 5 or _iter_count % 50 == 0:
                    logger.warning(
                        "[duplex-debug] iter=%d stage_id=%s outputs=%d finished=%s",
                        _iter_count,
                        stage_id,
                        len(output.outputs) if hasattr(output, "outputs") else -1,
                        getattr(output, "finished", None),
                    )

                if stage_id == 0 and output.outputs:
                    first = output.outputs[0]
                    token_ids = list(first.token_ids)
                    finish_reason = getattr(first, "finish_reason", None)

                    if finish_reason:
                        stage0_stop_count += 1

                    if stage0_stop_count >= 2 and token_ids and not finish_reason:
                        logger.debug("stage0 started new response after %d stops, breaking", stage0_stop_count)
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
            logger.warning(
                "[duplex-debug] respond() finished, total iterations=%d thinker_tokens=%d",
                _iter_count,
                len(self._pending_thinker_tokens),
            )
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
            self._trim_history()
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

        prompt_token_ids: list[int] = []
        audio_list: list[np.ndarray] = []

        for turn in self._history:
            if isinstance(turn, _UserTurn):
                prompt_token_ids.extend(tokenizer.encode(f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"))
                audio_list.append(turn.audio)
            elif isinstance(turn, _AssistantTurn):
                prompt_token_ids.extend(tokenizer.encode("<|im_start|>assistant\n"))
                prompt_token_ids.extend(turn.token_ids)
                im_end_ids = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
                if turn.token_ids and turn.token_ids[-1] == im_end_ids[0]:
                    prompt_token_ids.extend(im_end_ids[1:])
                else:
                    prompt_token_ids.extend(im_end_ids)

        prompt_token_ids.extend(
            tokenizer.encode(f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n<|im_start|>assistant\n")
        )
        audio_list.append(audio)

        mm_audio: np.ndarray | list[np.ndarray] = audio_list[0] if len(audio_list) == 1 else audio_list

        self._prompt_token_count = len(prompt_token_ids)
        logger.warning(
            "[duplex-debug] _build_streaming_input: prompt_tokens=%d audio_count=%d audio_shapes=%s",
            len(prompt_token_ids),
            len(audio_list),
            [a.shape for a in audio_list],
        )
        prompt = TokensPrompt(
            prompt_token_ids=prompt_token_ids,
            multi_modal_data={"audio": mm_audio},
        )
        parsed = parse_model_prompt(self._serving.model_config, prompt)
        (engine_input,) = await self._serving.renderer.render_cmpl_async([parsed])
        logger.warning(
            "[duplex-debug] _build_streaming_input: engine_input type=%s keys=%s",
            type(engine_input).__name__,
            list(engine_input.keys()) if isinstance(engine_input, dict) else dir(engine_input),
        )
        yield StreamingInput(prompt=engine_input)

    def _trim_history(self) -> None:
        while len(self._history) >= 2:
            text_total = sum(len(t.token_ids) for t in self._history if isinstance(t, _AssistantTurn))
            audio_total = sum(len(t.audio) / _INPUT_SAMPLE_RATE for t in self._history if isinstance(t, _UserTurn))
            if text_total <= self._MAX_HISTORY_TOKENS and audio_total <= self._MAX_HISTORY_AUDIO_S:
                break
            self._history.pop(0)
            if self._history and isinstance(self._history[0], _AssistantTurn):
                self._history.pop(0)

    def _build_sampling_params(self):
        from vllm_omni.entrypoints.utils import coerce_param_message_types

        params = list(self._engine.default_sampling_params_list)
        return coerce_param_message_types(params, is_streaming=True)
