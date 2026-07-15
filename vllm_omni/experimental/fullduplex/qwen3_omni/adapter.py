# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np
from vllm.engine.protocol import StreamingInput
from vllm.inputs import TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
)
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.experimental.fullduplex.core.adapter import (
    DuplexAdapter,
    DuplexCapability,
    OutputChunk,
)
from vllm_omni.experimental.fullduplex.core.session import DuplexSession
from vllm_omni.experimental.fullduplex.qwen3_omni.audio_utils import (
    extract_audio_chunks,
)

logger = init_logger(__name__)


@dataclass
class _UserTurn:
    audio: np.ndarray


@dataclass
class _AssistantTurn:
    token_ids: list[int]
    complete: bool


class Qwen3OmniDuplexAdapter(DuplexAdapter):
    """DuplexAdapter for Qwen3-Omni audio-in / audio+text-out streaming."""

    _MAX_HISTORY_TOKENS = 4096
    _MAX_HISTORY_AUDIO_S = 60.0

    def __init__(self, engine, serving) -> None:
        self._engine = engine
        self._serving = serving
        self._audio_buffer: list[np.ndarray] = []
        self._pending_text: str | None = None
        self._realtime_audio_ref: np.ndarray | None = None
        self._active_request_id: str | None = None
        self._history: list[_UserTurn | _AssistantTurn] = []
        self._pending_thinker_tokens: list[int] = []
        self._pending_audio: np.ndarray | None = None
        self._interrupted: bool = False

    def capabilities(self) -> DuplexCapability:
        return DuplexCapability(
            input_modalities=frozenset({"audio"}),
            output_modalities=frozenset({"audio", "text"}),
            proactive=False,
        )

    async def on_input(self, session: DuplexSession, modality: str, data: Any) -> None:
        if modality == "audio":
            self._audio_buffer.append(data)
        elif modality == "text":
            self._pending_text = data

    def should_respond(self, session: DuplexSession) -> bool:
        return len(self._audio_buffer) > 0

    async def respond(self, session: DuplexSession) -> AsyncIterator[OutputChunk]:
        if not self._audio_buffer:
            return

        audio = np.concatenate(self._audio_buffer)
        self._audio_buffer.clear()
        self._pending_text = None
        self._realtime_audio_ref = None
        self._pending_audio = audio
        self._pending_thinker_tokens = []
        self._interrupted = False

        request_id = f"dx-{uuid4()}"
        self._active_request_id = request_id
        input_stream: asyncio.Queue[list[int]] = asyncio.Queue()

        try:
            streaming_input_gen = self._build_streaming_input(audio, input_stream)
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
                        logger.info("stage0 started new response after %d stops, breaking", stage0_stop_count)
                        break

                    if token_ids:
                        input_stream.put_nowait(token_ids)
                        self._pending_thinker_tokens.extend(token_ids)
                    delta_text = first.text or ""
                    if delta_text:
                        yield OutputChunk("text", delta_text)

                chunks, sr, self._realtime_audio_ref = extract_audio_chunks(output, self._realtime_audio_ref)
                for chunk in chunks:
                    yield OutputChunk(
                        "audio",
                        {"pcm_f32": chunk, "sample_rate": sr},
                    )
        finally:
            if self._pending_audio is not None:
                self._history.append(_UserTurn(audio=self._pending_audio))
                self._pending_audio = None
            if self._pending_thinker_tokens:
                self._history.append(
                    _AssistantTurn(
                        token_ids=list(self._pending_thinker_tokens),
                        complete=not self._interrupted,
                    )
                )
            self._pending_thinker_tokens = []
            self._trim_history()
            self._active_request_id = None
            try:
                await self._engine.abort(request_id)
            except Exception:
                pass

    async def on_barge_in(self, session: DuplexSession) -> None:
        self._audio_buffer.clear()
        self._pending_text = None
        self._realtime_audio_ref = None
        self._interrupted = True

    async def _build_streaming_input(
        self,
        audio: np.ndarray,
        input_stream: asyncio.Queue[list[int]],
    ):
        model_config = self._serving.model_config
        tokenizer = cached_tokenizer_from_config(model_config)
        audio_placeholder = Qwen3OmniMoeThinkerForConditionalGeneration.get_placeholder_str("audio", 0)

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

        if isinstance(mm_audio, np.ndarray):
            logger.info(
                "[duplex-debug] single audio: shape=%s dtype=%s ndim=%d",
                mm_audio.shape,
                mm_audio.dtype,
                mm_audio.ndim,
            )
        else:
            logger.info(
                "[duplex-debug] %d audios: shapes=%s",
                len(mm_audio),
                [a.shape for a in mm_audio],
            )

        audio_pad_id = tokenizer.encode("<|audio_pad|>", add_special_tokens=False)
        pad_positions = [i for i, t in enumerate(prompt_token_ids) if t in audio_pad_id]
        logger.info(
            "[duplex-debug] prompt len=%d, audio_pad_id=%s, pad_positions=%s, first_20_tokens=%s",
            len(prompt_token_ids),
            audio_pad_id,
            pad_positions,
            prompt_token_ids[:20],
        )

        prompt = TokensPrompt(
            prompt_token_ids=prompt_token_ids,
            multi_modal_data={"audio": mm_audio},
        )
        parsed = parse_model_prompt(model_config, prompt)
        try:
            (engine_input,) = await self._serving.renderer.render_cmpl_async([parsed])
        except Exception:
            logger.exception("[duplex-debug] render_cmpl_async failed")
            raise
        yield StreamingInput(prompt=engine_input)

    def _trim_history(self) -> None:
        while len(self._history) >= 2:
            text_total = sum(len(t.token_ids) for t in self._history if isinstance(t, _AssistantTurn))
            audio_total = sum(len(t.audio) / 16000 for t in self._history if isinstance(t, _UserTurn))
            if text_total <= self._MAX_HISTORY_TOKENS and audio_total <= self._MAX_HISTORY_AUDIO_S:
                break
            self._history.pop(0)
            if self._history and isinstance(self._history[0], _AssistantTurn):
                self._history.pop(0)

    def _build_sampling_params(self):
        from vllm_omni.entrypoints.utils import coerce_param_message_types

        params = list(self._engine.default_sampling_params_list)
        return coerce_param_message_types(params, is_streaming=True)
