# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from http import HTTPStatus
from typing import cast
from uuid import uuid4

import numpy as np
from fastapi import WebSocket
from scipy.signal import resample_poly
from starlette.websockets import WebSocketDisconnect
from vllm.engine.protocol import StreamingInput
from vllm.inputs import TokensPrompt
from vllm.logger import init_logger
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.utils import coerce_param_message_types
from vllm_omni.experimental.fullduplex.omni.audio_utils import (
    extract_audio_chunks,
    pcm16_b64,
    pcm16_b64_to_f32,
)

logger = init_logger(__name__)

MODEL_INPUT_RATE = 16000
AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"


def _event_id() -> str:
    return f"event_{uuid4().hex[:24]}"


class RealtimeConnection:
    def __init__(self, websocket: WebSocket, serving) -> None:
        self.websocket = websocket
        self.connection_id = f"ws-{uuid4()}"
        self.serving = serving
        self.engine = cast(AsyncOmni, serving.engine_client)
        self.generation_task: asyncio.Task | None = None
        self._is_connected = False
        self._is_model_validated = False
        self._realtime_audio_ref: np.ndarray | None = None
        self._audio_buffer: list[np.ndarray] = []
        self._committed_audio: list[np.ndarray] = []
        self._client_event_id: str | None = None
        self._client_input_rate: int = 24000
        self._conversation_history: list[tuple[np.ndarray, str]] = []

    async def handle_connection(self):
        await self.websocket.accept()
        self._is_connected = True

        await self.send_json(
            {
                "event_id": _event_id(),
                "type": "session.created",
                "session": {
                    "id": self.connection_id,
                    "object": "realtime.session",
                },
            }
        )

        try:
            while True:
                message = await self.websocket.receive_text()
                try:
                    event = json.loads(message)
                    await self.handle_event(event)
                except json.JSONDecodeError:
                    await self._send_error("Invalid JSON", "invalid_json")
                except Exception as e:
                    logger.exception("Error handling event: %s", e)
                    await self._send_error(str(e), "processing_error")
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected: %s", self.connection_id)
        except Exception as e:
            logger.exception("Unexpected error in connection: %s", e)
        finally:
            self._is_connected = False
            self._audio_buffer.clear()
            if self.generation_task and not self.generation_task.done():
                self.generation_task.cancel()

    async def handle_event(self, event: dict):
        event_type = event.get("type")

        if event_type == "session.update":
            session = event.get("session", {})
            model = session.get("model")
            if model is not None:
                if not self.serving._is_model_supported(model):
                    err = self.serving.create_error_response(
                        message=f"The model `{model}` does not exist.",
                        err_type="NotFoundError",
                        status_code=HTTPStatus.NOT_FOUND,
                        param="model",
                    )
                    await self._send_error(err.error.message, "model_not_found")
                    return
            self._is_model_validated = True
            audio_cfg = session.get("audio", {})
            input_fmt = audio_cfg.get("input", {}).get("format", {})
            if isinstance(input_fmt, dict):
                rate = input_fmt.get("rate")
                if rate and isinstance(rate, int):
                    self._client_input_rate = rate

        elif event_type == "input_audio_buffer.append":
            audio_b64 = event.get("audio", "")
            try:
                audio_array = pcm16_b64_to_f32(audio_b64)
                if len(audio_array) == 0:
                    return
                self._audio_buffer.append(audio_array)
            except Exception as e:
                logger.error("Failed to decode audio: %s", e)
                await self._send_error("Invalid audio data", "invalid_audio")

        elif event_type == "input_audio_buffer.commit":
            if not self._audio_buffer:
                self._committed_audio = []
            else:
                audio = np.concatenate(self._audio_buffer)
                if self._client_input_rate != MODEL_INPUT_RATE:
                    audio = resample_poly(
                        audio,
                        up=MODEL_INPUT_RATE,
                        down=self._client_input_rate,
                    ).astype(np.float32)
                self._committed_audio = [audio]
            self._audio_buffer = []

        elif event_type == "response.create":
            client_event_id = event.get("event_id")
            metadata = event.get("response", {})
            if isinstance(metadata, dict):
                meta_inner = metadata.get("metadata", {})
                if isinstance(meta_inner, dict) and not client_event_id:
                    client_event_id = meta_inner.get("client_event_id")
            self._client_event_id = client_event_id

            if not self._committed_audio:
                logger.warning(
                    "response.create with no committed audio (client_event_id=%s), sending empty completed response",
                    client_event_id,
                )
                response_id = f"resp_{uuid4().hex[:24]}"
                item_id = f"item_{uuid4().hex[:24]}"
                await self._send_response_created(response_id, item_id)
                await self._send_response_done(
                    response_id,
                    item_id,
                    "completed",
                    0,
                    0,
                )
                return

            await self.start_generation(self._committed_audio)
            self._committed_audio = []

        elif event_type == "response.cancel":
            if self.generation_task and not self.generation_task.done():
                self.generation_task.cancel()

        elif event_type == "input_audio_buffer.clear":
            self._audio_buffer.clear()

    async def start_generation(self, committed_audio: list[np.ndarray]):
        if self.generation_task is not None and not self.generation_task.done():
            logger.warning("Generation already in progress, ignoring commit")
            self._audio_buffer = committed_audio + self._audio_buffer
            return

        current_audio = np.concatenate(committed_audio)
        input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
        streaming_input_gen = self._build_prompt(current_audio)

        self.generation_task = asyncio.create_task(
            self._run_generation(
                streaming_input_gen,
                input_stream,
                current_audio,
            ),
        )

    async def _build_prompt(
        self,
        current_audio: np.ndarray,
    ) -> AsyncGenerator[StreamingInput, None]:
        model_config = self.serving.model_config
        renderer = self.serving.renderer
        tokenizer = cached_tokenizer_from_config(model_config)

        parts: list[str] = []
        audio_arrays: list[np.ndarray] = []

        for user_audio, assistant_text in self._conversation_history:
            parts.append(f"<|im_start|>user\n{AUDIO_PLACEHOLDER}<|im_end|>\n")
            parts.append(
                f"<|im_start|>assistant\n{assistant_text}<|im_end|>\n",
            )
            audio_arrays.append(user_audio)

        parts.append(f"<|im_start|>user\n{AUDIO_PLACEHOLDER}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        audio_arrays.append(current_audio)

        prompt_text = "".join(parts)
        prompt_token_ids = tokenizer.encode(prompt_text)

        if len(audio_arrays) == 1:
            mm_data = {"audio": audio_arrays[0]}
        else:
            mm_data = {"audio": audio_arrays}

        prompt = TokensPrompt(
            prompt_token_ids=prompt_token_ids,
            multi_modal_data=mm_data,
        )
        parsed = parse_model_prompt(model_config, prompt)
        (engine_input,) = await renderer.render_cmpl_async([parsed])
        yield StreamingInput(prompt=engine_input)

    def _extract_audio_chunks_with_ref(self, output) -> tuple[list[np.ndarray], int]:
        chunks, sr, self._realtime_audio_ref = extract_audio_chunks(output, self._realtime_audio_ref)
        return chunks, sr

    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
        current_audio: np.ndarray | None = None,
    ):
        response_id = f"resp_{uuid4().hex[:24]}"
        item_id = f"item_{uuid4().hex[:24]}"
        full_text = ""
        prompt_token_ids_len = 0
        completion_tokens_len = 0
        self._realtime_audio_ref = None
        response_done_sent = False

        sampling_params_list = list(self.engine.default_sampling_params_list)
        sampling_params_list = coerce_param_message_types(
            sampling_params_list,
            is_streaming=True,
        )

        try:
            await self._send_response_created(response_id, item_id)

            request_id = f"rt-{self.connection_id}-{uuid4()}"
            result_gen = self.engine.generate(
                prompt=streaming_input_gen,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
            )

            stage0_stop_count = 0

            async for output in result_gen:
                stage_id = getattr(output, "stage_id", None)
                if stage_id == 0 and output.outputs:
                    first_output = output.outputs[0]
                    new_token_ids = list(first_output.token_ids)
                    finish_reason = getattr(first_output, "finish_reason", None)

                    if finish_reason:
                        stage0_stop_count += 1

                    if stage0_stop_count >= 2 and new_token_ids and not finish_reason:
                        break

                    if new_token_ids:
                        input_stream.put_nowait(new_token_ids)

                    if output.prompt_token_ids:
                        prompt_token_ids_len = max(
                            prompt_token_ids_len,
                            len(output.prompt_token_ids),
                        )

                    delta_text = first_output.text or ""
                    full_text += delta_text
                    completion_tokens_len += len(new_token_ids)

                    if delta_text:
                        await self.send_json(
                            {
                                "event_id": _event_id(),
                                "type": "response.output_text.delta",
                                "response_id": response_id,
                                "item_id": item_id,
                                "output_index": 0,
                                "content_index": 0,
                                "delta": delta_text,
                            }
                        )

                audio_chunks, _ = self._extract_audio_chunks_with_ref(output)
                for chunk in audio_chunks:
                    await self.send_json(
                        {
                            "event_id": _event_id(),
                            "type": "response.output_audio.delta",
                            "response_id": response_id,
                            "item_id": item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": pcm16_b64(chunk),
                        }
                    )

                if not self._is_connected:
                    break

            if current_audio is not None and full_text:
                self._conversation_history.append(
                    (current_audio, full_text),
                )
            await self._send_response_done(
                response_id,
                item_id,
                "completed",
                prompt_token_ids_len,
                completion_tokens_len,
            )
            response_done_sent = True

        except Exception as e:
            logger.exception("Error in generation: %s", e)
            await self.send_json(
                {
                    "event_id": _event_id(),
                    "type": "error",
                    "error": {"message": str(e), "type": "processing_error"},
                }
            )
        finally:
            self._audio_buffer.clear()

            if self._is_connected and not response_done_sent:
                try:
                    await self._send_response_done(
                        response_id,
                        item_id,
                        "failed",
                        prompt_token_ids_len,
                        completion_tokens_len,
                    )
                except Exception:
                    logger.exception("Failed to send response.done")

    async def _send_response_created(self, response_id: str, item_id: str) -> None:
        metadata: dict | None = None
        if self._client_event_id:
            metadata = {"client_event_id": self._client_event_id}
        await self.send_json(
            {
                "event_id": _event_id(),
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "realtime.response",
                    "status": "in_progress",
                    "output": [],
                    "metadata": metadata,
                },
            }
        )
        await self.send_json(
            {
                "event_id": _event_id(),
                "type": "response.output_item.added",
                "response_id": response_id,
                "output_index": 0,
                "item": {
                    "id": item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            }
        )
        await self.send_json(
            {
                "event_id": _event_id(),
                "type": "response.content_part.added",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "audio"},
            }
        )

    async def _send_response_done(
        self,
        response_id: str,
        item_id: str,
        status: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        await self.send_json(
            {
                "event_id": _event_id(),
                "type": "response.output_audio.done",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
            }
        )
        await self.send_json(
            {
                "event_id": _event_id(),
                "type": "response.output_item.done",
                "response_id": response_id,
                "output_index": 0,
                "item": {
                    "id": item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [],
                },
            }
        )
        await self.send_json(
            {
                "event_id": _event_id(),
                "type": "response.done",
                "response": {
                    "id": response_id,
                    "object": "realtime.response",
                    "status": status,
                    "output": [
                        {
                            "id": item_id,
                            "object": "realtime.item",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [],
                        }
                    ],
                    "usage": {
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                },
            }
        )

    async def _send_error(self, message: str, code: str | None = None) -> None:
        await self.send_json(
            {
                "event_id": _event_id(),
                "type": "error",
                "error": {"message": message, "code": code},
            }
        )

    async def send_json(self, payload: dict):
        await self.websocket.send_text(json.dumps(payload))
