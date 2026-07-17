# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import json
from http import HTTPStatus
from typing import Any
from uuid import uuid4

import numpy as np
from scipy.signal import resample_poly
from vllm.logger import init_logger

from vllm_omni.experimental.fullduplex.core import protocol as ev
from vllm_omni.experimental.fullduplex.core.adapter import AudioChunk
from vllm_omni.experimental.fullduplex.core.runtime import DuplexRuntime
from vllm_omni.experimental.fullduplex.core.session import (
    DuplexSession,
    DuplexSessionConfig,
)
from vllm_omni.experimental.fullduplex.omni.adapter import (
    OmniDuplexAdapter,
)
from vllm_omni.experimental.fullduplex.omni.audio_utils import (
    pcm16_b64,
    pcm16_b64_to_f32,
)

logger = init_logger(__name__)

_SENTINEL = object()
_MODEL_INPUT_RATE = 16000


def _event_id() -> str:
    return f"event_{uuid4().hex[:24]}"


class DuplexRealtimeHandler:
    """Bridges an OpenAI-realtime WebSocket to DuplexRuntime."""

    def __init__(self, websocket, engine, serving) -> None:
        self._ws = websocket
        self._engine = engine
        self._serving = serving
        self._event_queue: asyncio.Queue[dict | object] = asyncio.Queue()
        self._is_connected = False
        self._session_id = str(uuid4())
        self._adapter: OmniDuplexAdapter | None = None
        self._is_model_validated = False
        self._client_input_rate: int = 24000
        self._client_event_id: str | None = None
        self._response_id: str | None = None
        self._item_id: str | None = None
        self._sent_audio = False
        self._full_text = ""
        self._prompt_tokens = 0
        self._completion_tokens = 0

    async def handle_connection(self) -> None:
        from starlette.websockets import WebSocketDisconnect

        await self._ws.accept()
        self._is_connected = True

        await self._send_event(
            {
                "event_id": _event_id(),
                "type": "session.created",
                "session": {
                    "id": self._session_id,
                    "object": "realtime.session",
                },
            }
        )

        adapter = OmniDuplexAdapter(self._engine, self._serving)
        self._adapter = adapter
        config = DuplexSessionConfig(
            input_modalities=("audio",),
            output_modalities=("audio", "text"),
            proactive=False,
        )
        session = DuplexSession(self._session_id, config)
        runtime = DuplexRuntime(session, adapter)

        try:
            await asyncio.gather(
                self._read_loop(),
                runtime.run(self._event_iter(), self._emit),
            )
        except WebSocketDisconnect:
            logger.info("Duplex client disconnected: %s", self._session_id)
        except Exception:
            logger.exception("Duplex session error: %s", self._session_id)
        finally:
            self._is_connected = False

    async def _read_loop(self) -> None:
        from starlette.websockets import WebSocketDisconnect

        try:
            while self._is_connected:
                text = await self._ws.receive_text()
                event = json.loads(text)
                rt_event = await self._translate_client_event(event)
                if rt_event is not None:
                    await self._event_queue.put(rt_event)
        except WebSocketDisconnect:
            pass
        finally:
            await self._event_queue.put(_SENTINEL)

    async def _event_iter(self):
        while True:
            item = await self._event_queue.get()
            if item is _SENTINEL:
                return
            yield item

    async def _translate_client_event(self, event: dict) -> dict | None:
        etype = event.get("type", "")

        if etype == "session.update":
            session_cfg = event.get("session", event)
            model = session_cfg.get("model") or event.get("model")
            if model is not None:
                if not self._serving._is_model_supported(model):
                    err = self._serving.create_error_response(
                        message=f"The model `{model}` does not exist.",
                        err_type="NotFoundError",
                        status_code=HTTPStatus.NOT_FOUND,
                        param="model",
                    )
                    await self._send_error(err.error.message, "model_not_found")
                    return None
            self._is_model_validated = True
            audio_cfg = session_cfg.get("audio", {})
            input_fmt = audio_cfg.get("input", {}).get("format", {})
            if isinstance(input_fmt, dict):
                rate = input_fmt.get("rate")
                if rate and isinstance(rate, int):
                    self._client_input_rate = rate
            return None

        if etype == "input_audio_buffer.append":
            audio_b64 = event.get("audio", "")
            if not audio_b64:
                return None
            pcm_f32 = pcm16_b64_to_f32(audio_b64)
            if self._client_input_rate != _MODEL_INPUT_RATE:
                pcm_f32 = resample_poly(
                    pcm_f32,
                    up=_MODEL_INPUT_RATE,
                    down=self._client_input_rate,
                ).astype(np.float32)
            return {
                "type": ev.INPUT_APPEND,
                "modality": "audio",
                "data": pcm_f32,
            }

        if etype == "input_audio_buffer.commit":
            return {"type": ev.INPUT_COMMIT}

        if etype == "input_audio_buffer.clear":
            if self._adapter is not None:
                self._adapter._audio_buffer.clear()
            return None

        if etype == "response.create":
            client_event_id = event.get("event_id")
            metadata = event.get("response", {})
            if isinstance(metadata, dict):
                meta_inner = metadata.get("metadata", {})
                if isinstance(meta_inner, dict) and not client_event_id:
                    client_event_id = meta_inner.get("client_event_id")
            self._client_event_id = client_event_id

            if self._adapter is not None and not self._adapter._audio_buffer:
                await self._send_empty_completed_response()
                return None

            return {"type": ev.RESPONSE_CREATE}

        if etype == "response.cancel":
            return {"type": ev.RESPONSE_CANCEL}

        if etype == "conversation.item.truncate":
            cursor = event.get("audio_end_ms", 0)
            return {
                "type": ev.PLAYBACK_ACK,
                "cursor": cursor,
            }

        return None

    async def _emit(self, event: dict) -> None:
        if not self._is_connected:
            return

        etype = event.get("type")

        if etype == ev.RESPONSE_CREATED:
            self._sent_audio = False
            self._full_text = ""
            self._prompt_tokens = 0
            self._completion_tokens = 0
            response_id = f"resp_{uuid4().hex[:24]}"
            item_id = f"item_{uuid4().hex[:24]}"
            self._response_id = response_id
            self._item_id = item_id
            await self._send_response_created(response_id, item_id)

        elif etype == ev.RESPONSE_DELTA:
            modality = event.get("modality")
            data = event.get("data")
            response_id = self._response_id or ""
            item_id = self._item_id or ""

            if modality == "audio" and isinstance(data, AudioChunk):
                pcm_f32 = data.pcm_f32
                if pcm_f32 is not None:
                    self._sent_audio = True
                    await self._send_event(
                        {
                            "event_id": _event_id(),
                            "type": "response.output_audio.delta",
                            "response_id": response_id,
                            "item_id": item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": pcm16_b64(pcm_f32),
                        }
                    )

            elif modality == "text" and data:
                self._full_text += data
                await self._send_event(
                    {
                        "event_id": _event_id(),
                        "type": "response.output_text.delta",
                        "response_id": response_id,
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": data,
                    }
                )

        elif etype == ev.RESPONSE_DONE:
            response_id = self._response_id or ""
            item_id = self._item_id or ""
            await self._send_response_done(
                response_id,
                item_id,
                "completed",
                self._prompt_tokens,
                self._completion_tokens,
            )

        elif etype == ev.RESPONSE_CANCELLED:
            response_id = self._response_id or ""
            item_id = self._item_id or ""
            await self._send_response_done(
                response_id,
                item_id,
                "cancelled",
                self._prompt_tokens,
                self._completion_tokens,
            )

        elif etype == ev.ERROR:
            await self._send_error(
                event.get("message", "unknown error"),
                "processing_error",
            )

    async def _send_empty_completed_response(self) -> None:
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

    async def _send_response_created(self, response_id: str, item_id: str) -> None:
        metadata: dict | None = None
        if self._client_event_id:
            metadata = {"client_event_id": self._client_event_id}
        await self._send_event(
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
        await self._send_event(
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
        await self._send_event(
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
        await self._send_event(
            {
                "event_id": _event_id(),
                "type": "response.output_audio.done",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
            }
        )
        await self._send_event(
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
        await self._send_event(
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
        await self._send_event(
            {
                "event_id": _event_id(),
                "type": "error",
                "error": {"message": message, "code": code},
            }
        )

    async def _send_event(self, payload: dict[str, Any]) -> None:
        if self._is_connected:
            try:
                await self._ws.send_text(json.dumps(payload))
            except Exception:
                self._is_connected = False
