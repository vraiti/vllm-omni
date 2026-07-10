# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from uuid import uuid4

from vllm.logger import init_logger

from vllm_omni.experimental.fullduplex.core import protocol as ev
from vllm_omni.experimental.fullduplex.core.runtime import DuplexRuntime
from vllm_omni.experimental.fullduplex.core.session import (
    DuplexSession,
    DuplexSessionConfig,
)
from vllm_omni.experimental.fullduplex.qwen3_omni.adapter import (
    Qwen3OmniDuplexAdapter,
)
from vllm_omni.experimental.fullduplex.qwen3_omni.audio_utils import (
    pcm16_b64,
    pcm16_b64_to_f32,
)

logger = init_logger(__name__)

_SENTINEL = object()


class DuplexRealtimeHandler:
    """Bridges an OpenAI-realtime WebSocket to DuplexRuntime."""

    def __init__(self, websocket, engine, serving) -> None:
        self._ws = websocket
        self._engine = engine
        self._serving = serving
        self._event_queue: asyncio.Queue[dict | object] = asyncio.Queue()
        self._is_connected = False
        self._session_id = str(uuid4())
        self._adapter: Qwen3OmniDuplexAdapter | None = None
        self._sent_audio = False
        self._full_text = ""

    async def handle_connection(self) -> None:
        from starlette.websockets import WebSocketDisconnect

        await self._ws.accept()
        self._is_connected = True

        await self._send_event(
            {
                "type": "session.created",
                "id": self._session_id,
                "created": int(time.time()),
            }
        )

        adapter = Qwen3OmniDuplexAdapter(self._engine, self._serving)
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
                rt_event = self._translate_client_event(event)
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

    def _translate_client_event(self, event: dict) -> dict | None:
        etype = event.get("type", "")

        if etype == "session.update":
            return None

        if etype == "input_audio_buffer.append":
            audio_b64 = event.get("audio", "")
            if not audio_b64:
                return None
            pcm_f32 = pcm16_b64_to_f32(audio_b64)
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
            await self._send_event(
                {
                    "type": "response.created",
                    "response": {
                        "id": f"resp_{event.get('response_index', 0)}",
                        "status": "in_progress",
                    },
                }
            )

        elif etype == ev.RESPONSE_DELTA:
            modality = event.get("modality")
            data = event.get("data")

            if modality == "audio" and isinstance(data, dict):
                pcm_f32 = data.get("pcm_f32")
                sr = data.get("sample_rate", 24000)
                if pcm_f32 is not None:
                    self._sent_audio = True
                    await self._send_event(
                        {
                            "type": "response.audio.delta",
                            "audio": pcm16_b64(pcm_f32),
                            "format": "pcm16",
                            "sample_rate_hz": sr,
                        }
                    )

            elif modality == "text" and data:
                self._full_text += data
                await self._send_event({"type": "transcription.delta", "delta": data})

        elif etype == ev.RESPONSE_DONE:
            if self._full_text:
                await self._send_event({"type": "transcription.done", "text": self._full_text})
            await self._send_event(
                {
                    "type": "response.audio.done",
                    "has_audio": self._sent_audio,
                }
            )

        elif etype == ev.RESPONSE_CANCELLED:
            await self._send_event(
                {
                    "type": "response.done",
                    "response": {
                        "id": f"resp_{event.get('response_index', 0)}",
                        "status": "cancelled",
                    },
                }
            )

        elif etype == ev.ERROR:
            await self._send_event(
                {
                    "type": "error",
                    "error": {
                        "message": event.get("message", "unknown error"),
                        "type": "processing_error",
                    },
                }
            )

    async def _send_event(self, payload: dict[str, Any]) -> None:
        if self._is_connected:
            try:
                await self._ws.send_text(json.dumps(payload))
            except Exception:
                self._is_connected = False
