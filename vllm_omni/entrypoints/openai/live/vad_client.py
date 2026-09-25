# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Client for the external Silero VAD service (``vllm_omni.entrypoints.live_vad_service``).

One WebSocket per session. Binary frames carry 16 kHz PCM16 mono; the service
answers every frame with one JSON ``StreamingVADResult``. A ``reset`` text
frame drops endpointing state while keeping the service's audio clock.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from vllm.logger import init_logger

logger = init_logger(__name__)

VAD_SAMPLE_RATE_HZ = 16_000


class ServiceUnavailableError(RuntimeError):
    """An external Live service failed; the session must end."""


@dataclass(frozen=True)
class VadResult:
    speech_started: bool = False
    speech_stopped: bool = False
    speech_start_ms: int | None = None
    speech_end_ms: int | None = None
    speech_probability: float = 0.0


def http_base(url: str) -> str:
    parts = urlsplit(url)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    return urlunsplit((scheme, parts.netloc, "", "", ""))


def ws_endpoint(url: str) -> str:
    parts = urlsplit(url)
    path = parts.path if parts.path not in ("", "/") else "/v1/vad"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


async def check_health(session: aiohttp.ClientSession, base_url: str, timeout_s: float = 2.0) -> None:
    url = f"{http_base(base_url).rstrip('/')}/health"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_s)) as response:
            if response.status // 100 != 2:
                raise ServiceUnavailableError(f"{url} returned HTTP {response.status}")
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        raise ServiceUnavailableError(f"{url} is unreachable: {exc!r}") from exc


class VadClient:
    def __init__(self, http: aiohttp.ClientSession, url: str) -> None:
        self._http = http
        self._url = ws_endpoint(url)
        self._ws: aiohttp.ClientWebSocketResponse | None = None

    async def connect(self, timeout_s: float = 5.0) -> None:
        try:
            self._ws = await self._http.ws_connect(self._url, timeout=aiohttp.ClientWSTimeout(ws_close=timeout_s))
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise ServiceUnavailableError(f"VAD service {self._url} is unreachable: {exc!r}") from exc

    async def send(self, pcm16_16k: bytes) -> None:
        if self._ws is None or self._ws.closed:
            raise ServiceUnavailableError(f"VAD service connection {self._url} is closed")
        try:
            await self._ws.send_bytes(pcm16_16k)
        except (aiohttp.ClientError, ConnectionError) as exc:
            raise ServiceUnavailableError(f"VAD service send failed: {exc!r}") from exc

    async def reset(self) -> None:
        if self._ws is None or self._ws.closed:
            raise ServiceUnavailableError(f"VAD service connection {self._url} is closed")
        try:
            await self._ws.send_str(json.dumps({"type": "reset"}))
        except (aiohttp.ClientError, ConnectionError) as exc:
            raise ServiceUnavailableError(f"VAD service reset failed: {exc!r}") from exc

    async def results(self) -> AsyncIterator[VadResult]:
        """Results in send order; raises ``ServiceUnavailableError`` on a drop."""
        if self._ws is None:
            raise ServiceUnavailableError("VAD service is not connected")
        async for message in self._ws:
            if message.type == aiohttp.WSMsgType.TEXT:
                payload = json.loads(message.data)
                if payload.get("type") == "error":
                    raise ServiceUnavailableError(f"VAD service error: {payload.get('message')}")
                yield VadResult(
                    speech_started=bool(payload.get("speech_started")),
                    speech_stopped=bool(payload.get("speech_stopped")),
                    speech_start_ms=payload.get("speech_start_ms"),
                    speech_end_ms=payload.get("speech_end_ms"),
                    speech_probability=float(payload.get("speech_probability") or 0.0),
                )
            elif message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break
        raise ServiceUnavailableError(f"VAD service connection {self._url} dropped")

    async def close(self) -> None:
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
        self._ws = None
