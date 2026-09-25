# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Client for the Whisper ASR service (a plain ``vllm serve openai/whisper-*``)."""

from __future__ import annotations

import asyncio

import aiohttp
import numpy as np

from vllm_omni.entrypoints.openai.live.audio import to_wav_bytes
from vllm_omni.entrypoints.openai.live.timestamps import AsrSegment
from vllm_omni.entrypoints.openai.live.vad_client import ServiceUnavailableError, check_health


class AsrClient:
    def __init__(self, http: aiohttp.ClientSession, base_url: str, timeout_s: float = 10.0) -> None:
        self._http = http
        self.base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)

    async def check_health(self, timeout_s: float = 2.0) -> None:
        await check_health(self._http, self.base_url, timeout_s)

    async def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[AsrSegment]:
        """Transcribe one clip; returns segments with clip-relative seconds."""
        if samples.size == 0:
            return []
        form = aiohttp.FormData()
        form.add_field("file", to_wav_bytes(samples, sample_rate), filename="audio.wav", content_type="audio/wav")
        form.add_field("response_format", "verbose_json")
        form.add_field("timestamp_granularities[]", "segment")
        url = f"{self.base_url}/v1/audio/transcriptions"
        try:
            async with self._http.post(url, data=form, timeout=self._timeout) as response:
                if response.status // 100 != 2:
                    body = await response.text()
                    raise ServiceUnavailableError(f"{url} returned HTTP {response.status}: {body[:200]}")
                payload = await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise ServiceUnavailableError(f"{url} failed: {exc!r}") from exc
        segments = payload.get("segments") or []
        if not segments:
            text = (payload.get("text") or "").strip()
            duration = samples.size / sample_rate
            return [AsrSegment(0.0, duration, text)] if text else []
        return [
            AsrSegment(float(seg.get("start", 0.0)), float(seg.get("end", 0.0)), str(seg.get("text", "")))
            for seg in segments
        ]
