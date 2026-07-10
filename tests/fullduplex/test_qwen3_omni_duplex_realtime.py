# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E online tests for Qwen3-Omni duplex mode via /v1/realtime?mode=duplex.

Requires 2x H100 or equivalent GPU.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import wave

import pytest
import websockets

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SYNTH_PHRASE = "Hello, how are you doing today?"

default_stage_config = get_deploy_config_path("ci/qwen3_omni_moe.yaml")

duplex_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=default_stage_config,
            use_stage_cli=True,
        ),
        id="duplex",
    ),
]


def _pcm16_mono_16k_from_wav_bytes(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.readframes(wf.getnframes())


def _synthetic_pcm16_input() -> bytes:
    syn = generate_synthetic_audio(5, 1, sample_rate=16000, phrase_text=SYNTH_PHRASE)
    wav_bytes = base64.b64decode(syn["base64"])
    return _pcm16_mono_16k_from_wav_bytes(wav_bytes)


async def _run_duplex_roundtrip(
    host: str,
    port: int,
    model: str,
    pcm16: bytes,
    *,
    chunk_ms: int = 100,
) -> dict:
    uri = f"ws://{host}:{port}/v1/realtime?mode=duplex"
    incremental: list[bytes] = []
    output_sr = 24000
    text_chunks: list[str] = []
    delta_events = 0

    bytes_per_ms = 16000 * 2 // 1000
    chunk_bytes = max(bytes_per_ms * chunk_ms, 2)

    async with websockets.connect(uri, max_size=64 * 1024 * 1024) as ws:
        created = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        assert created["type"] == "session.created"

        for i in range(0, len(pcm16), chunk_bytes):
            chunk = pcm16[i : i + chunk_bytes]
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("utf-8"),
                    }
                )
            )

        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        while True:
            message = await asyncio.wait_for(ws.recv(), timeout=600)
            if isinstance(message, bytes):
                continue

            event = json.loads(message)
            etype = event.get("type")

            if etype == "response.created":
                continue

            if etype == "response.audio.delta":
                delta_events += 1
                sr = event.get("sample_rate_hz")
                if isinstance(sr, int) and sr > 0:
                    output_sr = sr
                audio_b64 = event.get("audio", "")
                if audio_b64:
                    incremental.append(base64.b64decode(audio_b64))
                continue

            if etype == "transcription.delta":
                d = event.get("delta", "")
                if d:
                    text_chunks.append(d)
                continue

            if etype == "transcription.done":
                continue

            if etype == "response.audio.done":
                break

            if etype == "response.done":
                break

            if etype == "error":
                raise AssertionError(f"Duplex WebSocket error: {event}")

    return {
        "output_pcm": b"".join(incremental),
        "output_sample_rate": output_sr,
        "transcription_text": "".join(text_chunks),
        "delta_events": delta_events,
    }


@hardware_test("h100", num_gpus=2)
@pytest.mark.parametrize("server_params", duplex_server_params, indirect=True)
@pytest.mark.asyncio
async def test_duplex_audio_roundtrip(server_params):
    host, port = server_params
    pcm16 = _synthetic_pcm16_input()
    result = await _run_duplex_roundtrip(host, port, MODEL, pcm16)

    assert result["delta_events"] >= 1, "Expected at least one audio delta"
    assert result["output_pcm"], "No output PCM received"
    assert len(result["output_pcm"]) >= 4096, "Output audio unexpectedly small"


@hardware_test("h100", num_gpus=2)
@pytest.mark.parametrize("server_params", duplex_server_params, indirect=True)
@pytest.mark.asyncio
async def test_duplex_barge_in(server_params):
    host, port = server_params
    pcm16 = _synthetic_pcm16_input()

    uri = f"ws://{host}:{port}/v1/realtime?mode=duplex"
    bytes_per_ms = 16000 * 2 // 1000
    chunk_bytes = max(bytes_per_ms * 100, 2)

    async with websockets.connect(uri, max_size=64 * 1024 * 1024) as ws:
        await asyncio.wait_for(ws.recv(), timeout=10)

        for i in range(0, len(pcm16), chunk_bytes):
            chunk = pcm16[i : i + chunk_bytes]
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("utf-8"),
                    }
                )
            )
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        got_created = False
        async for message in ws:
            if isinstance(message, bytes):
                continue
            event = json.loads(message)
            if event.get("type") == "response.created":
                got_created = True
                break
            if event.get("type") == "response.audio.delta":
                got_created = True
                break

        assert got_created, "Never received response.created"

        await ws.send(json.dumps({"type": "response.cancel"}))

        cancelled = False
        async for message in ws:
            if isinstance(message, bytes):
                continue
            event = json.loads(message)
            etype = event.get("type")
            if etype == "response.done" and event.get("response", {}).get("status") == "cancelled":
                cancelled = True
                break
            if etype in ("response.audio.done", "response.done"):
                break

        assert cancelled, "Expected cancelled response after barge-in"
