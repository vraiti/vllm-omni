# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Silero VAD service for OpenAI Live sessions (``vad: external`` models).

A thin FastAPI wrapper around ``SileroStreamingVAD``. Each WebSocket on
``/v1/vad`` is one session: binary frames are 16 kHz mono PCM16, each
answered with one JSON ``StreamingVADResult`` on the session's audio clock
(milliseconds of audio received). A ``{"type": "reset"}`` text frame drops
endpointing state and keeps the clock. The ONNX detector runs on CPU and is
shared across sessions.

    python -m vllm_omni.entrypoints.live_vad_service --port 15151
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from vllm.logger import init_logger

from vllm_omni.engine.duplex.vad import (
    SILERO_VAD_FILENAME,
    SILERO_VAD_REPO_ID,
    SILERO_VAD_REVISION,
    SileroStreamingVAD,
    SileroVADBackendProvider,
    SileroVADConfig,
)

logger = init_logger(__name__)

DEFAULT_PORT = 15151


def _resolve_model_path(model_path: str | None) -> str:
    """The pinned Silero ONNX artifact: explicit path, HF cache, or download."""
    if model_path:
        return str(Path(model_path).expanduser())
    from huggingface_hub import hf_hub_download

    return hf_hub_download(SILERO_VAD_REPO_ID, SILERO_VAD_FILENAME, revision=SILERO_VAD_REVISION)


def build_app(config: SileroVADConfig, model_path: str | None) -> FastAPI:
    provider = SileroVADBackendProvider(model_path=_resolve_model_path(model_path))
    provider.get()  # load and verify the detector before accepting sessions
    app = FastAPI(title="vLLM-Omni Live VAD service")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/v1/vad")
    async def vad_session(websocket: WebSocket) -> None:
        await websocket.accept()
        vad = SileroStreamingVAD(config, backend_provider=provider)
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if message.get("bytes") is not None:
                    raw = message["bytes"]
                    if len(raw) % 2:
                        await websocket.send_text(json.dumps({"type": "error", "message": "odd PCM16 frame length"}))
                        continue
                    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) * np.float32(1.0 / 32768.0)
                    result = await asyncio.to_thread(vad.process, samples)
                    await websocket.send_text(json.dumps({"type": "vad.result", **asdict(result)}))
                elif message.get("text") is not None:
                    try:
                        command = json.loads(message["text"])
                    except json.JSONDecodeError:
                        command = {}
                    if command.get("type") == "reset":
                        vad.reset()
                    else:
                        await websocket.send_text(json.dumps({"type": "error", "message": "unknown command"}))
        except WebSocketDisconnect:
            return

    return app


def make_arg_parser() -> argparse.ArgumentParser:
    defaults = SileroVADConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Silero VAD v6.2 ONNX file; defaults to the pinned Hugging Face artifact.",
    )
    parser.add_argument("--threshold", type=float, default=defaults.threshold)
    parser.add_argument("--prefix-padding-ms", type=int, default=defaults.prefix_padding_ms)
    parser.add_argument("--silence-duration-ms", type=int, default=defaults.silence_duration_ms)
    parser.add_argument("--min-speech-duration-ms", type=int, default=defaults.min_speech_duration_ms)
    return parser


def main() -> None:
    args = make_arg_parser().parse_args()
    config = SileroVADConfig(
        threshold=args.threshold,
        prefix_padding_ms=args.prefix_padding_ms,
        silence_duration_ms=args.silence_duration_ms,
        min_speech_duration_ms=args.min_speech_duration_ms,
    )
    app = build_app(config, args.model_path)
    logger.info("Live VAD service on ws://%s:%d/v1/vad (%s)", args.host, args.port, config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
