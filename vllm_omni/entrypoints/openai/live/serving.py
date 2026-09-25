# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Glue between the API server and ``LiveSessionHandler``."""

from __future__ import annotations

from typing import Any

import aiohttp
from fastapi import WebSocket
from starlette.datastructures import State
from vllm.logger import init_logger

from vllm_omni.config.live_session import LiveSessionConfig, LiveSessionDeployConfig
from vllm_omni.entrypoints.openai.live import protocol as proto
from vllm_omni.entrypoints.openai.live.handler import LiveSessionHandler
from vllm_omni.entrypoints.openai.live.processor import ProcessorContext, load_processor_class
from vllm_omni.metrics.live import LiveSessionMetrics

logger = init_logger(__name__)


def init_live_app_state(engine_client: Any, state: State) -> None:
    """Expose the model's Live config when the deployment enables it."""
    engine = getattr(engine_client, "engine", None)
    pipeline = getattr(engine, "pipeline_config", None)
    deploy = getattr(engine, "deploy_config", None)
    live: LiveSessionConfig | None = getattr(pipeline, "live_session_config", None)
    live_deploy: LiveSessionDeployConfig | None = getattr(deploy, "live_session_config", None)
    state.live_session_config = live if live_deploy is not None else None
    state.live_session_deploy_config = live_deploy if live is not None else None
    state.live_session_processor_cls = None
    state.live_session_http = None
    if live is not None and live_deploy is not None:
        # Resolve the processor at startup so a bad dotted path fails fast.
        state.live_session_processor_cls = load_processor_class(live.live_session_processor)
        logger.info(
            "Live sessions enabled on /v1/live/sessions (vad=%s, processor=%s)",
            live.vad,
            live.live_session_processor,
        )
    elif live_deploy is not None:
        logger.warning("Deploy config has live_session_config but the model declares no live_session_config")


async def reject_live_websocket(websocket: WebSocket, code: str, message: str) -> None:
    await websocket.accept()
    await websocket.send_text(proto.dump_event(proto.error_event(proto.LiveProtocolError(code, message))))
    await websocket.close(code=1008)


async def _chat_template_renderer(engine_client: Any) -> Any:
    try:
        tokenizer = await engine_client.get_tokenizer()
    except Exception:
        # Stages with skip_tokenizer_init (PersonaPlex) have none; their
        # processors bring their own.
        return None
    if tokenizer is None or getattr(tokenizer, "chat_template", None):
        return tokenizer
    # Omni checkpoints (e.g. Qwen3-Omni) ship the template on the processor.
    try:
        from vllm.transformers_utils.processor import cached_processor_from_config

        processor = cached_processor_from_config(engine_client.model_config)
        if getattr(processor, "chat_template", None):
            return processor
    except Exception:
        logger.warning("Could not load the HF processor for chat templating", exc_info=True)
    return tokenizer


async def serve_live_session(websocket: WebSocket) -> None:
    state = websocket.app.state
    live: LiveSessionConfig | None = getattr(state, "live_session_config", None)
    live_deploy: LiveSessionDeployConfig | None = getattr(state, "live_session_deploy_config", None)
    if live is None or live_deploy is None:
        await reject_live_websocket(
            websocket, "unsupported", "Live sessions are not enabled for this model or deployment."
        )
        return
    count = getattr(state, "api_server_count", None)
    if count is not None and not (type(count) is int and count == 1):
        await reject_live_websocket(
            websocket, "multi_api_live_unsupported", "Live sessions require a single API server process."
        )
        return

    engine_client = state.engine_client
    if state.live_session_http is None or state.live_session_http.closed:
        state.live_session_http = aiohttp.ClientSession()
    args = getattr(state, "args", None)
    tool_call_parser = (
        getattr(args, "tool_call_parser", None) if getattr(args, "enable_auto_tool_choice", False) else None
    )
    model_name = state.openai_serving_models.base_model_paths[0].name
    context = ProcessorContext(
        tokenizer=await _chat_template_renderer(engine_client),
        max_model_len=int(engine_client.model_config.max_model_len),
        model_name=model_name,
        tool_call_parser=tool_call_parser,
        extra={"model_path": str(getattr(engine_client.model_config, "model", "") or "")},
    )
    handler = LiveSessionHandler(
        websocket,
        engine=engine_client,
        model_name=model_name,
        live_config=live,
        deploy_config=live_deploy,
        processor_cls=state.live_session_processor_cls,
        processor_context=context,
        http=state.live_session_http,
        metrics=LiveSessionMetrics(model_name, live.vad, log_stats=bool(getattr(state, "log_stats", True))),
    )
    await handler.run()
