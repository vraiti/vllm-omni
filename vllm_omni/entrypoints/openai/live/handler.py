# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``LiveSessionHandler``: one OpenAI Live session over a WebSocket.

Each client event has a ``handle_*`` method (validation and wire replies) and,
where it changes session state, a ``do_*`` method that the VAD path and the
native-model output path call as well.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from vllm.logger import init_logger

from vllm_omni.config.live_session import LiveSessionConfig, LiveSessionDeployConfig
from vllm_omni.entrypoints.openai.live import protocol as proto
from vllm_omni.entrypoints.openai.live.asr_client import AsrClient
from vllm_omni.entrypoints.openai.live.audio import (
    InputAudioDecoder,
    OutputAudioEncoder,
    ResampleStream,
    RetainedAudio,
    float_to_pcm16,
    parse_audio_format,
    resample,
)
from vllm_omni.entrypoints.openai.live.driver import ResumableRequestDriver, finish_reason_str
from vllm_omni.entrypoints.openai.live.pacer import OutputPacer, PacedTurn
from vllm_omni.entrypoints.openai.live.processor import (
    ExternalVadProcessor,
    LiveSessionProcessor,
    NativeVadProcessor,
    ProcessorContext,
)
from vllm_omni.entrypoints.openai.live.protocol import LiveProtocolError
from vllm_omni.entrypoints.openai.live.session import (
    AssistantItem,
    FunctionCallItem,
    FunctionCallOutputItem,
    LiveSessionState,
    NativeUnitItem,
    OpenDelegation,
    SystemItem,
    UserAudioItem,
    UserTextItem,
)
from vllm_omni.entrypoints.openai.live.timestamps import TokenTimeline
from vllm_omni.entrypoints.openai.live.vad_client import (
    VAD_SAMPLE_RATE_HZ,
    ServiceUnavailableError,
    VadClient,
    VadResult,
    check_health,
)
from vllm_omni.metrics.live import LiveSessionMetrics

logger = init_logger(__name__)

SERVICE_HEALTH_TIMEOUT_S = 2.0
USAGE_INTERVAL_MS = 60_000
PACER_TICK_S = 0.05
# The pacer clock follows session_ms; if the client stops sending audio for
# longer than this, wall time keeps playback going.
PACER_STALL_S = 0.3
# End of a turn's audio when the final stage reports no boundary of its own:
# this long after the last chunk once the text segment has finished, or this
# long after the text finished when no audio arrived at all.
AUDIO_IDLE_S = 1.5
AUDIO_FIRST_CHUNK_TIMEOUT_S = 8.0
# Retained input audio before the current point while no speech is active;
# covers the VAD service's prefix padding.
VAD_RETAIN_MS = 3_000
# Left-trim target as a fraction of the context budget.
LEFT_TRIM_TARGET = 0.6
DEFAULT_MAX_TOKENS = 1024


@dataclass
class ExternalTurn:
    """One assistant generation turn on an external-VAD session."""

    epoch: int
    # 1-based index of this turn's input chunk within the epoch.
    segment_index: int
    assistant: AssistantItem
    pacer_turn: PacedTurn
    tool_extractor: Any = None
    raw_text: str = ""
    text_done: bool = False
    audio_done: bool = False
    tool_markup: bool = False
    audio: list[np.ndarray] = field(default_factory=list)
    audio_rate: int = 24_000
    first_audio_wall: float | None = None
    last_audio_wall: float | None = None
    text_done_wall: float | None = None
    prompt_tokens: int = 0
    start_wall: float = 0.0
    timeline: TokenTimeline | None = None
    timeline_task: asyncio.Task | None = None
    emitted_tokens: int = 0
    released_ms: float = 0.0
    finished: bool = False

    @property
    def spoken_tokens(self) -> list[int]:
        count = self.assistant.spoken_token_count
        return self.assistant.token_ids if count is None else self.assistant.token_ids[:count]


@dataclass
class NativeWindow:
    start_ms: int
    end_ms: int


class LiveSessionHandler:
    def __init__(
        self,
        websocket: WebSocket,
        *,
        engine: Any,
        model_name: str,
        live_config: LiveSessionConfig,
        deploy_config: LiveSessionDeployConfig,
        processor_cls: type[LiveSessionProcessor],
        processor_context: ProcessorContext,
        http: aiohttp.ClientSession,
        metrics: LiveSessionMetrics | None = None,
    ) -> None:
        self.ws = websocket
        self.metrics = metrics or LiveSessionMetrics(model_name, live_config.vad, log_stats=False)
        self.engine = engine
        self.model_name = model_name
        self.live_config = live_config
        self.deploy = deploy_config
        self.processor_cls = processor_cls
        self.processor_context = processor_context
        self.http = http
        self.external = live_config.vad == "external"

        self.state: LiveSessionState | None = None
        self.processor: LiveSessionProcessor | None = None
        self.driver: ResumableRequestDriver | None = None
        self.asr = AsrClient(http, deploy_config.asr_service_url, deploy_config.asr_timeout_s)

        self._send_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()
        self._ws_open = False
        self._closing = False
        self._recv_task: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self._decoder: InputAudioDecoder | None = None
        self._encoder: OutputAudioEncoder | None = None
        self._to_model: ResampleStream | None = None
        self._next_usage_ms = USAGE_INTERVAL_MS
        self._needs_full_render = True
        self._delegation_model = model_name

        # External VAD.
        self._vad: VadClient | None = None
        self._to_vad: ResampleStream | None = None
        self._vad_audio = RetainedAudio(VAD_SAMPLE_RATE_HZ)
        self._vad_offset_ms = 0.0  # session_ms - VAD clock, valid for the current unmuted span
        self._speech_active = False
        self._speech_start_ms: int | None = None
        self._pacer: OutputPacer | None = None
        self._turn: ExternalTurn | None = None
        self._pacer_stall_ms = 0.0
        self._last_append_wall = time.monotonic()
        # Segment accounting of the current epoch: chunks submitted, and audio
        # segments the final stage reported finished (once it reports any).
        self._segments_submitted = 0
        self._audio_segments_finished = 0
        self._audio_segment_signal = False

        # Native VAD.
        self._native_buffer: list[np.ndarray] = []
        self._native_buffer_samples = 0
        self._native_buffer_start_ms = 0.0
        self._native_pending: deque[NativeUnitItem] = deque()
        self._native_segment_open = False
        self._native_window: NativeWindow | None = None
        self._asr_window: list[np.ndarray] = []
        self._asr_window_start_ms: float | None = None
        self._asr_window_ms = 0.0

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        await self.ws.accept()
        self._ws_open = True
        self._recv_task = asyncio.current_task()
        try:
            while not self._closing:
                try:
                    text = await self.ws.receive_text()
                except WebSocketDisconnect:
                    self._ws_open = False
                    await self._close("connection_lost", send_closed=False)
                    return
                await self._dispatch(text)
        except asyncio.CancelledError:
            if not self._closing:
                raise
        except ServiceUnavailableError as exc:
            await self._fail(exc)
        except Exception as exc:
            logger.exception("Live session %s failed", self._session_id)
            await self._fail(exc)
        finally:
            await self._close("connection_lost", send_closed=False)

    @property
    def _session_id(self) -> str:
        return self.state.id if self.state is not None else "-"

    def _spawn(self, coro: Any, name: str) -> asyncio.Task:
        task = asyncio.create_task(self._guard(coro), name=f"live-{name}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _guard(self, coro: Any) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closing:
                logger.exception("Live session %s background task failed", self._session_id)
                await self._fail(exc)

    async def _fail(self, exc: BaseException) -> None:
        """Generic ``internal_server_error``, cleanup, close with 1011."""
        if self._closing:
            return
        if isinstance(exc, ServiceUnavailableError):
            logger.error("Live session %s: external service failure: %s", self._session_id, exc)
            self.metrics.error("service_unavailable")
        else:
            self.metrics.error("internal")
        await self._send_error(proto.internal_error())
        await self._close(None, send_closed=False, code=1011)

    async def _close(
        self,
        reason: str | None,
        *,
        client_event_id: str | None = None,
        send_closed: bool = True,
        code: int = 1000,
    ) -> None:
        """Run the session cleanup once; optionally reply ``session.closed``."""
        if self._closing:
            return
        self._closing = True
        self.metrics.session_finished()
        if reason is not None:
            logger.info("Live session %s closing: %s", self._session_id, reason)
        if self.driver is not None:
            await self.driver.close()
        if self._pacer is not None:
            self._pacer.discard_unreleased()
        current = asyncio.current_task()
        for task in list(self._tasks):
            if task is not current:
                task.cancel()
        if self._vad is not None:
            with contextlib.suppress(Exception):
                await self._vad.close()
        if send_closed and self.state is not None and reason is not None:
            await self._send(proto.usage_updated(self.state.usage_seconds))
            await self._send(
                proto.session_closed(self.state.to_resource(), reason, self.state.usage_seconds, client_event_id)
            )
        if self._ws_open:
            self._ws_open = False
            with contextlib.suppress(Exception):
                await self.ws.close(code=code)
        if self._recv_task is not None and self._recv_task is not current and not self._recv_task.done():
            self._recv_task.cancel()

    # ------------------------------------------------------------------ #
    # Wire                                                               #
    # ------------------------------------------------------------------ #

    async def _send(self, event: BaseModel) -> None:
        if not self._ws_open:
            return
        payload = proto.dump_event(event)
        try:
            async with self._send_lock:
                await self.ws.send_text(payload)
        except Exception:
            self._ws_open = False
            if not self._closing:
                self._spawn(self._close("connection_lost", send_closed=False), "drop")

    async def _send_error(self, err: LiveProtocolError, client_event_id: str | None = None) -> None:
        await self._send(proto.error_event(err, client_event_id))

    async def _dispatch(self, text: str) -> None:
        client_event_id = None
        try:
            payload = proto.parse_client_json(text)
            client_event_id = payload.get("event_id")
            event_type = payload["type"]
            if event_type in proto.SERVER_IGNORED_EVENTS:
                return
            if event_type not in proto.CLIENT_EVENT_MODELS:
                raise LiveProtocolError("unsupported_event", f"Unsupported event type '{event_type}'.")
            if event_type == "session.start":
                if self.state is not None:
                    raise LiveProtocolError("session_already_started", "The session has already started.")
                await self.handle_session_start(payload, client_event_id)
                return
            if self.state is None:
                raise LiveProtocolError("session_not_started", "Send session.start before other events.")
            if (
                event_type in proto.SERVER_UNSUPPORTED_EVENTS
                or event_type in self.live_config.unsupported_features.events
            ):
                raise LiveProtocolError("unsupported_event", f"'{event_type}' is not supported by this model.")
            handler = self._handlers[event_type]
            await handler(self, payload, client_event_id)
        except LiveProtocolError as err:
            await self._send_error(err, client_event_id)

    # ------------------------------------------------------------------ #
    # session.start                                                      #
    # ------------------------------------------------------------------ #

    async def handle_session_start(self, payload: dict[str, Any], client_event_id: str | None) -> None:
        # 1. External services.
        try:
            await self.asr.check_health(SERVICE_HEALTH_TIMEOUT_S)
            if self.external:
                await check_health(self.http, self.deploy.external_vad.vad_service_url, SERVICE_HEALTH_TIMEOUT_S)
        except ServiceUnavailableError as exc:
            logger.error("Live session.start rejected: %s", exc)
            await self._send_error(proto.internal_error(), client_event_id)
            await self._close(None, send_closed=False, code=1011)
            return

        # 2-3. Validation and defaults.
        try:
            state, processor = self._validate_session_start(payload)
        except LiveProtocolError as err:
            await self._send_error(err, client_event_id)
            await self._close(None, send_closed=False, code=1008)
            return
        self.state, self.processor = state, processor

        # 4. Timeline, audio paths, pacer, VAD connection.
        self._decoder = InputAudioDecoder(state.audio_format)
        self._encoder = OutputAudioEncoder(state.audio_format, processor.output_sample_rate)
        self._to_model = ResampleStream(state.audio_format.rate, processor.input_sample_rate)
        self.driver = ResumableRequestDriver(
            self.engine,
            state.id,
            processor.sampling_params_list(self.engine.default_sampling_params_list),
            on_output=self._on_engine_output,
            on_error=self._on_engine_error,
        )
        if self.external:
            self._to_vad = ResampleStream(state.audio_format.rate, VAD_SAMPLE_RATE_HZ)
            self._pacer = OutputPacer(self.deploy.external_vad.output_audio_delta_size_ms, self._on_pacer_release)
            self._vad = VadClient(self.http, self.deploy.external_vad.vad_service_url)
            try:
                await self._vad.connect()
            except ServiceUnavailableError as exc:
                await self._fail(exc)
                return
            self._spawn(self._vad_loop(), "vad")
            self._spawn(self._pacer_loop(), "pacer")

        # 5. Initial history. External models hold it until the first turn;
        # native models open the request on the first audio window.
        self._seed_history(state)

        # 6. Replies.
        self._spawn(self._expiry_loop(), "expiry")
        self.metrics.session_started()
        await self._send(proto.session_started(state.to_resource(), client_event_id))
        await self._send(proto.info("capabilities", json.dumps(self._capabilities())))

    def _capabilities(self) -> dict[str, Any]:
        unsupported = self.live_config.unsupported_features
        return {
            "vad": self.live_config.vad,
            "unsupported_features": {
                "events": sorted({*proto.SERVER_UNSUPPORTED_EVENTS, *unsupported.events}),
                "session_configs": sorted(
                    {*proto.SERVER_UNSUPPORTED_SESSION_CONFIGS, "store", *unsupported.session_configs}
                ),
            },
        }

    @staticmethod
    def _lookup(raw: Mapping[str, Any], path: str) -> Any:
        value: Any = raw
        for part in path.split("."):
            if not isinstance(value, Mapping):
                return None
            value = value.get(part)
        return value

    def _check_unsupported_configs(self, raw: Mapping[str, Any]) -> None:
        # Server-wide non-goals first.
        if raw.get("store") not in (None, False):
            raise proto.unsupported_session_config("store")
        if raw.get("client") is not None:
            raise proto.unsupported_session_config("client")
        delegation = raw.get("delegation")
        if isinstance(delegation, Mapping) and delegation.get("type") == "client":
            raise proto.unsupported_session_config("delegation")
        for path in self.live_config.unsupported_features.session_configs:
            value = self._lookup(raw, path)
            if value is not None and value != [] and value != {}:
                raise proto.unsupported_session_config(path)

    def _validate_delegation(self, delegation: Any, *, update: bool) -> dict[str, Any] | None:
        """Validate a ``delegation`` object; returns its JSON form."""
        if delegation is None:
            return None
        if not isinstance(delegation, Mapping) or delegation.get("type") != "responses":
            raise proto.unsupported_session_config("delegation")
        responses = delegation.get("responses")
        if responses is None:
            if update:
                return {"type": "responses"}
            raise proto.invalid_value("'delegation.responses' is required.", "delegation.responses")
        if not isinstance(responses, Mapping):
            raise proto.invalid_value("'delegation.responses' must be an object.", "delegation.responses")
        model = responses.get("model")
        if model is not None and model != self.model_name:
            raise proto.invalid_value(
                f"delegation.responses.model must be the served model '{self.model_name}'.",
                "delegation.responses.model",
            )
        for key in ("reasoning", "service_tier", "text"):
            if responses.get(key) is not None:
                raise proto.unsupported_session_config(f"delegation.responses.{key}")
        for tool in responses.get("tools") or []:
            if not isinstance(tool, Mapping) or tool.get("type") != "function":
                raise proto.unsupported_session_config("delegation.responses.tools")
        tool_choice = responses.get("tool_choice")
        if isinstance(tool_choice, Mapping) and tool_choice.get("type") != "function":
            raise proto.unsupported_session_config("delegation.responses.tool_choice")
        tools = responses.get("tools") or []
        if tools and not getattr(self.processor_cls, "supports_tools", False):
            raise proto.unsupported_session_config("delegation.responses.tools")
        return {"type": "responses", "responses": {k: v for k, v in responses.items() if v is not None}}

    def _validate_session_start(self, payload: dict[str, Any]) -> tuple[LiveSessionState, LiveSessionProcessor]:
        raw = payload.get("session")
        if not isinstance(raw, Mapping):
            raise proto.invalid_value("'session' must be an object.", "session")
        self._check_unsupported_configs(raw)
        delegation = self._validate_delegation(raw.get("delegation"), update=False)
        audio = raw.get("audio") or {}
        if not isinstance(audio, Mapping):
            raise proto.invalid_value("'session.audio' must be an object.", "audio")
        audio_format = parse_audio_format(audio.get("format"))
        # ``delegation.responses.model`` may be omitted on this server.
        checked = {**payload, "session": {**raw, "delegation": delegation}}
        if delegation is not None:
            checked["session"]["delegation"] = {
                "type": "responses",
                "responses": {"model": self.model_name, **delegation["responses"]},
            }
        event = proto.validate_client_event(checked)
        session = event.session  # type: ignore[attr-defined]
        if session.model != self.model_name:
            raise proto.invalid_value(f"Model '{session.model}' is not served here.", "model")

        processor = self.processor_cls(self.processor_context)
        voice = (audio.get("output") or {}).get("voice") if isinstance(audio.get("output"), Mapping) else None
        voice_name = voice.get("id") if isinstance(voice, Mapping) else voice
        if isinstance(voice_name, str) and voice_name in proto.OPENAI_BUILT_IN_VOICES:
            raise proto.invalid_value(
                f"OpenAI voice '{voice_name}' is not available; use a voice id of the served model.",
                "audio.output.voice",
            )
        processor.resolve_voice(voice)

        instructions = raw.get("instructions")
        if instructions is not None and processor.count_tokens(instructions) > proto.MAX_INSTRUCTIONS_TOKENS:
            raise proto.invalid_value(f"'instructions' exceeds {proto.MAX_INSTRUCTIONS_TOKENS} tokens.", "instructions")
        items = raw.get("input")
        if items is not None:
            if len(items) > proto.MAX_INPUT_MESSAGES:
                raise proto.invalid_value(f"'input' exceeds {proto.MAX_INPUT_MESSAGES} messages.", "input")
            total = 0
            for index, item in enumerate(items):
                content = item.get("content") if isinstance(item, Mapping) else None
                if not isinstance(content, list) or len(content) != 1 or not isinstance(content[0].get("text"), str):
                    raise proto.invalid_value("Each 'input' message must have exactly one text part.", f"input.{index}")
                total += processor.count_tokens(content[0]["text"])
            if total > proto.MAX_INPUT_TOKENS:
                raise proto.invalid_value(f"'input' exceeds {proto.MAX_INPUT_TOKENS} tokens.", "input")

        state = LiveSessionState(
            model=self.model_name,
            audio_format=audio_format,
            expires_at=int(time.time()) + self.deploy.session_lifetime_s,
            voice=voice,
            instructions=instructions,
            input=list(items) if items is not None else None,
            delegation=delegation,
        )
        return state, processor

    def _seed_history(self, state: LiveSessionState) -> None:
        if state.instructions and state.instructions.strip():
            state.history.append(SystemItem(state.instructions))
        for item in state.input or []:
            text = item["content"][0]["text"]
            role = item.get("role")
            if role == "developer":
                state.history.append(SystemItem(text))
            elif role == "user":
                state.history.append(UserTextItem(text))
            elif role == "assistant":
                state.history.append(AssistantItem(text=text))

    # ------------------------------------------------------------------ #
    # session.update                                                     #
    # ------------------------------------------------------------------ #

    async def handle_session_update(self, payload: dict[str, Any], client_event_id: str | None) -> None:
        state = self.state
        raw = payload.get("session")
        if not isinstance(raw, Mapping):
            raise proto.invalid_value("'session' must be an object.", "session")
        for key, value in raw.items():
            if key != "delegation" and value is not None:
                raise proto.unsupported_session_config(key)
        proto.validate_client_event(payload)
        delegation = raw.get("delegation")
        if delegation is not None:
            self._check_unsupported_configs({"delegation": delegation})
            update = self._validate_delegation(delegation, update=True)
            if not state.has_responses_delegation:
                raise proto.invalid_value("The delegation type cannot change after session.start.", "delegation.type")
            responses = dict(state.responses_config)
            changed_rendering = False
            for key, value in (update.get("responses") or {}).items():
                if responses.get(key) != value:
                    responses[key] = value
                    changed_rendering |= key in ("instructions", "tools", "tool_choice")
            state.delegation = {"type": "responses", "responses": responses}
            if changed_rendering:
                # History mutation: the next generation turn resubmits the
                # whole re-rendered history as a new request.
                self._needs_full_render = True
        await self._send(proto.session_updated(state.to_resource(), client_event_id))

    # ------------------------------------------------------------------ #
    # Input audio                                                        #
    # ------------------------------------------------------------------ #

    async def handle_input_audio_append(self, payload: dict[str, Any], client_event_id: str | None) -> None:
        event = proto.validate_client_event(payload)
        samples, duration_ms = self._decoder.decode(event.audio)  # type: ignore[attr-defined]
        await self.do_input_audio_append(samples, duration_ms)

    async def do_input_audio_append(self, samples: np.ndarray, duration_ms: float) -> None:
        state = self.state
        start_ms = state.session_ms
        now = time.monotonic()
        gap = now - self._last_append_wall
        if gap > PACER_STALL_S:
            self._pacer_stall_ms = max(0.0, self._pacer_stall_ms - (gap - PACER_STALL_S) * 1000.0)
        self._last_append_wall = now
        state.session_ms += duration_ms
        state.input_audio_ms += duration_ms
        self.metrics.input_audio(duration_ms / 1000.0)
        while state.session_ms >= self._next_usage_ms:
            self._next_usage_ms += USAGE_INTERVAL_MS
            await self._send(proto.usage_updated(state.usage_seconds))

        if self.external:
            if not state.muted:
                vad_samples = self._to_vad.process(samples)
                self._vad_offset_ms = start_ms - self._vad_audio.end_ms
                self._vad_audio.append(vad_samples)
                if vad_samples.size:
                    await self._vad.send(float_to_pcm16(vad_samples))
            else:
                self._to_vad.process(samples)  # keep the resampler's phase continuous
            if self._pacer is not None:
                await self._pacer.tick(self._pacer_clock())
        else:
            model_samples = self._to_model.process(samples)
            if state.muted:
                model_samples = np.zeros_like(model_samples)
            else:
                self._accumulate_transcription(model_samples, start_ms, state.session_ms)
            await self._native_buffer_append(model_samples, start_ms)

    async def handle_input_audio_mute(self, payload: dict[str, Any], client_event_id: str | None) -> None:
        state = self.state
        if not state.muted:
            state.muted = True
            if self.external:
                self._speech_active = False
                self._speech_start_ms = None
                await self._vad.reset()
            else:
                self._flush_transcription_window()
        await self._send(proto.input_audio_muted(client_event_id))

    async def handle_input_audio_unmute(self, payload: dict[str, Any], client_event_id: str | None) -> None:
        self.state.muted = False
        await self._send(proto.input_audio_unmuted(client_event_id))

    # ------------------------------------------------------------------ #
    # Conversation items                                                 #
    # ------------------------------------------------------------------ #

    async def handle_instructions_append(self, payload: dict[str, Any], client_event_id: str | None) -> None:
        if "delegation_id" not in payload:
            raise proto.invalid_value("'delegation_id' is required (null for session context).", "delegation_id")
        if payload["delegation_id"] is not None:
            raise proto.invalid_value("This server never creates client delegations.", "delegation_id")
        event = proto.validate_client_event(payload)
        content = event.content  # type: ignore[attr-defined]
        if self.processor.count_tokens(content) > proto.MAX_INSTRUCTIONS_APPEND_TOKENS:
            raise proto.invalid_value(f"'content' exceeds {proto.MAX_INSTRUCTIONS_APPEND_TOKENS} tokens.", "content")
        self.state.pending_items.append(SystemItem(content))
        await self._send(proto.instructions_appended(self.state.now_ms, client_event_id))

    async def handle_response_item_create(self, payload: dict[str, Any], client_event_id: str | None) -> None:
        state = self.state
        if not state.has_responses_delegation:
            raise LiveProtocolError("delegation_required", "response.item.create requires delegation.responses.")
        item = payload.get("item")
        if not isinstance(item, Mapping):
            raise proto.invalid_value("'item' must be an object.", "item")
        item_type = item.get("type") or ("message" if "role" in item else None)
        if item_type == "function_call_output":
            call_id = item.get("call_id")
            delegation = state.delegation_state
            if delegation is None or call_id not in delegation.call_ids:
                raise proto.invalid_value(f"Unknown call_id '{call_id}'.", "item.call_id")
            output = item.get("output")
            if not isinstance(output, str):
                output = json.dumps(output)
            delegation.outputs.add(call_id)
            state.pending_items.append(FunctionCallOutputItem(call_id=call_id, output=output))
        elif item_type == "message" and item.get("role") == "user":
            content = item.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = [p.get("text") for p in content if isinstance(p, Mapping) and p.get("type") == "input_text"]
                if len(parts) != len(content):
                    raise proto.invalid_value("Only input_text user content is supported.", "item.content")
                text = "".join(parts)
            else:
                raise proto.invalid_value("'item.content' is required.", "item.content")
            # Rendered into history at the next generation turn.
            state.pending_items.append(UserTextItem(text))
        else:
            raise LiveProtocolError("unsupported_event", f"Unsupported response item type '{item_type}'.")

    async def handle_response_create(self, payload: dict[str, Any], client_event_id: str | None) -> None:
        state = self.state
        if not state.has_responses_delegation:
            raise LiveProtocolError("delegation_required", "response.create requires delegation.responses.")
        if self._generation_in_progress():
            raise LiveProtocolError("generation_in_progress", "A generation turn is already in progress.")
        if state.delegation_state is not None:
            state.delegation_state.completed = True
            state.delegation_state = None
        await self.do_generation_turn(None)

    async def handle_session_close(self, payload: dict[str, Any], client_event_id: str | None) -> None:
        await self._close("close_requested", client_event_id=client_event_id)

    # ------------------------------------------------------------------ #
    # External VAD                                                       #
    # ------------------------------------------------------------------ #

    def _vad_to_session_ms(self, vad_ms: int | None) -> int:
        return int((vad_ms or 0) + self._vad_offset_ms)

    async def _vad_loop(self) -> None:
        async for result in self._vad.results():
            if self._closing:
                return
            if self.state.muted:
                continue
            await self._on_vad_result(result)
            if not self._speech_active:
                self._vad_audio.trim_before(self._vad_audio.end_ms - VAD_RETAIN_MS)

    async def _on_vad_result(self, result: VadResult) -> None:
        if result.speech_started:
            self._speech_active = True
            self._speech_start_ms = result.speech_start_ms
            if self._output_in_flight():
                await self.do_interrupt()
        if result.speech_stopped and self._speech_active:
            self._speech_active = False
            start_vad_ms = self._speech_start_ms or 0
            end_vad_ms = result.speech_end_ms if result.speech_end_ms is not None else int(self._vad_audio.end_ms)
            segment = self._vad_audio.slice(start_vad_ms, end_vad_ms)
            self._speech_start_ms = None
            if segment.size == 0:
                return
            model_rate = self.processor.input_sample_rate
            item = UserAudioItem(
                audio=resample(segment, VAD_SAMPLE_RATE_HZ, model_rate),
                sample_rate=model_rate,
                start_ms=self._vad_to_session_ms(start_vad_ms),
                end_ms=self._vad_to_session_ms(end_vad_ms),
            )
            self._spawn(self._transcribe_input(item), "asr-input")
            await self.do_generation_turn(item)

    async def _transcribe(self, audio: np.ndarray, sample_rate: int) -> list[Any]:
        self.metrics.asr_started()
        started = time.monotonic()
        latency = None
        try:
            segments = await self.asr.transcribe(audio, sample_rate)
            latency = time.monotonic() - started
            return segments
        finally:
            self.metrics.asr_finished(latency)

    async def _transcribe_input(self, item: UserAudioItem) -> None:
        segments = await self._transcribe(item.audio, item.sample_rate)
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        item.transcript = text
        if text:
            await self._send(proto.input_transcript_delta(text, item.start_ms, item.end_ms))

    def _output_in_flight(self) -> bool:
        """Whether the user would talk over output: unreleased or still generating."""
        if self._pacer is not None and self._pacer.has_unreleased:
            return True
        turn = self._turn
        if turn is None or turn.finished or turn.audio_done:
            return False
        if not turn.text_done:
            return True
        # The talker may still be producing audio for finished text.
        return turn.last_audio_wall is None or time.monotonic() - turn.last_audio_wall < AUDIO_IDLE_S

    def _generation_in_progress(self) -> bool:
        turn = self._turn
        return turn is not None and not turn.finished and not (turn.text_done and turn.audio_done)

    # ------------------------------------------------------------------ #
    # Generation turns                                                   #
    # ------------------------------------------------------------------ #

    def _max_tokens(self, params: Any) -> int:
        return int(getattr(params, "max_tokens", None) or DEFAULT_MAX_TOKENS)

    async def do_generation_turn(self, user_item: UserAudioItem | None) -> None:
        """Submit the user turn (or a continuation) to the resumable request."""
        processor = self.processor
        assert isinstance(processor, ExternalVadProcessor)
        async with self._turn_lock:
            state = self.state
            new_items = [*state.take_pending(), *([user_item] if user_item is not None else [])]
            state.history.extend(new_items)
            params = processor.stage0_sampling_params(state, self.driver.stage0_params)
            max_tokens = self._max_tokens(params)
            max_model_len = self.processor_context.max_model_len
            rendered = None
            if self.driver.active and not self._needs_full_render:
                rendered = processor.render_append(
                    state,
                    new_items,
                    last_sampled_token=self.driver.last_sampled_token,
                    stopped_on_stop_token=self.driver.last_finish_reason == "stop",
                )
                if self.driver.num_tokens + rendered.num_tokens + max_tokens > max_model_len:
                    rendered = None
            if rendered is None:
                budget = max_model_len - max_tokens
                if processor.estimate_history_tokens(state) > budget and processor.left_trim(
                    state, int(budget * LEFT_TRIM_TARGET)
                ):
                    logger.info("Live session %s: left-trimmed history to fit max_model_len", state.id)
                rendered = processor.render_full(state)
                await self.driver.start(rendered.prompt, params, rendered.num_tokens)
                self._needs_full_render = False
                self._segments_submitted = 1
                self._audio_segments_finished = 0
                self._audio_segment_signal = False
            else:
                self.driver.append(rendered.prompt, params, rendered.num_tokens)
                self._segments_submitted += 1
            self._begin_turn(rendered.num_tokens)

    def _begin_turn(self, prompt_tokens: int) -> None:
        state = self.state
        assistant = AssistantItem()
        state.history.append(assistant)
        extractor = None
        if state.tools and state.responses_config.get("tool_choice") != "none":
            from vllm_omni.entrypoints.openai.live.tools import ToolCallExtractor

            parser = self.processor_context.tool_call_parser or getattr(
                self.processor, "default_tool_call_parser", None
            )
            if parser:
                extractor = ToolCallExtractor(parser, self.processor_context.raw_tokenizer, state.tools)
        pacer_turn = self._pacer.open_turn(self.driver.epoch, self.processor.output_sample_rate)
        self._turn = ExternalTurn(
            epoch=self.driver.epoch,
            segment_index=self._segments_submitted,
            assistant=assistant,
            pacer_turn=pacer_turn,
            tool_extractor=extractor,
            audio_rate=self.processor.output_sample_rate,
            prompt_tokens=prompt_tokens,
            start_wall=time.monotonic(),
        )
        self._spawn(self._turn_watchdog(self._turn), "turn-watchdog")

    async def do_interrupt(self) -> None:
        """Barge-in: abort, discard unreleased audio, truncate at the playback cursor."""
        async with self._turn_lock:
            await self._interrupt(self._turn)

    async def _interrupt(self, turn: ExternalTurn | None) -> None:
        await self.driver.abort()
        self._needs_full_render = True
        if turn is None:
            self._pacer.discard_unreleased()
            return
        released_ms = turn.pacer_turn.released_ms
        self._pacer.discard_unreleased()
        self.metrics.set_unreleased(0.0)
        self.metrics.interruption()
        turn.text_done = turn.audio_done = True
        heard = 0
        if released_ms > 0 and turn.assistant.token_ids:
            timeline = await self._turn_timeline(turn)
            heard = timeline.token_index_at(released_ms) if timeline is not None else 0
            # Transcript for audio already released but not yet described.
            await self._emit_output_transcript(turn, released_ms, final=False, limit=heard)
        self.processor.truncate_assistant(turn.assistant, heard)
        if not turn.assistant.token_ids and not turn.assistant.text:
            with contextlib.suppress(ValueError):
                self.state.history.remove(turn.assistant)
        self._finish_turn(turn)

    def _finish_turn(self, turn: ExternalTurn) -> None:
        turn.finished = True
        if turn.timeline_task is not None and not turn.timeline_task.done():
            turn.timeline_task.cancel()
        if self._turn is turn:
            self._turn = None

    async def _turn_timeline(self, turn: ExternalTurn) -> TokenTimeline | None:
        if turn.timeline is not None:
            return turn.timeline
        if turn.timeline_task is None:
            turn.timeline_task = asyncio.create_task(self._build_timeline(turn))
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.shield(turn.timeline_task)
        return turn.timeline

    async def _build_timeline(self, turn: ExternalTurn) -> None:
        audio = np.concatenate(turn.audio) if turn.audio else np.empty(0, dtype=np.float32)
        segments = await self._transcribe(audio, turn.audio_rate)
        turn.timeline = TokenTimeline(turn.spoken_tokens, segments, self.processor.decode)

    async def _timeline_then_flush(self, turn: ExternalTurn) -> None:
        await self._turn_timeline(turn)
        if not turn.finished or turn.pacer_turn.final_released:
            await self._emit_output_transcript(turn, turn.released_ms, final=turn.pacer_turn.final_released)

    async def _turn_watchdog(self, turn: ExternalTurn) -> None:
        """Ends a turn's audio when the final stage reports no boundary."""
        while not turn.audio_done and not turn.finished:
            await asyncio.sleep(0.25)
            if not turn.text_done:
                continue
            now = time.monotonic()
            if turn.last_audio_wall is not None:
                idle = now - turn.last_audio_wall > AUDIO_IDLE_S
            else:
                idle = now - (turn.text_done_wall or now) > AUDIO_FIRST_CHUNK_TIMEOUT_S
            if idle:
                await self._on_turn_audio_done(turn)

    async def _on_turn_audio_done(self, turn: ExternalTurn) -> None:
        if turn.audio_done:
            return
        turn.audio_done = True
        self._pacer.finish(turn.pacer_turn)
        if turn.audio and turn.spoken_tokens:
            turn.timeline_task = asyncio.create_task(self._build_timeline(turn))
            self._spawn(self._timeline_then_flush(turn), "asr-output")
        elif turn.spoken_tokens and not turn.audio:
            # Text without audio: describe it at the current position.
            now = self.state.now_ms
            await self._send(proto.output_transcript_delta(self.processor.decode(turn.spoken_tokens), now, now))
            self._finish_turn(turn)
        else:
            self._finish_turn(turn)

    # ------------------------------------------------------------------ #
    # Engine output                                                      #
    # ------------------------------------------------------------------ #

    async def _on_engine_error(self, exc: BaseException) -> None:
        await self._fail(exc)

    async def _on_engine_output(self, epoch: int, output: Any) -> None:
        if self._closing or self.driver is None or epoch != self.driver.epoch:
            return
        if self.external:
            await self._on_external_output(output)
        else:
            await self._on_native_output(output)

    @staticmethod
    def _audio_chunks(output: Any) -> tuple[list[np.ndarray], int | None]:
        mm = getattr(output, "multimodal_output", None)
        if not isinstance(mm, Mapping):
            return [], None
        sr = mm.get("sr") or mm.get("sample_rate") or mm.get("audio_sample_rate")
        if isinstance(sr, (list, tuple)) and sr:
            sr = sr[-1]
        if hasattr(sr, "item"):
            sr = sr.item()
        key = "audio" if "audio" in mm else ("model_outputs" if "model_outputs" in mm else None)
        if key is None:
            return [], None
        raw = mm.get(key)
        # Per-step increments; code2wav already slices its left context out.
        values = (
            [raw[-1]] if isinstance(raw, (list, tuple)) and raw else ([] if isinstance(raw, (list, tuple)) else [raw])
        )
        chunks = []
        for value in values:
            if hasattr(value, "detach"):
                value = value.detach().float().cpu().numpy()
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size:
                chunks.append(arr)
        return chunks, int(sr) if sr else None

    async def _on_external_output(self, output: Any) -> None:
        is_audio = getattr(output, "final_output_type", None) == "audio"
        segment_done = is_audio and bool(getattr(output, "segment_finished", False))
        # Audio still arriving for an earlier segment of this epoch (e.g. the
        # talker voicing tool-call markup after its turn was closed).
        stale_audio = self._audio_segment_signal and self._audio_segments_finished < (
            (self._turn.segment_index if self._turn is not None else self._segments_submitted) - 1
        )
        if segment_done:
            self._audio_segment_signal = True
            self._audio_segments_finished += 1
        turn = self._turn
        if turn is None or turn.finished or turn.epoch != self.driver.epoch or (is_audio and stale_audio):
            return
        if getattr(output, "stage_id", None) == 0 and getattr(output, "outputs", None):
            completion = output.outputs[0]
            token_ids = list(getattr(completion, "token_ids", None) or ())
            turn.assistant.token_ids.extend(token_ids)
            turn.raw_text += getattr(completion, "text", "") or ""
            if turn.tool_extractor is not None and not turn.tool_markup:
                turn.tool_markup = turn.tool_extractor.markup_started(turn.raw_text)
            if finish_reason_str(getattr(completion, "finish_reason", None)) is not None:
                await self._on_turn_text_done(turn)
            return
        if getattr(output, "final_output_type", None) != "audio":
            return
        chunks, sample_rate = self._audio_chunks(output)
        if sample_rate and sample_rate != turn.audio_rate:
            chunks = [resample(c, sample_rate, turn.audio_rate) for c in chunks]
        if turn.tool_markup or turn.audio_done:
            # The talker speaks tool-call markup; drop audio from there on.
            return
        for chunk in chunks:
            turn.audio.append(chunk)
            self._pacer.push(turn.pacer_turn, chunk)
        if chunks:
            now = time.monotonic()
            if turn.first_audio_wall is None:
                self.metrics.first_audio(now - turn.start_wall)
            turn.first_audio_wall = turn.first_audio_wall or now
            turn.last_audio_wall = now
        if segment_done or getattr(output, "finished", False):
            await self._on_turn_audio_done(turn)

    async def _on_turn_text_done(self, turn: ExternalTurn) -> None:
        turn.text_done = True
        turn.text_done_wall = time.monotonic()
        assistant = turn.assistant
        # Stop tokens are not part of the spoken text.
        raw_tokenizer = self.processor_context.raw_tokenizer
        text_with_markup = raw_tokenizer.decode(assistant.token_ids, skip_special_tokens=False)
        calls = turn.tool_extractor.extract(text_with_markup) if turn.tool_extractor is not None else []
        if calls:
            start = turn.tool_extractor.start_token
            spoken = len(assistant.token_ids)
            for k in range(1, len(assistant.token_ids) + 1):
                if start in raw_tokenizer.decode(assistant.token_ids[:k], skip_special_tokens=False):
                    spoken = k - 1
                    break
            assistant.spoken_token_count = spoken
        assistant.text = self.processor.decode(turn.spoken_tokens)
        if calls:
            await self._open_delegation(turn, calls)
            # Nothing after the markup is spoken; the continuation is a new turn.
            await self._on_turn_audio_done(turn)

    async def _open_delegation(self, turn: ExternalTurn, calls: list[Any]) -> None:
        from vllm_omni.entrypoints.openai.live.tools import ResponsesEventStream

        state = self.state
        for call in calls:
            state.history.append(FunctionCallItem(call_id=call.call_id, name=call.name, arguments=call.arguments))
        delegation = OpenDelegation(
            delegation_id=proto.new_id("dlg"),
            response_id=proto.new_id("resp"),
            call_ids=[call.call_id for call in calls],
        )
        state.delegation_state = delegation
        await self._send(proto.delegation_created(delegation.delegation_id, delegation.response_id, state.now_ms))
        stream = ResponsesEventStream(delegation.response_id, self.model_name)
        for event in stream.function_call_events(
            calls, input_tokens=turn.prompt_tokens, output_tokens=len(turn.assistant.token_ids)
        ):
            await self._send(proto.response_event(delegation.delegation_id, event))

    # ------------------------------------------------------------------ #
    # Pacing and output transcripts                                      #
    # ------------------------------------------------------------------ #

    def _pacer_clock(self) -> float:
        gap = time.monotonic() - self._last_append_wall
        stall = max(0.0, gap - PACER_STALL_S) * 1000.0
        self._pacer_stall_ms = max(self._pacer_stall_ms, stall)
        return self.state.session_ms + self._pacer_stall_ms

    async def _pacer_loop(self) -> None:
        while not self._closing:
            await asyncio.sleep(PACER_TICK_S)
            await self._pacer.tick(self._pacer_clock())
            self.metrics.set_unreleased(self._pacer.unreleased_ms / 1000.0)

    async def _on_pacer_release(
        self, pacer_turn: PacedTurn, chunk: np.ndarray, start_ms: float, end_ms: float, last: bool
    ) -> None:
        if chunk.size:
            await self._send(proto.output_audio_delta(self._encoder.encode(chunk)))
            self.state.output_audio_ms += chunk.size * 1000.0 / pacer_turn.sample_rate
            self.metrics.output_audio(chunk.size / pacer_turn.sample_rate)
        turn = self._turn if self._turn is not None and self._turn.pacer_turn is pacer_turn else None
        if turn is None:
            return
        turn.released_ms = end_ms
        if turn.timeline is not None:
            await self._emit_output_transcript(turn, end_ms, final=last)
        if last and turn.timeline is not None:
            self._finish_turn(turn)

    async def _emit_output_transcript(
        self, turn: ExternalTurn, released_ms: float, *, final: bool, limit: int | None = None
    ) -> None:
        timeline = turn.timeline
        if timeline is None or turn.pacer_turn.release_start_ms is None:
            return
        spoken = len(timeline.token_ids)
        end_index = spoken if final else min(timeline.token_index_at(released_ms), spoken)
        if limit is not None:
            end_index = min(end_index, limit)
        begin = turn.emitted_tokens
        if end_index <= begin:
            if final:
                self._finish_turn(turn)
            return
        text = timeline.text_between(begin, end_index)
        turn.emitted_tokens = end_index
        base = turn.pacer_turn.release_start_ms
        start = base + timeline.token_start_ms(begin)
        last_time = timeline.times_ms[end_index - 1]
        end = base + (released_ms if last_time == float("inf") else min(last_time, released_ms))
        if text:
            await self._send(proto.output_transcript_delta(text, int(start), int(end)))
        if final:
            self._finish_turn(turn)

    # ------------------------------------------------------------------ #
    # Native VAD                                                         #
    # ------------------------------------------------------------------ #

    def _accumulate_transcription(self, samples: np.ndarray, start_ms: float, end_ms: float) -> None:
        if self._asr_window_start_ms is None:
            self._asr_window_start_ms = start_ms
        self._asr_window.append(samples)
        self._asr_window_ms = end_ms
        if end_ms - self._asr_window_start_ms >= self.deploy.native_vad.audio_transcription_interval_ms:
            self._flush_transcription_window()

    def _flush_transcription_window(self) -> None:
        if not self._asr_window or self._asr_window_start_ms is None:
            return
        audio = np.concatenate(self._asr_window)
        item = UserAudioItem(
            audio=audio,
            sample_rate=self.processor.input_sample_rate,
            start_ms=int(self._asr_window_start_ms),
            end_ms=int(self._asr_window_ms),
        )
        self._asr_window, self._asr_window_start_ms = [], None
        self._spawn(self._transcribe_input(item), "asr-input")

    async def _native_buffer_append(self, samples: np.ndarray, start_ms: float) -> None:
        if self._native_buffer_samples == 0:
            self._native_buffer_start_ms = start_ms
        self._native_buffer.append(samples)
        self._native_buffer_samples += samples.size
        rate = self.processor.input_sample_rate
        window = int(rate * self.live_config.audio_buffer_ms / 1000)
        while self._native_buffer_samples >= window:
            joined = np.concatenate(self._native_buffer)
            chunk, rest = joined[:window], joined[window:]
            start = self._native_buffer_start_ms
            end = start + self.live_config.audio_buffer_ms
            self._native_buffer = [rest] if rest.size else []
            self._native_buffer_samples = rest.size
            self._native_buffer_start_ms = end
            await self.do_native_turn(chunk, int(start), int(end))

    async def do_native_turn(self, chunk: np.ndarray, start_ms: int, end_ms: int) -> None:
        """One ``audio_buffer_ms`` window into the model.

        Windows are submitted one segment at a time: each update re-emits the
        previous segment's last sampled token, which is only known once that
        segment has finished.
        """
        processor = self.processor
        assert isinstance(processor, NativeVadProcessor)
        self._native_pending.append(
            NativeUnitItem(audio=chunk, sample_rate=processor.input_sample_rate, start_ms=start_ms, end_ms=end_ms)
        )
        if len(self._native_pending) > 3:
            logger.warning(
                "Live session %s: model is %d windows behind real time", self.state.id, len(self._native_pending)
            )
        await self._pump_native()

    async def _pump_native(self) -> None:
        if self._native_segment_open or not self._native_pending or self._closing:
            return
        processor = self.processor
        state = self.state
        unit = self._native_pending.popleft()
        state.history.append(unit)
        max_model_len = self.processor_context.max_model_len
        resume = False
        rendered = None
        if self.driver.active and not self._needs_full_render:
            rendered = processor.render_native_turn(
                state, unit, first=False, last_sampled_token=self.driver.last_sampled_token
            )
            if self.driver.num_tokens + rendered.num_tokens + 64 > max_model_len:
                rendered = None
        if rendered is None:
            if self.driver.epoch > 0:
                # Context exhausted (or the request ended): resubmit the trimmed history.
                units = sum(1 for item in state.history if isinstance(item, NativeUnitItem))
                processor.left_trim(state, max(1, int(units * LEFT_TRIM_TARGET / 1.2)))
                rendered = processor.render_native_resume(state)
                resume = True
            else:
                rendered = processor.render_native_turn(state, unit, first=True, last_sampled_token=None)
        params = processor.stage0_sampling_params(state, self.driver.stage0_params)
        extra = processor.native_extra_args(state, resume=resume)
        if extra:
            params.extra_args = {**(params.extra_args or {}), **extra}
        self._native_window = NativeWindow(unit.start_ms, unit.end_ms)
        self._native_segment_open = processor.serialize_windows
        if resume or not self.driver.active:
            await self.driver.start(rendered.prompt, params, rendered.num_tokens)
            self._needs_full_render = False
        else:
            self.driver.append(rendered.prompt, params, rendered.num_tokens)
        if not processor.serialize_windows:
            await self._pump_native()

    async def _on_native_output(self, output: Any) -> None:
        processor = self.processor
        assert isinstance(processor, NativeVadProcessor)
        window = self._native_window or NativeWindow(self.state.now_ms, self.state.now_ms)
        if getattr(output, "stage_id", None) == 0 and getattr(output, "outputs", None):
            completion = output.outputs[0]
            token_ids = list(getattr(completion, "token_ids", None) or ())
            processor.observe_stage0_output(self.state, token_ids, output)
            text = processor.spoken_text(token_ids, getattr(completion, "text", "") or "")
            if text:
                await self._send(proto.output_transcript_delta(text, window.start_ms, window.end_ms))
            if finish_reason_str(getattr(completion, "finish_reason", None)) is not None:
                self._native_segment_open = False
                await self._pump_native()
            return
        stream_text = processor.observe_multimodal_output(self.state, output)
        if stream_text:
            await self._send(proto.output_transcript_delta(stream_text, window.start_ms, window.end_ms))
        if getattr(output, "final_output_type", None) != "audio":
            return
        chunks, sample_rate = self._audio_chunks(output)
        for chunk in chunks:
            if sample_rate and sample_rate != processor.output_sample_rate:
                chunk = resample(chunk, sample_rate, processor.output_sample_rate)
            await self._send(proto.output_audio_delta(self._encoder.encode(chunk)))
            self.state.output_audio_ms += chunk.size * 1000.0 / processor.output_sample_rate
            self.metrics.output_audio(chunk.size / processor.output_sample_rate)

    # ------------------------------------------------------------------ #
    # Timers                                                             #
    # ------------------------------------------------------------------ #

    async def _expiry_loop(self) -> None:
        delay = self.state.expires_at - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        await self._close("expired")

    _handlers = {
        "session.update": handle_session_update,
        "session.input_audio.append": handle_input_audio_append,
        "session.input_audio.mute": handle_input_audio_mute,
        "session.input_audio.unmute": handle_input_audio_unmute,
        "session.instructions.append": handle_instructions_append,
        "response.item.create": handle_response_item_create,
        "response.create": handle_response_create,
        "session.close": handle_session_close,
    }
