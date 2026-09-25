# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Live wire protocol: client-event parsing and server-event construction.

Server events are built from the ``openai.types.live`` Pydantic models, so
every event is validated against the published schema before it is sent.
Client events are validated with the same models on dispatch; ``event_id``
is optional on every client event.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, get_args

from openai.types.live import (
    BuiltInVoice,
    CommentaryAppendEvent,
    DelegationCreatedEvent,
    Error,
    ErrorEvent,
    InfoEvent,
    InputAudioAppendEvent,
    InputAudioMutedEvent,
    InputAudioMuteEvent,
    InputAudioUnmutedEvent,
    InputAudioUnmuteEvent,
    InputTranscriptDeltaEvent,
    InstructionsAppendedEvent,
    InstructionsAppendEvent,
    OutputAudioDeltaEvent,
    OutputTranscriptDeltaEvent,
    ResponseCreateEvent,
    ResponseEvent,
    SessionClosedEvent,
    SessionCloseEvent,
    SessionResource,
    SessionStartedEvent,
    SessionStartEvent,
    SessionUpdatedEvent,
    SessionUpdateEvent,
    SessionUsage,
    SessionUsageUpdatedEvent,
    ThinkingAppendEvent,
)
from openai.types.live.delegation_created_event import Delegation as CreatedDelegation
from pydantic import BaseModel, ValidationError

# Validation models for the client events this server understands. Events not
# listed here are ``unsupported_event``. ``response.item.create`` is validated
# by hand: the Responses input-item union is broader than what Live sessions
# accept and a plain dict gives better error messages.
CLIENT_EVENT_MODELS: dict[str, type[BaseModel] | None] = {
    "session.start": SessionStartEvent,
    "session.update": SessionUpdateEvent,
    "session.input_audio.append": InputAudioAppendEvent,
    "session.input_audio.mute": InputAudioMuteEvent,
    "session.input_audio.unmute": InputAudioUnmuteEvent,
    "session.instructions.append": InstructionsAppendEvent,
    "session.thinking.append": ThinkingAppendEvent,
    "session.commentary.append": CommentaryAppendEvent,
    "response.item.create": None,
    "response.create": ResponseCreateEvent,
    "session.close": SessionCloseEvent,
}

# Server-wide non-goals, rejected before the per-model unsupported list.
SERVER_UNSUPPORTED_EVENTS = frozenset({"session.thinking.append"})
# Accepted and dropped without a reply: LiveKit's GPTLiveModel sends this for
# ``generate_reply()``; an error would only add noise to its logs.
SERVER_IGNORED_EVENTS = frozenset({"session.commentary.append"})
# Server-wide non-goal session-config paths (any non-null value is rejected).
SERVER_UNSUPPORTED_SESSION_CONFIGS = ("client", "delegation.client")

OPENAI_BUILT_IN_VOICES = frozenset(get_args(BuiltInVoice))

# Live limits (instructions / initial input) and our append limit.
MAX_INSTRUCTIONS_TOKENS = 16_384
MAX_INPUT_MESSAGES = 128
MAX_INPUT_TOKENS = 8_192
MAX_INSTRUCTIONS_APPEND_TOKENS = 500


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class LiveProtocolError(Exception):
    """A client-visible ``error`` event."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        param: str | None = None,
        error_type: str = "invalid_request_error",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.param = param
        self.error_type = error_type


def unsupported_session_config(param: str) -> LiveProtocolError:
    return LiveProtocolError(
        "unsupported_session_config",
        f"'{param}' is not supported by this server or model.",
        param=param,
    )


def invalid_value(message: str, param: str | None = None) -> LiveProtocolError:
    return LiveProtocolError("invalid_value", message, param=param)


def internal_error() -> LiveProtocolError:
    return LiveProtocolError(
        "internal_server_error",
        "The server had an error while processing the session.",
        error_type="server_error",
    )


def parse_client_json(text: str) -> dict[str, Any]:
    """Decode one client frame into a dict with a string ``type``."""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise LiveProtocolError("invalid_json", f"Client event is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
        raise LiveProtocolError("invalid_event", "Client event must be a JSON object with a string 'type'.")
    event_id = payload.get("event_id")
    if event_id is not None and not isinstance(event_id, str):
        raise LiveProtocolError("invalid_event", "'event_id' must be a string.", param="event_id")
    return payload


def validate_client_event(payload: dict[str, Any]) -> BaseModel | dict[str, Any]:
    """Validate a known client event against its ``openai.types.live`` model."""
    model = CLIENT_EVENT_MODELS[payload["type"]]
    if model is None:
        return payload
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(part) for part in first.get("loc", ()) if not isinstance(part, int))
        raise invalid_value(f"Invalid '{payload['type']}': {first.get('msg', exc)}", param=loc or None) from exc


def dump_event(event: BaseModel) -> str:
    """Serialize a server event; fields left unset are omitted, explicit ``None`` is kept."""
    return event.model_dump_json(exclude_unset=True)


# --------------------------------------------------------------------------- #
# Server events                                                               #
# --------------------------------------------------------------------------- #


def error_event(err: LiveProtocolError, client_event_id: str | None = None) -> ErrorEvent:
    error_kwargs: dict[str, Any] = {"code": err.code, "message": err.message, "type": err.error_type}
    if err.param is not None:
        error_kwargs["param"] = err.param
    if client_event_id is not None:
        error_kwargs["client_event_id"] = client_event_id
    kwargs: dict[str, Any] = {"error": Error(**error_kwargs), "event_id": new_id("evt"), "type": "error"}
    if client_event_id is not None:
        kwargs["client_event_id"] = client_event_id
    return ErrorEvent(**kwargs)


def _with_client_event_id(kwargs: dict[str, Any], client_event_id: str | None) -> dict[str, Any]:
    if client_event_id is not None:
        kwargs["client_event_id"] = client_event_id
    return kwargs


def session_started(session: SessionResource, client_event_id: str | None) -> SessionStartedEvent:
    return SessionStartedEvent(
        **_with_client_event_id(
            {"event_id": new_id("evt"), "session": session, "type": "session.started"}, client_event_id
        )
    )


def session_updated(session: SessionResource, client_event_id: str | None) -> SessionUpdatedEvent:
    return SessionUpdatedEvent(
        **_with_client_event_id(
            {"event_id": new_id("evt"), "session": session, "type": "session.updated"}, client_event_id
        )
    )


def session_closed(
    session: SessionResource, reason: str, usage_seconds: float, client_event_id: str | None
) -> SessionClosedEvent:
    return SessionClosedEvent(
        **_with_client_event_id(
            {
                "event_id": new_id("evt"),
                "reason": reason,
                "session": session,
                "type": "session.closed",
                "usage": SessionUsage(seconds=usage_seconds),
            },
            client_event_id,
        )
    )


def info(code: str, message: str) -> InfoEvent:
    return InfoEvent(code=code, event_id=new_id("evt"), message=message, type="info")


def input_audio_muted(client_event_id: str | None) -> InputAudioMutedEvent:
    return InputAudioMutedEvent(
        **_with_client_event_id({"event_id": new_id("evt"), "type": "session.input_audio.muted"}, client_event_id)
    )


def input_audio_unmuted(client_event_id: str | None) -> InputAudioUnmutedEvent:
    return InputAudioUnmutedEvent(
        **_with_client_event_id({"event_id": new_id("evt"), "type": "session.input_audio.unmuted"}, client_event_id)
    )


def instructions_appended(offset_ms: int, client_event_id: str | None) -> InstructionsAppendedEvent:
    return InstructionsAppendedEvent(
        **_with_client_event_id(
            {
                "event_id": new_id("evt"),
                "start_ms": offset_ms,
                "end_ms": offset_ms,
                "type": "session.instructions.appended",
            },
            client_event_id,
        )
    )


def output_audio_delta(audio_b64: str) -> OutputAudioDeltaEvent:
    return OutputAudioDeltaEvent(delta=audio_b64, type="session.output_audio.delta")


def output_transcript_delta(delta: str, start_ms: int, end_ms: int) -> OutputTranscriptDeltaEvent:
    return OutputTranscriptDeltaEvent(
        delta=delta,
        start_ms=start_ms,
        end_ms=max(start_ms, end_ms),
        event_id=new_id("evt"),
        type="session.output_transcript.delta",
    )


def input_transcript_delta(delta: str, start_ms: int, end_ms: int) -> InputTranscriptDeltaEvent:
    return InputTranscriptDeltaEvent(
        delta=delta,
        start_ms=start_ms,
        end_ms=max(start_ms, end_ms),
        event_id=new_id("evt"),
        type="session.input_transcript.delta",
    )


def usage_updated(usage_seconds: float) -> SessionUsageUpdatedEvent:
    return SessionUsageUpdatedEvent(
        event_id=new_id("evt"), type="session.usage.updated", usage=SessionUsage(seconds=usage_seconds)
    )


def delegation_created(delegation_id: str, response_id: str, offset_ms: int) -> DelegationCreatedEvent:
    return DelegationCreatedEvent(
        delegation=CreatedDelegation(id=delegation_id, target="responses", type="delegation", response_id=response_id),
        event_id=new_id("evt"),
        offset_ms=offset_ms,
        type="session.delegation.created",
    )


def response_event(delegation_id: str, event: dict[str, Any]) -> ResponseEvent:
    return ResponseEvent(event=event, event_id=new_id("evt"), type="response.event", delegation_id=delegation_id)
