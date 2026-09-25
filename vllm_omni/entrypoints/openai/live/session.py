# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-session state and the conversation history owned by the API server."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from openai.types.live import SessionResource

from vllm_omni.entrypoints.openai.live.audio import LiveAudioFormat
from vllm_omni.entrypoints.openai.live.protocol import new_id

# --------------------------------------------------------------------------- #
# History items                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class SystemItem:
    """Developer context: ``instructions``, ``input`` developer messages, and
    ``session.instructions.append`` text."""

    text: str


@dataclass
class UserTextItem:
    text: str


@dataclass
class UserAudioItem:
    """One user speech segment at the model's input rate."""

    audio: np.ndarray
    sample_rate: int
    start_ms: int
    end_ms: int
    transcript: str | None = None


@dataclass
class AssistantItem:
    """One assistant turn.

    ``token_ids`` is the model's raw stage-0 output (what the resumable
    request actually contains); ``text`` is the spoken/visible text. History
    supplied as text (``input``) has no token ids.
    """

    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    # Tool-call markup begins at this token index; later tokens are not spoken.
    spoken_token_count: int | None = None
    truncated: bool = False


@dataclass
class FunctionCallItem:
    call_id: str
    name: str
    arguments: str


@dataclass
class FunctionCallOutputItem:
    call_id: str
    output: str


@dataclass
class NativeUnitItem:
    """One native-VAD generation turn (one ``audio_buffer_ms`` window).

    ``streams`` holds per-modality model outputs for processors that resubmit
    through ``multi_modal_data`` (PersonaPlex); ``token_ids`` holds stage-0
    output for token-stream models (MiniCPM-o).
    """

    audio: np.ndarray
    sample_rate: int
    start_ms: int
    end_ms: int
    token_ids: list[int] = field(default_factory=list)
    streams: dict[str, Any] = field(default_factory=dict)


HistoryItem = (
    SystemItem
    | UserTextItem
    | UserAudioItem
    | AssistantItem
    | FunctionCallItem
    | FunctionCallOutputItem
    | NativeUnitItem
)


# --------------------------------------------------------------------------- #
# Delegation                                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class OpenDelegation:
    """A ``session.delegation.created`` awaiting tool results."""

    delegation_id: str
    response_id: str
    call_ids: list[str] = field(default_factory=list)
    outputs: set[str] = field(default_factory=set)
    completed: bool = False

    @property
    def all_outputs_received(self) -> bool:
        return bool(self.call_ids) and set(self.call_ids) <= self.outputs


# --------------------------------------------------------------------------- #
# Session                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class LiveSessionState:
    model: str
    audio_format: LiveAudioFormat
    expires_at: int
    voice: Any = None  # as supplied by the client, echoed in the resource
    instructions: str | None = None
    input: list[dict[str, Any]] | None = None
    delegation: dict[str, Any] | None = None
    id: str = field(default_factory=lambda: new_id("sess"))
    started_monotonic: float = field(default_factory=time.monotonic)

    # Timeline and usage.
    session_ms: float = 0.0
    input_audio_ms: float = 0.0
    output_audio_ms: float = 0.0
    muted: bool = False

    # Conversation history as rendered into the model, plus items accepted
    # since the last generation turn (they ride on the next append).
    history: list[HistoryItem] = field(default_factory=list)
    pending_items: list[HistoryItem] = field(default_factory=list)

    delegation_state: OpenDelegation | None = None

    @property
    def now_ms(self) -> int:
        return int(self.session_ms)

    @property
    def usage_seconds(self) -> float:
        return round((self.input_audio_ms + self.output_audio_ms) / 1000.0, 3)

    @property
    def has_responses_delegation(self) -> bool:
        return isinstance(self.delegation, dict) and self.delegation.get("type") == "responses"

    @property
    def responses_config(self) -> dict[str, Any]:
        if not self.has_responses_delegation:
            return {}
        return self.delegation.get("responses") or {}

    @property
    def tools(self) -> list[dict[str, Any]]:
        return [tool for tool in (self.responses_config.get("tools") or []) if tool.get("type") == "function"]

    def take_pending(self) -> list[HistoryItem]:
        items, self.pending_items = self.pending_items, []
        return items

    def to_resource(self) -> SessionResource:
        audio: dict[str, Any] = {"format": self.audio_format.to_dict()}
        if self.voice is not None:
            audio["output"] = {"voice": self.voice}
        return SessionResource(
            id=self.id,
            expires_at=self.expires_at,
            model=self.model,
            status="active",
            audio=audio,
            client=None,
            delegation=self.delegation,
            input=self.input,
            instructions=self.instructions,
            store=False,
        )
