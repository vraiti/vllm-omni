# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configuration for OpenAI Live sessions (``/v1/live/sessions``).

Two halves: ``LiveSessionConfig`` is a ``PipelineConfig`` attribute that
describes what a model can do on a Live session; ``LiveSessionDeployConfig``
is the optional ``live_session_config`` deploy-YAML object that enables the
endpoint for a deployment and points it at the external VAD and ASR services.

Kept free of engine and API-server imports: ``stage_config`` imports it, and
pipeline modules construct ``LiveSessionConfig`` at import time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Literal

LIVE_VAD_TYPES = ("external", "native")


@dataclass(frozen=True)
class UnsupportedFeatures:
    """Live client events and session-config paths a model rejects.

    ``events`` are Live client event types (``session.instructions.append``);
    ``session_configs`` are dotted paths into ``session.start`` /
    ``session.update`` (``delegation.responses.tools``). The server-wide
    non-goals are rejected before this list is consulted and need not be
    listed.
    """

    events: tuple[str, ...] = ()
    session_configs: tuple[str, ...] = ()


@dataclass(frozen=True)
class LiveSessionConfig:
    """Per-model capabilities for ``/v1/live/sessions``."""

    vad: Literal["external", "native"]
    # Dotted path (``module:Class`` or ``module.Class``) of the model's
    # ``LiveSessionProcessor``; resolved lazily by the API server so pipeline
    # modules stay import-light.
    live_session_processor: str
    # Native VAD only: the API server submits one placeholder token per input
    # frame and carries the real inputs in ``multi_modal_data``.
    multi_modal_data_bypass: bool = False
    # Native VAD only: user audio buffered per generation turn.
    audio_buffer_ms: int | None = None
    # Stage id -> ``module:Class`` of a vLLM ``LogitsProcessor``, appended to
    # that stage's ``logits_processors`` engine arg on Live deployments.
    logits_processor: Mapping[int, str] = field(default_factory=dict)
    unsupported_features: UnsupportedFeatures = field(default_factory=UnsupportedFeatures)

    def get_validation_errors(self) -> list[str]:
        errors: list[str] = []
        if self.vad not in LIVE_VAD_TYPES:
            errors.append(f"live_session_config.vad must be one of {LIVE_VAD_TYPES}, got {self.vad!r}")
        if not self.live_session_processor:
            errors.append("live_session_config.live_session_processor is required")
        if self.vad == "native":
            if self.audio_buffer_ms is None or self.audio_buffer_ms <= 0:
                errors.append("live_session_config.audio_buffer_ms must be positive for vad: native")
        else:
            if self.audio_buffer_ms is not None:
                errors.append("live_session_config.audio_buffer_ms is only valid for vad: native")
            if self.multi_modal_data_bypass:
                errors.append("live_session_config.multi_modal_data_bypass is only valid for vad: native")
        for stage_id, fqcn in self.logits_processor.items():
            if not isinstance(stage_id, int) or stage_id < 0:
                errors.append(f"live_session_config.logits_processor key {stage_id!r} must be a stage id")
            if ":" not in fqcn:
                errors.append(f"live_session_config.logits_processor[{stage_id}] must be 'module:Class', got {fqcn!r}")
        return errors


@dataclass(frozen=True)
class ExternalVadDeployConfig:
    # Pacer release granularity and therefore the playback-cursor precision.
    output_audio_delta_size_ms: int = 500
    vad_service_url: str = "ws://localhost:15151"


@dataclass(frozen=True)
class NativeVadDeployConfig:
    # Window of unmuted user audio per input-transcription request.
    audio_transcription_interval_ms: int = 10000


def _build_section(cls: type, value: Any, name: str) -> Any:
    if value is None:
        return cls()
    if not isinstance(value, Mapping):
        raise ValueError(f"live_session_config.{name} must be a mapping")
    known = {f.name for f in fields(cls)}
    unknown = set(value) - known
    if unknown:
        raise ValueError(f"Unknown live_session_config.{name} keys: {sorted(unknown)}")
    return cls(**value)


@dataclass(frozen=True)
class LiveSessionDeployConfig:
    """The ``live_session_config`` deploy-YAML object.

    Its presence enables ``/v1/live/sessions`` for the deployment.
    """

    # Wall-clock limit on a session's total duration; fills ``expires_at``.
    session_lifetime_s: int = 1800
    external_vad: ExternalVadDeployConfig = field(default_factory=ExternalVadDeployConfig)
    native_vad: NativeVadDeployConfig = field(default_factory=NativeVadDeployConfig)
    asr_service_url: str = "http://localhost:15152"
    # Per-request timeout for the ASR service.
    asr_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if self.session_lifetime_s <= 0:
            raise ValueError("live_session_config.session_lifetime_s must be positive")
        if self.external_vad.output_audio_delta_size_ms <= 0:
            raise ValueError("live_session_config.external_vad.output_audio_delta_size_ms must be positive")
        if self.native_vad.audio_transcription_interval_ms <= 0:
            raise ValueError("live_session_config.native_vad.audio_transcription_interval_ms must be positive")
        if self.asr_timeout_s <= 0:
            raise ValueError("live_session_config.asr_timeout_s must be positive")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> LiveSessionDeployConfig:
        raw = dict(raw or {})
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown live_session_config keys: {sorted(unknown)}")
        raw["external_vad"] = _build_section(ExternalVadDeployConfig, raw.get("external_vad"), "external_vad")
        raw["native_vad"] = _build_section(NativeVadDeployConfig, raw.get("native_vad"), "native_vad")
        return cls(**raw)
