# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prometheus metrics for OpenAI Live sessions (``/v1/live/sessions``)."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from vllm_omni.metrics import definitions as defs

_LABELS = [*defs.PIPELINE_LABELS, "vad"]

_active_sessions = Gauge(defs.LIVE_ACTIVE_SESSIONS, "Number of active Live sessions.", labelnames=_LABELS)
_audio_seconds = Counter(
    defs.LIVE_AUDIO_SECONDS,
    "Live session audio, in seconds, by direction (input or output).",
    labelnames=[*_LABELS, "direction"],
)
_pacer_unreleased = Gauge(
    defs.LIVE_PACER_UNRELEASED_S,
    "Generated output audio queued in external-VAD pacers and not yet released, in seconds.",
    labelnames=_LABELS,
)
_asr_pending = Gauge(defs.LIVE_ASR_PENDING_REQUESTS, "In-flight requests to the ASR service.", labelnames=_LABELS)
_asr_latency = Histogram(
    defs.LIVE_ASR_LATENCY_S,
    "ASR service round trip per request, in seconds.",
    labelnames=_LABELS,
    buckets=defs.SECONDS_BUCKETS,
)
_first_audio_latency = Histogram(
    defs.LIVE_FIRST_AUDIO_LATENCY_S,
    "External VAD: end of user speech to the first output audio from the model, in seconds.",
    labelnames=_LABELS,
    buckets=defs.SECONDS_BUCKETS,
)
_interruptions = Counter(defs.LIVE_INTERRUPTIONS, "Server-side barge-in truncations.", labelnames=_LABELS)
_errors = Counter(
    defs.LIVE_ERRORS, "Live sessions ended by an internal error, by reason.", labelnames=[*_LABELS, "reason"]
)


class LiveSessionMetrics:
    """Per-session handle; every method is a no-op when ``log_stats`` is off."""

    def __init__(self, model_name: str, vad: str, log_stats: bool = True) -> None:
        self._log = log_stats
        labels = {"model_name": model_name, "vad": vad}
        self._labels = labels
        self._active = _active_sessions.labels(**labels)
        self._input = _audio_seconds.labels(**labels, direction="input")
        self._output = _audio_seconds.labels(**labels, direction="output")
        self._unreleased = _pacer_unreleased.labels(**labels)
        self._asr_pending = _asr_pending.labels(**labels)
        self._asr_latency = _asr_latency.labels(**labels)
        self._first_audio = _first_audio_latency.labels(**labels)
        self._interruptions = _interruptions.labels(**labels)
        self._unreleased_s = 0.0
        self._started = False

    def session_started(self) -> None:
        if self._log and not self._started:
            self._started = True
            self._active.inc()

    def session_finished(self) -> None:
        if self._log and self._started:
            self._started = False
            self._active.dec()
            self.set_unreleased(0.0)

    def input_audio(self, seconds: float) -> None:
        if self._log:
            self._input.inc(seconds)

    def output_audio(self, seconds: float) -> None:
        if self._log:
            self._output.inc(seconds)

    def set_unreleased(self, seconds: float) -> None:
        """This session's share of the unreleased-audio gauge."""
        if self._log:
            self._unreleased.inc(seconds - self._unreleased_s)
            self._unreleased_s = seconds

    def asr_started(self) -> None:
        if self._log:
            self._asr_pending.inc()

    def asr_finished(self, seconds: float | None) -> None:
        if self._log:
            self._asr_pending.dec()
            if seconds is not None:
                self._asr_latency.observe(seconds)

    def first_audio(self, seconds: float) -> None:
        if self._log:
            self._first_audio.observe(seconds)

    def interruption(self) -> None:
        if self._log:
            self._interruptions.inc()

    def error(self, reason: str) -> None:
        if self._log:
            _errors.labels(**self._labels, reason=reason).inc()
