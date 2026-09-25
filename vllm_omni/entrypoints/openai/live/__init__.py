# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OpenAI Live API (``/v1/live/sessions``) for vLLM-Omni.

One ``LiveSessionHandler`` per WebSocket drives one resumable request per
session through ``AsyncOmni``. Turn-based models are driven by an external
VAD service (``vad: external``); native full-duplex models decide themselves
when to listen and speak (``vad: native``). See ``docs/serving/live_sessions_api.md``.
"""
