# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PersonaPlex ``LiveSessionProcessor`` (native VAD, 80 ms frames).

Each ``audio_buffer_ms`` window is one 1920-sample frame at 24 kHz, appended
to the resumable request as a runner data-plane payload
(``model_intermediate_buffer["duplex"]``) that
``PersonaPlexStage0DuplexRuntime`` consumes: it Mimi-encodes the frame with a
per-session streaming encoder, builds the delayed 17-row frame embedding,
prefills the voice prompt and persona on the first frame, and teacher-forces
the depformer. The talker samples one text token per frame
(``max_tokens=1``), the depformer emits the agent codes, and Code2Wav decodes
them. The prompt tokens are placeholders that only reserve scheduler slots.

Context is owned by the stage-0 runtime (keyed by session and incarnation).
A resubmission after the context is exhausted starts a new incarnation: the
voice prompt and persona are prefilled again and the conversation so far is
carried as a text summary in the persona.
"""

from __future__ import annotations

import base64
import uuid
from pathlib import PurePath
from typing import Any, ClassVar

import numpy as np

from vllm_omni.entrypoints.openai.live.processor import NativeVadProcessor, RenderedPrompt
from vllm_omni.entrypoints.openai.live.protocol import LiveProtocolError
from vllm_omni.entrypoints.openai.live.session import (
    AssistantItem,
    LiveSessionState,
    NativeUnitItem,
    SystemItem,
    UserTextItem,
)
from vllm_omni.model_executor.models.personaplex.duplex.config import DEFAULT_PERSONA

_FRAME_SAMPLES = 1920
_DEFAULT_VOICE = "NATF2.pt"


class PersonaPlexLiveSessionProcessor(NativeVadProcessor):
    input_sample_rate: ClassVar[int] = 24_000
    output_sample_rate: ClassVar[int] = 24_000
    # The stage-0 runtime carries the sampled text/agent codes between frames
    # itself, so frames can be queued without waiting for the previous one.
    serialize_windows: ClassVar[bool] = False

    def __init__(self, context) -> None:
        super().__init__(context)
        self.model_path = str(context.extra.get("model_path") or "")
        self.voice_file = _DEFAULT_VOICE
        self.session_key = f"live-{uuid.uuid4().hex}"
        self.incarnation = 0
        self.seq = 0
        self._persona: str | None = None
        self._prefill_slots: int | None = None
        self._spm = None

    # ---- configuration -----------------------------------------------------------

    def resolve_voice(self, voice: Any) -> str | None:
        voice_id = super().resolve_voice(voice)
        if voice_id is None:
            return None
        name = voice_id if voice_id.endswith(".pt") else f"{voice_id}.pt"
        path = PurePath(name)
        if path.name != name or any(part == ".." for part in path.parts):
            raise LiveProtocolError("invalid_value", "Invalid PersonaPlex voice.", param="audio.output.voice")
        try:
            from vllm_omni.model_executor.models.personaplex.duplex.stage0 import load_personaplex_voice_state

            load_personaplex_voice_state(self.model_path, name)
        except (FileNotFoundError, ValueError) as exc:
            raise LiveProtocolError(
                "invalid_value", f"Unknown PersonaPlex voice '{voice_id}'.", param="audio.output.voice"
            ) from exc
        self.voice_file = name
        return voice_id

    def count_tokens(self, text: str) -> int:
        if self._spm is None:
            from vllm_omni.model_executor.models.personaplex.duplex.stage0 import load_personaplex_tokenizer

            self._spm = load_personaplex_tokenizer(self.model_path)
        return len(self._spm(text))

    def stage0_sampling_params(self, state: LiveSessionState, base: Any) -> Any:
        params = super().stage0_sampling_params(state, base)
        # One temporal frame per append.
        params.max_tokens = 1
        return params

    # ---- rendering -------------------------------------------------------------

    def persona(self, state: LiveSessionState) -> str:
        developer = [item.text for item in state.history if isinstance(item, SystemItem)]
        history = []
        for item in state.history:
            if isinstance(item, UserTextItem):
                history.append(f"User: {item.text}")
            elif isinstance(item, AssistantItem):
                history.append(f"You: {item.text}")
        persona = " ".join(developer) or DEFAULT_PERSONA
        if history:
            persona += " Conversation so far: " + " ".join(history)
        return persona

    def _prefill(self, persona: str) -> int:
        if self._prefill_slots is None or persona != self._persona:
            from vllm_omni.model_executor.models.personaplex.duplex.stage0 import personaplex_prefill_slots

            self._prefill_slots = personaplex_prefill_slots(self.model_path, self.voice_file, persona)
            self._persona = persona
        return self._prefill_slots

    def _frame_prompt(self, state: LiveSessionState, unit: NativeUnitItem, *, first: bool) -> RenderedPrompt:
        audio = np.ascontiguousarray(unit.audio, dtype="<f4")
        if audio.size != _FRAME_SAMPLES:
            audio = np.pad(audio[:_FRAME_SAMPLES], (0, max(0, _FRAME_SAMPLES - audio.size)))
        if first:
            self.incarnation += 1
            self.seq = 0
        self.seq += 1
        persona = self.persona(state)
        budget = self._prefill(persona) + 1 if first else 1
        duplex = {
            "session_id": self.session_key,
            "incarnation": self.incarnation,
            "epoch": 0,
            "seq": self.seq,
            "mode": "append_audio_chunk",
            "data_plane": True,
            "final": False,
            "payload": {
                "format": "pcm_f32le",
                "sample_rate_hz": self.input_sample_rate,
                "audio": base64.b64encode(audio.astype("<f4").tobytes()).decode("ascii"),
            },
            "runtime_config": {
                "personaplex_voice_prompt": self.voice_file,
                "personaplex_persona": persona,
            },
        }
        prompt = {"prompt_token_ids": [0] * budget, "model_intermediate_buffer": {"duplex": duplex}}
        return RenderedPrompt(prompt, budget + 1)

    def render_native_turn(
        self,
        state: LiveSessionState,
        unit: NativeUnitItem,
        *,
        first: bool,
        last_sampled_token: int | None,
    ) -> RenderedPrompt:
        return self._frame_prompt(state, unit, first=first)

    def render_native_resume(self, state: LiveSessionState) -> RenderedPrompt:
        units = [item for item in state.history if isinstance(item, NativeUnitItem)]
        return self._frame_prompt(state, units[-1], first=True)

    def left_trim(self, state: LiveSessionState, keep_units: int) -> bool:
        # Only the newest frame is resubmitted; the stage-0 runtime restarts
        # from the voice/persona prefill.
        return super().left_trim(state, 1)

    def spoken_text(self, token_ids: list[int], text: str) -> str:
        # Stage 0 is not a final-output stage; nothing reaches the frontend.
        return ""
