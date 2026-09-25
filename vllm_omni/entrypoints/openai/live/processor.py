# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``LiveSessionProcessor``: model-specific rendering of the session history.

A processor turns the API server's conversation history into engine prompts
for the session's resumable request, and edits that history when it must be
mutated (interruption truncation, left-trimming). ``GenericChatMLProcessor``
covers external-VAD models whose chat template renders every item.
"""

from __future__ import annotations

import copy
import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from vllm_omni.entrypoints.openai.live.protocol import LiveProtocolError
from vllm_omni.entrypoints.openai.live.session import (
    AssistantItem,
    FunctionCallItem,
    FunctionCallOutputItem,
    HistoryItem,
    LiveSessionState,
    NativeUnitItem,
    SystemItem,
    UserAudioItem,
    UserTextItem,
)


@dataclass
class ProcessorContext:
    """What a processor may use from the serving process."""

    tokenizer: Any  # HF tokenizer, or a processor exposing apply_chat_template
    max_model_len: int
    model_name: str
    tool_call_parser: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def raw_tokenizer(self) -> Any:
        return getattr(self.tokenizer, "tokenizer", self.tokenizer)


@dataclass
class RenderedPrompt:
    """One engine prompt (first chunk or streaming update) and its token cost."""

    prompt: dict[str, Any]
    num_tokens: int


def load_processor_class(path: str) -> type[LiveSessionProcessor]:
    module_name, _, attr = path.partition(":") if ":" in path else path.rpartition(".")
    cls = getattr(importlib.import_module(module_name), attr)
    if not (isinstance(cls, type) and issubclass(cls, LiveSessionProcessor)):
        raise TypeError(f"{path} is not a LiveSessionProcessor")
    return cls


class LiveSessionProcessor(ABC):
    input_sample_rate: ClassVar[int] = 16_000
    output_sample_rate: ClassVar[int] = 24_000

    def __init__(self, context: ProcessorContext) -> None:
        self.context = context

    # ---- session configuration -------------------------------------------------

    def resolve_voice(self, voice: Any) -> str | None:
        """Validate ``audio.output.voice`` and return the model voice id.

        OpenAI built-in names are rejected by the handler before this runs.
        """
        if voice is None:
            return None
        if isinstance(voice, dict):
            voice = voice.get("id")
        if not isinstance(voice, str) or not voice:
            raise LiveProtocolError("invalid_value", "Invalid voice.", param="audio.output.voice")
        return voice

    def count_tokens(self, text: str) -> int:
        return len(self.context.raw_tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        return self.context.raw_tokenizer.decode(token_ids, skip_special_tokens=True)

    def stage0_sampling_params(self, state: LiveSessionState, base: Any) -> Any:
        """Per-append stage-0 sampling params (a clone of ``base``)."""
        return base.clone() if hasattr(base, "clone") else copy.deepcopy(base)


class ExternalVadProcessor(LiveSessionProcessor):
    """Turn-based models: one generation turn per user utterance."""

    # Stop tokens whose re-emission is already the start of the rendered turn
    # suffix. Anything else the scheduler discarded is prepended verbatim.
    supports_tools: ClassVar[bool] = False
    default_tool_call_parser: ClassVar[str | None] = None

    @abstractmethod
    def render_full(self, state: LiveSessionState) -> RenderedPrompt:
        """The whole history plus the assistant generation prompt."""

    @abstractmethod
    def render_append(
        self,
        state: LiveSessionState,
        new_items: list[HistoryItem],
        *,
        last_sampled_token: int | None,
        stopped_on_stop_token: bool,
    ) -> RenderedPrompt:
        """Streaming update after the previous assistant segment.

        The scheduler folds the previous segment into the prompt except its
        last sampled token, so the update re-emits that token (or the
        template's own turn closing when the segment ended on a stop token),
        then ``new_items``, then the assistant generation prompt.
        """

    @abstractmethod
    def estimate_history_tokens(self, state: LiveSessionState) -> int: ...

    def truncate_assistant(self, item: AssistantItem, token_index: int) -> None:
        spoken = item.spoken_token_count if item.spoken_token_count is not None else len(item.token_ids)
        keep = max(0, min(token_index, spoken))
        item.token_ids = item.token_ids[:keep]
        item.spoken_token_count = keep
        item.text = self.decode(item.token_ids)
        item.truncated = True

    def left_trim(self, state: LiveSessionState, target_tokens: int) -> bool:
        """Drop the oldest whole turns until the history fits ``target_tokens``.

        Leading developer context (``instructions`` and ``input`` developer
        messages) is kept. Returns whether anything was dropped.
        """
        history = state.history
        head = 0
        while head < len(history) and isinstance(history[head], SystemItem):
            head += 1
        dropped = False
        while self.estimate_history_tokens(state) > target_tokens and head < len(history):
            # A turn starts at a user item; drop through to the next one.
            end = head + 1
            while end < len(history) and not isinstance(history[end], (UserAudioItem, UserTextItem)):
                end += 1
            del history[head:end]
            dropped = True
        return dropped


class NativeVadProcessor(LiveSessionProcessor):
    """Native full-duplex models: one append per ``audio_buffer_ms`` window."""

    # Submit the next window only after the previous segment finished: the
    # update must re-emit that segment's last sampled token. Models whose
    # stage-0 runtime carries inter-frame state itself can queue windows.
    serialize_windows: ClassVar[bool] = True

    @abstractmethod
    def render_native_turn(
        self,
        state: LiveSessionState,
        unit: NativeUnitItem,
        *,
        first: bool,
        last_sampled_token: int | None,
    ) -> RenderedPrompt:
        """Prompt for one input window; ``first`` also carries the session prefix."""

    @abstractmethod
    def render_native_resume(self, state: LiveSessionState) -> RenderedPrompt:
        """Full resubmission of the (left-trimmed) history as a new request."""

    def native_extra_args(self, state: LiveSessionState, *, resume: bool) -> dict[str, Any]:
        """Per-append ``SamplingParams.extra_args`` for the stage-0 model."""
        return {}

    def observe_stage0_output(self, state: LiveSessionState, token_ids: list[int], output: Any) -> None:
        """Record stage-0 output of the current window into history."""
        for item in reversed(state.history):
            if isinstance(item, NativeUnitItem):
                item.token_ids.extend(token_ids)
                break

    def spoken_text(self, token_ids: list[int], text: str) -> str:
        """The part of a stage-0 delta that is the model's speech transcript."""
        return text

    def observe_multimodal_output(self, state: LiveSessionState, output: Any) -> str:
        """Record per-stream model outputs; returns transcript text, if any."""
        return ""

    def silence(self, duration_ms: int) -> np.ndarray:
        return np.zeros(int(self.input_sample_rate * duration_ms / 1000), dtype=np.float32)

    def left_trim(self, state: LiveSessionState, keep_units: int) -> bool:
        units = [i for i, item in enumerate(state.history) if isinstance(item, NativeUnitItem)]
        if len(units) <= keep_units:
            return False
        drop = set(units[: len(units) - keep_units])
        state.history = [item for i, item in enumerate(state.history) if i not in drop]
        return True


class GenericChatMLProcessor(ExternalVadProcessor):
    """Renders the history through the model's chat template.

    Appends are computed by rendering a placeholder assistant turn followed by
    the new items and keeping the text after the placeholder: that is exactly
    the template's turn closing plus the new turns plus the generation prompt,
    for any template that renders messages in order.
    """

    supports_tools: ClassVar[bool] = True
    # Chat-template content part for one audio clip.
    audio_content_part: ClassVar[dict[str, Any]] = {"type": "audio"}
    # Beyond this many audio turns, older ones re-render as their transcripts
    # (``limit_mm_per_prompt``).
    max_audio_items: ClassVar[int] = 32
    # Rough cost of one second of audio in thinker tokens, for budgeting.
    audio_tokens_per_second: ClassVar[float] = 13.0
    _MARKER = "@@LIVE_SESSION_ASSISTANT_MARKER@@"

    # ---- chat template ---------------------------------------------------------

    def _apply_template(
        self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None, add_generation_prompt: bool
    ) -> str:
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
        if tools:
            kwargs["tools"] = tools
        return self.context.tokenizer.apply_chat_template(messages, **kwargs)

    def _encode(self, text: str) -> list[int]:
        return list(self.context.raw_tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Live function tools (flat) -> chat-template tools (nested)."""
        converted = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description") or "",
                        "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
            )
        return converted

    def template_tools(self, state: LiveSessionState) -> list[dict[str, Any]] | None:
        if state.responses_config.get("tool_choice") == "none":
            return None
        return self.convert_tools(state.tools) or None

    def assistant_text(self, item: AssistantItem) -> str:
        if item.token_ids:
            spoken = item.spoken_token_count if item.spoken_token_count is not None else len(item.token_ids)
            return self.decode(item.token_ids[:spoken])
        return item.text

    def to_messages(
        self, items: list[HistoryItem], *, audio_ok: set[int] | None = None
    ) -> tuple[list[dict[str, Any]], list[tuple[np.ndarray, int]]]:
        """History items -> chat messages plus the audio clips they reference.

        ``audio_ok`` holds the ``id()`` of audio items to render as audio; the
        rest fall back to their transcripts. ``None`` renders every clip.
        """
        messages: list[dict[str, Any]] = []
        audio: list[tuple[np.ndarray, int]] = []
        for item in items:
            if isinstance(item, SystemItem):
                messages.append({"role": "system", "content": item.text})
            elif isinstance(item, UserTextItem):
                messages.append({"role": "user", "content": item.text})
            elif isinstance(item, UserAudioItem):
                if audio_ok is None or id(item) in audio_ok:
                    messages.append({"role": "user", "content": [dict(self.audio_content_part)]})
                    audio.append((item.audio, item.sample_rate))
                else:
                    messages.append({"role": "user", "content": item.transcript or ""})
            elif isinstance(item, AssistantItem):
                messages.append({"role": "assistant", "content": self.assistant_text(item)})
            elif isinstance(item, FunctionCallItem):
                call = {
                    "id": item.call_id,
                    "type": "function",
                    "function": {"name": item.name, "arguments": item.arguments},
                }
                if messages and messages[-1]["role"] == "assistant":
                    messages[-1].setdefault("tool_calls", []).append(call)
                else:
                    # "" not None: chat templates iterate non-string content.
                    messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
            elif isinstance(item, FunctionCallOutputItem):
                messages.append({"role": "tool", "tool_call_id": item.call_id, "content": item.output})
            else:
                raise TypeError(f"{type(self).__name__} cannot render {type(item).__name__}")
        return self._merge_leading_system(messages), audio

    @staticmethod
    def _merge_leading_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Chat templates treat only ``messages[0]`` as the system prompt."""
        merged: list[dict[str, Any]] = []
        for message in messages:
            if (
                message["role"] == "system"
                and merged
                and all(m["role"] == "system" for m in merged)
                and isinstance(merged[-1]["content"], str)
            ):
                merged[-1] = {"role": "system", "content": f"{merged[-1]['content']}\n\n{message['content']}"}
            else:
                merged.append(message)
        return merged

    def _prompt(self, token_ids: list[int], audio: list[tuple[np.ndarray, int]], state: LiveSessionState) -> dict:
        prompt: dict[str, Any] = {"prompt_token_ids": token_ids}
        if audio:
            prompt["multi_modal_data"] = {"audio": audio if len(audio) > 1 else audio[0]}
        info = self.additional_information(state)
        if info:
            prompt["additional_information"] = info
        return prompt

    def additional_information(self, state: LiveSessionState) -> dict[str, Any]:
        return {}

    def audio_token_count(self, num_samples: int, sample_rate: int) -> int:
        return int(num_samples / sample_rate * self.audio_tokens_per_second) + 2

    # ---- rendering -------------------------------------------------------------

    def _audio_items_to_render(self, items: list[HistoryItem]) -> set[int]:
        audio_items = [item for item in items if isinstance(item, UserAudioItem)]
        return {id(item) for item in audio_items[-self.max_audio_items :]}

    def render_full(self, state: LiveSessionState) -> RenderedPrompt:
        audio_ok = self._audio_items_to_render(state.history)
        messages, audio = self.to_messages(state.history, audio_ok=audio_ok)
        text = self._apply_template(messages, tools=self.template_tools(state), add_generation_prompt=True)
        token_ids = self._encode(text)
        num_tokens = len(token_ids) + sum(self.audio_token_count(a.size, sr) for a, sr in audio)
        return RenderedPrompt(self._prompt(token_ids, audio, state), num_tokens)

    def render_append(
        self,
        state: LiveSessionState,
        new_items: list[HistoryItem],
        *,
        last_sampled_token: int | None,
        stopped_on_stop_token: bool,
    ) -> RenderedPrompt:
        new_messages, audio = self.to_messages(new_items)
        scaffold = [
            {"role": "user", "content": "."},
            {"role": "assistant", "content": self._MARKER},
            *new_messages,
        ]
        text = self._apply_template(scaffold, tools=None, add_generation_prompt=True)
        marker_at = text.rfind(self._MARKER)
        if marker_at < 0:
            raise RuntimeError("chat template dropped the assistant marker; cannot compute a streaming append")
        token_ids = self._encode(text[marker_at + len(self._MARKER) :])
        if last_sampled_token is not None and not stopped_on_stop_token:
            # The segment was cut by max_tokens: its last sampled token is
            # content the scheduler dropped; restore it before the closing.
            token_ids = [last_sampled_token, *token_ids]
        num_tokens = len(token_ids) + sum(self.audio_token_count(a.size, sr) for a, sr in audio)
        return RenderedPrompt(self._prompt(token_ids, audio, state), num_tokens)

    def estimate_history_tokens(self, state: LiveSessionState) -> int:
        audio_ok = self._audio_items_to_render(state.history)
        total = 0
        for item in state.history:
            if isinstance(item, UserAudioItem):
                if id(item) in audio_ok:
                    total += self.audio_token_count(item.audio.size, item.sample_rate) + 6
                else:
                    total += self.count_tokens(item.transcript or "") + 6
            elif isinstance(item, AssistantItem):
                total += (len(item.token_ids) if item.token_ids else self.count_tokens(item.text)) + 6
            elif isinstance(item, (SystemItem, UserTextItem)):
                total += self.count_tokens(item.text) + 6
            elif isinstance(item, FunctionCallItem):
                total += self.count_tokens(item.name + item.arguments) + 16
            elif isinstance(item, FunctionCallOutputItem):
                total += self.count_tokens(item.output) + 12
        tools = self.template_tools(state)
        if tools:
            total += self.count_tokens(str(tools)) + 64
        return total

    def stage0_sampling_params(self, state: LiveSessionState, base: Any) -> Any:
        params = super().stage0_sampling_params(state, base)
        max_output_tokens = state.responses_config.get("max_output_tokens")
        if isinstance(max_output_tokens, int) and max_output_tokens > 0:
            params.max_tokens = max_output_tokens
        return params
