# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tool calls on ``delegation.responses`` sessions.

The served model is its own Responses backend: its stage-0 text is parsed
with vLLM's tool parser for the model, and each tool call becomes a
``session.delegation.created`` plus a nested Responses event stream inside
``response.event``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest, ChatCompletionToolsParam
from vllm.logger import init_logger
from vllm.tool_parsers import ToolParserManager

from vllm_omni.entrypoints.openai.live.processor import GenericChatMLProcessor
from vllm_omni.entrypoints.openai.live.protocol import new_id

logger = init_logger(__name__)


@dataclass(frozen=True)
class ParsedToolCall:
    call_id: str
    item_id: str
    name: str
    arguments: str


class ToolCallExtractor:
    """Finds tool-call markup in one assistant turn's raw text."""

    def __init__(self, parser_name: str, tokenizer: Any, live_tools: list[dict[str, Any]]) -> None:
        tools = [ChatCompletionToolsParam(**tool) for tool in GenericChatMLProcessor.convert_tools(live_tools)]
        parser_cls = ToolParserManager.get_tool_parser(parser_name)
        self._parser = parser_cls(tokenizer, tools=tools)
        self._request = ChatCompletionRequest(messages=[], tools=tools, skip_special_tokens=False)
        # Marker that starts tool-call markup, for cutting speech early.
        self.start_token: str | None = getattr(self._parser, "tool_call_start_token", None)

    def markup_started(self, text: str) -> bool:
        return self.start_token is not None and self.start_token in text

    def extract(self, text: str) -> list[ParsedToolCall]:
        info = self._parser.extract_tool_calls(text, self._request)
        if not info.tools_called:
            return []
        calls = []
        for call in info.tool_calls:
            calls.append(
                ParsedToolCall(
                    call_id=getattr(call, "id", None) or new_id("call"),
                    item_id=new_id("fc"),
                    name=call.function.name,
                    arguments=call.function.arguments or "{}",
                )
            )
        return calls


class ResponsesEventStream:
    """Builds the nested Responses events of one delegation."""

    def __init__(self, response_id: str, model: str) -> None:
        self.response_id = response_id
        self.model = model
        self.created_at = int(time.time())
        self._sequence = 0

    def _event(self, event_type: str, **fields: Any) -> dict[str, Any]:
        event = {"type": event_type, "sequence_number": self._sequence, **fields}
        self._sequence += 1
        return event

    def _snapshot(self, status: str, usage: dict[str, int] | None = None) -> dict[str, Any]:
        # Lifecycle snapshots omit instructions, tools, input, and output.
        snapshot: dict[str, Any] = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
        }
        if usage is not None:
            snapshot["usage"] = usage
        return snapshot

    def function_call_events(
        self, calls: list[ParsedToolCall], *, input_tokens: int, output_tokens: int
    ) -> list[dict[str, Any]]:
        events = [self._event("response.created", response=self._snapshot("in_progress"))]
        for index, call in enumerate(calls):
            item = {
                "type": "function_call",
                "id": call.item_id,
                "call_id": call.call_id,
                "name": call.name,
                "arguments": "",
                "status": "in_progress",
            }
            events.append(self._event("response.output_item.added", output_index=index, item=item))
            events.append(
                self._event(
                    "response.function_call_arguments.delta",
                    item_id=call.item_id,
                    output_index=index,
                    delta=call.arguments,
                )
            )
            events.append(
                self._event(
                    "response.function_call_arguments.done",
                    item_id=call.item_id,
                    output_index=index,
                    name=call.name,
                    arguments=call.arguments,
                )
            )
            events.append(
                self._event(
                    "response.output_item.done",
                    output_index=index,
                    item={**item, "arguments": call.arguments, "status": "completed"},
                )
            )
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        events.append(self._event("response.completed", response=self._snapshot("completed", usage)))
        return events
