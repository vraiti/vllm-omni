# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Output token <-> audio timeline mapping for external-VAD turns.

The turn's audio is transcribed by the ASR service, which returns segments
with timestamps. The ASR text is aligned against the model's own output text
(normalized characters, so the LLM -> TTS -> ASR round trip may differ in
punctuation, casing, numerals, and the odd word). Every aligned character of
the model text gets a time interpolated inside its ASR segment; a token's
time is the time of its last aligned character. Tokens the alignment cannot
place inherit the time of the next placed token, so truncation never keeps a
token that precedes nothing the user is known to have heard; trailing
unplaced tokens are never considered heard.
"""

from __future__ import annotations

import bisect
import difflib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

NEVER = math.inf


@dataclass(frozen=True)
class AsrSegment:
    start_s: float
    end_s: float
    text: str


def _normalize(text: str) -> tuple[str, list[int]]:
    """Lowercased alphanumerics plus the index of each in ``text``."""
    chars: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(text):
        if ch.isalnum():
            chars.append(ch.lower())
            index.append(i)
    return "".join(chars), index


class TokenTimeline:
    """Per-token end times (ms, relative to the start of the turn's audio)."""

    def __init__(
        self,
        token_ids: Sequence[int],
        segments: Sequence[AsrSegment],
        decode: Callable[[list[int]], str],
    ) -> None:
        self.token_ids = list(token_ids)
        self._decode = decode
        # Character span of each token in the decoded model text.
        token_ends: list[int] = []
        for i in range(len(self.token_ids)):
            token_ends.append(len(decode(self.token_ids[: i + 1])))
        model_text = decode(self.token_ids)

        # ASR characters with interpolated times.
        asr_chars: list[str] = []
        asr_times: list[float] = []
        for seg in segments:
            norm, _ = _normalize(seg.text)
            if not norm:
                continue
            span_ms = max(0.0, (seg.end_s - seg.start_s) * 1000.0)
            for j, ch in enumerate(norm):
                asr_chars.append(ch)
                asr_times.append(seg.start_s * 1000.0 + span_ms * (j + 1) / len(norm))

        model_norm, model_index = _normalize(model_text)
        char_time: dict[int, float] = {}  # model_text index -> ms
        if model_norm and asr_chars:
            matcher = difflib.SequenceMatcher(None, model_norm, "".join(asr_chars), autojunk=False)
            for block in matcher.get_matching_blocks():
                for k in range(block.size):
                    char_time[model_index[block.a + k]] = asr_times[block.b + k]

        times: list[float] = []
        start = 0
        for end in token_ends:
            placed = [char_time[c] for c in range(start, end) if c in char_time]
            times.append(max(placed) if placed else NEVER)
            start = end
        # Unplaced tokens inherit the next placed token's time; enforce monotonicity.
        next_time = NEVER
        for i in range(len(times) - 1, -1, -1):
            if times[i] == NEVER:
                times[i] = next_time
            else:
                next_time = min(times[i], next_time)
                times[i] = next_time
        self.times_ms = times

    def token_index_at(self, cursor_ms: float) -> int:
        """Number of leading tokens whose audio ended at or before ``cursor_ms``."""
        return bisect.bisect_right(self.times_ms, cursor_ms)

    def text_between(self, i: int, j: int) -> str:
        if j <= i:
            return ""
        return self._decode(self.token_ids[:j])[len(self._decode(self.token_ids[:i])) :]

    def token_start_ms(self, i: int) -> float:
        """Start of token ``i``: the end of the previous token (0 for the first)."""
        if i <= 0:
            return 0.0
        prev = self.times_ms[i - 1]
        return 0.0 if prev == NEVER else prev
