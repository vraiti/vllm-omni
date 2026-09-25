# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-time output pacing for external-VAD sessions.

Turn-based models generate audio faster than real time. The pacer queues it
and releases ``output_audio_delta_size_ms`` chunks at 1x against the session
timeline, one chunk ahead of the playhead so the client never starves. The
released position is the server's playback cursor: an interruption discards
everything not yet released and truncates the turn at that cursor.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PacedTurn:
    turn_id: int
    sample_rate: int
    queued: list[np.ndarray] = field(default_factory=list)
    queued_samples: int = 0
    released_samples: int = 0
    # Session-timeline position of the turn's first released sample.
    release_start_ms: float | None = None
    generation_done: bool = False
    discarded: bool = False
    final_released: bool = False

    @property
    def released_ms(self) -> float:
        return self.released_samples * 1000.0 / self.sample_rate

    @property
    def complete(self) -> bool:
        return self.discarded or (self.generation_done and self.queued_samples == 0)


# (turn, audio, turn-relative start ms, turn-relative end ms, last chunk of turn)
ReleaseCallback = Callable[[PacedTurn, np.ndarray, float, float, bool], Awaitable[None]]


class OutputPacer:
    def __init__(self, chunk_ms: int, on_release: ReleaseCallback) -> None:
        self.chunk_ms = chunk_ms
        self._on_release = on_release
        self._turns: deque[PacedTurn] = deque()
        self._lock = asyncio.Lock()
        # Session-timeline end of everything released so far.
        self.cursor_ms = 0.0

    def open_turn(self, turn_id: int, sample_rate: int) -> PacedTurn:
        turn = PacedTurn(turn_id=turn_id, sample_rate=sample_rate)
        self._turns.append(turn)
        return turn

    def push(self, turn: PacedTurn, samples: np.ndarray) -> None:
        if turn.discarded or samples.size == 0:
            return
        turn.queued.append(np.asarray(samples, dtype=np.float32))
        turn.queued_samples += samples.size

    def finish(self, turn: PacedTurn) -> None:
        turn.generation_done = True

    @property
    def has_unreleased(self) -> bool:
        return any(turn.queued_samples > 0 for turn in self._turns)

    @property
    def unreleased_ms(self) -> float:
        return sum(turn.queued_samples * 1000.0 / turn.sample_rate for turn in self._turns)

    def discard_unreleased(self) -> None:
        for turn in self._turns:
            turn.queued.clear()
            turn.queued_samples = 0
            turn.discarded = True
        self._turns.clear()

    def _take(self, turn: PacedTurn, num_samples: int) -> np.ndarray:
        joined = np.concatenate(turn.queued) if len(turn.queued) > 1 else turn.queued[0]
        chunk, rest = joined[:num_samples], joined[num_samples:]
        turn.queued = [rest] if rest.size else []
        turn.queued_samples = rest.size
        return chunk

    async def tick(self, now_ms: float) -> None:
        """Release every chunk that is due at session time ``now_ms``."""
        async with self._lock:
            await self._release_due(now_ms)

    async def _release_due(self, now_ms: float) -> None:
        while self._turns:
            turn = self._turns[0]
            if turn.complete:
                self._turns.popleft()
                if not turn.discarded and not turn.final_released and turn.release_start_ms is not None:
                    # Generation ended exactly on a chunk boundary.
                    turn.final_released = True
                    await self._on_release(
                        turn, np.empty(0, dtype=np.float32), turn.released_ms, turn.released_ms, True
                    )
                continue
            chunk_samples = max(1, int(turn.sample_rate * self.chunk_ms / 1000))
            ready = turn.queued_samples >= chunk_samples or (turn.generation_done and turn.queued_samples > 0)
            if not ready:
                return
            if turn.release_start_ms is None:
                turn.release_start_ms = max(now_ms, self.cursor_ms)
            # Chunk k is due at release_start + k * chunk_ms.
            due_ms = turn.release_start_ms + turn.released_ms
            if now_ms + 1e-6 < due_ms:
                return
            chunk = self._take(turn, chunk_samples)
            start_ms = turn.released_ms
            turn.released_samples += chunk.size
            end_ms = turn.released_ms
            self.cursor_ms = turn.release_start_ms + end_ms
            turn.final_released = turn.complete
            await self._on_release(turn, chunk, start_ms, end_ms, turn.final_released)
