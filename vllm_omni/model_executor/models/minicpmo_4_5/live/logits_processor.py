# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniCPM-o 4.5 duplex decoding rules for OpenAI Live sessions.

Mirrors the official ``MiniCPMODuplex.streaming_generate`` loop (explicit
listen/speak mode) on a vLLM resumable request, where each streaming update is
one ``<unit>`` of audio and each segment is the model's output for it:

* the unit ends on a chunk terminator (``<|listen|>``, ``<|chunk_eos|>``,
  ``<|chunk_tts_eos|>``), which the Live processor also passes as
  ``stop_token_ids``;
* ``<|listen|>`` is only allowed once the current turn has ended
  (``<|turn_eos|>``); before that its probability goes to ``<|tts_bos|>``,
  where the official loop replaces a sampled listen with ``<|tts_bos|>``;
* the unit's last allowed token is forced to ``<|chunk_eos|>``;
* forbidden tokens (``<|tts_pad|>`` and the tokenizer's bad tokens) never
  appear.

Opt-in per request through ``SamplingParams.extra_args["minicpmo_live"]``,
which carries the token ids and ``turn_ended`` at the start of the segment
(the Live processor tracks it across segments). Requests without it are
untouched, so chat requests on the same engine are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates

_REQUIRED_KEYS = ("listen", "tts_bos", "chunk_eos", "turn_eos", "max_unit_tokens", "turn_ended")


@dataclass
class _UnitState:
    listen: int
    tts_bos: int
    chunk_eos: int
    turn_eos: int
    chunk_tts_eos: int | None
    max_unit_tokens: int
    forbidden: tuple[int, ...]
    turn_ended: bool
    output_token_ids: list[int]
    seen: int = 0

    def advance(self) -> None:
        tokens = self.output_token_ids
        for token in tokens[self.seen :]:
            if token == self.turn_eos:
                self.turn_ended = True
            elif token not in (self.listen, self.chunk_eos, self.chunk_tts_eos):
                # Any spoken token re-opens the turn (official "normal speak").
                self.turn_ended = False
        self.seen = len(tokens)


class MiniCPMODuplexLogitsProcessor(LogitsProcessor):
    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        spec = (sampling_params.extra_args or {}).get("minicpmo_live")
        if spec is None:
            return
        missing = [key for key in _REQUIRED_KEYS if key not in spec]
        if missing:
            raise ValueError(f"extra_args.minicpmo_live is missing {missing}")
        if int(spec["max_unit_tokens"]) < 1:
            raise ValueError("extra_args.minicpmo_live.max_unit_tokens must be >= 1")

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool) -> None:
        self.device = device
        self._states: dict[int, _UnitState] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    @staticmethod
    def _new_state(params: SamplingParams, _prompt: list[int] | None, output: list[int]) -> _UnitState | None:
        spec = (params.extra_args or {}).get("minicpmo_live") if params is not None else None
        if not spec:
            return None
        return _UnitState(
            listen=int(spec["listen"]),
            tts_bos=int(spec["tts_bos"]),
            chunk_eos=int(spec["chunk_eos"]),
            turn_eos=int(spec["turn_eos"]),
            chunk_tts_eos=int(spec["chunk_tts_eos"]) if spec.get("chunk_tts_eos") is not None else None,
            max_unit_tokens=int(spec["max_unit_tokens"]),
            forbidden=tuple(int(t) for t in spec.get("forbidden", ())),
            turn_ended=bool(spec["turn_ended"]),
            output_token_ids=output,
        )

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        process_dict_updates(self._states, batch_update, self._new_state)
        for state in self._states.values():
            state.advance()

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._states:
            return logits
        neg_inf = float("-inf")
        for row, state in self._states.items():
            if row >= logits.shape[0]:
                continue
            if len(state.output_token_ids) >= state.max_unit_tokens - 1:
                # Official loop: the last slot of a unit is always <|chunk_eos|>.
                keep = logits[row, state.chunk_eos].clone()
                logits[row] = neg_inf
                logits[row, state.chunk_eos] = keep if torch.isfinite(keep) else 0.0
                continue
            if state.forbidden:
                logits[row, list(state.forbidden)] = neg_inf
            if not state.turn_ended:
                # A listen mid-turn becomes <|tts_bos|>: move its probability mass.
                logits[row, state.tts_bos] = torch.logaddexp(logits[row, state.tts_bos], logits[row, state.listen])
                logits[row, state.listen] = neg_inf
        return logits
