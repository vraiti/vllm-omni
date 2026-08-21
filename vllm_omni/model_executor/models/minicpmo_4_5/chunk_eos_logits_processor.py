# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from MiniCPM-o-Demo/MiniCPMO45/modeling_minicpmo_unified.py's
# streaming_generate loop: `for j in range(max_new_speak_tokens_per_chunk):
# if j == max_new_speak_tokens_per_chunk - 1: ... force chunk_eos`. That
# reference loop restarts fresh for every ~1s input audio chunk; here the
# same cap is expressed as "at most N-1 freely-sampled output tokens since
# the last <|chunk_eos|> token", since our persistent-streaming request has
# no discrete per-chunk decode loop to restart -- new audio chunks are
# APPENDed into the same request without ever adding a batch row, so
# <|chunk_eos|>'s own position in the live output-token list is the only
# per-request signal this logits processor has to key off of.
"""Force MiniCPM-o's per-chunk `<|chunk_eos|>` cap during persistent
streaming generation, driven per request by
``SamplingParams.extra_args["minicpmo_chunk_eos"]``:

    {
        "chunk_eos_token_id": <int>,
        "max_new_tokens_per_chunk": <int, default 20>,
    }
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

NEG_INF = float("-inf")


class MiniCPMOChunkEosForceLogitsProcessor(LogitsProcessor):
    """Force `<|chunk_eos|>` once a request has generated
    ``max_new_tokens_per_chunk - 1`` tokens since the last `<|chunk_eos|>`
    (or since the start of generation, if none has been sampled yet).

    Per batch row:
      ``_req[idx]``           -> {"chunk_eos_token_id", "max_new_tokens_per_chunk"}
      ``_output_tokens[idx]`` -> live reference to the request's output ids
      ``_boundary_pos[idx]``  -> index in that list of the last `<|chunk_eos|>`
                                 seen, or -1 if none yet.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool) -> None:
        self._req: dict[int, dict[str, Any]] = {}
        self._output_tokens: dict[int, list[int]] = {}
        self._boundary_pos: dict[int, int] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return

        for idx in batch_update.removed:
            self._req.pop(idx, None)
            self._output_tokens.pop(idx, None)
            self._boundary_pos.pop(idx, None)

        for idx, params, _, output_token_ids in batch_update.added:
            extra_args = params.extra_args if params else None
            chunk_eos = extra_args.get("minicpmo_chunk_eos") if extra_args else None
            if chunk_eos:
                self._req[idx] = {
                    "chunk_eos_token_id": chunk_eos["chunk_eos_token_id"],
                    "max_new_tokens_per_chunk": chunk_eos.get("max_new_tokens_per_chunk", 20),
                }
                self._output_tokens[idx] = output_token_ids
                self._boundary_pos[idx] = -1
            else:
                self._req.pop(idx, None)
                self._output_tokens.pop(idx, None)
                self._boundary_pos.pop(idx, None)

        if self._req:
            for src, dst, direction in batch_update.moved:
                src_req = self._req.pop(src, None)
                dst_req = self._req.pop(dst, None)
                src_tokens = self._output_tokens.pop(src, None)
                dst_tokens = self._output_tokens.pop(dst, None)
                src_pos = self._boundary_pos.pop(src, None)
                dst_pos = self._boundary_pos.pop(dst, None)
                if src_req is not None:
                    self._req[dst] = src_req
                if src_tokens is not None:
                    self._output_tokens[dst] = src_tokens
                if src_pos is not None:
                    self._boundary_pos[dst] = src_pos
                if direction == MoveDirectionality.SWAP:
                    if dst_req is not None:
                        self._req[src] = dst_req
                    if dst_tokens is not None:
                        self._output_tokens[src] = dst_tokens
                    if dst_pos is not None:
                        self._boundary_pos[src] = dst_pos

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._req:
            return logits

        num_rows = logits.shape[0]
        for idx, state in self._req.items():
            # Rows beyond this step's logits (pure-prefill / partially
            # scheduled steps) have nothing to force; the persistent-batch
            # index re-enters range on the decode steps that sample.
            if idx >= num_rows:
                continue
            tokens = self._output_tokens.get(idx)
            if tokens is None:
                continue

            chunk_eos_id = state["chunk_eos_token_id"]
            n = len(tokens)
            boundary = self._boundary_pos[idx]
            if n > 0 and tokens[-1] == chunk_eos_id and boundary != n - 1:
                boundary = n - 1
                self._boundary_pos[idx] = boundary

            count_since_boundary = n - boundary - 1
            if count_since_boundary >= state["max_new_tokens_per_chunk"] - 1:
                logits[idx].fill_(NEG_INF)
                logits[idx, chunk_eos_id] = 0.0

        return logits
