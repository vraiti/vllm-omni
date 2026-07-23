# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from base64 import b64decode
from binascii import Error as BinasciiError
from typing import Any

from vllm_omni.experimental.fullduplex.base.data_plane import (
    coerce_int,
    completion_token_ids,
    first_completion,
    multimodal_output,
    special_token_ids,
)
from vllm_omni.experimental.fullduplex.base.runtime_extension import (
    BaseDuplexRuntimeExtension,
)
from vllm_omni.experimental.fullduplex.engine.duplex_runtime import (
    DuplexAppendPlan,
    DuplexInputMode,
    DuplexOutputAction,
    DuplexOutputDecision,
)
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.qwen3omni.policy import Qwen3OmniDuplexPolicy

_CHUNK_SAMPLES = Qwen3OmniDuplexPolicy.CHUNK_SAMPLES
_SAMPLES_PER_AUDIO_TOKEN = Qwen3OmniDuplexPolicy.SAMPLES_PER_AUDIO_TOKEN
_IM_END_TOKEN_ID = Qwen3OmniDuplexPolicy.IM_END_TOKEN_ID


def _pcm_sample_count(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    audio = payload.get("audio") or payload.get("data")
    if payload.get("format") != "pcm_f32le" or not isinstance(audio, str):
        return None
    try:
        raw = b64decode(audio, validate=True)
    except (BinasciiError, ValueError):
        return None
    return len(raw) // 4


def qwen3omni_scheduler_token_budget(payload: object, *, default: int = 64) -> int:
    sample_count = _pcm_sample_count(payload)
    if sample_count is None:
        return max(1, int(default))
    return max(16, min(768, sample_count // _SAMPLES_PER_AUDIO_TOKEN + 8))


def qwen3omni_first_append_context_reserve(runtime_config: object) -> int:
    if not isinstance(runtime_config, dict):
        return 48
    exact = runtime_config.get("duplex_first_append_context_tokens")
    if isinstance(exact, int) and exact >= 0:
        return exact
    return 48


def build_qwen3omni_data_plane_prompt(
    *,
    request_id: str,
    fence: DuplexFence,
    session_config: dict[str, Any],
    runtime_config: dict[str, Any],
    seq: int,
    turn_seq: int,
    mode: DuplexInputMode,
    payload: object,
    final: bool,
) -> dict[str, Any]:
    token_budget = qwen3omni_scheduler_token_budget(payload)
    if seq <= 1:
        token_budget += qwen3omni_first_append_context_reserve(runtime_config)
    if final:
        token_budget += 12

    raw_token_id = runtime_config.get("duplex_scheduler_token_id")
    try:
        token_id = max(0, int(raw_token_id))
    except (TypeError, ValueError):
        token_id = 0

    return {
        "prompt_token_ids": [token_id] * token_budget,
        "model_intermediate_buffer": {
            "request_id": request_id,
            "global_request_id": [fence.session_id],
            "duplex": {
                "fence": fence,
                "session_id": fence.session_id,
                "incarnation": fence.incarnation,
                "epoch": fence.epoch,
                "seq": seq,
                "turn_id": fence.turn_id,
                "response_seq": fence.response_seq,
                "turn_seq": turn_seq,
                "mode": mode.value,
                "payload": payload,
                "final": final,
                "data_plane": True,
                "session_config": dict(session_config),
                "runtime_config": dict(runtime_config),
                "scheduler_token_budget": token_budget,
                "scheduler_token_id": token_id,
            },
        },
    }


class Qwen3OmniDuplexRuntimeExtension(BaseDuplexRuntimeExtension):
    def plan_append(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, Any],
        runtime_config: dict[str, Any],
        seq: int,
        turn_seq: int,
        mode: DuplexInputMode,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        del sampling_params
        return DuplexAppendPlan(
            prompt=build_qwen3omni_data_plane_prompt(
                request_id=request_id,
                fence=fence,
                session_config=session_config,
                runtime_config=runtime_config,
                seq=seq,
                turn_seq=turn_seq,
                mode=mode,
                payload=payload,
                final=final,
            )
        )

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, Any],
        output: object,
    ) -> DuplexOutputDecision | None:
        if stage_id != 0 or not segment_finished:
            return None

        completion = first_completion(output)
        token_ids = completion_token_ids(completion) or list(segment_token_ids)
        stop_reason = getattr(completion, "stop_reason", None) if completion is not None else None

        im_end_id = _IM_END_TOKEN_ID
        stids = special_token_ids(segment_output_metadata)
        output_metadata = multimodal_output(output, completion)
        stids.update(special_token_ids(output_metadata))
        if "im_end_token_id" in stids:
            im_end_id = stids["im_end_token_id"]

        is_im_end = coerce_int(stop_reason) == im_end_id or (token_ids and token_ids[-1] == im_end_id)
        if not is_im_end:
            return None

        metadata = dict(output_metadata)
        for key, value in stids.items():
            metadata.setdefault(f"meta.{key}", value)
        metadata.update(
            {
                "duplex_direct_response": True,
                "duplex_native_decision": "turn_complete",
                "turn_end": True,
                "end_of_turn": True,
            }
        )
        return DuplexOutputDecision(
            action=DuplexOutputAction.DIRECT_RESPONSE,
            metadata=metadata,
        )
