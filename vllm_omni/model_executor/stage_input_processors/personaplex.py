# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Talker -> Code2Wav input processors for PersonaPlex.

The talker (stage 0) emits, per frame, a logical ``frame_t`` under
``("codes","audio")``: ``[text, agent_code[0:8], user_code[0:8]]``. Only the
agent rows are decoded to PCM by Mimi; the text and user rows are retained so
the replay frontend and the one-frame acoustic delay have an unambiguous
history. These processors take the accumulated logical frames ``[F, 17]``,
keep agent ``cb 0..7``, and flatten them codebook-major (``[8 * F]``) — the
exact layout :class:`PersonaPlexCode2Wav` consumes. The older raw
``[F, dep_q]`` shape remains accepted by the async-chunk compatibility path.

Mirrors the Qwen3-TTS processors (sync ``full_payload`` + ``token_only``; an
async-chunk variant for the streaming path), but with PersonaPlex's agent-codebook
slice instead of Qwen3-TTS's residual layout.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from vllm_omni.data_entry_keys import (
    CodesStruct,
    EmbeddingsStruct,
    MetaStruct,
    OmniPayloadStruct,
)

_NUM_ACTIVE_CODEBOOKS = 8  # agent cb 0..7 (the PCM-bearing rows)


def _empty_finished_payload(
    *,
    prefill: torch.Tensor | None = None,
    request_id: str | None = None,
    chunk_seq: int | None = None,
    cache_epoch: int | None = None,
    codec_streaming: bool | None = None,
) -> OmniPayloadStruct:
    # A prefill-only handoff still needs to wake the generation stage. Keep a
    # single malformed codec placeholder in that case; Code2Wav recognizes
    # the non-divisible length and skips decoding, while still returning the
    # embedded replay prefix. A genuinely empty handoff remains empty.
    audio = (
        torch.zeros(1, dtype=torch.long)
        if isinstance(prefill, torch.Tensor) and prefill.numel() > 0
        else torch.empty(0, dtype=torch.long)
    )
    return OmniPayloadStruct(
        embed=EmbeddingsStruct(prefill=prefill) if isinstance(prefill, torch.Tensor) else None,
        codes=CodesStruct(audio=audio),
        meta=MetaStruct(
            finished=torch.tensor(True, dtype=torch.bool),
            request_id=request_id,
            chunk_seq=chunk_seq,
            cache_epoch=cache_epoch,
            codec_streaming=(request_id is not None if codec_streaming is None else codec_streaming),
        ),
    )


def _value_from_sources(sources: tuple[Any, ...], dotted: str, nested_key: str) -> Any:
    """Read a flattened or nested field from the first payload that has it."""
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        value = source.get(dotted)
        if value is not None:
            return value
        root = dotted.split(".", 1)[0]
        nested = source.get(root)
        if isinstance(nested, dict):
            value = nested.get(nested_key)
            if value is not None:
                return value
    return None


def _request_replay_metadata(request: Any, *sources: Any) -> tuple[str | None, int | None, int | None]:
    all_sources = (
        *sources,
        getattr(request, "additional_information", None),
        getattr(request, "additional_information_cpu", None),
    )
    request_id = _value_from_sources(all_sources, "meta.request_id", "request_id")
    if request_id is None:
        request_id = _value_from_sources(all_sources, "request_id", "request_id")
    chunk_seq = _value_from_sources(all_sources, "meta.chunk_seq", "chunk_seq")
    cache_epoch = _value_from_sources(all_sources, "meta.cache_epoch", "cache_epoch")
    try:
        chunk_seq = int(chunk_seq) if chunk_seq is not None else None
    except (TypeError, ValueError):
        chunk_seq = None
    try:
        cache_epoch = int(cache_epoch) if cache_epoch is not None else None
    except (TypeError, ValueError):
        cache_epoch = None
    return (str(request_id) if request_id is not None else None, chunk_seq, cache_epoch)


def _request_codec_streaming(request: Any, *sources: Any) -> bool:
    request_id, _, _ = _request_replay_metadata(request, *sources)
    all_sources = (
        *sources,
        getattr(request, "additional_information", None),
        getattr(request, "additional_information_cpu", None),
    )
    value = _value_from_sources(all_sources, "meta.codec_streaming", "codec_streaming")
    return request_id is not None if value is None else bool(value)


def _request_closes_session(request: Any, *sources: Any) -> bool:
    all_sources = (
        *sources,
        getattr(request, "additional_information", None),
        getattr(request, "additional_information_cpu", None),
    )
    value = _value_from_sources(all_sources, "duplex.close_session", "close_session")
    return bool(value)


def _agent_codes_to_codebook_major(audio: torch.Tensor) -> torch.Tensor:
    """``[F, dep_q]`` raw depformer agent codes -> de-delayed flat codebook-major.

    The talker emits the raw per-frame depformer codes ``gen[t]`` (cb 0..7). Mimi
    needs them ACOUSTICALLY DE-DELAYED to a common time step (Moshi agent delays
    ``[0, 1, 1, 1, 1, 1, 1, 1]``): acoustic frame t's cb_k is predicted at step
    ``t + delay[k]``, so output frame t = ``[gen[t][0], gen[t+1][1:8]]`` (cb0 from
    the current step, cb1..7 from the NEXT step, since the delayed codebooks lag).
    Without this the codebooks are misaligned by one frame and Mimi decodes garble.
    The last frame has no successor for cb1..7, so it is dropped (the delay warmup).
    """
    if audio.ndim != 2 or audio.shape[0] < 2:
        return torch.empty(0, dtype=torch.long)
    audio = audio.to(torch.long)
    # Replay requests publish logical frame_t rows as
    # [text, agent[0:8], user[0:8]]. Keep accepting the older raw depformer
    # layout [agent[0:8], user[0:8]] for the native async-chunk path.
    if audio.shape[1] >= 1 + 2 * _NUM_ACTIVE_CODEBOOKS:
        agent = audio[:, 1 : 1 + _NUM_ACTIVE_CODEBOOKS]
    else:
        agent = audio[:, : min(_NUM_ACTIVE_CODEBOOKS, int(audio.shape[1]))]
    k = min(_NUM_ACTIVE_CODEBOOKS, int(agent.shape[1]))
    agent = agent[:, :k]
    valid = (agent >= 0).all(dim=1)
    agent = agent[valid]
    if agent.shape[0] < 2:
        return torch.empty(0, dtype=torch.long)
    # De-delay: cb0 from frame t, cb1..7 from frame t+1 (drop the last frame).
    cb0 = agent[:-1, 0:1]  # [F-1, 1]
    cb_rest = agent[1:, 1:k]  # [F-1, k-1]
    dd = torch.cat([cb0, cb_rest], dim=1)  # [F-1, k] de-delayed
    # [F-1, k] -> [k, F-1] -> flat [k * (F-1)] (codebook-major), as Code2Wav expects.
    return dd.transpose(0, 1).contiguous().reshape(-1)


def talker2code2wav_token_only(
    source_outputs: list,
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list:
    """Sync ``process_engine_inputs``: build the code2wav placeholder inputs.

    Returns one :class:`OmniTokensPrompt` per finished talker request, with
    ``prompt_token_ids`` sized to the flat codebook-major codec length
    (``num_active_codebooks * num_agent_frames``). The actual codec ids are
    delivered via the worker connector payload from ``talker2code2wav_full_payload``.
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    del prompt, _requires_multimodal_data
    inputs: list = []
    for talker_output in source_outputs:
        if not getattr(talker_output, "finished", False):
            continue
        output = talker_output.outputs[0]
        # The per-request output is the "latent" (engine_output_type="latent"); the
        # codes ship via the connector full_payload. Size the placeholder from the
        # generated token count (one AR token == one Mimi frame). prompt was 1 frame.
        token_ids = getattr(output, "cumulative_token_ids", None) or getattr(output, "token_ids", None) or []
        n_frames = max(len(token_ids) - 1, 0)
        output_mm = getattr(output, "multimodal_output", None)
        if isinstance(output_mm, Mapping):
            output_codes = output_mm.get("codes")
            output_audio = output_codes.get("audio") if isinstance(output_codes, Mapping) else None
            if output_audio is None:
                output_audio = output_mm.get("codes.audio")
            if isinstance(output_audio, torch.Tensor) and output_audio.ndim == 2:
                n_frames = max(n_frames, int(output_audio.shape[0]) - 1)
                # The first logical frame has no de-delayed PCM frame yet,
                # but Stage 1 still must execute once so it can propagate the
                # replay prefill returned by the producer. A one-token
                # placeholder makes Code2Wav run without asking it to decode
                # a fabricated eight-codebook frame.
        prompt_len = _NUM_ACTIVE_CODEBOOKS * n_frames
        # PersonaPlex emits one logical frame per ordinary request. Even when
        # the producer has only a prefill (or an empty frame), wake the
        # generation stage with a harmless one-token placeholder so the
        # connector payload can still carry replay state and finish.
        if prompt_len == 0:
            prompt_len = 1
        inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return inputs


def talker2code2wav_full_payload(
    transfer_manager: Any = None,
    pooling_output: Any = None,
    request: Any = None,
    is_finished: bool = False,
    **kwargs: Any,
) -> OmniPayloadStruct:
    """Producer: collect the talker's accumulated agent codes -> Code2Wav input.

    Called by the connector with (transfer_manager, pooling_output, request,
    is_finished). Prefer the request's additional_information payload under
    ("codes","audio") (talker_mtp_output_key), while accepting pooling_output
    and the legacy multimodal_output keyword as compatibility fallbacks.
    """
    del is_finished

    # Connector output is authoritative for the current request. The request's
    # original prompt is retained as a fallback for metadata and legacy tests,
    # but its replay prefill is the prefix that was submitted *before* this
    # frame and must not shadow the newly returned Stage 0 prefix.
    sources = (
        pooling_output,
        # Compatibility with the legacy producer call contract.
        kwargs.get("multimodal_output"),
        getattr(request, "additional_information", None),
        getattr(request, "additional_information_cpu", None),
    )

    def _codes_from(src: Any) -> torch.Tensor | None:
        if not isinstance(src, Mapping):
            return None
        nested = src.get("codes")
        audio = nested.get("audio") if isinstance(nested, Mapping) else None
        return audio if audio is not None else src.get("codes.audio")

    def _prefill_from(src: Any) -> torch.Tensor | None:
        if not isinstance(src, Mapping):
            return None
        nested = src.get("embed")
        prefill = nested.get("prefill") if isinstance(nested, Mapping) else None
        if prefill is None:
            prefill = src.get("embed.prefill")
        return prefill if isinstance(prefill, torch.Tensor) and prefill.numel() > 0 else None

    audio = None
    for source in sources:
        audio = _codes_from(source)
        if audio is not None:
            break
    prefill = None
    for source in sources:
        prefill = _prefill_from(source)
        if prefill is not None:
            break
    request_id, chunk_seq, cache_epoch = _request_replay_metadata(request, *sources)
    closes_session = _request_closes_session(request, *sources)
    replay_key = request_id or str(getattr(request, "external_req_id", getattr(request, "request_id", "?")))

    # The normal realtime frontend submits one ordinary request per user frame.
    # The model output is one frame, while Code2Wav needs the cumulative raw
    # frame prefix to undo the one-frame acoustic delay. Keep that logical-frame
    # history on this PersonaPlex-specific transfer manager; no general runner
    # or scheduler state is involved.
    if isinstance(audio, torch.Tensor) and audio.numel() > 0:
        audio = audio if audio.ndim == 2 else audio.reshape(1, -1)
        if audio.shape[1] >= 1 + _NUM_ACTIVE_CODEBOOKS and transfer_manager is not None:
            replay_store = getattr(transfer_manager, "_personaplex_replay_frames", None)
            if not isinstance(replay_store, dict):
                replay_store = {}
                setattr(transfer_manager, "_personaplex_replay_frames", replay_store)
            state = replay_store.get(replay_key)
            if not isinstance(state, dict) or (
                cache_epoch is not None and state.get("cache_epoch") not in (None, cache_epoch)
            ):
                state = {"frames": [], "cache_epoch": cache_epoch, "last_seq": 0}
                replay_store[replay_key] = state
            if chunk_seq is None or chunk_seq > int(state.get("last_seq", 0)):
                state["frames"].append(audio[-1].detach().to(device="cpu", dtype=torch.long))
                if chunk_seq is not None:
                    state["last_seq"] = chunk_seq
            audio = torch.stack(state["frames"], dim=0)

    flat = (
        _agent_codes_to_codebook_major(audio)
        if isinstance(audio, torch.Tensor) and audio.numel() > 0
        else torch.empty(0, dtype=torch.long)
    )
    codec_streaming = _request_codec_streaming(request, *sources)
    if closes_session and transfer_manager is not None:
        replay_store = getattr(transfer_manager, "_personaplex_replay_frames", None)
        if isinstance(replay_store, dict):
            replay_store.pop(replay_key, None)
    if flat.numel() == 0 and prefill is None:
        return _empty_finished_payload(
            request_id=request_id,
            chunk_seq=chunk_seq,
            cache_epoch=cache_epoch,
            codec_streaming=codec_streaming,
        )
    meta_kwargs: dict[str, Any] = {
        "finished": torch.tensor(True, dtype=torch.bool),
        "request_id": request_id,
        "chunk_seq": chunk_seq,
        "cache_epoch": cache_epoch,
        "codec_streaming": codec_streaming,
    }
    if flat.numel() > 0:
        meta_kwargs["next_stage_prompt_len"] = int(flat.numel())
    return OmniPayloadStruct(
        embed=EmbeddingsStruct(prefill=prefill.detach().to(device="cpu")) if prefill is not None else None,
        codes=CodesStruct(audio=flat),
        meta=MetaStruct(**meta_kwargs),
    )


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: Any,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Streaming: accumulate per-frame agent codes, emit a codebook-major chunk.

    Minimal fixed-chunk variant (no left-context / ref-code complexity, which
    PersonaPlex does not use). Frames are buffered on the transfer manager until a
    chunk's worth is ready (or the request finishes), then flushed.
    """
    request_id = getattr(request, "external_req_id", getattr(request, "request_id", "?"))
    # The adapter passes ``is_finished=True`` for both a resumable segment
    # boundary and the terminal request boundary. PersonaPlex must preserve the
    # delayed cb1..7 tail across the former; only a non-resumable stop flushes
    # the stream.
    finished = bool(is_finished and not getattr(request, "resumable", False))
    request_payload = getattr(transfer_manager, "request_payload", None)
    if not isinstance(request_payload, dict):
        request_payload = {}
        transfer_manager.request_payload = request_payload
    state = request_payload.setdefault(request_id, {})
    frames = state.setdefault("personaplex_frames", [])

    # Codes live in the server-side request's additional_information under
    # ("codes","audio") (talker_mtp_output_key), not in multimodal_output (latent).
    def _codes_from(src: Any) -> torch.Tensor | None:
        if isinstance(src, dict):
            nested = src.get("codes")
            a = nested.get("audio") if isinstance(nested, dict) else None
            return a if a is not None else src.get("codes.audio")
        return None

    # Explicit None fallback: `a or b` would evaluate bool(a) on a multi-element
    # Tensor and raise "Boolean value of Tensor ... is ambiguous".
    audio = _codes_from(getattr(request, "additional_information", None))
    if audio is None:
        audio = _codes_from(multimodal_output)
    if isinstance(audio, torch.Tensor) and audio.numel() > 0:
        a = audio if audio.ndim == 2 else audio.reshape(1, -1)
        frames.append(a[-1].to(torch.long).cpu())  # latest frame's codes

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk = int(cfg.get("codec_chunk_frames", 25))
    initial_chunk = int(cfg.get("initial_codec_chunk_frames") or 0)
    if chunk <= 0 or initial_chunk < 0:
        raise ValueError(
            "PersonaPlex codec chunk sizes must be positive/non-negative: "
            f"codec_chunk_frames={chunk}, initial_codec_chunk_frames={initial_chunk}"
        )
    target_frames = initial_chunk if not state.get("personaplex_emitted") and initial_chunk > 0 else chunk

    # De-delay needs one successor raw frame: N output acoustic frames require
    # N + 1 raw depformer rows.
    available_frames = max(0, len(frames) - 1)
    if available_frames < target_frames and not finished:
        # Each full-duplex input frame is a resumable stage-0 segment. Returning
        # None would make the generic chunk adapter synthesize a
        # segment-finished marker, wake Code2Wav with its one-token placeholder,
        # and discard the buffered de-delay tail. Explicitly keep this transport
        # chunk non-terminal until enough successor frames exist.
        pending = torch.tensor(False, dtype=torch.bool)
        return OmniPayloadStruct(
            meta=MetaStruct(
                finished=pending,
                is_segment_finished=pending,
            )
        )
    emit_frames = available_frames if finished else target_frames
    if emit_frames <= 0:
        if finished:
            request_payload.pop(request_id, None)
            return _empty_finished_payload()
        return None

    stacked = torch.stack(frames[: emit_frames + 1], dim=0)  # [F+1, dep_q]
    flat = _agent_codes_to_codebook_major(stacked)
    if finished:
        request_payload.pop(request_id, None)
    else:
        # Row ``emit_frames`` is the successor used by the last emitted frame
        # and the cb0 source for the next frame.
        state["personaplex_frames"] = frames[emit_frames:]
        state["personaplex_emitted"] = True
    return OmniPayloadStruct(
        codes=CodesStruct(audio=flat),
        meta=MetaStruct(finished=torch.tensor(bool(finished), dtype=torch.bool)),
    )


__all__ = [
    "talker2code2wav_token_only",
    "talker2code2wav_full_payload",
    "talker2code2wav_async_chunk",
]
