# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
from collections.abc import Mapping

import numpy as np


def tensor_to_numpy(value) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        arr = value
    elif hasattr(value, "detach"):
        arr = value.detach().float().cpu().numpy()
    else:
        try:
            arr = np.asarray(value)
        except Exception:
            return None
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr.astype(np.float32, copy=False)


def numpy_audio_prefix_match(prev: np.ndarray, curr: np.ndarray) -> bool:
    n = prev.shape[0]
    if n == 0:
        return True
    if curr.shape[0] < n:
        return False
    return bool(np.allclose(curr[:n], prev, rtol=1e-3, atol=2e-4))


def raw_waveform_to_deltas(arr: np.ndarray, ref: np.ndarray | None) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Convert one streaming PCM f32 chunk into incremental pieces.

    Returns (delta_chunks, updated_ref).
    """
    if arr.size == 0:
        return [], ref
    if ref is None:
        return [arr], arr.copy()
    if numpy_audio_prefix_match(ref, arr):
        delta = arr[ref.shape[0] :]
        new_ref = arr.copy()
        return ([delta] if delta.size > 0 else []), new_ref
    new_ref = np.concatenate([ref, arr])
    return [arr], new_ref


def extract_audio_chunks(output, ref: np.ndarray | None) -> tuple[list[np.ndarray], int, np.ndarray | None]:
    """Extract audio chunks from an engine output.

    Returns (chunks, sample_rate, updated_ref).
    """
    mm = getattr(output, "multimodal_output", None)
    if mm is None:
        return [], 24000, ref
    if not isinstance(mm, Mapping):
        return [], 24000, ref

    sr = mm.get("sr") or mm.get("sample_rate") or mm.get("audio_sample_rate") or 24000
    if isinstance(sr, (list, tuple)) and sr:
        sr = sr[-1]
    if hasattr(sr, "item"):
        sr = sr.item()
    sample_rate_hz = int(sr)
    key = "audio" if "audio" in mm else ("model_outputs" if "model_outputs" in mm else None)
    if key is None:
        return [], sample_rate_hz, ref

    raw_audio = mm.get(key)
    chunks: list[np.ndarray] = []
    if isinstance(raw_audio, (list, tuple)):
        if len(raw_audio) > 0:
            arr = tensor_to_numpy(raw_audio[-1])
            if arr is not None and arr.size > 0:
                deltas, ref = raw_waveform_to_deltas(arr, ref)
                chunks.extend(deltas)
    else:
        arr = tensor_to_numpy(raw_audio)
        if arr is not None and arr.size > 0:
            deltas, ref = raw_waveform_to_deltas(arr, ref)
            chunks.extend(deltas)
    return chunks, sample_rate_hz, ref


def pcm16_b64(audio_f32: np.ndarray) -> str:
    clipped = np.clip(audio_f32, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    return base64.b64encode(pcm16.tobytes()).decode("utf-8")


def pcm16_b64_to_f32(b64_data: str) -> np.ndarray:
    raw = base64.b64decode(b64_data)
    pcm16 = np.frombuffer(raw, dtype=np.int16)
    return pcm16.astype(np.float32) / 32767.0
