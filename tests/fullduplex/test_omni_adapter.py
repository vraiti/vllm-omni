# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from dataclasses import dataclass, field

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.core import protocol as ev
from vllm_omni.experimental.fullduplex.core.runtime import DuplexRuntime
from vllm_omni.experimental.fullduplex.core.session import (
    DuplexSession,
    DuplexSessionConfig,
)
from vllm_omni.experimental.fullduplex.omni.adapter import (
    OmniDuplexAdapter,
)
from vllm_omni.experimental.fullduplex.omni.audio_utils import (
    numpy_audio_prefix_match,
    pcm16_b64,
    pcm16_b64_to_f32,
    raw_waveform_to_deltas,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Audio utility tests
# ---------------------------------------------------------------------------


class TestPcm16Roundtrip:
    def test_roundtrip_preserves_signal(self):
        original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        b64 = pcm16_b64(original)
        recovered = pcm16_b64_to_f32(b64)
        np.testing.assert_allclose(recovered, original, atol=1e-4)

    def test_clipping(self):
        loud = np.array([2.0, -3.0], dtype=np.float32)
        b64 = pcm16_b64(loud)
        recovered = pcm16_b64_to_f32(b64)
        np.testing.assert_allclose(recovered, [1.0, -1.0], atol=1e-4)


class TestRawWaveformToDeltas:
    def test_first_chunk_returns_as_is(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        deltas, ref = raw_waveform_to_deltas(arr, None)
        assert len(deltas) == 1
        np.testing.assert_array_equal(deltas[0], arr)
        assert ref is not None

    def test_cumulative_mode_extracts_suffix(self):
        first = np.array([1.0, 2.0], dtype=np.float32)
        _, ref = raw_waveform_to_deltas(first, None)
        second = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        deltas, ref = raw_waveform_to_deltas(second, ref)
        assert len(deltas) == 1
        np.testing.assert_array_equal(deltas[0], [3.0, 4.0])

    def test_true_delta_mode(self):
        first = np.array([1.0, 2.0], dtype=np.float32)
        _, ref = raw_waveform_to_deltas(first, None)
        second = np.array([5.0, 6.0], dtype=np.float32)
        deltas, ref = raw_waveform_to_deltas(second, ref)
        assert len(deltas) == 1
        np.testing.assert_array_equal(deltas[0], [5.0, 6.0])

    def test_empty_array_returns_nothing(self):
        deltas, ref = raw_waveform_to_deltas(np.array([], dtype=np.float32), None)
        assert deltas == []
        assert ref is None


class TestPrefixMatch:
    def test_exact_prefix(self):
        prev = np.array([1.0, 2.0], dtype=np.float32)
        curr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert numpy_audio_prefix_match(prev, curr) is True

    def test_no_prefix(self):
        prev = np.array([1.0, 2.0], dtype=np.float32)
        curr = np.array([9.0, 8.0, 7.0], dtype=np.float32)
        assert numpy_audio_prefix_match(prev, curr) is False


# ---------------------------------------------------------------------------
# Fake engine/serving for adapter tests
# ---------------------------------------------------------------------------


@dataclass
class _FakeOutput:
    stage_id: int = 0
    outputs: list = field(default_factory=list)
    multimodal_output: dict | None = None
    prompt_token_ids: list | None = None


@dataclass
class _FakeCompletionOutput:
    token_ids: list = field(default_factory=list)
    text: str = ""
    finish_reason: str | None = None


class _FakeEngine:
    """Minimal fake that yields predetermined outputs."""

    def __init__(self, output_sequence: list[_FakeOutput], delay: float = 0.0):
        self._outputs = output_sequence
        self._delay = delay
        self.default_sampling_params_list = []
        self.aborted: list[str] = []

    async def generate(self, prompt, request_id, sampling_params_list=None):
        async for _ in prompt:
            pass
        for out in self._outputs:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield out

    async def abort(self, request_id):
        self.aborted.append(request_id)


class _FakeServing:
    pass


def _make_audio_output(samples: list[float], sr: int = 24000, stage_id: int = 2):
    return _FakeOutput(
        stage_id=stage_id,
        multimodal_output={
            "audio": np.array(samples, dtype=np.float32),
            "sr": sr,
        },
    )


def _make_text_output(text: str, token_ids: list[int] | None = None):
    return _FakeOutput(
        stage_id=0,
        outputs=[
            _FakeCompletionOutput(
                token_ids=token_ids or [1],
                text=text,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Adapter + Runtime integration tests
# ---------------------------------------------------------------------------

_AUDIO_CFG = DuplexSessionConfig(
    input_modalities=("audio",),
    output_modalities=("audio", "text"),
    proactive=False,
)


def _audio_chunk(n: int = 160) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


async def _feed(events):
    for e in events:
        yield e


def _collector():
    out: list[dict] = []

    async def emit(event: dict) -> None:
        out.append(event)

    return out, emit


@pytest.mark.asyncio
async def test_basic_audio_roundtrip():
    engine = _FakeEngine(
        [
            _make_text_output("hello", [101]),
            _make_audio_output([0.1, 0.2, 0.3]),
        ]
    )
    adapter = OmniDuplexAdapter(engine, _FakeServing())
    session = DuplexSession("s", _AUDIO_CFG)
    rt = DuplexRuntime(session, adapter)
    out, emit = _collector()

    await rt.run(
        _feed(
            [
                {"type": ev.INPUT_APPEND, "modality": "audio", "data": _audio_chunk()},
                {"type": ev.INPUT_COMMIT},
                {"type": ev.RESPONSE_CREATE},
                {"type": ev.CLOSE},
            ]
        ),
        emit,
    )

    types = [e["type"] for e in out]
    assert ev.RESPONSE_CREATED in types
    assert ev.RESPONSE_DONE in types

    text_deltas = [e["data"] for e in out if e["type"] == ev.RESPONSE_DELTA and e.get("modality") == "text"]
    assert "hello" in text_deltas

    audio_deltas = [e for e in out if e["type"] == ev.RESPONSE_DELTA and e.get("modality") == "audio"]
    assert len(audio_deltas) > 0


@pytest.mark.asyncio
async def test_barge_in_aborts_generation():
    engine = _FakeEngine(
        [_make_audio_output([float(i)]) for i in range(10)],
        delay=0.02,
    )
    adapter = OmniDuplexAdapter(engine, _FakeServing())
    session = DuplexSession("s", _AUDIO_CFG)
    rt = DuplexRuntime(session, adapter)
    out, emit = _collector()

    async def feed():
        yield {"type": ev.INPUT_APPEND, "modality": "audio", "data": _audio_chunk()}
        yield {"type": ev.INPUT_COMMIT}
        yield {"type": ev.RESPONSE_CREATE}
        await asyncio.sleep(0.03)
        yield {"type": ev.RESPONSE_CANCEL}
        yield {"type": ev.CLOSE}

    await rt.run(feed(), emit)

    audio_deltas = [e for e in out if e["type"] == ev.RESPONSE_DELTA and e.get("modality") == "audio"]
    assert len(audio_deltas) < 10
    done = [e for e in out if e["type"] == ev.RESPONSE_DONE]
    assert len(done) == 1
    assert done[0]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_new_response_supersedes_inflight():
    engine = _FakeEngine(
        [_make_audio_output([float(i)]) for i in range(5)],
        delay=0.02,
    )
    adapter = OmniDuplexAdapter(engine, _FakeServing())
    session = DuplexSession("s", _AUDIO_CFG)
    rt = DuplexRuntime(session, adapter)
    out, emit = _collector()

    async def feed():
        yield {"type": ev.INPUT_APPEND, "modality": "audio", "data": _audio_chunk()}
        yield {"type": ev.INPUT_COMMIT}
        yield {"type": ev.RESPONSE_CREATE}
        await asyncio.sleep(0.03)
        yield {"type": ev.INPUT_APPEND, "modality": "audio", "data": _audio_chunk()}
        yield {"type": ev.INPUT_COMMIT}
        yield {"type": ev.RESPONSE_CREATE}
        await asyncio.sleep(0.2)
        yield {"type": ev.CLOSE}

    await rt.run(feed(), emit)

    created = [e for e in out if e["type"] == ev.RESPONSE_CREATED]
    done = [e for e in out if e["type"] == ev.RESPONSE_DONE]
    assert len(created) == 2
    assert len(done) == 2
    assert done[0]["status"] == "cancelled"
    assert done[0]["response_index"] == created[0]["response_index"]
    assert done[1]["status"] == "completed"
    assert done[1]["response_index"] == created[1]["response_index"]


@pytest.mark.asyncio
async def test_text_output_alongside_audio():
    engine = _FakeEngine(
        [
            _make_text_output("world", [42]),
            _make_audio_output([0.5, -0.5]),
        ]
    )
    adapter = OmniDuplexAdapter(engine, _FakeServing())
    session = DuplexSession("s", _AUDIO_CFG)
    rt = DuplexRuntime(session, adapter)
    out, emit = _collector()

    await rt.run(
        _feed(
            [
                {"type": ev.INPUT_APPEND, "modality": "audio", "data": _audio_chunk()},
                {"type": ev.INPUT_COMMIT},
                {"type": ev.RESPONSE_CREATE},
                {"type": ev.CLOSE},
            ]
        ),
        emit,
    )

    text_deltas = [e["data"] for e in out if e["type"] == ev.RESPONSE_DELTA and e.get("modality") == "text"]
    audio_deltas = [e for e in out if e["type"] == ev.RESPONSE_DELTA and e.get("modality") == "audio"]
    assert "world" in text_deltas
    assert len(audio_deltas) > 0


@pytest.mark.asyncio
async def test_on_barge_in_clears_buffer():
    engine = _FakeEngine([_make_audio_output([1.0])])
    adapter = OmniDuplexAdapter(engine, _FakeServing())

    adapter._audio_buffer.append(_audio_chunk())

    session = DuplexSession("s", _AUDIO_CFG)
    await adapter.on_barge_in(session)

    assert len(adapter._audio_buffer) == 0


@pytest.mark.asyncio
async def test_should_respond_requires_audio():
    adapter = OmniDuplexAdapter(_FakeEngine([]), _FakeServing())
    session = DuplexSession("s", _AUDIO_CFG)

    assert adapter.should_respond(session) is False

    adapter._audio_buffer.append(_audio_chunk())
    assert adapter.should_respond(session) is True


@pytest.mark.asyncio
async def test_capabilities():
    adapter = OmniDuplexAdapter(_FakeEngine([]), _FakeServing())
    caps = adapter.capabilities()
    assert "audio" in caps.input_modalities
    assert "audio" in caps.output_modalities
    assert "text" in caps.output_modalities
    assert caps.proactive is False
