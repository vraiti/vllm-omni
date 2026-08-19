# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.personaplex import (
    talker2code2wav_async_chunk,
    talker2code2wav_full_payload,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _raw_agent_codes() -> torch.Tensor:
    return torch.arange(16, dtype=torch.long).reshape(2, 8)


@pytest.mark.parametrize("source", ["pooling_output", "additional_information_cpu"])
def test_full_payload_accepts_worker_and_cached_payload_sources(source: str) -> None:
    audio = _raw_agent_codes()
    payload = talker2code2wav_full_payload(
        pooling_output={"codes": {"audio": audio}} if source == "pooling_output" else None,
        request=(
            SimpleNamespace()
            if source == "pooling_output"
            else SimpleNamespace(additional_information_cpu={"codes": {"audio": audio}})
        ),
    )

    expected = torch.cat([audio[:-1, :1], audio[1:, 1:]], dim=1).reshape(-1)
    assert torch.equal(payload.codes.audio, expected)


def test_async_chunk_keeps_delay_tail_across_resumable_segments() -> None:
    manager = SimpleNamespace(
        connector=SimpleNamespace(
            config={
                "extra": {
                    "initial_codec_chunk_frames": 1,
                    "codec_chunk_frames": 5,
                }
            }
        )
    )
    request = SimpleNamespace(
        request_id="req",
        external_req_id="req",
        resumable=True,
        is_finished=lambda: True,
        additional_information=None,
    )
    first_frame = torch.arange(8, dtype=torch.long).reshape(1, 8)
    second_frame = torch.arange(8, 16, dtype=torch.long).reshape(1, 8)

    request.additional_information = {"codes": {"audio": first_frame}}
    first = talker2code2wav_async_chunk(
        manager,
        multimodal_output=None,
        request=request,
        is_finished=True,
    )
    request.additional_information = {"codes": {"audio": second_frame}}
    second = talker2code2wav_async_chunk(
        manager,
        multimodal_output=None,
        request=request,
        is_finished=True,
    )

    assert first is not None
    assert first.codes is None
    assert first.meta is not None
    assert first.meta.finished.item() is False
    assert first.meta.is_segment_finished.item() is False
    expected = torch.cat([first_frame[:, :1], second_frame[:, 1:]], dim=1).reshape(-1)
    assert torch.equal(second.codes.audio, expected)
    assert manager.request_payload["req"]["personaplex_frames"][0].equal(second_frame.reshape(-1))
