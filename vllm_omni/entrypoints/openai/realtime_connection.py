from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import cast
from uuid import uuid4

import numpy as np
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection as VllmRealtimeConnection
from vllm.entrypoints.speech_to_text.realtime.protocol import TranscriptionDelta, TranscriptionDone
from vllm.logger import init_logger

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.utils import coerce_param_message_types
from vllm_omni.experimental.fullduplex.qwen3_omni.audio_utils import (
    extract_audio_chunks,
    pcm16_b64,
)

logger = init_logger(__name__)


class RealtimeConnection(VllmRealtimeConnection):
    """Omni realtime connection with audio-only server events.

    Reuses upstream vLLM websocket/session lifecycle and only customizes
    generation output handling to emit audio deltas.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = cast(AsyncOmni, self.serving.engine_client)
        self._realtime_audio_ref: np.ndarray | None = None

    async def start_generation(self):
        await super().start_generation()

    def _extract_audio_chunks_with_ref(self, output) -> tuple[list[np.ndarray], int]:
        chunks, sr, self._realtime_audio_ref = extract_audio_chunks(output, self._realtime_audio_ref)
        return chunks, sr

    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
    ):
        request_id = f"rt-{self.connection_id}-{uuid4()}"
        sent_audio = False
        audio_done_sent = False
        full_text = ""
        prompt_token_ids_len = 0
        completion_tokens_len = 0
        self._realtime_audio_ref = None

        # Coerce cumulative outputs to delta outputs; this ensures
        # we don't emit redundant MM data & drain after emitting.
        sampling_params_list = list(self.engine.default_sampling_params_list)
        sampling_params_list = coerce_param_message_types(
            sampling_params_list,
            is_streaming=True,
        )

        sp0 = sampling_params_list[0]
        logger.info(
            "[rt-debug] stage0 sampling_params: stop_token_ids=%s eos_token_id=%s ignore_eos=%s max_tokens=%s",
            getattr(sp0, "stop_token_ids", "N/A"),
            getattr(sp0, "eos_token_id", "N/A"),
            getattr(sp0, "ignore_eos", "N/A"),
            getattr(sp0, "max_tokens", "N/A"),
        )

        try:
            result_gen = self.engine.generate(
                prompt=streaming_input_gen,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
            )

            async for output in result_gen:
                stage_id = getattr(output, "stage_id", None)
                if stage_id == 0 and output.outputs:
                    first_output = output.outputs[0]
                    new_token_ids = list(first_output.token_ids)
                    finish_reason = getattr(first_output, "finish_reason", None)
                    if new_token_ids or finish_reason:
                        logger.info(
                            "[rt-debug] stage0 tokens=%s finish=%s",
                            new_token_ids,
                            finish_reason,
                        )
                    if new_token_ids:
                        input_stream.put_nowait(new_token_ids)

                    if output.prompt_token_ids:
                        prompt_token_ids_len = max(
                            prompt_token_ids_len,
                            len(output.prompt_token_ids),
                        )

                    delta_text = first_output.text or ""
                    full_text += delta_text
                    completion_tokens_len += len(new_token_ids)

                    if delta_text:
                        await self.send(TranscriptionDelta(delta=delta_text))

                audio_chunks, sample_rate = self._extract_audio_chunks_with_ref(output)

                for chunk in audio_chunks:
                    sent_audio = True
                    await self.send_json(
                        {
                            "type": "response.audio.delta",
                            "audio": pcm16_b64(chunk),
                            "format": "pcm16",
                            "sample_rate_hz": sample_rate,
                        }
                    )

                if not self._is_connected:
                    break

            usage = UsageInfo(
                prompt_tokens=prompt_token_ids_len,
                completion_tokens=completion_tokens_len,
                total_tokens=prompt_token_ids_len + completion_tokens_len,
            )
            await self.send(TranscriptionDone(text=full_text, usage=usage))

            if sent_audio:
                await self.send_json({"type": "response.audio.done", "has_audio": True})
                audio_done_sent = True
        except Exception as e:
            logger.exception("Error in generation: %s", e)
            await self.send_error(str(e), "processing_error")
        finally:
            if self._is_connected and not audio_done_sent:
                try:
                    await self.send_json({"type": "response.audio.done", "has_audio": sent_audio})
                except Exception:
                    logger.exception("Failed to send response.audio.done")

    async def send_json(self, payload: dict):
        await self.websocket.send_text(json.dumps(payload))
