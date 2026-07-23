# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import binascii


class PcmAppendReservation:
    __slots__ = (
        "_active",
        "_force_listen",
        "_is_speech",
        "_owner",
        "_raw",
        "_sample_rate_hz",
        "_turn_had_speech",
        "_video_frames",
        "operation_id",
        "payload",
    )

    def __init__(
        self,
        *,
        owner: BasePcmAppendBuffer,
        operation_id: str,
        payload: dict[str, object] | None,
        raw: bytes,
        sample_rate_hz: int,
        force_listen: bool,
        is_speech: bool,
        turn_had_speech: bool = False,
        video_frames: list[str] | None = None,
    ) -> None:
        self._owner = owner
        self.operation_id = operation_id
        self.payload = payload
        self._raw = raw
        self._sample_rate_hz = sample_rate_hz
        self._force_listen = force_listen
        self._is_speech = is_speech
        self._turn_had_speech = turn_had_speech
        self._video_frames = list(video_frames or [])
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    @property
    def byte_count(self) -> int:
        return len(self._raw)

    def commit(self) -> None:
        self._owner._commit_reservation(self)

    def rollback(self) -> None:
        self._owner._rollback_reservation(self)


class BasePcmAppendBuffer:
    """Accumulates short PCM chunks into model-sized appends."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._sample_rate_hz: int | None = None
        self._force_listen = False
        self._is_speech = False
        self._turn_had_speech = False
        self._reservation_seq = 0
        self._reservations: list[PcmAppendReservation] = []
        self._frame_queue: list[str] = []

    def clear(self) -> None:
        for reservation in self._reservations:
            reservation._active = False
        self._reservations.clear()
        self._buffer.clear()
        self._frame_queue.clear()
        self._sample_rate_hz = None
        self._force_listen = False
        self._is_speech = False
        self._turn_had_speech = False

    def clear_force_listen(self) -> None:
        self._force_listen = False

    def has_pending(self) -> bool:
        return bool(self._buffer)

    def has_reserved(self) -> bool:
        return bool(self._reservations)

    @property
    def pending_byte_count(self) -> int:
        return len(self._buffer)

    def _reserve_passthrough(
        self,
        payload: dict[str, object],
        *,
        operation_id: str,
    ) -> PcmAppendReservation:
        sample_rate_hz = payload.get("sample_rate_hz")
        if isinstance(payload, dict) and "video_frames" in payload:
            payload = {key: value for key, value in payload.items() if key != "video_frames"}
        reservation = PcmAppendReservation(
            owner=self,
            operation_id=operation_id,
            payload=payload,
            raw=b"",
            sample_rate_hz=sample_rate_hz if isinstance(sample_rate_hz, int) else 0,
            force_listen=bool(payload.get("force_listen", False)),
            is_speech=bool(payload.get("is_speech", False)),
            turn_had_speech=bool(payload.get("is_speech", False)),
        )
        self._reservations.append(reservation)
        return reservation

    def prepare_append(
        self,
        payload: dict[str, object],
        *,
        operation_id: str,
        chunk_period_ms: int,
        flush: bool = False,
        allow_emit: bool = True,
    ) -> PcmAppendReservation | None:
        fmt = payload.get("format")
        sample_rate_hz = payload.get("sample_rate_hz")
        audio = payload.get("audio")
        if fmt != "pcm_f32le" or not isinstance(sample_rate_hz, int) or not isinstance(audio, str):
            return self._reserve_passthrough(payload, operation_id=operation_id)
        try:
            raw = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError):
            return self._reserve_passthrough(payload, operation_id=operation_id)
        if len(raw) % 4 != 0:
            return self._reserve_passthrough(payload, operation_id=operation_id)

        if self._sample_rate_hz is not None and self._sample_rate_hz != sample_rate_hz:
            raise ValueError("duplex audio append sample_rate_hz changed within a session")
        self._sample_rate_hz = sample_rate_hz
        self._buffer.extend(raw)
        frames_in = payload.get("video_frames")
        if isinstance(frames_in, list):
            self._frame_queue.extend(frame for frame in frames_in if isinstance(frame, str) and frame)
        self._turn_had_speech = self._turn_had_speech or bool(payload.get("is_speech", False))
        self._force_listen = self._force_listen or bool(payload.get("force_listen", False))
        self._is_speech = self._is_speech or bool(payload.get("is_speech", False))
        if not allow_emit:
            return None

        min_samples = max(1, int(sample_rate_hz * max(1, int(chunk_period_ms)) / 1000))
        buffered_samples = len(self._buffer) // 4
        if not flush and buffered_samples < min_samples:
            return None

        if flush:
            emit_samples = buffered_samples
            remainder = emit_samples % min_samples
            pad_samples = (min_samples - remainder) if remainder else 0
        else:
            emit_samples = min_samples
            pad_samples = 0
        emit_bytes = emit_samples * 4
        reserved_raw = bytes(self._buffer[:emit_bytes])
        emit_raw = reserved_raw + b"\x00" * (pad_samples * 4)
        del self._buffer[:emit_bytes]

        out = dict(payload)
        out.pop("force_speak", None)
        out.pop("video_frames", None)
        out["audio"] = base64.b64encode(emit_raw).decode("ascii")
        out["sample_rate_hz"] = sample_rate_hz
        attached_frames: list[str] = []
        if emit_samples + pad_samples >= min_samples and self._frame_queue:
            attached_frames = [self._frame_queue.pop(0)]
            out["video_frames"] = attached_frames
        out["force_listen"] = self._force_listen
        out["is_speech"] = self._is_speech
        if not self._buffer:
            self._force_listen = False
            self._is_speech = False
        reservation = PcmAppendReservation(
            owner=self,
            operation_id=operation_id,
            payload=out,
            raw=reserved_raw,
            sample_rate_hz=sample_rate_hz,
            force_listen=bool(out.get("force_listen", False)),
            is_speech=bool(out.get("is_speech", False)),
            turn_had_speech=self._turn_had_speech,
            video_frames=attached_frames,
        )
        self._reservations.append(reservation)
        return reservation

    def prepare_commit(
        self,
        *,
        operation_id: str,
        chunk_period_ms: int,
    ) -> PcmAppendReservation:
        had_speech = self._turn_had_speech
        reservation: PcmAppendReservation | None = None
        if had_speech and self._buffer:
            payload: dict[str, object] = {
                "type": "audio",
                "audio": "",
                "format": "pcm_f32le",
                "sample_rate_hz": self._sample_rate_hz or 16000,
                "force_listen": self._force_listen,
                "is_speech": self._is_speech,
            }
            reservation = self.prepare_append(
                payload,
                operation_id=operation_id,
                chunk_period_ms=chunk_period_ms,
                flush=True,
            )
            assert reservation is not None
            assert reservation.payload is not None
            reservation.payload["final"] = True
        if reservation is None:
            reservation = PcmAppendReservation(
                owner=self,
                operation_id=operation_id,
                payload=None,
                raw=b"",
                sample_rate_hz=self._sample_rate_hz or 0,
                force_listen=self._force_listen,
                is_speech=self._is_speech,
                turn_had_speech=had_speech,
            )
            self._reservations.append(reservation)

        self._sample_rate_hz = None
        self._force_listen = False
        self._is_speech = False
        self._turn_had_speech = False
        return reservation

    def append(
        self,
        payload: dict[str, object],
        *,
        chunk_period_ms: int,
        flush: bool = False,
        allow_emit: bool = True,
    ) -> dict[str, object] | None:
        self._reservation_seq += 1
        reservation = self.prepare_append(
            payload,
            operation_id=f"immediate-{self._reservation_seq}",
            chunk_period_ms=chunk_period_ms,
            flush=flush,
            allow_emit=allow_emit,
        )
        if reservation is None:
            return None
        reservation.commit()
        assert reservation.payload is not None
        return reservation.payload

    def _commit_reservation(self, reservation: PcmAppendReservation) -> None:
        if not reservation._active:
            return
        if not self._reservations or self._reservations[0] is not reservation:
            raise RuntimeError("PCM append reservations must commit in wire order")
        self._reservations.pop(0)
        reservation._active = False

    def _rollback_reservation(self, reservation: PcmAppendReservation) -> None:
        if not reservation._active:
            return
        try:
            index = self._reservations.index(reservation)
        except ValueError:
            reservation._active = False
            return
        rolled_back = self._reservations[index:]
        restored = b"".join(item._raw for item in rolled_back)
        self._buffer[:0] = restored
        restored_frames = [frame for item in rolled_back for frame in item._video_frames]
        if restored_frames:
            self._frame_queue[:0] = restored_frames
        self._sample_rate_hz = self._sample_rate_hz or reservation._sample_rate_hz
        self._force_listen = self._force_listen or any(item._force_listen for item in rolled_back)
        self._is_speech = self._is_speech or any(item._is_speech for item in rolled_back)
        self._turn_had_speech = self._turn_had_speech or any(item._turn_had_speech for item in rolled_back)
        for item in rolled_back:
            item._active = False
        del self._reservations[index:]

    def flush(self, *, chunk_period_ms: int) -> dict[str, object] | None:
        if not self._buffer:
            return None
        payload: dict[str, object] = {
            "type": "audio",
            "audio": "",
            "format": "pcm_f32le",
            "sample_rate_hz": self._sample_rate_hz or 16000,
            "force_listen": self._force_listen,
            "is_speech": self._is_speech,
        }
        return self.append(payload, chunk_period_ms=chunk_period_ms, flush=True)

    def commit(self, *, chunk_period_ms: int) -> dict[str, object] | None:
        self._reservation_seq += 1
        reservation = self.prepare_commit(
            operation_id=f"immediate-commit-{self._reservation_seq}",
            chunk_period_ms=chunk_period_ms,
        )
        reservation.commit()
        return reservation.payload


__all__ = [
    "BasePcmAppendBuffer",
    "PcmAppendReservation",
]
