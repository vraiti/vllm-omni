# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreOutputs

if TYPE_CHECKING:
    from vllm_omni.engine.stage_client import StagePoolLLMClient

logger = init_logger(__name__)


class EngineGroupDemux:
    """Demultiplexes a shared engine's output stream to per-stage queues.

    When multiple logical stages share one vLLM engine (an "engine group"),
    this component:
    1. Runs one background poller per replica, reading from the shared
       engine client's ``get_output_async()``.
    2. Routes each ``EngineCoreOutput`` to the correct stage's
       ``asyncio.Queue`` based on an explicit request-id to stage-id
       registry.
    3. Each stage pool reads from its own queue -- no stage is special.
    """

    def __init__(
        self,
        group_name: str,
        stage_ids: list[int],
        clients: list[Any],
        stats_owner_stage: int,
    ) -> None:
        self._group_name = group_name
        self._stage_ids = list(stage_ids)
        self._clients = list(clients)
        self._stats_owner_stage = stats_owner_stage

        self._request_stage: dict[str, int] = {}
        self._stage_queues: dict[int, asyncio.Queue[EngineCoreOutputs]] = {sid: asyncio.Queue() for sid in stage_ids}
        self._poller_tasks: list[asyncio.Task[None]] = []
        self._shutdown = asyncio.Event()

    def register_request(self, request_id: str, stage_id: int) -> None:
        self._request_stage[request_id] = stage_id

    def unregister_request(self, request_id: str) -> None:
        self._request_stage.pop(request_id, None)

    def get_stage_queue(self, stage_id: int) -> asyncio.Queue[EngineCoreOutputs]:
        return self._stage_queues[stage_id]

    async def start(self) -> None:
        for replica_id, client in enumerate(self._clients):
            if client is None:
                continue
            task = asyncio.create_task(
                self._poll_replica(replica_id, client),
                name=f"demux-{self._group_name}-r{replica_id}",
            )
            self._poller_tasks.append(task)

    async def stop(self) -> None:
        self._shutdown.set()
        for task in self._poller_tasks:
            task.cancel()
        for task in self._poller_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._poller_tasks.clear()

    async def _poll_replica(
        self,
        replica_id: int,
        client: StagePoolLLMClient,
    ) -> None:
        while not self._shutdown.is_set():
            try:
                raw: EngineCoreOutputs = await client.get_output_async()
            except asyncio.CancelledError:
                return
            except Exception:
                if self._shutdown.is_set():
                    return
                raise

            if not raw.outputs:
                continue

            buckets: dict[int, list[Any]] = defaultdict(list)
            for eco in raw.outputs:
                sid = self._request_stage.get(eco.request_id)
                if sid is None:
                    logger.warning(
                        "[EngineGroupDemux %s] Unknown request_id %s; dropping output",
                        self._group_name,
                        eco.request_id,
                    )
                    continue
                buckets[sid].append(eco)

            for sid, outputs in buckets.items():
                queue = self._stage_queues.get(sid)
                if queue is None:
                    continue
                stats = raw.scheduler_stats if sid == self._stats_owner_stage else None
                partitioned = EngineCoreOutputs(
                    engine_index=raw.engine_index,
                    outputs=outputs,
                    scheduler_stats=stats,
                    timestamp=raw.timestamp,
                )
                queue.put_nowait(partitioned)
