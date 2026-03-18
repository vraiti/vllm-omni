# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Kubernetes-native orchestrator for vLLM-Omni using Watch API and HTTP."""

import asyncio
import base64
import os
from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from vllm.logger import init_logger

from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.k8s.rbac_utils import K8sWatchClient
from vllm_omni.inputs.data import OmniPromptType, OmniSamplingParams
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


class K8sOmniOrchestrator:
    """Kubernetes-native orchestrator for vLLM-Omni.

    Replaces Ray-based orchestration with pure Kubernetes primitives:
    - REST API for control plane (HTTP POST/DELETE)
    - Watch API for service discovery (event-driven pod detection)
    - Manual round-robin load balancing
    - NIXL over UCX for data plane (GPU-to-GPU transfers)

    Args:
        model: Model name or path
        namespace: Kubernetes namespace (default: from NAMESPACE env var or "default")
        label_selector: Pod label selector (default: "app=vllm-omni")
        num_stages: Number of pipeline stages (default: 3)
        stage_worker_port: Stage worker HTTP port (default: 8080)
        http_timeout: HTTP request timeout in seconds (default: 300)
        **kwargs: Additional arguments (for compatibility)
    """

    def __init__(
        self,
        model: str,
        namespace: str | None = None,
        label_selector: str | None = None,
        num_stages: int = 3,
        stage_worker_port: int = 8080,
        http_timeout: float = 300.0,
        **kwargs: Any,
    ):
        self.model = model
        self.namespace = namespace or os.getenv("NAMESPACE", "default")
        self.label_selector = label_selector or "app=vllm-omni"
        self.num_stages = num_stages
        self.stage_worker_port = stage_worker_port
        self.http_timeout = http_timeout

        # HTTP client for stage worker communication
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(http_timeout))

        # Pod discovery state: stage_id -> list of pod IPs
        self.stage_pods: dict[int, list[str]] = defaultdict(list)
        self.stage_pods_lock = asyncio.Lock()

        # Round-robin state: stage_id -> current index
        self.rr_index: dict[int, int] = defaultdict(int)

        # Request tracking
        self.request_states: dict[str, ClientRequestState] = {}

        # Watch client for pod discovery
        self.watch_client = K8sWatchClient(
            namespace=self.namespace,
            label_selector=self.label_selector,
        )

        # Watch task
        self._watch_task: asyncio.Task | None = None
        self._running = False

        # Compatibility attributes for AsyncOmni interface
        self.stage_configs: list[dict[str, Any]] | None = None
        self.input_processor = None
        self.io_processor = None
        self.model_config = None

        logger.info(
            f"Initialized K8sOmniOrchestrator: model={model}, namespace={self.namespace}, "
            f"label_selector={self.label_selector}, num_stages={num_stages}"
        )

    async def start(self):
        """Start the orchestrator and begin watching pods."""
        if self._running:
            logger.warning("Orchestrator already running")
            return

        self._running = True

        # Start Watch API task
        self._watch_task = asyncio.create_task(self._watch_pods())

        logger.info("Orchestrator started")

    async def shutdown(self):
        """Shutdown the orchestrator and cleanup resources."""
        if not self._running:
            return

        self._running = False

        # Cancel watch task
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

        # Close HTTP client
        await self.http.aclose()

        logger.info("Orchestrator shutdown complete")

    async def _watch_pods(self):
        """Background task to watch pod events and update stage_pods mapping.

        Subscribes to Kubernetes Watch API for real-time pod discovery.
        Updates stage_pods dict when pods are added/deleted.
        """
        logger.info("Starting pod watch task")

        try:
            async for event in self.watch_client.watch_pods():
                await self._handle_pod_event(event)
        except asyncio.CancelledError:
            logger.info("Pod watch task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in pod watch task: {e}", exc_info=True)
            # Retry after delay
            if self._running:
                logger.info("Restarting pod watch task in 5 seconds...")
                await asyncio.sleep(5)
                if self._running:
                    self._watch_task = asyncio.create_task(self._watch_pods())

    async def _handle_pod_event(self, event: dict[str, Any]):
        """Handle pod event from Watch API.

        Args:
            event: Event dictionary with 'type' and 'object' fields
        """
        event_type = event.get("type")
        pod = event.get("object")

        if not pod:
            return

        # Extract metadata
        metadata = pod.get("metadata", {})
        labels = metadata.get("labels", {})
        pod_name = metadata.get("name", "unknown")

        # Get stage ID from label
        stage_id_str = labels.get("stage")
        if not stage_id_str:
            logger.debug(f"Pod {pod_name} has no 'stage' label, skipping")
            return

        try:
            stage_id = int(stage_id_str)
        except ValueError:
            logger.warning(f"Invalid stage label '{stage_id_str}' on pod {pod_name}")
            return

        # Get pod IP
        status = pod.get("status", {})
        pod_ip = status.get("podIP")
        phase = status.get("phase")

        if event_type == "ADDED" and phase == "Running" and pod_ip:
            async with self.stage_pods_lock:
                if pod_ip not in self.stage_pods[stage_id]:
                    self.stage_pods[stage_id].append(pod_ip)
                    logger.info(f"Added pod {pod_name} ({pod_ip}) to stage-{stage_id}")

        elif event_type == "DELETED":
            async with self.stage_pods_lock:
                if pod_ip and pod_ip in self.stage_pods[stage_id]:
                    self.stage_pods[stage_id].remove(pod_ip)
                    logger.info(f"Removed pod {pod_name} ({pod_ip}) from stage-{stage_id}")

        elif event_type == "MODIFIED":
            # Handle pod becoming ready or not ready
            if phase == "Running" and pod_ip:
                async with self.stage_pods_lock:
                    if pod_ip not in self.stage_pods[stage_id]:
                        self.stage_pods[stage_id].append(pod_ip)
                        logger.info(f"Pod {pod_name} ({pod_ip}) became ready for stage-{stage_id}")
            elif phase != "Running" and pod_ip:
                async with self.stage_pods_lock:
                    if pod_ip in self.stage_pods[stage_id]:
                        self.stage_pods[stage_id].remove(pod_ip)
                        logger.info(f"Pod {pod_name} ({pod_ip}) is no longer ready for stage-{stage_id}")

    async def _pick_pod(self, stage_id: int) -> str:
        """Pick a pod for the given stage using round-robin.

        Args:
            stage_id: Stage ID to pick pod for

        Returns:
            Pod IP address

        Raises:
            RuntimeError: If no pods available for stage
        """
        async with self.stage_pods_lock:
            pods = self.stage_pods.get(stage_id, [])

            if not pods:
                raise RuntimeError(f"No pods available for stage {stage_id}")

            # Round-robin selection
            idx = self.rr_index[stage_id] % len(pods)
            self.rr_index[stage_id] += 1

            pod_ip = pods[idx]
            logger.debug(f"Selected pod {pod_ip} for stage {stage_id} (index {idx}/{len(pods)})")

            return pod_ip

    async def _process_stage(
        self,
        stage_id: int,
        request_id: str,
        data: bytes | None = None,
        nixl_source_ip: str | None = None,
        nixl_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process request through a single stage (blocking HTTP POST).

        Args:
            stage_id: Target stage ID
            request_id: Unique request identifier
            data: Raw input data (Stage-0 only)
            nixl_source_ip: Source pod IP for NIXL transfer (Stage-1+)
            nixl_metadata: NIXL buffer metadata (Stage-1+)

        Returns:
            Response dict with 'pod_ip', 'nixl_metadata', and optionally 'result'

        Raises:
            RuntimeError: If HTTP request fails
        """
        # Pick pod using round-robin
        pod_ip = await self._pick_pod(stage_id)

        # Build request
        request_data: dict[str, Any] = {
            "request_id": request_id,
            "stage_id": stage_id,
        }

        if stage_id == 0:
            # Stage-0: send base64-encoded data
            if data is None:
                raise ValueError("Stage-0 requires data")
            request_data["data"] = base64.b64encode(data).decode()
        else:
            # Stage-1+: send NIXL metadata
            if nixl_source_ip is None or nixl_metadata is None:
                raise ValueError(f"Stage-{stage_id} requires nixl_source_ip and nixl_metadata")
            request_data["nixl_source_ip"] = nixl_source_ip
            request_data["nixl_metadata"] = nixl_metadata

        # Send HTTP POST (blocking until stage completes)
        url = f"http://{pod_ip}:{self.stage_worker_port}/v1/generate"

        logger.debug(f"POST {url} (request_id={request_id})")

        try:
            response = await self.http.post(url, json=request_data)
            response.raise_for_status()

            result = response.json()
            logger.debug(f"Stage-{stage_id} completed for request {request_id}")

            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from stage-{stage_id} pod {pod_ip}: {e}")
            raise RuntimeError(f"Stage-{stage_id} returned HTTP {e.response.status_code}") from e
        except httpx.RequestError as e:
            logger.error(f"Request error to stage-{stage_id} pod {pod_ip}: {e}")
            raise RuntimeError(f"Failed to connect to stage-{stage_id} pod {pod_ip}") from e

    async def _cleanup_buffer(
        self,
        pod_ip: str,
        request_id: str,
        from_stage: int,
        to_stage: int,
    ):
        """Clean up buffer on stage worker (async, fire-and-forget).

        Args:
            pod_ip: Pod IP address
            request_id: Request identifier
            from_stage: Source stage ID
            to_stage: Destination stage ID
        """
        url = f"http://{pod_ip}:{self.stage_worker_port}/v1/buffers/{request_id}"
        params = {"from_stage": from_stage, "to_stage": to_stage}

        logger.debug(f"DELETE {url} (async cleanup)")

        try:
            # Fire-and-forget DELETE (don't wait for response)
            asyncio.create_task(self._cleanup_buffer_impl(url, params))
        except Exception as e:
            logger.warning(f"Failed to initiate buffer cleanup: {e}")

    async def _cleanup_buffer_impl(self, url: str, params: dict[str, int]):
        """Implementation of async buffer cleanup."""
        try:
            response = await self.http.delete(url, params=params)
            response.raise_for_status()
            logger.debug(f"Buffer cleanup successful: {url}")
        except Exception as e:
            logger.warning(f"Buffer cleanup failed for {url}: {e}")

    async def process_pipeline(
        self,
        prompt: str | bytes,
        request_id: str,
    ) -> dict[str, Any]:
        """Process request through full pipeline (blocking).

        Coordinates execution through all stages:
        1. Stage-0: Processes input, stores in NIXL buffer
        2. Stage-1: RDMA GET from Stage-0, processes, stores in NIXL buffer
        3. Cleanup Stage-0 buffer (async)
        4. Stage-2: RDMA GET from Stage-1, processes, returns final output
        5. Cleanup Stage-1 buffer (async)

        Args:
            prompt: Input prompt (text or bytes)
            request_id: Unique request identifier

        Returns:
            Final output from last stage

        Raises:
            RuntimeError: If any stage fails
        """
        logger.info(f"Processing pipeline for request {request_id}")

        # Convert prompt to bytes
        if isinstance(prompt, str):
            data = prompt.encode("utf-8")
        else:
            data = prompt

        # Stage-0: Initial processing
        stage_0_result = await self._process_stage(
            stage_id=0,
            request_id=request_id,
            data=data,
        )

        stage_0_pod_ip = stage_0_result["pod_ip"]
        stage_0_metadata = stage_0_result["nixl_metadata"]

        # Stage-1: Intermediate processing
        stage_1_result = await self._process_stage(
            stage_id=1,
            request_id=request_id,
            nixl_source_ip=stage_0_pod_ip,
            nixl_metadata=stage_0_metadata,
        )

        # Cleanup Stage-0 buffer (async)
        await self._cleanup_buffer(
            pod_ip=stage_0_pod_ip,
            request_id=request_id,
            from_stage=0,
            to_stage=1,
        )

        stage_1_pod_ip = stage_1_result["pod_ip"]
        stage_1_metadata = stage_1_result["nixl_metadata"]

        # Stage-2: Final processing
        stage_2_result = await self._process_stage(
            stage_id=2,
            request_id=request_id,
            nixl_source_ip=stage_1_pod_ip,
            nixl_metadata=stage_1_metadata,
        )

        # Cleanup Stage-1 buffer (async)
        await self._cleanup_buffer(
            pod_ip=stage_1_pod_ip,
            request_id=request_id,
            from_stage=1,
            to_stage=2,
        )

        logger.info(f"Pipeline completed for request {request_id}")

        return stage_2_result

    async def generate(
        self,
        prompt: OmniPromptType,
        request_id: str,
        sampling_params_list: list[OmniSamplingParams] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Generate outputs for the given prompt (async generator for compatibility).

        This method provides compatibility with AsyncOmni interface by wrapping
        the blocking pipeline execution in an async generator.

        Args:
            prompt: Input prompt
            request_id: Unique request identifier
            sampling_params_list: Sampling parameters (currently unused in K8s mode)
            **kwargs: Additional arguments (for compatibility)

        Yields:
            OmniRequestOutput with final result
        """
        # Track request
        req_state = ClientRequestState(request_id)
        self.request_states[request_id] = req_state

        try:
            # Convert prompt to string/bytes
            if isinstance(prompt, dict):
                # Extract text from dict prompt
                prompt_text = prompt.get("prompt", "")
            elif isinstance(prompt, str):
                prompt_text = prompt
            else:
                prompt_text = str(prompt)

            # Process through pipeline
            result = await self.process_pipeline(prompt_text, request_id)

            # Yield final output
            output = OmniRequestOutput(
                request_id=request_id,
                stage_id=self.num_stages - 1,
                final_output_type="audio",  # TODO: Get from config
                request_output=result.get("result"),
            )

            yield output

        finally:
            # Cleanup request state
            self.request_states.pop(request_id, None)

    async def health(self) -> dict[str, Any]:
        """Get orchestrator health status.

        Returns:
            Health information including pod counts per stage
        """
        async with self.stage_pods_lock:
            pod_counts = {stage_id: len(pods) for stage_id, pods in self.stage_pods.items()}

        return {
            "status": "healthy",
            "namespace": self.namespace,
            "label_selector": self.label_selector,
            "num_stages": self.num_stages,
            "pod_counts": pod_counts,
        }

    async def get_vllm_config(self):
        """Get vLLM configuration (compatibility method for AsyncOmni interface).

        Returns None for K8s orchestrator as config is distributed across stage workers.
        """
        return None

    async def get_supported_tasks(self) -> set[str]:
        """Get supported tasks (compatibility method for AsyncOmni interface).

        Returns:
            Set of supported task types
        """
        # K8s orchestrator supports standard multi-modal tasks
        return {"generate", "chat", "audio"}
