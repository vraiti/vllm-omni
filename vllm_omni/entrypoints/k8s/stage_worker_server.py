# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stage worker REST API server for Kubernetes deployment."""

import asyncio
import base64
import os
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

from vllm_omni.distributed.omni_connectors import build_stage_connectors
from vllm_omni.distributed.omni_connectors.connectors.base import OmniConnectorBase
from vllm_omni.engine.arg_utils import AsyncOmniEngineArgs
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.entrypoints.async_omni_llm import AsyncOmniLLM
from vllm_omni.entrypoints.k8s.stage_config_loader import load_stage_config
from vllm_omni.entrypoints.stage_utils import _to_dict, set_stage_devices
from vllm_omni.entrypoints.utils import filter_dataclass_kwargs
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniSamplingParams

logger = init_logger(__name__)


# Request/Response Models
class NixlMetadata(BaseModel):
    """NIXL transfer metadata."""

    buffer_ptr: int = Field(description="GPU buffer pointer")
    size: int = Field(description="Buffer size in bytes")
    pod_ip: str = Field(description="Source pod IP address")


class GenerateRequest(BaseModel):
    """Request for generation endpoint."""

    request_id: str = Field(description="Unique request identifier")
    stage_id: int = Field(description="Target stage ID")
    data: str | None = Field(default=None, description="Base64-encoded input data (Stage-0 only)")
    nixl_source_ip: str | None = Field(default=None, description="Source pod IP for NIXL transfer")
    nixl_metadata: dict[str, Any] | None = Field(default=None, description="NIXL buffer metadata")


class GenerateResponse(BaseModel):
    """Response from generation endpoint."""

    pod_ip: str = Field(description="This pod's IP address")
    nixl_metadata: dict[str, Any] = Field(description="NIXL buffer metadata for output")
    result: Any | None = Field(default=None, description="Final output (if terminal stage)")


class BufferCleanupResponse(BaseModel):
    """Response from buffer cleanup endpoint."""

    status: Literal["ok", "error"] = Field(description="Cleanup status")
    message: str | None = Field(default=None, description="Optional status message")


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy"] = Field(description="Health status")
    stage_id: int = Field(description="Stage ID")
    pod_ip: str = Field(description="Pod IP address")
    connector_health: dict[str, Any] | None = Field(default=None, description="Connector health info")


# Global state
class StageWorkerState:
    """Global state for stage worker."""

    def __init__(self):
        self.stage_id: int | None = None
        self.pod_ip: str | None = None
        self.engine: AsyncOmniLLM | AsyncOmniDiffusion | None = None
        self.connector: OmniConnectorBase | None = None
        self.stage_type: str | None = None
        self.is_final_stage: bool = False


state = StageWorkerState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    logger.info(f"Starting stage worker server for stage {state.stage_id}")
    yield
    logger.info(f"Shutting down stage worker server for stage {state.stage_id}")

    # Cleanup resources
    if state.connector:
        try:
            state.connector.close()
        except Exception as e:
            logger.warning(f"Error closing connector: {e}")

    if state.engine:
        try:
            # Shutdown engine if it has a shutdown method
            if hasattr(state.engine, "shutdown"):
                await state.engine.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down engine: {e}")


def create_stage_worker_app(
    model: str,
    stage_config_path: str,
    stage_id: int,
    pod_ip: str | None = None,
) -> FastAPI:
    """Create FastAPI application for stage worker.

    Args:
        model: Model name or path
        stage_config_path: Path to stage configuration YAML
        stage_id: Stage ID for this worker
        pod_ip: Pod IP address (from K8s downward API)

    Returns:
        Configured FastAPI application
    """
    # Load stage configuration
    config_data = load_stage_config(stage_config_path, stage_id)
    stage_config = config_data["stage_config"]
    runtime_config = config_data["runtime_config"]

    # Get pod IP from env if not provided
    if pod_ip is None:
        pod_ip = os.getenv("POD_IP", "127.0.0.1")

    # Set global state
    state.stage_id = stage_id
    state.pod_ip = pod_ip
    state.stage_type = stage_config.get("stage_type", "llm")
    state.is_final_stage = stage_config.get("final_output", False)

    # Set stage devices
    runtime = stage_config.get("runtime", {})
    devices = runtime.get("devices")
    if devices:
        set_stage_devices(stage_id, devices)
        logger.info(f"Set stage {stage_id} devices: {devices}")

    # Build engine args
    engine_args_dict = stage_config.get("engine_args", {})
    engine_args = AsyncOmniEngineArgs.from_cli_args({
        "model": model,
        **engine_args_dict,
    })

    # Filter kwargs for VllmConfig
    filtered_kwargs = filter_dataclass_kwargs(VllmConfig, vars(engine_args))
    vllm_config = VllmConfig(**filtered_kwargs)

    # Get executor class
    from vllm.v1.executor.gpu_executor import GPUExecutor as executor_class

    # Build connector
    connectors_config = {}
    # Add NIXL connector configuration
    nixl_config = {
        "transport": os.getenv("NIXL_TRANSPORT", "tcp"),
        "pod_ip": pod_ip,
        "pool_size_gb": engine_args_dict.get("nixl_pool_size_gb", 1),
    }
    connectors_config["nixl"] = {"name": "NixlConnector", "extra": nixl_config}

    connectors = build_stage_connectors(connectors_config)
    if "nixl" in connectors:
        state.connector = connectors["nixl"]
        logger.info(f"Initialized NIXL connector: transport={nixl_config['transport']}, pod_ip={pod_ip}")
    else:
        logger.warning("NIXL connector not available")

    # Initialize engine based on stage type
    if state.stage_type == "llm":
        state.engine = AsyncOmniLLM(
            engine_args=engine_args,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=True,
            usage_context=UsageContext.OPENAI_API_SERVER,
            start_engine_loop=True,
        )
        logger.info(f"Initialized AsyncOmniLLM for stage {stage_id}")
    elif state.stage_type == "diffusion":
        state.engine = AsyncOmniDiffusion(
            engine_args=engine_args,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=True,
            usage_context=UsageContext.OPENAI_API_SERVER,
            start_engine_loop=True,
        )
        logger.info(f"Initialized AsyncOmniDiffusion for stage {stage_id}")
    else:
        raise ValueError(f"Unknown stage type: {state.stage_type}")

    # Create FastAPI app
    app = FastAPI(
        title=f"vLLM-Omni Stage Worker (Stage {stage_id})",
        description="Kubernetes-native stage worker REST API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Register endpoints
    @app.post("/v1/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """Process generation request (blocking until complete).

        For Stage-0:
            - Receives base64-encoded input data
            - Processes through engine
            - Returns NIXL buffer metadata

        For Stage-1+:
            - Receives NIXL metadata from previous stage
            - Performs RDMA GET to retrieve input
            - Processes through engine
            - Returns NIXL buffer metadata
        """
        try:
            logger.info(f"Received generate request: {request.request_id}")

            # Get input data
            if request.stage_id == 0:
                # Stage-0: decode base64 input
                if request.data is None:
                    raise HTTPException(status_code=400, detail="Stage-0 requires 'data' field")

                input_data = base64.b64decode(request.data)
                logger.debug(f"Stage-0 input: {len(input_data)} bytes")

            else:
                # Stage-1+: RDMA GET from previous stage
                if request.nixl_metadata is None or request.nixl_source_ip is None:
                    raise HTTPException(status_code=400, detail="Stages 1+ require 'nixl_metadata' and 'nixl_source_ip'")

                if state.connector is None:
                    raise HTTPException(status_code=500, detail="NIXL connector not initialized")

                # Perform RDMA transfer
                metadata = {**request.nixl_metadata, "pod_ip": request.nixl_source_ip}
                result = state.connector.get(
                    from_stage=str(request.stage_id - 1),
                    to_stage=str(request.stage_id),
                    get_key=request.request_id,
                    metadata=metadata,
                )

                if result is None:
                    raise HTTPException(status_code=404, detail="Failed to retrieve data from previous stage")

                input_data, _ = result
                logger.debug(f"Stage-{request.stage_id} received data via RDMA")

            # Process through engine
            # TODO: Integrate with actual engine generate() method
            # This is a placeholder - actual implementation needs to:
            # 1. Convert input_data to proper engine input format
            # 2. Call engine.generate() with appropriate sampling params
            # 3. Collect output
            logger.info(f"Processing request {request.request_id} through engine...")

            # Simulate processing (replace with actual engine call)
            await asyncio.sleep(0.1)
            output_data = b"dummy_output_" + request.request_id.encode()

            # Store output in NIXL buffer (if not final stage)
            nixl_metadata = {}
            if state.connector and not state.is_final_stage:
                success, size, metadata = state.connector.put(
                    from_stage=str(request.stage_id),
                    to_stage=str(request.stage_id + 1),
                    put_key=request.request_id,
                    data=output_data,
                )

                if not success:
                    raise HTTPException(status_code=500, detail="Failed to store output in buffer")

                nixl_metadata = metadata
                logger.debug(f"Stored output in NIXL buffer: {size} bytes")

            # Return response
            response = GenerateResponse(
                pod_ip=state.pod_ip,
                nixl_metadata=nixl_metadata,
                result=output_data.decode() if state.is_final_stage else None,
            )

            logger.info(f"Completed request {request.request_id}")
            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing request {request.request_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/v1/buffers/{request_id}", response_model=BufferCleanupResponse)
    async def cleanup_buffers(
        request_id: str,
        from_stage: int = Query(..., description="Source stage ID"),
        to_stage: int = Query(..., description="Destination stage ID"),
    ):
        """Clean up buffers for a completed request (async, non-blocking)."""
        try:
            logger.info(f"Cleaning up buffers for request {request_id} (from {from_stage} to {to_stage})")

            if state.connector is None:
                return BufferCleanupResponse(status="ok", message="No connector initialized")

            # Async cleanup
            state.connector.cleanup(request_id)

            return BufferCleanupResponse(status="ok")

        except Exception as e:
            logger.error(f"Error cleaning up buffers for {request_id}: {e}", exc_info=True)
            return BufferCleanupResponse(status="error", message=str(e))

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        connector_health = None
        if state.connector:
            try:
                connector_health = state.connector.health()
            except Exception as e:
                logger.warning(f"Error getting connector health: {e}")

        is_healthy = state.engine is not None
        status = "healthy" if is_healthy else "unhealthy"

        return HealthResponse(
            status=status,
            stage_id=state.stage_id,
            pod_ip=state.pod_ip,
            connector_health=connector_health,
        )

    logger.info(f"Created stage worker app for stage {stage_id} (type={state.stage_type}, pod_ip={pod_ip})")

    return app
