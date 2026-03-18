# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import threading
import time as _time_mod
from typing import Any

import torch

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError:
    nixl_agent = None
    nixl_agent_config = None

# Stale buffer TTL: buffers older than this are automatically reclaimed
# to prevent memory leaks when receiver crashes or gives up.
_BUFFER_TTL_SECONDS = 300  # 5 minutes


class BufferAllocator:
    """
    Manages the allocation of memory segments within the registered pool.
    Thread-safe implementation using a simple free list.
    """

    def __init__(self, total_size: int, alignment: int = 4096):
        self.total_size = total_size
        self.alignment = alignment
        self.lock = threading.Lock()
        # Free list: [(start, size), ...] sorted by start
        self.free_blocks = [(0, total_size)]

    def alloc(self, size: int) -> int:
        """
        Allocates a block of 'size' bytes.
        Returns the starting offset.
        """
        # Align size upwards
        aligned_size = (size + self.alignment - 1) // self.alignment * self.alignment

        with self.lock:
            for i, (start, block_size) in enumerate(self.free_blocks):
                if block_size >= aligned_size:
                    # Found a block
                    new_start = start + aligned_size
                    new_size = block_size - aligned_size

                    if new_size > 0:
                        self.free_blocks[i] = (new_start, new_size)
                    else:
                        self.free_blocks.pop(i)
                    return start

        raise MemoryError(f"Out of memory in buffer pool. Requested {size} bytes (aligned {aligned_size}).")

    def free(self, offset: int, size: int):
        """
        Frees a previously allocated block.
        """
        aligned_size = (size + self.alignment - 1) // self.alignment * self.alignment

        with self.lock:
            # Check for double-free and corruption
            for start, length in self.free_blocks:
                # Case 1: Exact match = double free, safe to ignore
                if offset == start and aligned_size == length:
                    logger.warning(f"Double free detected at offset {offset}, size {aligned_size}. Ignoring.")
                    return
                # Case 2: Block is fully contained within an existing free block = also double free
                if offset >= start and offset + aligned_size <= start + length:
                    logger.warning(
                        f"Double free detected: block {offset}-{offset + aligned_size} "
                        f"is already within free block {start}-{start + length}. Ignoring."
                    )
                    return
                # Case 3: Partial overlap (but not fully contained) = memory corruption
                if not (offset + aligned_size <= start or start + length <= offset):
                    raise RuntimeError(
                        f"Memory corruption detected: freeing {offset}-{offset + aligned_size} "
                        f"partially overlaps with free block {start}-{start + length}"
                    )

            self.free_blocks.append((offset, aligned_size))
            self.free_blocks.sort()  # Sort by offset

            # Merge adjacent blocks
            i = 0
            while i < len(self.free_blocks) - 1:
                curr_start, curr_size = self.free_blocks[i]
                next_start, next_size = self.free_blocks[i + 1]

                if curr_start + curr_size == next_start:
                    self.free_blocks[i] = (curr_start, curr_size + next_size)
                    self.free_blocks.pop(i + 1)
                else:
                    i += 1

    def get_stats(self) -> dict[str, Any]:
        """Return allocator statistics."""
        with self.lock:
            free_bytes = sum(size for _, size in self.free_blocks)
            allocated_bytes = self.total_size - free_bytes
            return {
                "total_bytes": self.total_size,
                "allocated_bytes": allocated_bytes,
                "free_bytes": free_bytes,
                "num_free_blocks": len(self.free_blocks),
                "fragmentation": len(self.free_blocks) > 1,
            }


class ManagedBuffer:
    """
    A temporary view into the global memory pool.
    Must be kept alive while the data view is being used.
    """

    def __init__(self, allocator: BufferAllocator, offset: int, size: int, pool_tensor: torch.Tensor):
        self.allocator = allocator
        self.offset = offset
        self.size = size
        self.pool_tensor = pool_tensor
        self._released = False

    def release(self):
        """Explicitly release the buffer back to the pool."""
        if not self._released:
            self.allocator.free(self.offset, self.size)
            self._released = True

    def __del__(self):
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    @property
    def tensor(self) -> torch.Tensor:
        """
        Returns a 1D uint8 zero-copy view of the buffer.
        """
        return self.pool_tensor[self.offset : self.offset + self.size]

    def as_tensor(self, dtype: torch.dtype, shape: tuple) -> torch.Tensor:
        """
        Returns a typed, shaped zero-copy view.
        Validates size, shape, and alignment.
        """
        itemsize = torch.tensor([], dtype=dtype).element_size()

        # Calculate expected size
        expected_bytes = itemsize
        for dim in shape:
            if dim < 0:
                raise ValueError("Dynamic dimension (-1) is not supported in as_tensor")
            expected_bytes *= dim

        if expected_bytes != self.size:
            raise ValueError(
                f"Shape {shape} with dtype {dtype} requires {expected_bytes} bytes, but buffer size is {self.size}"
            )

        # Check alignment for the dtype
        if self.offset % itemsize != 0:
            raise ValueError(f"Buffer offset {self.offset} is not aligned to {dtype} ({itemsize} bytes)")

        # Create zero-copy view
        raw_view = self.pool_tensor[self.offset : self.offset + self.size]
        typed_view = raw_view.view(dtype)
        return typed_view.reshape(shape)


class NixlConnector(OmniConnectorBase):
    """
    NIXL connector for GPU-to-GPU RDMA transfers using UCX backend.
    Supports both RDMA and TCP transport modes.
    """

    supports_raw_data = True

    def __init__(self, config: dict[str, Any]):
        if nixl_agent is None or nixl_agent_config is None:
            raise ImportError(
                "NIXL library not found. Install with: pip install nixl\n"
                "For UCX support, ensure UCX is installed and configured."
            )

        # Extract configuration
        self.pool_size_gb = config.get("pool_size_gb", 1)
        self.nixl_port = config.get("nixl_port", 5600)

        # Transport configuration: tcp or rdma
        # Priority: config dict > env var > default (tcp for safety)
        transport = config.get("transport", os.getenv("NIXL_TRANSPORT", "tcp")).lower()
        if transport not in ("tcp", "rdma"):
            raise ValueError(f"Invalid NIXL transport: {transport}. Must be 'tcp' or 'rdma'.")
        self.transport = transport

        # Get pod IP from environment (injected by K8s downward API)
        self.pod_ip = config.get("pod_ip", os.getenv("POD_IP", "127.0.0.1"))

        # Initialize NIXL agent with UCX backend
        # UCX will use the configured transport (RDMA or TCP)
        agent_config = nixl_agent_config(backends=["UCX"])
        self.nixl_agent = nixl_agent(f"{self.pod_ip}:{self.nixl_port}", agent_config)

        logger.info(
            f"Initialized NIXL agent: pod_ip={self.pod_ip}, port={self.nixl_port}, "
            f"transport={self.transport}, pool_size={self.pool_size_gb}GB"
        )

        # Allocate GPU memory pool
        pool_size_bytes = self.pool_size_gb * 1024**3
        self.pool = torch.empty(pool_size_bytes, dtype=torch.uint8, device="cuda")
        self.pool.pin_memory()  # Pin for faster transfers

        # Initialize allocator
        self.allocator = BufferAllocator(pool_size_bytes)

        # Buffer tracking: key -> (ManagedBuffer, timestamp)
        self.buffers: dict[str, tuple[ManagedBuffer, float]] = {}
        self.buffers_lock = threading.Lock()

        # Start background cleanup thread
        self._cleanup_running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_stale_buffers, daemon=True)
        self._cleanup_thread.start()

        logger.info(
            f"NixlConnector initialized: pool_size={pool_size_bytes / 1024**3:.2f}GB, "
            f"transport={self.transport}, pod_ip={self.pod_ip}"
        )

    def put(self, from_stage: str, to_stage: str, put_key: str, data: Any) -> tuple[bool, int, dict[str, Any] | None]:
        """
        Store data in GPU buffer and return metadata for RDMA transfer.

        Args:
            from_stage: Source stage identifier
            to_stage: Destination stage identifier
            put_key: Unique request identifier
            data: Torch tensor or bytes to store

        Returns:
            (success, size, metadata) where metadata contains:
                - buffer_ptr: GPU buffer pointer
                - size: Buffer size in bytes
                - pod_ip: This pod's IP address
        """
        key = self._make_key(put_key, from_stage, to_stage)

        # Convert data to tensor if needed
        if isinstance(data, bytes):
            # Convert bytes to tensor
            data_tensor = torch.frombuffer(data, dtype=torch.uint8).cuda()
        elif isinstance(data, torch.Tensor):
            # Ensure on GPU
            if data.device.type != "cuda":
                data_tensor = data.cuda()
            else:
                data_tensor = data
        else:
            # Serialize and convert to tensor
            serialized = self.serialize_obj(data)
            data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).cuda()

        # Flatten to 1D uint8
        if data_tensor.dtype != torch.uint8:
            data_tensor = data_tensor.view(torch.uint8)
        data_tensor = data_tensor.contiguous().view(-1)

        size = data_tensor.numel()

        try:
            # Allocate buffer
            offset = self.allocator.alloc(size)
            managed_buffer = ManagedBuffer(self.allocator, offset, size, self.pool)

            # Copy data to pool
            managed_buffer.tensor.copy_(data_tensor)

            # Store buffer with timestamp
            with self.buffers_lock:
                self.buffers[key] = (managed_buffer, _time_mod.time())

            # Get buffer pointer for RDMA
            buffer_ptr = self.pool.data_ptr() + offset

            metadata = {
                "buffer_ptr": buffer_ptr,
                "size": size,
                "pod_ip": self.pod_ip,
            }

            logger.debug(f"PUT {key}: size={size}, offset={offset}, ptr={buffer_ptr:x}")
            return True, size, metadata

        except MemoryError as e:
            logger.error(f"Failed to allocate buffer for {key}: {e}")
            return False, 0, None

    def get(self, from_stage: str, to_stage: str, get_key: str, metadata: dict | None = None) -> tuple[Any, int] | None:
        """
        Retrieve data via RDMA GET from remote pod.

        Args:
            from_stage: Source stage identifier
            to_stage: Destination stage identifier
            get_key: Unique request identifier
            metadata: Transfer metadata from PUT operation containing:
                - buffer_ptr: Remote GPU buffer pointer
                - size: Buffer size in bytes
                - pod_ip: Remote pod's IP address

        Returns:
            (data, size) tuple or None if not found
        """
        if metadata is None:
            logger.error(f"GET {get_key}: metadata is required for NIXL transfers")
            return None

        try:
            remote_ptr = metadata["buffer_ptr"]
            size = metadata["size"]
            remote_ip = metadata["pod_ip"]

            # Allocate local buffer
            offset = self.allocator.alloc(size)
            managed_buffer = ManagedBuffer(self.allocator, offset, size, self.pool)
            local_ptr = self.pool.data_ptr() + offset

            # Perform RDMA GET from remote pod
            # NIXL API: get(remote_addr, local_addr, size, remote_endpoint)
            remote_endpoint = f"{remote_ip}:{self.nixl_port}"

            logger.debug(f"GET {get_key}: RDMA transfer {size} bytes from {remote_endpoint}")

            # Execute RDMA transfer
            self.nixl_agent.get(remote_ptr, local_ptr, size, remote_endpoint)

            # Return tensor view
            data_tensor = managed_buffer.tensor

            # Clean up local buffer after use
            managed_buffer.release()

            logger.debug(f"GET {get_key}: completed RDMA transfer of {size} bytes")
            return data_tensor, size

        except Exception as e:
            logger.error(f"RDMA GET failed for {get_key}: {e}")
            return None

    def cleanup(self, request_id: str) -> None:
        """
        Clean up buffers for a specific request.

        Args:
            request_id: Request identifier to clean up
        """
        with self.buffers_lock:
            # Find and remove all buffers for this request
            keys_to_remove = [key for key in self.buffers.keys() if request_id in key]

            for key in keys_to_remove:
                managed_buffer, _ = self.buffers.pop(key)
                managed_buffer.release()
                logger.debug(f"Cleaned up buffer: {key}")

            if keys_to_remove:
                logger.info(f"Cleaned up {len(keys_to_remove)} buffers for request {request_id}")

    def health(self) -> dict[str, Any]:
        """
        Return health status and metrics.

        Returns:
            Dictionary with connector health information
        """
        allocator_stats = self.allocator.get_stats()

        with self.buffers_lock:
            num_buffers = len(self.buffers)

        return {
            "status": "healthy",
            "transport": self.transport,
            "pod_ip": self.pod_ip,
            "nixl_port": self.nixl_port,
            "pool_size_gb": self.pool_size_gb,
            "num_active_buffers": num_buffers,
            **allocator_stats,
        }

    def close(self) -> None:
        """
        Release resources held by this connector.
        """
        # Stop cleanup thread
        self._cleanup_running = False
        if hasattr(self, "_cleanup_thread") and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=2.0)

        # Release all buffers
        with self.buffers_lock:
            for key, (managed_buffer, _) in self.buffers.items():
                managed_buffer.release()
            self.buffers.clear()

        # Shutdown NIXL agent
        if hasattr(self, "nixl_agent"):
            try:
                # NIXL agent cleanup if available
                if hasattr(self.nixl_agent, "close"):
                    self.nixl_agent.close()
            except Exception as e:
                logger.warning(f"Error closing NIXL agent: {e}")

        logger.info("NixlConnector closed")

    def _cleanup_stale_buffers(self):
        """
        Background thread to clean up buffers older than TTL.
        Prevents memory leaks when receivers crash or abandon requests.
        """
        while self._cleanup_running:
            try:
                _time_mod.sleep(60)  # Check every minute

                now = _time_mod.time()
                stale_keys = []

                with self.buffers_lock:
                    for key, (_, timestamp) in self.buffers.items():
                        if now - timestamp > _BUFFER_TTL_SECONDS:
                            stale_keys.append(key)

                # Release stale buffers
                if stale_keys:
                    logger.warning(f"Cleaning up {len(stale_keys)} stale buffers (TTL={_BUFFER_TTL_SECONDS}s)")
                    with self.buffers_lock:
                        for key in stale_keys:
                            if key in self.buffers:
                                managed_buffer, _ = self.buffers.pop(key)
                                managed_buffer.release()

            except Exception as e:
                logger.error(f"Error in stale buffer cleanup: {e}")
