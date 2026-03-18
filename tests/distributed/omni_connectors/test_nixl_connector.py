# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import (
    BufferAllocator,
    ManagedBuffer,
    NixlConnector,
)
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

# Skip tests if NIXL is not available
try:
    from nixl._api import nixl_agent, nixl_agent_config

    NIXL_AVAILABLE = True
except ImportError:
    NIXL_AVAILABLE = False

# Skip tests if CUDA is not available
CUDA_AVAILABLE = torch.cuda.is_available()

pytestmark = [
    pytest.mark.skipif(not NIXL_AVAILABLE, reason="NIXL library not available"),
    pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
]


class TestBufferAllocator:
    """Test BufferAllocator class."""

    def test_allocator_basic(self):
        """Test basic allocation and deallocation."""
        allocator = BufferAllocator(total_size=1024 * 1024, alignment=4096)

        # Allocate some memory
        offset1 = allocator.alloc(1024)
        assert offset1 == 0

        offset2 = allocator.alloc(1024)
        assert offset2 == 4096  # Aligned to 4096

        # Free first block
        allocator.free(offset1, 1024)

        # Allocate again, should reuse first block
        offset3 = allocator.alloc(1024)
        assert offset3 == 0

    def test_allocator_out_of_memory(self):
        """Test allocation failure when pool is exhausted."""
        allocator = BufferAllocator(total_size=1024, alignment=1)

        # Allocate all available memory
        offset = allocator.alloc(1024)
        assert offset == 0

        # Try to allocate more
        with pytest.raises(MemoryError):
            allocator.alloc(1)

    def test_allocator_double_free(self):
        """Test double-free detection."""
        allocator = BufferAllocator(total_size=1024, alignment=1)

        offset = allocator.alloc(100)
        allocator.free(offset, 100)

        # Double free should be detected and ignored (warning logged)
        allocator.free(offset, 100)  # Should not raise

    def test_allocator_stats(self):
        """Test allocator statistics."""
        allocator = BufferAllocator(total_size=1024 * 1024, alignment=4096)

        stats = allocator.get_stats()
        assert stats["total_bytes"] == 1024 * 1024
        assert stats["free_bytes"] == 1024 * 1024
        assert stats["allocated_bytes"] == 0

        # Allocate some memory
        allocator.alloc(1024)

        stats = allocator.get_stats()
        assert stats["allocated_bytes"] == 4096  # Aligned
        assert stats["free_bytes"] == 1024 * 1024 - 4096


class TestManagedBuffer:
    """Test ManagedBuffer class."""

    def test_managed_buffer_basic(self):
        """Test basic buffer operations."""
        pool = torch.zeros(1024, dtype=torch.uint8, device="cuda")
        allocator = BufferAllocator(total_size=1024, alignment=1)

        offset = allocator.alloc(100)
        buffer = ManagedBuffer(allocator, offset, 100, pool)

        # Get tensor view
        view = buffer.tensor
        assert view.shape == (100,)
        assert view.dtype == torch.uint8

        # Modify and check
        view[:] = 42
        assert pool[offset : offset + 100].eq(42).all()

        # Release
        buffer.release()

    def test_managed_buffer_as_tensor(self):
        """Test typed tensor views."""
        pool = torch.zeros(1024, dtype=torch.uint8, device="cuda")
        allocator = BufferAllocator(total_size=1024, alignment=8)

        # Allocate 32 bytes (8 float32 values)
        offset = allocator.alloc(32)
        buffer = ManagedBuffer(allocator, offset, 32, pool)

        # Get as float32 tensor
        float_view = buffer.as_tensor(torch.float32, (2, 4))
        assert float_view.shape == (2, 4)
        assert float_view.dtype == torch.float32

    def test_managed_buffer_context_manager(self):
        """Test context manager protocol."""
        pool = torch.zeros(1024, dtype=torch.uint8, device="cuda")
        allocator = BufferAllocator(total_size=1024, alignment=1)

        offset = allocator.alloc(100)

        with ManagedBuffer(allocator, offset, 100, pool) as buffer:
            view = buffer.tensor
            view[:] = 123

        # Buffer should be released after context
        assert buffer._released is True


@pytest.fixture
def mock_nixl_agent():
    """Mock NIXL agent for testing without actual RDMA."""
    with patch("vllm_omni.distributed.omni_connectors.connectors.nixl_connector.nixl_agent") as mock_agent_cls, patch(
        "vllm_omni.distributed.omni_connectors.connectors.nixl_connector.nixl_agent_config"
    ) as mock_config_cls:
        mock_config = MagicMock()
        mock_config_cls.return_value = mock_config

        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent

        # Mock RDMA GET operation
        def mock_get(remote_ptr, local_ptr, size, remote_endpoint):
            # Simulate successful RDMA transfer (no-op in mock)
            pass

        mock_agent.get = MagicMock(side_effect=mock_get)

        yield mock_agent


@pytest.fixture
def nixl_connector_tcp(mock_nixl_agent):
    """Create NixlConnector with TCP transport."""
    config = {
        "pool_size_gb": 1,
        "nixl_port": 5600,
        "transport": "tcp",
        "pod_ip": "127.0.0.1",
    }
    connector = NixlConnector(config)
    yield connector
    connector.close()


@pytest.fixture
def nixl_connector_rdma(mock_nixl_agent):
    """Create NixlConnector with RDMA transport."""
    config = {
        "pool_size_gb": 1,
        "nixl_port": 5600,
        "transport": "rdma",
        "pod_ip": "127.0.0.1",
    }
    connector = NixlConnector(config)
    yield connector
    connector.close()


class TestNixlConnector:
    """Test NixlConnector class."""

    def test_connector_creation_tcp(self, nixl_connector_tcp):
        """Test connector creation with TCP transport."""
        assert nixl_connector_tcp.transport == "tcp"
        assert nixl_connector_tcp.pod_ip == "127.0.0.1"
        assert nixl_connector_tcp.nixl_port == 5600
        assert nixl_connector_tcp.supports_raw_data is True

    def test_connector_creation_rdma(self, nixl_connector_rdma):
        """Test connector creation with RDMA transport."""
        assert nixl_connector_rdma.transport == "rdma"
        assert nixl_connector_rdma.pod_ip == "127.0.0.1"
        assert nixl_connector_rdma.nixl_port == 5600

    def test_connector_invalid_transport(self, mock_nixl_agent):
        """Test invalid transport configuration."""
        config = {"transport": "invalid", "pod_ip": "127.0.0.1"}
        with pytest.raises(ValueError, match="Invalid NIXL transport"):
            NixlConnector(config)

    def test_connector_env_var_transport(self, mock_nixl_agent):
        """Test transport configuration from environment variable."""
        with patch.dict(os.environ, {"NIXL_TRANSPORT": "rdma"}):
            config = {"pod_ip": "127.0.0.1"}
            connector = NixlConnector(config)
            assert connector.transport == "rdma"
            connector.close()

    def test_connector_default_transport(self, mock_nixl_agent):
        """Test default transport is TCP."""
        config = {"pod_ip": "127.0.0.1"}
        with patch.dict(os.environ, {}, clear=True):
            # Ensure NIXL_TRANSPORT is not set
            os.environ.pop("NIXL_TRANSPORT", None)
            connector = NixlConnector(config)
            assert connector.transport == "tcp"
            connector.close()

    def test_put_tensor_data(self, nixl_connector_tcp):
        """Test PUT operation with tensor data."""
        data = torch.randn(100, dtype=torch.float32, device="cuda")

        success, size, metadata = nixl_connector_tcp.put("stage_0", "stage_1", "req_1", data)

        assert success is True
        assert size > 0
        assert "buffer_ptr" in metadata
        assert "size" in metadata
        assert "pod_ip" in metadata
        assert metadata["pod_ip"] == "127.0.0.1"

    def test_put_bytes_data(self, nixl_connector_tcp):
        """Test PUT operation with bytes data."""
        data = b"Hello, NIXL!"

        success, size, metadata = nixl_connector_tcp.put("stage_0", "stage_1", "req_2", data)

        assert success is True
        assert size == len(data)
        assert "buffer_ptr" in metadata

    def test_put_python_object(self, nixl_connector_tcp):
        """Test PUT operation with Python object (serialized)."""
        data = {"key": "value", "list": [1, 2, 3]}

        success, size, metadata = nixl_connector_tcp.put("stage_0", "stage_1", "req_3", data)

        assert success is True
        assert size > 0
        assert "buffer_ptr" in metadata

    def test_get_operation(self, nixl_connector_tcp, mock_nixl_agent):
        """Test GET operation (RDMA transfer)."""
        # First, put some data
        data = torch.randn(100, dtype=torch.float32, device="cuda")
        success, size, put_metadata = nixl_connector_tcp.put("stage_0", "stage_1", "req_4", data)
        assert success is True

        # Now, GET from remote pod
        get_metadata = {
            "buffer_ptr": 0x12345678,  # Mock remote pointer
            "size": 1024,
            "pod_ip": "10.1.2.3",
        }

        result = nixl_connector_tcp.get("stage_0", "stage_1", "req_5", metadata=get_metadata)

        assert result is not None
        retrieved_data, ret_size = result
        assert isinstance(retrieved_data, torch.Tensor)
        assert ret_size == 1024

        # Verify RDMA GET was called
        mock_nixl_agent.get.assert_called_once()

    def test_cleanup_buffers(self, nixl_connector_tcp):
        """Test cleanup of buffers for a request."""
        # Put some data
        data1 = torch.randn(100, dtype=torch.float32, device="cuda")
        data2 = torch.randn(200, dtype=torch.float32, device="cuda")

        nixl_connector_tcp.put("stage_0", "stage_1", "req_6", data1)
        nixl_connector_tcp.put("stage_1", "stage_2", "req_6", data2)

        # Check buffers exist
        with nixl_connector_tcp.buffers_lock:
            num_buffers_before = len(nixl_connector_tcp.buffers)
            assert num_buffers_before == 2

        # Cleanup
        nixl_connector_tcp.cleanup("req_6")

        # Check buffers removed
        with nixl_connector_tcp.buffers_lock:
            num_buffers_after = len(nixl_connector_tcp.buffers)
            assert num_buffers_after == 0

    def test_health_check(self, nixl_connector_tcp):
        """Test health check endpoint."""
        health = nixl_connector_tcp.health()

        assert health["status"] == "healthy"
        assert health["transport"] == "tcp"
        assert health["pod_ip"] == "127.0.0.1"
        assert health["nixl_port"] == 5600
        assert "num_active_buffers" in health
        assert "total_bytes" in health
        assert "free_bytes" in health

    def test_stale_buffer_cleanup(self, nixl_connector_tcp):
        """Test automatic cleanup of stale buffers."""
        # Put some data
        data = torch.randn(100, dtype=torch.float32, device="cuda")
        nixl_connector_tcp.put("stage_0", "stage_1", "req_7", data)

        # Manually set timestamp to old value
        key = nixl_connector_tcp._make_key("req_7", "stage_0", "stage_1")
        with nixl_connector_tcp.buffers_lock:
            if key in nixl_connector_tcp.buffers:
                managed_buffer, _ = nixl_connector_tcp.buffers[key]
                nixl_connector_tcp.buffers[key] = (managed_buffer, time.time() - 400)  # 400 seconds old

        # Trigger cleanup (normally runs in background thread)
        nixl_connector_tcp._cleanup_stale_buffers()

        # Check buffer was removed
        with nixl_connector_tcp.buffers_lock:
            assert key not in nixl_connector_tcp.buffers

    def test_factory_creation(self):
        """Test creating NixlConnector via factory."""
        spec = ConnectorSpec(
            name="NixlConnector",
            extra={
                "transport": "tcp",
                "pod_ip": "127.0.0.1",
            },
        )

        with patch(
            "vllm_omni.distributed.omni_connectors.connectors.nixl_connector.nixl_agent"
        ), patch("vllm_omni.distributed.omni_connectors.connectors.nixl_connector.nixl_agent_config"):
            connector = OmniConnectorFactory.create_connector(spec)
            assert isinstance(connector, NixlConnector)
            connector.close()

    def test_context_manager(self, mock_nixl_agent):
        """Test context manager protocol."""
        config = {"transport": "tcp", "pod_ip": "127.0.0.1"}

        with NixlConnector(config) as connector:
            assert connector.transport == "tcp"
            health = connector.health()
            assert health["status"] == "healthy"

        # Connector should be closed after context
        # (close is idempotent, so this should be safe)


@pytest.mark.skipif(not NIXL_AVAILABLE, reason="NIXL library not available")
def test_nixl_import_error():
    """Test error when NIXL is not available."""
    with patch.dict("sys.modules", {"nixl._api": None}):
        with patch(
            "vllm_omni.distributed.omni_connectors.connectors.nixl_connector.nixl_agent", None
        ), patch("vllm_omni.distributed.omni_connectors.connectors.nixl_connector.nixl_agent_config", None):
            config = {"transport": "tcp"}
            with pytest.raises(ImportError, match="NIXL library not found"):
                NixlConnector(config)
