# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from vllm_omni.entrypoints.k8s.stage_worker_server import create_stage_worker_app

# Sample stage config for testing
SAMPLE_STAGE_CONFIG = """
stage_args:
  - stage_id: 0
    stage_type: llm
    runtime:
      process: true
      devices: "0"
      max_batch_size: 1
    engine_args:
      model_stage: thinker
      model_arch: Qwen2_5OmniForConditionalGeneration
      worker_type: ar
      gpu_memory_utilization: 0.8
      enforce_eager: true
      trust_remote_code: true
      engine_output_type: latent
      enable_prefix_caching: false
    is_comprehension: true
    final_output: false
    default_sampling_params:
      temperature: 0.0
      top_p: 1.0
      max_tokens: 2048

  - stage_id: 1
    stage_type: llm
    runtime:
      process: true
      devices: "1"
      max_batch_size: 1
    engine_args:
      model_stage: talker
      model_arch: Qwen2_5OmniForConditionalGeneration
      worker_type: ar
      gpu_memory_utilization: 0.8
      enforce_eager: true
      trust_remote_code: true
      engine_output_type: latent
    engine_input_source: [0]
    final_output: true
    final_output_type: audio

runtime:
  enabled: true
  defaults:
    window_size: -1
    max_inflight: 1
"""


@pytest.fixture
def stage_config_file():
    """Create temporary stage config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(SAMPLE_STAGE_CONFIG)
        config_path = f.name

    yield config_path

    # Cleanup
    Path(config_path).unlink()


@pytest.fixture
def mock_engine():
    """Mock AsyncOmniLLM engine."""
    with patch("vllm_omni.entrypoints.k8s.stage_worker_server.AsyncOmniLLM") as mock:
        engine = MagicMock()
        mock.return_value = engine
        yield engine


@pytest.fixture
def mock_connector():
    """Mock NIXL connector."""
    with patch("vllm_omni.entrypoints.k8s.stage_worker_server.build_stage_connectors") as mock:
        connector = MagicMock()

        # Mock put() to return success
        connector.put.return_value = (
            True,  # success
            1024,  # size
            {"buffer_ptr": 0x12345678, "size": 1024, "pod_ip": "10.1.2.3"},  # metadata
        )

        # Mock get() to return data
        import torch

        dummy_data = torch.tensor([1, 2, 3], dtype=torch.uint8)
        connector.get.return_value = (dummy_data, 1024)

        # Mock health()
        connector.health.return_value = {
            "status": "healthy",
            "transport": "tcp",
            "num_active_buffers": 0,
        }

        # Mock cleanup()
        connector.cleanup.return_value = None

        mock.return_value = {"nixl": connector}
        yield connector


@pytest.fixture
def mock_vllm_config():
    """Mock VllmConfig."""
    with patch("vllm_omni.entrypoints.k8s.stage_worker_server.VllmConfig") as mock:
        config = MagicMock()
        mock.return_value = config
        yield config


@pytest.fixture
def mock_set_stage_devices():
    """Mock set_stage_devices."""
    with patch("vllm_omni.entrypoints.k8s.stage_worker_server.set_stage_devices") as mock:
        yield mock


class TestStageWorkerServer:
    """Tests for stage worker REST API."""

    def test_health_endpoint(
        self, stage_config_file, mock_engine, mock_connector, mock_vllm_config, mock_set_stage_devices
    ):
        """Test health check endpoint."""
        app = create_stage_worker_app(
            model="test-model", stage_config_path=stage_config_file, stage_id=0, pod_ip="10.1.2.3"
        )

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["stage_id"] == 0
        assert data["pod_ip"] == "10.1.2.3"
        assert data["connector_health"] is not None

    def test_generate_stage_0(
        self, stage_config_file, mock_engine, mock_connector, mock_vllm_config, mock_set_stage_devices
    ):
        """Test generate endpoint for Stage-0 (initial stage)."""
        app = create_stage_worker_app(
            model="test-model", stage_config_path=stage_config_file, stage_id=0, pod_ip="10.1.2.3"
        )

        client = TestClient(app)

        # Prepare request
        input_data = b"Hello, world!"
        request_data = {
            "request_id": "req-123",
            "stage_id": 0,
            "data": base64.b64encode(input_data).decode(),
        }

        response = client.post("/v1/generate", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["pod_ip"] == "10.1.2.3"
        assert "nixl_metadata" in data
        assert data["result"] is None  # Not final stage

        # Verify connector.put was called
        mock_connector.put.assert_called_once()

    def test_generate_stage_1(
        self, stage_config_file, mock_engine, mock_connector, mock_vllm_config, mock_set_stage_devices
    ):
        """Test generate endpoint for Stage-1 (intermediate stage)."""
        app = create_stage_worker_app(
            model="test-model", stage_config_path=stage_config_file, stage_id=1, pod_ip="10.1.2.4"
        )

        client = TestClient(app)

        # Prepare request with NIXL metadata
        request_data = {
            "request_id": "req-123",
            "stage_id": 1,
            "nixl_source_ip": "10.1.2.3",
            "nixl_metadata": {"buffer_ptr": 0x12345678, "size": 1024},
        }

        response = client.post("/v1/generate", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["pod_ip"] == "10.1.2.4"
        assert "result" in data  # Final stage returns result

        # Verify connector.get was called
        mock_connector.get.assert_called_once()

    def test_generate_missing_data_stage_0(
        self, stage_config_file, mock_engine, mock_connector, mock_vllm_config, mock_set_stage_devices
    ):
        """Test generate endpoint returns 400 when Stage-0 missing data."""
        app = create_stage_worker_app(
            model="test-model", stage_config_path=stage_config_file, stage_id=0, pod_ip="10.1.2.3"
        )

        client = TestClient(app)

        request_data = {
            "request_id": "req-123",
            "stage_id": 0,
            # Missing 'data' field
        }

        response = client.post("/v1/generate", json=request_data)
        assert response.status_code == 400

    def test_generate_missing_metadata_stage_1(
        self, stage_config_file, mock_engine, mock_connector, mock_vllm_config, mock_set_stage_devices
    ):
        """Test generate endpoint returns 400 when Stage-1 missing metadata."""
        app = create_stage_worker_app(
            model="test-model", stage_config_path=stage_config_file, stage_id=1, pod_ip="10.1.2.4"
        )

        client = TestClient(app)

        request_data = {
            "request_id": "req-123",
            "stage_id": 1,
            # Missing 'nixl_metadata' and 'nixl_source_ip'
        }

        response = client.post("/v1/generate", json=request_data)
        assert response.status_code == 400

    def test_cleanup_buffers(
        self, stage_config_file, mock_engine, mock_connector, mock_vllm_config, mock_set_stage_devices
    ):
        """Test buffer cleanup endpoint."""
        app = create_stage_worker_app(
            model="test-model", stage_config_path=stage_config_file, stage_id=0, pod_ip="10.1.2.3"
        )

        client = TestClient(app)

        response = client.delete("/v1/buffers/req-123?from_stage=0&to_stage=1")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

        # Verify connector.cleanup was called
        mock_connector.cleanup.assert_called_once_with("req-123")

    def test_pod_ip_from_env(
        self, stage_config_file, mock_engine, mock_connector, mock_vllm_config, mock_set_stage_devices, monkeypatch
    ):
        """Test pod IP is read from environment variable."""
        monkeypatch.setenv("POD_IP", "10.5.6.7")

        app = create_stage_worker_app(model="test-model", stage_config_path=stage_config_file, stage_id=0, pod_ip=None)

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["pod_ip"] == "10.5.6.7"

    def test_set_stage_devices_called(
        self, stage_config_file, mock_engine, mock_connector, mock_vllm_config, mock_set_stage_devices
    ):
        """Test set_stage_devices is called with correct arguments."""
        app = create_stage_worker_app(
            model="test-model", stage_config_path=stage_config_file, stage_id=0, pod_ip="10.1.2.3"
        )

        # Verify set_stage_devices was called
        mock_set_stage_devices.assert_called_once_with(0, "0")
