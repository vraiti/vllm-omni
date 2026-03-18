# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end tests for K8s API server integration."""

import asyncio
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_k8s_orchestrator():
    """Mock K8sOmniOrchestrator for testing."""
    with patch("vllm_omni.entrypoints.openai.api_server.K8sOmniOrchestrator") as mock_class:
        orchestrator = AsyncMock()

        # Mock methods
        orchestrator.start = AsyncMock()
        orchestrator.shutdown = AsyncMock()
        orchestrator.get_vllm_config = AsyncMock(return_value=None)
        orchestrator.get_supported_tasks = AsyncMock(return_value={"generate", "chat", "audio"})
        orchestrator.health = AsyncMock(
            return_value={
                "status": "healthy",
                "namespace": "vllm-omni",
                "label_selector": "app=vllm-omni",
                "num_stages": 3,
                "pod_counts": {0: 1, 1: 1, 2: 1},
            }
        )

        # Mock generate method
        async def mock_generate(*args, **kwargs):
            from vllm_omni.outputs import OmniRequestOutput

            yield OmniRequestOutput(
                request_id="test-req",
                stage_id=2,
                final_output_type="audio",
                request_output="test_audio_data",
            )

        orchestrator.generate = mock_generate

        # Set attributes
        orchestrator.model = "Qwen/Qwen2.5-Omni-7B"
        orchestrator.stage_configs = None
        orchestrator.input_processor = None
        orchestrator.io_processor = None
        orchestrator.model_config = None

        mock_class.return_value = orchestrator
        yield orchestrator


@pytest.fixture
def mock_watch_client():
    """Mock K8sWatchClient."""
    with patch("vllm_omni.entrypoints.k8s.k8s_omni_orchestrator.K8sWatchClient") as mock:
        client = MagicMock()

        # Mock watch_pods to return empty async generator
        async def mock_watch():
            await asyncio.sleep(100)
            yield  # Never actually yields

        client.watch_pods.return_value = mock_watch()
        mock.return_value = client
        yield client


@pytest.mark.asyncio
class TestK8sAPIServerIntegration:
    """Tests for K8s API server integration."""

    async def test_build_async_omni_k8s_mode(self, mock_k8s_orchestrator, mock_watch_client, monkeypatch):
        """Test building engine client in K8s mode."""
        from vllm_omni.entrypoints.openai.api_server import build_async_omni_from_stage_config

        # Set environment variable
        monkeypatch.setenv("VLLM_USE_K8S_ORCHESTRATOR", "true")
        monkeypatch.setenv("NAMESPACE", "test-ns")
        monkeypatch.setenv("LABEL_SELECTOR", "app=test")

        # Create args
        args = Namespace(model="test-model", num_stages=3)

        # Build engine client
        async with build_async_omni_from_stage_config(args) as engine_client:
            # Verify it's a K8sOmniOrchestrator
            assert engine_client == mock_k8s_orchestrator

            # Verify start was called
            mock_k8s_orchestrator.start.assert_called_once()

        # Verify shutdown was called
        mock_k8s_orchestrator.shutdown.assert_called_once()

    async def test_build_async_omni_standard_mode(self, monkeypatch):
        """Test building engine client in standard mode (AsyncOmni)."""
        from vllm_omni.entrypoints.openai.api_server import build_async_omni_from_stage_config

        # Ensure K8s mode is disabled
        monkeypatch.setenv("VLLM_USE_K8S_ORCHESTRATOR", "false")

        # Mock AsyncOmni
        with patch("vllm_omni.entrypoints.openai.api_server.AsyncOmni") as mock_async_omni:
            async_omni = MagicMock()
            async_omni.shutdown = MagicMock()
            mock_async_omni.return_value = async_omni

            args = Namespace(model="test-model")

            async with build_async_omni_from_stage_config(args) as engine_client:
                # Verify it's an AsyncOmni
                assert engine_client == async_omni

                # Verify AsyncOmni was created
                mock_async_omni.assert_called_once()

            # Verify shutdown was called
            async_omni.shutdown.assert_called_once()

    async def test_omni_init_app_state_k8s(self, mock_k8s_orchestrator, monkeypatch):
        """Test app state initialization with K8s orchestrator."""
        from vllm_omni.entrypoints.openai.api_server import omni_init_app_state

        # Create args
        args = Namespace(
            model="test-model",
            served_model_name=None,
            enable_log_requests=False,
            disable_log_stats=False,
            chat_template=None,
            tool_server=None,
            lora_modules=[],
            enable_server_load_tracking=False,
        )

        # Create state
        state = MagicMock()

        # Initialize app state
        await omni_init_app_state(mock_k8s_orchestrator, state, args)

        # Verify state was set
        assert state.engine_client == mock_k8s_orchestrator
        assert state.vllm_config is None  # K8s orchestrator doesn't have vllm_config

    async def test_k8s_orchestrator_compatibility_methods(self, mock_k8s_orchestrator):
        """Test compatibility methods on K8sOmniOrchestrator."""
        # Test get_vllm_config
        vllm_config = await mock_k8s_orchestrator.get_vllm_config()
        assert vllm_config is None

        # Test get_supported_tasks
        tasks = await mock_k8s_orchestrator.get_supported_tasks()
        assert "generate" in tasks
        assert "chat" in tasks
        assert "audio" in tasks

        # Test health
        health = await mock_k8s_orchestrator.health()
        assert health["status"] == "healthy"
        assert "pod_counts" in health

    async def test_generate_with_k8s_orchestrator(self, mock_k8s_orchestrator):
        """Test generate method with K8s orchestrator."""
        outputs = []
        async for output in mock_k8s_orchestrator.generate(
            prompt="Hello", request_id="test-req", sampling_params_list=None
        ):
            outputs.append(output)

        assert len(outputs) == 1
        assert outputs[0].request_id == "test-req"
        assert outputs[0].stage_id == 2
        assert outputs[0].final_output_type == "audio"


@pytest.mark.integration
class TestK8sAPIServerE2E:
    """Integration tests requiring actual K8s cluster (marked for manual testing)."""

    @pytest.mark.skip(reason="Requires actual Kubernetes cluster")
    async def test_full_k8s_deployment(self):
        """Test full K8s deployment end-to-end.

        This test requires:
        1. Kubernetes cluster with GPU nodes
        2. vllm-omni namespace created
        3. Stage workers deployed
        4. Orchestrator deployed

        Run manually in test environment:
            pytest tests/e2e/test_k8s_api_server.py::TestK8sAPIServerE2E::test_full_k8s_deployment -v -s
        """
        import httpx

        # Get orchestrator service endpoint
        # In real deployment: oc get svc omni-orchestrator -n vllm-omni
        orchestrator_url = "http://localhost:8000"  # Replace with actual endpoint

        async with httpx.AsyncClient() as client:
            # Test health endpoint
            response = await client.get(f"{orchestrator_url}/health")
            assert response.status_code == 200

            # Test chat completion
            response = await client.post(
                f"{orchestrator_url}/v1/chat/completions",
                json={
                    "model": "Qwen/Qwen2.5-Omni-7B",
                    "messages": [{"role": "user", "content": "Hello, how are you?"}],
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert "choices" in data
            assert len(data["choices"]) > 0
