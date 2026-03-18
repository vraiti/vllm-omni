# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm_omni.entrypoints.k8s.k8s_omni_orchestrator import K8sOmniOrchestrator


@pytest.fixture
def mock_watch_client():
    """Mock K8sWatchClient."""
    with patch("vllm_omni.entrypoints.k8s.k8s_omni_orchestrator.K8sWatchClient") as mock:
        client = MagicMock()

        # Mock watch_pods to return async generator
        async def mock_watch():
            # Yield some test events
            yield {
                "type": "ADDED",
                "object": {
                    "metadata": {
                        "name": "stage-0-pod-1",
                        "labels": {"stage": "0"},
                    },
                    "status": {
                        "phase": "Running",
                        "podIP": "10.1.2.3",
                    },
                },
            }
            yield {
                "type": "ADDED",
                "object": {
                    "metadata": {
                        "name": "stage-1-pod-1",
                        "labels": {"stage": "1"},
                    },
                    "status": {
                        "phase": "Running",
                        "podIP": "10.1.2.4",
                    },
                },
            }
            # Keep generator alive
            await asyncio.sleep(100)

        client.watch_pods.return_value = mock_watch()
        mock.return_value = client
        yield client


@pytest.fixture
def mock_http_client():
    """Mock httpx.AsyncClient."""
    with patch("vllm_omni.entrypoints.k8s.k8s_omni_orchestrator.httpx.AsyncClient") as mock:
        client = AsyncMock()

        # Mock successful POST responses
        async def mock_post(url, json=None):
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status = MagicMock()

            # Parse stage ID from URL
            if "10.1.2.3" in url:
                # Stage-0 response
                response.json.return_value = {
                    "pod_ip": "10.1.2.3",
                    "nixl_metadata": {"buffer_ptr": 0x123, "size": 1024},
                }
            elif "10.1.2.4" in url:
                # Stage-1 response
                response.json.return_value = {
                    "pod_ip": "10.1.2.4",
                    "nixl_metadata": {"buffer_ptr": 0x456, "size": 2048},
                }
            elif "10.1.2.5" in url:
                # Stage-2 response (final)
                response.json.return_value = {
                    "pod_ip": "10.1.2.5",
                    "nixl_metadata": {},
                    "result": "final_output_data",
                }

            return response

        # Mock successful DELETE responses
        async def mock_delete(url, params=None):
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status = MagicMock()
            response.json.return_value = {"status": "ok"}
            return response

        client.post = mock_post
        client.delete = mock_delete
        client.aclose = AsyncMock()

        mock.return_value = client
        yield client


@pytest.mark.asyncio
class TestK8sOmniOrchestrator:
    """Tests for K8sOmniOrchestrator."""

    async def test_orchestrator_init(self, mock_watch_client):
        """Test orchestrator initialization."""
        orch = K8sOmniOrchestrator(
            model="test-model",
            namespace="test-ns",
            label_selector="app=test",
        )

        assert orch.model == "test-model"
        assert orch.namespace == "test-ns"
        assert orch.label_selector == "app=test"
        assert orch.num_stages == 3
        assert len(orch.stage_pods) == 0

    async def test_start_and_shutdown(self, mock_watch_client, mock_http_client):
        """Test orchestrator start and shutdown."""
        orch = K8sOmniOrchestrator(model="test-model")

        # Start
        await orch.start()
        assert orch._running is True
        assert orch._watch_task is not None

        # Give watch task time to process events
        await asyncio.sleep(0.1)

        # Shutdown
        await orch.shutdown()
        assert orch._running is False

    async def test_pod_event_handling(self, mock_watch_client, mock_http_client):
        """Test handling of pod events."""
        orch = K8sOmniOrchestrator(model="test-model")

        # Test ADDED event
        event = {
            "type": "ADDED",
            "object": {
                "metadata": {
                    "name": "test-pod",
                    "labels": {"stage": "0"},
                },
                "status": {
                    "phase": "Running",
                    "podIP": "10.1.2.3",
                },
            },
        }

        await orch._handle_pod_event(event)

        async with orch.stage_pods_lock:
            assert 0 in orch.stage_pods
            assert "10.1.2.3" in orch.stage_pods[0]

        # Test DELETED event
        event_delete = {
            "type": "DELETED",
            "object": {
                "metadata": {
                    "name": "test-pod",
                    "labels": {"stage": "0"},
                },
                "status": {
                    "podIP": "10.1.2.3",
                },
            },
        }

        await orch._handle_pod_event(event_delete)

        async with orch.stage_pods_lock:
            assert "10.1.2.3" not in orch.stage_pods[0]

    async def test_pick_pod_round_robin(self, mock_watch_client, mock_http_client):
        """Test round-robin pod selection."""
        orch = K8sOmniOrchestrator(model="test-model")

        # Add multiple pods for stage 0
        async with orch.stage_pods_lock:
            orch.stage_pods[0] = ["10.1.2.3", "10.1.2.4", "10.1.2.5"]

        # Pick pods and verify round-robin
        pod1 = await orch._pick_pod(0)
        pod2 = await orch._pick_pod(0)
        pod3 = await orch._pick_pod(0)
        pod4 = await orch._pick_pod(0)  # Should wrap around

        assert pod1 == "10.1.2.3"
        assert pod2 == "10.1.2.4"
        assert pod3 == "10.1.2.5"
        assert pod4 == "10.1.2.3"  # Wrapped around

    async def test_pick_pod_no_pods_available(self, mock_watch_client, mock_http_client):
        """Test error when no pods available."""
        orch = K8sOmniOrchestrator(model="test-model")

        with pytest.raises(RuntimeError, match="No pods available"):
            await orch._pick_pod(0)

    async def test_process_stage_0(self, mock_watch_client, mock_http_client):
        """Test processing Stage-0 request."""
        orch = K8sOmniOrchestrator(model="test-model")

        # Add pod for stage 0
        async with orch.stage_pods_lock:
            orch.stage_pods[0] = ["10.1.2.3"]

        # Process stage
        result = await orch._process_stage(
            stage_id=0,
            request_id="req-123",
            data=b"test input",
        )

        assert result["pod_ip"] == "10.1.2.3"
        assert "nixl_metadata" in result
        assert result["nixl_metadata"]["buffer_ptr"] == 0x123

    async def test_process_stage_1(self, mock_watch_client, mock_http_client):
        """Test processing Stage-1 request."""
        orch = K8sOmniOrchestrator(model="test-model")

        # Add pod for stage 1
        async with orch.stage_pods_lock:
            orch.stage_pods[1] = ["10.1.2.4"]

        # Process stage
        result = await orch._process_stage(
            stage_id=1,
            request_id="req-123",
            nixl_source_ip="10.1.2.3",
            nixl_metadata={"buffer_ptr": 0x123, "size": 1024},
        )

        assert result["pod_ip"] == "10.1.2.4"
        assert "nixl_metadata" in result

    async def test_process_pipeline(self, mock_watch_client, mock_http_client):
        """Test full pipeline processing."""
        orch = K8sOmniOrchestrator(model="test-model")

        # Add pods for all stages
        async with orch.stage_pods_lock:
            orch.stage_pods[0] = ["10.1.2.3"]
            orch.stage_pods[1] = ["10.1.2.4"]
            orch.stage_pods[2] = ["10.1.2.5"]

        # Process pipeline
        result = await orch.process_pipeline(
            prompt="Hello, world!",
            request_id="req-123",
        )

        assert result["result"] == "final_output_data"

    async def test_generate(self, mock_watch_client, mock_http_client):
        """Test generate method (async generator)."""
        orch = K8sOmniOrchestrator(model="test-model")

        # Add pods for all stages
        async with orch.stage_pods_lock:
            orch.stage_pods[0] = ["10.1.2.3"]
            orch.stage_pods[1] = ["10.1.2.4"]
            orch.stage_pods[2] = ["10.1.2.5"]

        # Generate outputs
        outputs = []
        async for output in orch.generate(
            prompt="Hello, world!",
            request_id="req-123",
        ):
            outputs.append(output)

        assert len(outputs) == 1
        assert outputs[0].request_id == "req-123"
        assert outputs[0].stage_id == 2  # Final stage
        assert outputs[0].request_output == "final_output_data"

    async def test_health(self, mock_watch_client, mock_http_client):
        """Test health check."""
        orch = K8sOmniOrchestrator(model="test-model")

        # Add some pods
        async with orch.stage_pods_lock:
            orch.stage_pods[0] = ["10.1.2.3", "10.1.2.4"]
            orch.stage_pods[1] = ["10.1.2.5"]

        health = await orch.health()

        assert health["status"] == "healthy"
        assert health["num_stages"] == 3
        assert health["pod_counts"][0] == 2
        assert health["pod_counts"][1] == 1

    async def test_cleanup_buffer(self, mock_watch_client, mock_http_client):
        """Test async buffer cleanup."""
        orch = K8sOmniOrchestrator(model="test-model")

        # Cleanup should not raise
        await orch._cleanup_buffer(
            pod_ip="10.1.2.3",
            request_id="req-123",
            from_stage=0,
            to_stage=1,
        )

        # Give async task time to complete
        await asyncio.sleep(0.1)

    async def test_http_error_handling(self, mock_watch_client):
        """Test HTTP error handling."""
        with patch("vllm_omni.entrypoints.k8s.k8s_omni_orchestrator.httpx.AsyncClient") as mock:
            client = AsyncMock()

            # Mock HTTP error
            async def mock_post_error(url, json=None):
                import httpx

                response = MagicMock()
                response.status_code = 500
                response.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "Server error", request=MagicMock(), response=response
                )
                return response

            client.post = mock_post_error
            client.aclose = AsyncMock()
            mock.return_value = client

            orch = K8sOmniOrchestrator(model="test-model")

            # Add pod
            async with orch.stage_pods_lock:
                orch.stage_pods[0] = ["10.1.2.3"]

            # Process should raise RuntimeError
            with pytest.raises(RuntimeError, match="Stage-0 returned HTTP 500"):
                await orch._process_stage(
                    stage_id=0,
                    request_id="req-123",
                    data=b"test",
                )
