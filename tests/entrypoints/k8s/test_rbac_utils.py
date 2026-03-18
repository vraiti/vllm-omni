# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_kubernetes():
    """Mock kubernetes library."""
    with patch("vllm_omni.entrypoints.k8s.rbac_utils.KUBERNETES_AVAILABLE", True), patch(
        "vllm_omni.entrypoints.k8s.rbac_utils.config"
    ) as mock_config, patch("vllm_omni.entrypoints.k8s.rbac_utils.client") as mock_client, patch(
        "vllm_omni.entrypoints.k8s.rbac_utils.watch"
    ) as mock_watch:
        # Mock config.load_incluster_config
        mock_config.load_incluster_config = MagicMock()

        # Mock CoreV1Api
        v1_api = MagicMock()
        mock_client.CoreV1Api.return_value = v1_api

        # Mock watch.Watch
        watch_instance = MagicMock()
        mock_watch.Watch.return_value = watch_instance

        yield {
            "config": mock_config,
            "client": mock_client,
            "watch": mock_watch,
            "v1_api": v1_api,
            "watch_instance": watch_instance,
        }


@pytest.mark.asyncio
class TestK8sWatchClient:
    """Tests for K8sWatchClient."""

    async def test_watch_client_init_incluster(self, mock_kubernetes):
        """Test initialization with in-cluster config."""
        from vllm_omni.entrypoints.k8s.rbac_utils import K8sWatchClient

        client = K8sWatchClient(namespace="test-ns", label_selector="app=test")

        assert client.namespace == "test-ns"
        assert client.label_selector == "app=test"
        mock_kubernetes["config"].load_incluster_config.assert_called_once()

    async def test_watch_client_init_kubeconfig_fallback(self, mock_kubernetes):
        """Test fallback to kubeconfig when in-cluster fails."""
        from vllm_omni.entrypoints.k8s.rbac_utils import K8sWatchClient

        # Mock in-cluster config to fail
        mock_kubernetes["config"].load_incluster_config.side_effect = Exception("Not in cluster")
        mock_kubernetes["config"].load_kube_config = MagicMock()

        client = K8sWatchClient()

        mock_kubernetes["config"].load_kube_config.assert_called_once()

    async def test_watch_pods(self, mock_kubernetes):
        """Test watching pods."""
        from vllm_omni.entrypoints.k8s.rbac_utils import K8sWatchClient

        # Mock pod events
        mock_events = [
            {
                "type": "ADDED",
                "object": MagicMock(
                    metadata=MagicMock(name="pod-1", labels={"stage": "0"}),
                    status=MagicMock(phase="Running", pod_ip="10.1.2.3"),
                ),
            },
            {
                "type": "DELETED",
                "object": MagicMock(
                    metadata=MagicMock(name="pod-1", labels={"stage": "0"}),
                    status=MagicMock(pod_ip="10.1.2.3"),
                ),
            },
        ]

        # Mock watch stream
        def mock_stream(*args, **kwargs):
            for event in mock_events:
                yield event

        mock_kubernetes["watch_instance"].stream = mock_stream

        client = K8sWatchClient()

        # Collect events
        events = []
        async for event in client.watch_pods():
            events.append(event)
            if len(events) >= 2:
                break  # Stop after 2 events

        assert len(events) == 2
        assert events[0]["type"] == "ADDED"
        assert events[1]["type"] == "DELETED"

    async def test_list_pods(self, mock_kubernetes):
        """Test listing pods."""
        from vllm_omni.entrypoints.k8s.rbac_utils import K8sWatchClient

        # Mock pod list
        mock_pod_list = MagicMock()
        mock_pod_list.items = [
            MagicMock(
                metadata=MagicMock(name="pod-1", namespace="test-ns", labels={"stage": "0"}),
                status=MagicMock(phase="Running", pod_ip="10.1.2.3"),
            ),
            MagicMock(
                metadata=MagicMock(name="pod-2", namespace="test-ns", labels={"stage": "1"}),
                status=MagicMock(phase="Running", pod_ip="10.1.2.4"),
            ),
        ]

        mock_kubernetes["v1_api"].list_namespaced_pod.return_value = mock_pod_list

        client = K8sWatchClient(namespace="test-ns")

        pods = await client.list_pods()

        assert len(pods) == 2
        assert pods[0]["metadata"]["name"] == "pod-1"
        assert pods[0]["status"]["podIP"] == "10.1.2.3"
        assert pods[1]["metadata"]["name"] == "pod-2"

    async def test_get_pod(self, mock_kubernetes):
        """Test getting a pod by name."""
        from vllm_omni.entrypoints.k8s.rbac_utils import K8sWatchClient

        # Mock pod
        mock_pod = MagicMock(
            metadata=MagicMock(name="pod-1", namespace="test-ns", labels={"stage": "0"}),
            status=MagicMock(phase="Running", pod_ip="10.1.2.3"),
        )

        mock_kubernetes["v1_api"].read_namespaced_pod.return_value = mock_pod

        client = K8sWatchClient(namespace="test-ns")

        pod = await client.get_pod("pod-1")

        assert pod is not None
        assert pod["metadata"]["name"] == "pod-1"
        assert pod["status"]["podIP"] == "10.1.2.3"

    async def test_get_pod_not_found(self, mock_kubernetes):
        """Test getting a non-existent pod."""
        from vllm_omni.entrypoints.k8s.rbac_utils import K8sWatchClient

        # Mock 404 error
        from kubernetes.client.rest import ApiException

        mock_kubernetes["v1_api"].read_namespaced_pod.side_effect = ApiException(status=404)

        client = K8sWatchClient(namespace="test-ns")

        pod = await client.get_pod("nonexistent")

        assert pod is None


class TestCheckRBACPermissions:
    """Tests for check_rbac_permissions."""

    def test_check_rbac_success(self, mock_kubernetes):
        """Test successful RBAC check."""
        from vllm_omni.entrypoints.k8s.rbac_utils import check_rbac_permissions

        # Mock successful list
        mock_kubernetes["v1_api"].list_namespaced_pod.return_value = MagicMock()

        result = check_rbac_permissions(namespace="test-ns")

        assert result is True

    def test_check_rbac_forbidden(self, mock_kubernetes):
        """Test RBAC check with forbidden error."""
        from vllm_omni.entrypoints.k8s.rbac_utils import check_rbac_permissions
        from kubernetes.client.rest import ApiException

        # Mock 403 error
        mock_kubernetes["v1_api"].list_namespaced_pod.side_effect = ApiException(status=403)

        result = check_rbac_permissions(namespace="test-ns")

        assert result is False

    def test_check_rbac_kubernetes_unavailable(self):
        """Test RBAC check when kubernetes library not available."""
        with patch("vllm_omni.entrypoints.k8s.rbac_utils.KUBERNETES_AVAILABLE", False):
            from vllm_omni.entrypoints.k8s.rbac_utils import check_rbac_permissions

            result = check_rbac_permissions()

            assert result is False
