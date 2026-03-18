# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Kubernetes Watch API utilities for service discovery."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    from kubernetes import client, config, watch
    from kubernetes.client.rest import ApiException

    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False
    logger.warning("kubernetes library not available. Install with: pip install kubernetes")


class K8sWatchClient:
    """Kubernetes Watch API client for pod discovery.

    Provides async interface to Kubernetes Watch API for real-time
    pod event notifications (ADDED, MODIFIED, DELETED).

    Args:
        namespace: Kubernetes namespace to watch
        label_selector: Label selector for filtering pods (e.g., "app=vllm-omni")
        timeout_seconds: Watch timeout before reconnect (default: 300)
    """

    def __init__(
        self,
        namespace: str = "default",
        label_selector: str = "app=vllm-omni",
        timeout_seconds: int = 300,
    ):
        if not KUBERNETES_AVAILABLE:
            raise ImportError("kubernetes library required. Install with: pip install kubernetes")

        self.namespace = namespace
        self.label_selector = label_selector
        self.timeout_seconds = timeout_seconds

        # Load Kubernetes config (in-cluster or kubeconfig)
        try:
            # Try in-cluster config first (when running in pod)
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config")
        except config.ConfigException:
            # Fall back to kubeconfig (for local development)
            try:
                config.load_kube_config()
                logger.info("Loaded kubeconfig")
            except config.ConfigException as e:
                raise RuntimeError("Failed to load Kubernetes config. Run inside cluster or set KUBECONFIG.") from e

        # Create API client
        self.v1 = client.CoreV1Api()

        logger.info(f"Initialized K8sWatchClient: namespace={namespace}, label_selector={label_selector}")

    async def watch_pods(self) -> AsyncGenerator[dict[str, Any], None]:
        """Watch pod events asynchronously.

        Yields pod events as they occur (ADDED, MODIFIED, DELETED).
        Automatically reconnects on timeout or error.

        Yields:
            Event dictionaries with 'type' and 'object' fields
        """
        while True:
            try:
                w = watch.Watch()

                logger.info(f"Starting pod watch: namespace={self.namespace}, selector={self.label_selector}")

                # Watch pods (blocking call, so run in executor)
                for event in w.stream(
                    self.v1.list_namespaced_pod,
                    namespace=self.namespace,
                    label_selector=self.label_selector,
                    timeout_seconds=self.timeout_seconds,
                ):
                    # Yield event
                    yield event

                    # Allow other tasks to run
                    await asyncio.sleep(0)

            except asyncio.CancelledError:
                logger.info("Pod watch cancelled")
                raise

            except ApiException as e:
                if e.status == 403:
                    logger.error(
                        f"Forbidden: Insufficient permissions to watch pods in namespace '{self.namespace}'. "
                        "Ensure ServiceAccount has 'get', 'list', 'watch' permissions on pods resource."
                    )
                    raise RuntimeError("Insufficient Kubernetes RBAC permissions") from e
                else:
                    logger.error(f"Kubernetes API error: {e}")
                    # Retry after delay
                    await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Error watching pods: {e}", exc_info=True)
                # Retry after delay
                await asyncio.sleep(5)

    async def list_pods(self) -> list[dict[str, Any]]:
        """List pods matching the label selector.

        Returns:
            List of pod dictionaries
        """
        try:
            # Run list in executor (blocking call)
            loop = asyncio.get_event_loop()
            pod_list = await loop.run_in_executor(
                None,
                lambda: self.v1.list_namespaced_pod(
                    namespace=self.namespace,
                    label_selector=self.label_selector,
                ),
            )

            # Convert to dict
            pods = []
            for pod in pod_list.items:
                pod_dict = {
                    "metadata": {
                        "name": pod.metadata.name,
                        "namespace": pod.metadata.namespace,
                        "labels": pod.metadata.labels or {},
                    },
                    "status": {
                        "phase": pod.status.phase,
                        "podIP": pod.status.pod_ip,
                    },
                }
                pods.append(pod_dict)

            return pods

        except ApiException as e:
            logger.error(f"Failed to list pods: {e}")
            raise RuntimeError(f"Failed to list pods in namespace '{self.namespace}'") from e

    async def get_pod(self, pod_name: str) -> dict[str, Any] | None:
        """Get pod by name.

        Args:
            pod_name: Name of the pod

        Returns:
            Pod dictionary or None if not found
        """
        try:
            loop = asyncio.get_event_loop()
            pod = await loop.run_in_executor(
                None,
                lambda: self.v1.read_namespaced_pod(
                    name=pod_name,
                    namespace=self.namespace,
                ),
            )

            return {
                "metadata": {
                    "name": pod.metadata.name,
                    "namespace": pod.metadata.namespace,
                    "labels": pod.metadata.labels or {},
                },
                "status": {
                    "phase": pod.status.phase,
                    "podIP": pod.status.pod_ip,
                },
            }

        except ApiException as e:
            if e.status == 404:
                return None
            logger.error(f"Failed to get pod {pod_name}: {e}")
            raise RuntimeError(f"Failed to get pod '{pod_name}'") from e


def check_rbac_permissions(namespace: str = "default") -> bool:
    """Check if current ServiceAccount has required RBAC permissions.

    Args:
        namespace: Namespace to check permissions for

    Returns:
        True if permissions are sufficient, False otherwise
    """
    if not KUBERNETES_AVAILABLE:
        logger.warning("kubernetes library not available, cannot check RBAC")
        return False

    try:
        # Load config
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        v1 = client.CoreV1Api()

        # Try to list pods (requires 'list' permission)
        try:
            v1.list_namespaced_pod(namespace=namespace, limit=1)
            logger.info(f"RBAC check passed: can list pods in namespace '{namespace}'")
            return True
        except ApiException as e:
            if e.status == 403:
                logger.error(
                    f"RBAC check failed: Forbidden to list pods in namespace '{namespace}'. "
                    "Required permissions: 'get', 'list', 'watch' on pods resource."
                )
                return False
            raise

    except Exception as e:
        logger.error(f"RBAC check failed with error: {e}")
        return False
