"""
E2E test: Prometheus metrics under multi-replica Qwen3-Omni configuration.

Verifies that the /metrics endpoint exposes the expected vllm_omni:* and
vllm:* metric families with correct labels when serving a 3-stage pipeline
with 2 talker replicas on 2 GPUs.
"""

from __future__ import annotations

import os
import re

import pytest
import requests

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEPLOY = get_deploy_config_path("ci/qwen3_omni_moe_metrics_2gpu.yaml")
NUM_ENGINES = 4  # 1 thinker + 2 talker replicas + 1 code2wav
NUM_REQUESTS = 2

_PIPELINE_METRICS = [
    "vllm_omni:num_requests_running",
    "vllm_omni:num_requests_waiting",
    "vllm_omni:num_requests_success",
    "vllm_omni:num_requests_fail",
    "vllm_omni:e2e_request_latency_seconds",
    "vllm_omni:request_queue_time_seconds",
]

test_params = [
    OmniServerParams(
        model=MODEL,
        stage_config_path=DEPLOY,
        server_args=[],
    )
]


def _scrape(omni_server) -> str:
    url = f"http://{omni_server.host}:{omni_server.port}/metrics"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.text


def _sample_value(output: str, metric_line: str) -> float | None:
    for line in output.splitlines():
        if line.startswith(metric_line):
            return float(line.split()[-1])
    return None


def _count_data_lines(output: str, metric_name: str) -> int:
    count = 0
    for line in output.splitlines():
        if line.startswith(metric_name + "{") or line.startswith(metric_name + " "):
            if not line.startswith("# "):
                count += 1
    return count


def _send_text_requests(omni_server, n: int) -> None:
    url = f"http://{omni_server.host}:{omni_server.port}/v1/chat/completions"
    payload = {
        "model": omni_server.model,
        "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
        "stream": False,
        "modalities": ["text"],
        "max_tokens": 32,
    }
    for _ in range(n):
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()


@pytest.fixture(scope="module")
def metrics_before(omni_server) -> str:
    return _scrape(omni_server)


@pytest.fixture(scope="module")
def metrics_after(omni_server, metrics_before) -> str:
    _send_text_requests(omni_server, NUM_REQUESTS)
    return _scrape(omni_server)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
class TestMetricFamilies:
    def test_omni_pipeline_metrics_present(self, metrics_before: str) -> None:
        for name in _PIPELINE_METRICS:
            assert f"# HELP {name}" in metrics_before, f"missing metric family: {name}"

    def test_omni_pipeline_metrics_carry_model_name(self, metrics_before: str) -> None:
        for name in _PIPELINE_METRICS:
            pattern = rf'^{re.escape(name)}.*model_name="{re.escape(MODEL)}"'
            assert re.search(pattern, metrics_before, re.MULTILINE), f"{name} missing model_name label"

    def test_vllm_per_engine_metrics_present(self, metrics_before: str) -> None:
        assert "# HELP vllm:num_requests_running" in metrics_before

    def test_vllm_metrics_have_correct_engine_count(self, metrics_before: str) -> None:
        count = _count_data_lines(metrics_before, "vllm:num_requests_running")
        assert count == NUM_ENGINES, f"expected {NUM_ENGINES} engine label sets, got {count}"


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
class TestMetricValues:
    def test_success_counter_increments(self, metrics_before: str, metrics_after: str) -> None:
        prefix = f'vllm_omni:num_requests_success_total{{model_name="{MODEL}"}}'
        before = _sample_value(metrics_before, prefix) or 0.0
        after = _sample_value(metrics_after, prefix)
        assert after is not None, "success counter not found after requests"
        assert after >= before + NUM_REQUESTS, f"expected success count >= {before + NUM_REQUESTS}, got {after}"
