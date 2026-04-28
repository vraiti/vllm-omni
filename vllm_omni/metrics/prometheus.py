from dataclasses import dataclass

from prometheus_client import Counter, Gauge, Histogram

_labelnames = ["model_name"]
_stage_labelnames = ["model_name", "stage_id"]

_DIFFUSION_METRIC_DEFS: dict[str, tuple[str, str]] = {
    "preprocess_time_ms": (
        "vllm_omni:diffusion_preprocess_time_ms",
        "Diffusion preprocess time per request in milliseconds.",
    ),
    "diffusion_engine_exec_time_ms": (
        "vllm_omni:diffusion_exec_time_ms",
        "Diffusion model execution time per request in milliseconds.",
    ),
    "postprocess_time_ms": (
        "vllm_omni:diffusion_postprocess_time_ms",
        "Diffusion postprocess time per request in milliseconds.",
    ),
    "diffusion_engine_total_time_ms": (
        "vllm_omni:diffusion_step_time_ms",
        "Total diffusion step time per request in milliseconds.",
    ),
}


@dataclass
class StagePrometheusStats:
    kv_cache_usage: float = 0.0


class OmniPrometheusMetrics:
    """Label-bound wrapper around the raw Prometheus metrics.

    Metric collectors are registered here (not at module level) so that
    upstream vLLM's ``unregister_vllm_metrics()`` — which strips every
    collector whose ``_name`` contains ``"vllm"`` — has already run
    before these are created.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._running = Gauge(
            "vllm_omni:num_requests_running",
            "Number of requests currently running across all pipeline stages.",
            labelnames=_labelnames,
        ).labels(model_name=model_name)
        self._waiting = Gauge(
            "vllm_omni:num_requests_waiting",
            "Number of requests waiting to be scheduled.",
            labelnames=_labelnames,
        ).labels(model_name=model_name)
        self._success = Counter(
            "vllm_omni:num_requests_success",
            "Number of requests that completed without error.",
            labelnames=_labelnames,
        ).labels(model_name=model_name)
        self._fail = Counter(
            "vllm_omni:num_requests_fail",
            "Number of requests that returned an error.",
            labelnames=_labelnames,
        ).labels(model_name=model_name)
        self._e2e_latency = Histogram(
            "vllm_omni:e2e_request_latency_seconds",
            "Histogram of end-to-end request latency in seconds.",
            labelnames=_labelnames,
        ).labels(model_name=model_name)
        self._queue_time = Histogram(
            "vllm_omni:request_queue_time_seconds",
            "Histogram of request queue wait time in seconds.",
            labelnames=_labelnames,
        ).labels(model_name=model_name)
        self._kv_cache_usage_parent = Gauge(
            "vllm_omni:kv_cache_usage_percent",
            "Fraction of KV cache blocks currently in use, per pipeline stage.",
            labelnames=_stage_labelnames,
        )
        self._kv_cache_by_stage: dict[int, Gauge] = {}
        self._diffusion_parents: dict[str, Histogram] = {}
        for key, (metric_name, desc) in _DIFFUSION_METRIC_DEFS.items():
            self._diffusion_parents[key] = Histogram(
                metric_name, desc, labelnames=_stage_labelnames,
            )
        self._diffusion_by_stage: dict[tuple[str, int], Histogram] = {}

    def set_running(self, n: int) -> None:
        self._running.set(n)

    def set_waiting(self, n: int) -> None:
        self._waiting.set(n)

    def request_succeeded(self, e2e_seconds: float, queue_seconds: float | None = None) -> None:
        self._success.inc()
        self._e2e_latency.observe(e2e_seconds)
        if queue_seconds is not None:
            self._queue_time.observe(queue_seconds)

    def request_failed(self) -> None:
        self._fail.inc()

    def set_stage_stats(self, stage_id: int, stats: StagePrometheusStats) -> None:
        gauge = self._kv_cache_by_stage.get(stage_id)
        if gauge is None:
            gauge = self._kv_cache_usage_parent.labels(
                model_name=self._model_name, stage_id=str(stage_id),
            )
            self._kv_cache_by_stage[stage_id] = gauge
        gauge.set(stats.kv_cache_usage)

    def observe_diffusion_metrics(self, stage_id: int, metrics: dict[str, float]) -> None:
        for key, parent in self._diffusion_parents.items():
            value = metrics.get(key)
            if value is None:
                continue
            bound = self._diffusion_by_stage.get((key, stage_id))
            if bound is None:
                bound = parent.labels(
                    model_name=self._model_name, stage_id=str(stage_id),
                )
                self._diffusion_by_stage[(key, stage_id)] = bound
            bound.observe(value)


class OmniRequestCounter:
    """Running-request counter written by the orchestrator thread, read by the client thread."""

    def __init__(self) -> None:
        self.value = 0

    def increment(self) -> None:
        self.value += 1

    def decrement(self) -> None:
        self.value = max(0, self.value - 1)
