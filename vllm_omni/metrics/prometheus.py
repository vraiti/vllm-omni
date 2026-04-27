import threading

from prometheus_client import Counter, Gauge, Histogram

_labelnames = ["model_name"]

_num_requests_running = Gauge(
    "vllm_omni:num_requests_running",
    "Number of requests currently running across all pipeline stages.",
    labelnames=_labelnames,
)

_num_requests_waiting = Gauge(
    "vllm_omni:num_requests_waiting",
    "Number of requests waiting to be scheduled.",
    labelnames=_labelnames,
)

_num_requests_success = Counter(
    "vllm_omni:num_requests_success",
    "Number of requests that completed without error.",
    labelnames=_labelnames,
)

_num_requests_fail = Counter(
    "vllm_omni:num_requests_fail",
    "Number of requests that returned an error.",
    labelnames=_labelnames,
)

_e2e_request_latency_seconds = Histogram(
    "vllm_omni:e2e_request_latency_seconds",
    "Histogram of end-to-end request latency in seconds.",
    labelnames=_labelnames,
)

_request_queue_time_seconds = Histogram(
    "vllm_omni:request_queue_time_seconds",
    "Histogram of request queue wait time in seconds.",
    labelnames=_labelnames,
)


class OmniPrometheusMetrics:
    """Label-bound wrapper around the raw Prometheus metrics."""

    def __init__(self, model_name: str) -> None:
        self._running = _num_requests_running.labels(model_name=model_name)
        self._waiting = _num_requests_waiting.labels(model_name=model_name)
        self._success = _num_requests_success.labels(model_name=model_name)
        self._fail = _num_requests_fail.labels(model_name=model_name)
        self._e2e_latency = _e2e_request_latency_seconds.labels(model_name=model_name)
        self._queue_time = _request_queue_time_seconds.labels(model_name=model_name)

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


class OmniRequestCounter:
    """Thread-safe counter shared between the client and orchestrator threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 0

    def increment(self) -> None:
        with self._lock:
            self._value += 1

    def decrement(self) -> None:
        with self._lock:
            self._value = max(0, self._value - 1)

    @property
    def value(self) -> int:
        with self._lock:
            return self._value
