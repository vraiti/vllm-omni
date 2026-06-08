import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import pytest
from vllm.tracing import (
    SpanKind,
    extract_trace_context,
    instrument,
    instrument_manual,
)
from vllm.tracing.otel import is_otel_available

from vllm_omni.tracing import OmniSpanAttributes

from .conftest import FakeTraceService

pytestmark = pytest.mark.skipif(not is_otel_available(), reason="OTel packages required")

FAKE_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
FAKE_PARENT_ID = "00f067aa0be902b7"


class TestOmniSpanAttributes:
    def test_attribute_constants_are_namespaced(self):
        assert OmniSpanAttributes.ENGINE_IDX == "vllm_omni.engine.idx"
        assert OmniSpanAttributes.STAGE_ID == "vllm_omni.stage.id"
        assert OmniSpanAttributes.STAGE_NAME == "vllm_omni.stage.name"
        assert OmniSpanAttributes.STAGE_REPLICA_ID == "vllm_omni.stage.replica_id"
        assert OmniSpanAttributes.PIPELINE_TIMINGS == "vllm_omni.pipeline.timings"

    def test_diffusion_attributes_are_namespaced(self):
        assert OmniSpanAttributes.DIFFUSION_PREPROCESS_MS == "vllm_omni.diffusion.preprocess_ms"
        assert OmniSpanAttributes.DIFFUSION_EXEC_MS == "vllm_omni.diffusion.exec_ms"
        assert OmniSpanAttributes.DIFFUSION_POSTPROCESS_MS == "vllm_omni.diffusion.postprocess_ms"
        assert OmniSpanAttributes.DIFFUSION_TOTAL_MS == "vllm_omni.diffusion.total_ms"


class TestInstrumentDecorator:
    @pytest.fixture(autouse=True)
    def _setup(self, init_test_tracer):
        self.trace_service: FakeTraceService = init_test_tracer

    def test_sync_function_creates_span(self):
        @instrument(span_name="sync_op")
        def sync_op():
            return 42

        result = sync_op()
        assert result == 42
        assert self.trace_service.wait_for_spans(count=1)
        spans = self.trace_service.get_all_spans()
        assert any(s["name"] == "sync_op" for s in spans)

    def test_async_function_creates_span(self):
        @instrument(span_name="async_op")
        async def async_op():
            return 99

        result = asyncio.run(async_op())
        assert result == 99
        assert self.trace_service.wait_for_spans(count=1)
        spans = self.trace_service.get_all_spans()
        assert any(s["name"] == "async_op" for s in spans)

    def test_nested_spans_have_parent_child_relationship(self):
        @instrument(span_name="child_fn")
        def child_fn():
            pass

        @instrument(span_name="parent_fn")
        def parent_fn():
            child_fn()

        parent_fn()
        assert self.trace_service.wait_for_spans(count=2)
        spans = self.trace_service.get_all_spans()
        parent = next(s for s in spans if s["name"] == "parent_fn")
        child = next(s for s in spans if s["name"] == "child_fn")
        assert child["parent_span_id"] == parent["span_id"]
        assert child["trace_id"] == parent["trace_id"]

    def test_span_has_valid_timestamps(self):
        @instrument(span_name="timed_fn")
        def timed_fn():
            pass

        timed_fn()
        assert self.trace_service.wait_for_spans(count=1)
        span = self.trace_service.get_all_spans()[0]
        assert span["start_time_unix_nano"] > 0
        assert span["end_time_unix_nano"] >= span["start_time_unix_nano"]


class TestInstrumentManual:
    @pytest.fixture(autouse=True)
    def _setup(self, init_test_tracer):
        self.trace_service: FakeTraceService = init_test_tracer

    def test_creates_span_with_explicit_timestamps(self):
        start_ns = int(time.time() * 1e9)
        end_ns = start_ns + 500_000_000

        instrument_manual(
            span_name="manual_span",
            start_time=start_ns,
            end_time=end_ns,
            attributes={"test.key": "test_value"},
        )

        assert self.trace_service.wait_for_spans(count=1)
        span = self.trace_service.get_all_spans()[0]
        assert span["name"] == "manual_span"
        assert span["start_time_unix_nano"] == start_ns
        assert span["end_time_unix_nano"] == end_ns
        assert span["attributes"]["test.key"] == "test_value"

    def test_span_with_no_end_time_ends_immediately(self):
        start_ns = int(time.time() * 1e9)

        instrument_manual(
            span_name="auto_end",
            start_time=start_ns,
        )

        assert self.trace_service.wait_for_spans(count=1)
        span = self.trace_service.get_all_spans()[0]
        assert span["name"] == "auto_end"
        assert span["end_time_unix_nano"] >= span["start_time_unix_nano"]

    def test_span_respects_parent_context(self):
        traceparent = f"00-{FAKE_TRACE_ID}-{FAKE_PARENT_ID}-01"
        ctx = extract_trace_context({"traceparent": traceparent})

        instrument_manual(
            span_name="child_manual",
            start_time=int(time.time() * 1e9),
            context=ctx,
        )

        assert self.trace_service.wait_for_spans(count=1)
        span = self.trace_service.get_all_spans()[0]
        assert span["trace_id"] == FAKE_TRACE_ID
        assert span["parent_span_id"] == FAKE_PARENT_ID

    def test_span_kind_is_propagated(self):
        instrument_manual(
            span_name="server_span",
            start_time=int(time.time() * 1e9),
            kind=SpanKind.SERVER,
        )

        assert self.trace_service.wait_for_spans(count=1)
        span = self.trace_service.get_all_spans()[0]
        # SPAN_KIND_SERVER = 2 in the protobuf enum
        assert span["kind"] == 2

    def test_multiple_attributes_types(self):
        instrument_manual(
            span_name="typed_attrs",
            start_time=int(time.time() * 1e9),
            attributes={
                "str_attr": "hello",
                "int_attr": 42,
                "float_attr": 3.14,
                "bool_attr": True,
            },
        )

        assert self.trace_service.wait_for_spans(count=1)
        attrs = self.trace_service.get_all_spans()[0]["attributes"]
        assert attrs["str_attr"] == "hello"
        assert attrs["int_attr"] == 42
        assert abs(attrs["float_attr"] - 3.14) < 0.001
        assert attrs["bool_attr"] is True


class TestTraceContextPropagation:
    @pytest.fixture(autouse=True)
    def _setup(self, init_test_tracer):
        self.trace_service: FakeTraceService = init_test_tracer

    def test_extract_from_traceparent_header(self):
        traceparent = f"00-{FAKE_TRACE_ID}-{FAKE_PARENT_ID}-01"
        ctx = extract_trace_context({"traceparent": traceparent})
        assert ctx is not None

        @instrument(span_name="with_parent")
        def traced():
            pass

        from opentelemetry import context

        token = context.attach(ctx)
        try:
            traced()
        finally:
            context.detach(token)

        assert self.trace_service.wait_for_spans(count=1)
        span = self.trace_service.get_all_spans()[0]
        assert span["trace_id"] == FAKE_TRACE_ID
        assert span["parent_span_id"] == FAKE_PARENT_ID

    def test_extract_from_none_returns_none(self):
        ctx = extract_trace_context(None)
        assert ctx is None

    def test_extract_from_empty_dict_returns_none(self):
        ctx = extract_trace_context({})
        assert ctx is None

    def test_header_injection_round_trip(self):
        from opentelemetry import trace as otel_trace
        from opentelemetry.context import Context
        from opentelemetry.trace.propagation import set_span_in_context
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )

        tracer = otel_trace.get_tracer("test.roundtrip")
        span = tracer.start_span("origin")
        ctx: Context = set_span_in_context(span)

        carrier: dict[str, str] = {}
        TraceContextTextMapPropagator().inject(carrier, context=ctx)
        assert "traceparent" in carrier

        extracted = extract_trace_context(carrier)
        assert extracted is not None

        instrument_manual(
            span_name="downstream",
            start_time=int(time.time() * 1e9),
            context=extracted,
        )
        span.end()

        assert self.trace_service.wait_for_spans(count=2)
        spans = self.trace_service.get_all_spans()
        origin = next(s for s in spans if s["name"] == "origin")
        downstream = next(s for s in spans if s["name"] == "downstream")
        assert downstream["trace_id"] == origin["trace_id"]
        assert downstream["parent_span_id"] == origin["span_id"]


@dataclass
class MockOrchestratorRequestState:
    request_id: str = "req-001"
    trace_headers: dict[str, str] | None = None
    trace_start_ns: int = 0
    pipeline_timings: dict[str, float] = field(default_factory=dict)
    _omni_span: Any = None


class TestOmniRequestSpan:
    @pytest.fixture(autouse=True)
    def _setup(self, init_test_tracer):
        self.trace_service: FakeTraceService = init_test_tracer

    def _make_orchestrator(self):
        from vllm_omni.engine.orchestrator import Orchestrator

        orch = object.__new__(Orchestrator)
        orch._tracing_enabled = True
        return orch

    def test_start_span_sets_omni_span(self):
        orch = self._make_orchestrator()
        req_state = MockOrchestratorRequestState(
            trace_start_ns=int(time.time() * 1e9),
        )

        orch._start_omni_request_span(req_state)

        assert req_state._omni_span is not None
        assert req_state.trace_headers is not None
        assert "traceparent" in req_state.trace_headers

    def test_start_span_inherits_parent_trace(self):
        orch = self._make_orchestrator()
        traceparent = f"00-{FAKE_TRACE_ID}-{FAKE_PARENT_ID}-01"
        req_state = MockOrchestratorRequestState(
            trace_headers={"traceparent": traceparent},
            trace_start_ns=int(time.time() * 1e9),
        )

        orch._start_omni_request_span(req_state)
        req_state._omni_span.end()

        assert self.trace_service.wait_for_spans(count=1)
        span = self.trace_service.get_all_spans()[0]
        assert span["name"] == "omni_request"
        assert span["trace_id"] == FAKE_TRACE_ID
        assert span["parent_span_id"] == FAKE_PARENT_ID

    def test_end_span_sets_attributes(self):
        orch = self._make_orchestrator()
        req_state = MockOrchestratorRequestState(
            trace_start_ns=int(time.time() * 1e9),
            pipeline_timings={"stage0_ms": 100.5, "stage1_ms": 200.3},
        )

        orch._start_omni_request_span(req_state)
        orch._end_omni_request_span("req-001", req_state)

        assert self.trace_service.wait_for_spans(count=1)
        span = self.trace_service.get_all_spans()[0]
        assert span["name"] == "omni_request"
        assert span["attributes"]["gen_ai.request.id"] == "req-001"
        assert "vllm_omni.pipeline.timings" in span["attributes"]

    def test_end_span_without_start_is_noop(self):
        orch = self._make_orchestrator()
        req_state = MockOrchestratorRequestState()
        # _omni_span is None, should not raise
        orch._end_omni_request_span("req-001", req_state)


class TestLlmRequestSpan:
    @pytest.fixture(autouse=True)
    def _setup(self, init_test_tracer):
        self.trace_service: FakeTraceService = init_test_tracer

    def test_llm_request_span_attributes(self):
        from vllm.tracing import SpanAttributes

        now = time.time()
        arrival_time_ns = int(now * 1e9)
        traceparent = f"00-{FAKE_TRACE_ID}-{FAKE_PARENT_ID}-01"
        trace_context = extract_trace_context({"traceparent": traceparent})

        attributes = {
            SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN: 0.05,
            SpanAttributes.GEN_AI_LATENCY_E2E: 1.5,
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE: 0.01,
            SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS: 10,
            SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS: 50,
            SpanAttributes.GEN_AI_REQUEST_ID: "req-llm-001",
            OmniSpanAttributes.ENGINE_IDX: 0,
            OmniSpanAttributes.STAGE_ID: 0,
            OmniSpanAttributes.STAGE_REPLICA_ID: 0,
            OmniSpanAttributes.STAGE_NAME: "thinker",
        }

        instrument_manual(
            span_name="llm_request",
            start_time=arrival_time_ns,
            attributes=attributes,
            context=trace_context,
            kind=SpanKind.INTERNAL,
        )

        assert self.trace_service.wait_for_spans(count=1)
        span = self.trace_service.get_all_spans()[0]
        assert span["name"] == "llm_request"
        assert span["trace_id"] == FAKE_TRACE_ID
        assert span["attributes"]["gen_ai.request.id"] == "req-llm-001"
        assert span["attributes"]["gen_ai.usage.prompt_tokens"] == 10
        assert span["attributes"]["gen_ai.usage.completion_tokens"] == 50
        assert span["attributes"][OmniSpanAttributes.STAGE_NAME] == "thinker"
        assert span["attributes"][OmniSpanAttributes.ENGINE_IDX] == 0


class TestDiffusionRequestSpan:
    @pytest.fixture(autouse=True)
    def _setup(self, init_test_tracer):
        self.trace_service: FakeTraceService = init_test_tracer

    def test_diffusion_span_with_timing_attributes(self):
        now = time.time()
        start_ns = int(now * 1e9)
        total_ms = 2500.0
        end_ns = int((now + total_ms / 1000) * 1e9)

        attributes = {
            OmniSpanAttributes.STAGE_ID: 0,
            OmniSpanAttributes.STAGE_NAME: "diffusion",
            OmniSpanAttributes.STAGE_REPLICA_ID: 0,
            OmniSpanAttributes.ENGINE_IDX: 0,
            OmniSpanAttributes.DIFFUSION_PREPROCESS_MS: 50.0,
            OmniSpanAttributes.DIFFUSION_EXEC_MS: 2300.0,
            OmniSpanAttributes.DIFFUSION_POSTPROCESS_MS: 150.0,
            OmniSpanAttributes.DIFFUSION_TOTAL_MS: total_ms,
        }

        instrument_manual(
            span_name="diffusion_request",
            start_time=start_ns,
            end_time=end_ns,
            attributes=attributes,
            kind=SpanKind.INTERNAL,
        )

        assert self.trace_service.wait_for_spans(count=1)
        span = self.trace_service.get_all_spans()[0]
        assert span["name"] == "diffusion_request"
        attrs = span["attributes"]
        assert attrs[OmniSpanAttributes.STAGE_NAME] == "diffusion"
        assert abs(attrs[OmniSpanAttributes.DIFFUSION_PREPROCESS_MS] - 50.0) < 0.01
        assert abs(attrs[OmniSpanAttributes.DIFFUSION_EXEC_MS] - 2300.0) < 0.01
        assert abs(attrs[OmniSpanAttributes.DIFFUSION_POSTPROCESS_MS] - 150.0) < 0.01
        assert abs(attrs[OmniSpanAttributes.DIFFUSION_TOTAL_MS] - total_ms) < 0.01
        assert span["start_time_unix_nano"] == start_ns
        assert span["end_time_unix_nano"] == end_ns

    def test_diffusion_span_inherits_trace_context(self):
        traceparent = f"00-{FAKE_TRACE_ID}-{FAKE_PARENT_ID}-01"
        ctx = extract_trace_context({"traceparent": traceparent})

        instrument_manual(
            span_name="diffusion_request",
            start_time=int(time.time() * 1e9),
            attributes={
                OmniSpanAttributes.STAGE_ID: 0,
                OmniSpanAttributes.STAGE_NAME: "diffusion",
            },
            context=ctx,
            kind=SpanKind.INTERNAL,
        )

        assert self.trace_service.wait_for_spans(count=1)
        span = self.trace_service.get_all_spans()[0]
        assert span["trace_id"] == FAKE_TRACE_ID
        assert span["parent_span_id"] == FAKE_PARENT_ID
