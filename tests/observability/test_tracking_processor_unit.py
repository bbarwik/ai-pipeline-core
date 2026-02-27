"""Unit tests for PipelineSpanProcessor."""

import dataclasses
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4


from ai_pipeline_core.observability._span_data import SpanData, _attr_to_json
from ai_pipeline_core.observability._tracking._processor import (
    PipelineSpanProcessor,
    _build_start_span_data,
)


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------


class TestAttrToJson:
    def test_none_returns_empty(self):
        assert _attr_to_json(None) == ""

    def test_string_preserved_as_is(self):
        assert _attr_to_json('{"key": "val"}') == '{"key": "val"}'

    def test_dict_serialized_to_json(self):
        import json

        result = _attr_to_json({"key": "val"})
        assert json.loads(result) == {"key": "val"}

    def test_list_serialized_to_json(self):
        import json

        result = _attr_to_json([1, 2, 3])
        assert json.loads(result) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Processor tests
# ---------------------------------------------------------------------------


class MockBackend:
    """Mock SpanBackend for testing."""

    def __init__(self):
        self.start_calls: list[SpanData] = []
        self.end_calls: list[SpanData] = []
        self.shutdown_called = False

    def on_span_start(self, span_data: SpanData) -> None:
        self.start_calls.append(span_data)

    def on_span_end(self, span_data: SpanData) -> None:
        self.end_calls.append(span_data)

    def shutdown(self) -> None:
        self.shutdown_called = True


def _make_span_context(span_id=1, trace_id=2):
    return SimpleNamespace(span_id=span_id, trace_id=trace_id)


def _make_span(*, span_id=1, trace_id=2, name="test_span", attrs=None, parent=None):
    ctx = _make_span_context(span_id, trace_id)
    span = MagicMock()
    span.context = ctx
    span.get_span_context.return_value = ctx
    span.name = name
    span.attributes = attrs or {}
    span.parent = parent
    return span


def _make_readable_span(
    *,
    span_id=1,
    trace_id=2,
    name="test_span",
    attrs=None,
    parent=None,
    start_time=1_000_000_000,
    end_time=2_000_000_000,
    status_code=None,
    events=None,
    resource=None,
):
    from opentelemetry.trace import StatusCode

    ctx = _make_span_context(span_id, trace_id)
    span = MagicMock()
    span.get_span_context.return_value = ctx
    span.name = name
    span.attributes = attrs or {}
    span.parent = parent
    span.start_time = start_time
    span.end_time = end_time
    span.status = SimpleNamespace(status_code=status_code or StatusCode.OK, description=None)
    span.events = events or []
    span.resource = resource
    return span


class TestOnStart:
    def test_notifies_backend(self):
        backend = MockBackend()
        processor = PipelineSpanProcessor(backends=(backend,))
        span = _make_span(span_id=100, trace_id=200, name="my_span")
        processor.on_start(span)
        assert len(backend.start_calls) == 1
        data = backend.start_calls[0]
        assert data.span_id == format(100, "016x")
        assert data.name == "my_span"

    def test_null_context_skips(self):
        backend = MockBackend()
        processor = PipelineSpanProcessor(backends=(backend,))
        span = MagicMock()
        span.context = None
        processor.on_start(span)
        assert len(backend.start_calls) == 0

    def test_llm_span_filtered(self):
        backend = MockBackend()
        processor = PipelineSpanProcessor(backends=(backend,))
        span = _make_span(attrs={"lmnr.span.type": "LLM"})
        processor.on_start(span)
        assert len(backend.start_calls) == 0

    def test_llm_span_not_filtered_when_verbose(self):
        backend = MockBackend()
        processor = PipelineSpanProcessor(backends=(backend,), verbose=True)
        span = _make_span(attrs={"lmnr.span.type": "LLM"})
        processor.on_start(span)
        assert len(backend.start_calls) == 1


class TestOnEnd:
    def test_dispatches_span_data(self):
        backend = MockBackend()
        processor = PipelineSpanProcessor(backends=(backend,))
        span = _make_span(span_id=10, trace_id=20)
        processor.on_start(span)
        readable = _make_readable_span(
            span_id=10,
            trace_id=20,
            attrs={"gen_ai.usage.cost": 0.5, "gen_ai.usage.input_tokens": 100},
        )
        processor.on_end(readable)
        assert len(backend.end_calls) == 1
        data = backend.end_calls[0]
        assert data.cost == 0.5
        assert data.tokens_input == 100
        assert data.status == "completed"

    def test_error_status(self):
        from opentelemetry.trace import StatusCode

        backend = MockBackend()
        processor = PipelineSpanProcessor(backends=(backend,))
        span = _make_span(span_id=10, trace_id=20)
        processor.on_start(span)
        readable = _make_readable_span(span_id=10, trace_id=20, status_code=StatusCode.ERROR)
        processor.on_end(readable)
        assert backend.end_calls[0].status == "failed"

    def test_filtered_llm_still_dispatched_to_backend(self):
        """Filtered LLM spans (span_order=0) are still sent to backend.on_end for metrics."""
        backend = MockBackend()
        processor = PipelineSpanProcessor(backends=(backend,))
        span = _make_span(span_id=10, trace_id=20, attrs={"lmnr.span.type": "LLM"})
        processor.on_start(span)
        readable = _make_readable_span(span_id=10, trace_id=20, attrs={"lmnr.span.type": "LLM"})
        processor.on_end(readable)
        assert len(backend.end_calls) == 1
        assert backend.end_calls[0].span_order == 0

    def test_null_context_skips(self):
        backend = MockBackend()
        processor = PipelineSpanProcessor(backends=(backend,))
        span = MagicMock()
        span.get_span_context.return_value = None
        processor.on_end(span)
        assert len(backend.end_calls) == 0


class TestSpanOrdering:
    def test_monotonic_span_order(self):
        backend = MockBackend()
        processor = PipelineSpanProcessor(backends=(backend,))
        for i in range(3):
            span = _make_span(span_id=i + 10, trace_id=100)
            processor.on_start(span)
        assert len(backend.start_calls) == 3
        orders = [d.span_order for d in backend.start_calls]
        assert orders == [1, 2, 3]


class TestShutdownAndFlush:
    def test_shutdown_delegates(self):
        backend = MockBackend()
        processor = PipelineSpanProcessor(backends=(backend,))
        processor.shutdown()
        assert backend.shutdown_called

    def test_force_flush_returns_true(self):
        backend = MockBackend()
        processor = PipelineSpanProcessor(backends=(backend,))
        assert processor.force_flush() is True


class TestBuildStartSpanData:
    def test_minimal_span_data(self):
        data = _build_start_span_data(
            span_id="abc",
            trace_id="def",
            parent_span_id=None,
            name="test",
            span_order=42,
        )
        assert data.span_id == "abc"
        assert data.span_order == 42
        assert data.status == "running"
        assert data.execution_id is None


# ---------------------------------------------------------------------------
# Run context propagation (C1 fix verification)
# ---------------------------------------------------------------------------


class TestRunContextPropagation:
    def test_set_run_context_injects_execution_id(self):
        """After set_run_context, child spans without execution_id get it injected."""
        backend = MockBackend()
        processor = PipelineSpanProcessor(backends=(backend,))
        uid = uuid4()
        processor.set_run_context(execution_id=uid, run_id="r1", flow_name="f1", run_scope="s1")
        span = _make_span(span_id=10, trace_id=20)
        processor.on_start(span)
        readable = _make_readable_span(span_id=10, trace_id=20)
        processor.on_end(readable)
        data = backend.end_calls[0]
        assert data.execution_id == uid
        assert data.run_id == "r1"
        assert data.flow_name == "f1"
        assert data.run_scope == "s1"

    def test_does_not_overwrite_existing_execution_id(self):
        """Span with its own execution_id in attributes keeps it."""
        backend = MockBackend()
        processor = PipelineSpanProcessor(backends=(backend,))
        processor_uid = uuid4()
        span_uid = uuid4()
        processor.set_run_context(execution_id=processor_uid, run_id="r1", flow_name="f1", run_scope="s1")
        span = _make_span(span_id=10, trace_id=20)
        processor.on_start(span)
        readable = _make_readable_span(
            span_id=10,
            trace_id=20,
            attrs={"pipeline.execution_id": str(span_uid)},
        )
        processor.on_end(readable)
        data = backend.end_calls[0]
        assert data.execution_id == span_uid

    def test_clear_run_context_stops_injection(self):
        backend = MockBackend()
        processor = PipelineSpanProcessor(backends=(backend,))
        uid = uuid4()
        processor.set_run_context(execution_id=uid, run_id="r1", flow_name="f1", run_scope="s1")
        processor.clear_run_context()
        span = _make_span(span_id=10, trace_id=20)
        processor.on_start(span)
        readable = _make_readable_span(span_id=10, trace_id=20)
        processor.on_end(readable)
        data = backend.end_calls[0]
        assert data.execution_id is None


# ---------------------------------------------------------------------------
# Span ordering with filtered LLM spans
# ---------------------------------------------------------------------------


class TestSpanOrderWithFiltered:
    def test_no_gaps_after_filtered(self):
        """Regular spans maintain consecutive ordering even when LLM spans are filtered."""
        backend = MockBackend()
        processor = PipelineSpanProcessor(backends=(backend,))
        # Regular span 1
        processor.on_start(_make_span(span_id=1, trace_id=100))
        # Filtered LLM span
        processor.on_start(_make_span(span_id=2, trace_id=100, attrs={"lmnr.span.type": "LLM"}))
        # Regular span 2
        processor.on_start(_make_span(span_id=3, trace_id=100))
        # Orders should be [1, 2] for regular spans (LLM skipped)
        orders = [d.span_order for d in backend.start_calls]
        assert orders == [1, 2]

    def test_filtered_span_gets_order_zero(self):
        backend = MockBackend()
        processor = PipelineSpanProcessor(backends=(backend,))
        span = _make_span(span_id=10, trace_id=20, attrs={"lmnr.span.type": "LLM"})
        processor.on_start(span)
        readable = _make_readable_span(span_id=10, trace_id=20, attrs={"lmnr.span.type": "LLM"})
        processor.on_end(readable)
        assert backend.end_calls[0].span_order == 0


# ---------------------------------------------------------------------------
# Multiple backends
# ---------------------------------------------------------------------------


class TestMultipleBackends:
    def test_both_receive_same_span_data(self):
        b1 = MockBackend()
        b2 = MockBackend()
        processor = PipelineSpanProcessor(backends=(b1, b2))
        span = _make_span(span_id=10, trace_id=20)
        processor.on_start(span)
        readable = _make_readable_span(span_id=10, trace_id=20)
        processor.on_end(readable)
        assert len(b1.end_calls) == 1
        assert len(b2.end_calls) == 1
        assert b1.end_calls[0].span_id == b2.end_calls[0].span_id

    def test_backend_error_does_not_propagate(self):
        """If one backend raises, the other still receives the span."""

        class FailBackend:
            def on_span_start(self, span_data: SpanData) -> None:
                raise RuntimeError("fail")

            def on_span_end(self, span_data: SpanData) -> None:
                raise RuntimeError("fail")

            def shutdown(self) -> None:
                pass

        fail = FailBackend()
        good = MockBackend()
        processor = PipelineSpanProcessor(backends=(fail, good))
        span = _make_span(span_id=10, trace_id=20)
        processor.on_start(span)
        assert len(good.start_calls) == 1
        readable = _make_readable_span(span_id=10, trace_id=20)
        processor.on_end(readable)
        assert len(good.end_calls) == 1


# ---------------------------------------------------------------------------
# Concurrent access
# ---------------------------------------------------------------------------


class TestConcurrentAccess:
    def test_concurrent_on_start_on_end(self):
        """20 threads doing on_start + on_end — no crashes or KeyErrors."""
        backend = MockBackend()
        processor = PipelineSpanProcessor(backends=(backend,))
        barrier = threading.Barrier(20)

        def worker(idx: int) -> None:
            span = _make_span(span_id=idx + 1000, trace_id=1)
            barrier.wait()  # Maximize contention
            processor.on_start(span)
            readable = _make_readable_span(span_id=idx + 1000, trace_id=1)
            processor.on_end(readable)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(backend.end_calls) == 20


# ---------------------------------------------------------------------------
# dataclasses.replace on frozen SpanData
# ---------------------------------------------------------------------------


class TestSpanDataReplace:
    def test_replace_preserves_original(self):
        original = _build_start_span_data(
            span_id="abc",
            trace_id="def",
            parent_span_id=None,
            name="test",
            span_order=1,
        )
        uid = uuid4()
        replaced = dataclasses.replace(original, execution_id=uid, run_id="new_run")
        assert replaced.execution_id == uid
        assert replaced.run_id == "new_run"
        assert original.execution_id is None
        assert original.run_id == ""
