"""Unit tests for TrackingSpanProcessor."""

from datetime import UTC
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

clickhouse_connect = pytest.importorskip("clickhouse_connect")

from ai_pipeline_core.observability._tracking._models import SpanType
from ai_pipeline_core.observability._tracking._processor import (
    TrackingSpanProcessor,
    _classify_span,
    _hex_span_id,
    _hex_trace_id,
    _ns_to_datetime,
)


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------


class TestHexSpanId:
    def test_formats_255(self):
        assert _hex_span_id(255) == "00000000000000ff"

    def test_formats_zero(self):
        assert _hex_span_id(0) == "0000000000000000"

    def test_formats_large(self):
        result = _hex_span_id(2**63)
        assert len(result) == 16


class TestHexTraceId:
    def test_formats_42(self):
        result = _hex_trace_id(42)
        assert len(result) == 32
        assert result.endswith("2a")

    def test_formats_zero(self):
        assert _hex_trace_id(0) == "0" * 32


class TestNsToDatetime:
    def test_one_second(self):
        dt = _ns_to_datetime(1_000_000_000)
        assert dt.year == 1970
        assert dt.second == 1
        assert dt.tzinfo == UTC

    def test_zero(self):
        dt = _ns_to_datetime(0)
        assert dt.year == 1970


class TestClassifySpan:
    def test_llm(self):
        assert _classify_span({"lmnr.span.type": "LLM"}) == SpanType.LLM

    def test_flow(self):
        assert _classify_span({"prefect.flow.name": "my_flow"}) == SpanType.FLOW

    def test_task(self):
        assert _classify_span({"prefect.task.name": "my_task"}) == SpanType.TASK

    def test_default(self):
        assert _classify_span({}) == SpanType.TRACE


# ---------------------------------------------------------------------------
# Processor tests
# ---------------------------------------------------------------------------


def _make_processor():
    mock_service = MagicMock()
    processor = TrackingSpanProcessor(mock_service)
    return processor, mock_service


def _make_span_context(span_id=1, trace_id=2):
    return SimpleNamespace(span_id=span_id, trace_id=trace_id)


def _make_span(*, span_id=1, trace_id=2, name="test_span", attrs=None, parent=None):
    ctx = _make_span_context(span_id, trace_id)
    span = MagicMock()
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
    span.status = SimpleNamespace(status_code=status_code or StatusCode.OK)
    span.events = events or []
    return span


class TestOnStart:
    def test_tracks_span_start(self):
        processor, service = _make_processor()
        span = _make_span(span_id=100, trace_id=200, name="my_span")
        processor.on_start(span)
        service.track_span_start.assert_called_once()
        call_kwargs = service.track_span_start.call_args[1]
        assert call_kwargs["span_id"] == _hex_span_id(100)
        assert call_kwargs["trace_id"] == _hex_trace_id(200)
        assert call_kwargs["name"] == "my_span"

    def test_null_context_skips(self):
        processor, service = _make_processor()
        span = MagicMock()
        span.get_span_context.return_value = None
        processor.on_start(span)
        service.track_span_start.assert_not_called()

    def test_exception_swallowed(self):
        processor, service = _make_processor()
        service.track_span_start.side_effect = RuntimeError("boom")
        span = _make_span()
        processor.on_start(span)  # should not raise

    def test_parent_span_id_extracted(self):
        processor, service = _make_processor()
        parent_ctx = SimpleNamespace(span_id=42)
        span = _make_span(parent=parent_ctx)
        processor.on_start(span)
        call_kwargs = service.track_span_start.call_args[1]
        assert call_kwargs["parent_span_id"] == _hex_span_id(42)


class TestOnEnd:
    def test_full_span_end(self):

        processor, service = _make_processor()
        attrs = {
            "gen_ai.usage.cost": 0.5,
            "gen_ai.usage.input_tokens": 100,
            "gen_ai.usage.output_tokens": 50,
            "gen_ai.request.model": "gpt-4",
        }
        span = _make_readable_span(span_id=10, trace_id=20, attrs=attrs)
        processor.on_end(span)
        service.track_span_end.assert_called_once()
        kw = service.track_span_end.call_args[1]
        assert kw["cost"] == 0.5
        assert kw["tokens_input"] == 100
        assert kw["tokens_output"] == 50
        assert kw["llm_model"] == "gpt-4"
        assert kw["status"] == "completed"

    def test_error_status(self):
        from opentelemetry.trace import StatusCode

        processor, service = _make_processor()
        span = _make_readable_span(status_code=StatusCode.ERROR)
        processor.on_end(span)
        kw = service.track_span_end.call_args[1]
        assert kw["status"] == "failed"

    def test_forwards_events(self):
        processor, service = _make_processor()
        event = SimpleNamespace(
            name="log_event",
            timestamp=1_500_000_000,
            attributes={"log.level": "INFO", "key": "val"},
        )
        span = _make_readable_span(events=[event])
        processor.on_end(span)
        service.track_span_events.assert_called_once()
        events_arg = service.track_span_events.call_args[1]["events"]
        assert len(events_arg) == 1
        assert events_arg[0][0] == "log_event"
        assert events_arg[0][3] == "INFO"

    def test_doc_sha256s_extracted(self):
        processor, service = _make_processor()
        attrs = {
            "pipeline.input_document_sha256s": ["sha_in_1", "sha_in_2"],
            "pipeline.output_document_sha256s": ["sha_out_1"],
        }
        span = _make_readable_span(attrs=attrs)
        processor.on_end(span)
        kw = service.track_span_end.call_args[1]
        assert kw["input_document_sha256s"] == ["sha_in_1", "sha_in_2"]
        assert kw["output_document_sha256s"] == ["sha_out_1"]

    def test_null_context_skips(self):
        processor, service = _make_processor()
        span = MagicMock()
        span.get_span_context.return_value = None
        processor.on_end(span)
        service.track_span_end.assert_not_called()

    def test_exception_swallowed(self):
        processor, service = _make_processor()
        service.track_span_end.side_effect = RuntimeError("boom")
        span = _make_readable_span()
        processor.on_end(span)  # should not raise


class TestShutdownAndFlush:
    def test_shutdown_delegates(self):
        processor, service = _make_processor()
        processor.shutdown()
        service.shutdown.assert_called_once()

    def test_force_flush_returns_true(self):
        processor, _ = _make_processor()
        assert processor.force_flush() is True
