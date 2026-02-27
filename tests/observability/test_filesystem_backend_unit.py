"""Unit tests for FilesystemBackend — queue, flush, shutdown, filtered metrics."""

import threading
from datetime import UTC, datetime
from unittest.mock import patch

from ai_pipeline_core.observability._debug._backend import FilesystemBackend, _FlushRequest
from ai_pipeline_core.observability._span_data import SpanData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPOCH = datetime(2024, 1, 1, tzinfo=UTC)


def _make_span_data(*, span_order: int = 1, trace_id: str = "trace1", span_id: str = "span1", cost: float = 0.0) -> SpanData:
    return SpanData(
        execution_id=None,
        span_id=span_id,
        trace_id=trace_id,
        parent_span_id=None,
        name="test",
        run_id="",
        flow_name="",
        run_scope="",
        span_type="trace",
        status="completed",
        start_time=_EPOCH,
        end_time=_EPOCH,
        duration_ms=100,
        span_order=span_order,
        cost=cost,
        tokens_input=0,
        tokens_output=0,
        tokens_cached=0,
        llm_model=None,
        error_message="",
        input_json="",
        output_json="",
        replay_payload="",
        attributes={},
        events=(),
        input_doc_sha256s=(),
        output_doc_sha256s=(),
    )


def _create_backend(tmp_path) -> FilesystemBackend:
    """Create a FilesystemBackend with a mock materializer to avoid filesystem interaction."""
    with patch("ai_pipeline_core.observability._debug._backend.TraceMaterializer"):
        from ai_pipeline_core.observability._debug._config import TraceDebugConfig

        config = TraceDebugConfig(path=tmp_path / "trace")
        backend = FilesystemBackend(config)
    return backend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOnSpanStart:
    def test_delegates_to_materializer(self, tmp_path):
        backend = _create_backend(tmp_path)
        try:
            span_data = _make_span_data()
            backend.on_span_start(span_data)
            backend._materializer.on_span_start.assert_called_once_with(span_data)
        finally:
            backend.shutdown()


class TestOnSpanEnd:
    def test_queues_span(self, tmp_path):
        backend = _create_backend(tmp_path)
        try:
            span_data = _make_span_data(span_order=5)
            backend.on_span_end(span_data)
            backend.flush()
            backend._materializer.add_span.assert_called_once_with(span_data)
        finally:
            backend.shutdown()

    def test_filtered_metrics_span_order_zero(self, tmp_path):
        backend = _create_backend(tmp_path)
        try:
            span_data = _make_span_data(span_order=0, cost=0.1)
            backend.on_span_end(span_data)
            backend._materializer.record_filtered_llm_metrics.assert_called_once_with(span_data)
            backend._materializer.add_span.assert_not_called()
        finally:
            backend.shutdown()

    def test_after_shutdown_noop(self, tmp_path):
        backend = _create_backend(tmp_path)
        backend.shutdown()
        span_data = _make_span_data()
        # Should not raise
        backend.on_span_end(span_data)
        backend._materializer.add_span.assert_not_called()


class TestFlush:
    def test_blocks_until_drained(self, tmp_path):
        backend = _create_backend(tmp_path)
        try:
            for i in range(5):
                backend.on_span_end(_make_span_data(span_order=i + 1, span_id=f"s{i}"))
            result = backend.flush(timeout=10.0)
            assert result is True
            assert backend._materializer.add_span.call_count == 5
        finally:
            backend.shutdown()

    def test_after_shutdown_returns_true(self, tmp_path):
        backend = _create_backend(tmp_path)
        backend.shutdown()
        assert backend.flush() is True


class TestShutdown:
    def test_drains_queue(self, tmp_path):
        backend = _create_backend(tmp_path)
        backend.on_span_end(_make_span_data(span_order=1))
        backend.shutdown()
        backend._materializer.finalize_all.assert_called_once()

    def test_idempotent(self, tmp_path):
        backend = _create_backend(tmp_path)
        backend.shutdown()
        backend.shutdown()  # Should not raise


class TestFlushRequest:
    def test_no_args_creates_event(self):
        req = _FlushRequest()
        assert isinstance(req.done, threading.Event)
        assert req.done.is_set() is False


class TestWriterLoopErrorHandling:
    def test_materializer_error_continues(self, tmp_path):
        backend = _create_backend(tmp_path)
        try:
            backend._materializer.add_span.side_effect = [OSError("disk full"), None]
            backend.on_span_end(_make_span_data(span_order=1, span_id="s1"))
            backend.on_span_end(_make_span_data(span_order=2, span_id="s2"))
            backend.flush()
            assert backend._materializer.add_span.call_count == 2
        finally:
            backend.shutdown()
