"""Tests for SpanEventLoggingHandler."""

import logging
from unittest.mock import MagicMock, patch

from ai_pipeline_core.observability._logging_bridge import SpanEventLoggingHandler, get_bridge_handler


class TestSpanEventLoggingHandler:
    """Test logging handler."""

    def test_handler_level_is_info(self):
        handler = SpanEventLoggingHandler()
        assert handler.level == logging.INFO

    def test_emit_adds_span_event(self):
        handler = SpanEventLoggingHandler()
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        record = logging.LogRecord(
            name="test.logger",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Test warning message",
            args=(),
            exc_info=None,
        )

        with patch("ai_pipeline_core.observability._logging_bridge.otel_trace") as mock_trace:
            mock_trace.get_current_span.return_value = mock_span
            handler.emit(record)

        mock_span.add_event.assert_called_once()
        call_kwargs = mock_span.add_event.call_args
        assert call_kwargs[1]["name"] == "log"
        attrs = call_kwargs[1]["attributes"]
        assert attrs["log.level"] == "WARNING"
        assert "Test warning message" in attrs["log.message"]

    def test_emit_skips_non_recording_span(self):
        handler = SpanEventLoggingHandler()
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="msg",
            args=(),
            exc_info=None,
        )

        with patch("ai_pipeline_core.observability._logging_bridge.otel_trace") as mock_trace:
            mock_trace.get_current_span.return_value = mock_span
            handler.emit(record)

        mock_span.add_event.assert_not_called()

    def test_emit_never_raises(self):
        handler = SpanEventLoggingHandler()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="msg",
            args=(),
            exc_info=None,
        )
        with patch("ai_pipeline_core.observability._logging_bridge.otel_trace") as mock_trace:
            mock_trace.get_current_span.side_effect = RuntimeError("boom")
            # Should not raise
            handler.emit(record)

    def test_emit_deduplicates_via_record_flag(self):
        """Same record emitted twice (e.g. propagation) only produces one span event."""
        handler = SpanEventLoggingHandler()
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="duplicate check",
            args=(),
            exc_info=None,
        )

        with patch("ai_pipeline_core.observability._logging_bridge.otel_trace") as mock_trace:
            mock_trace.get_current_span.return_value = mock_span
            handler.emit(record)
            handler.emit(record)  # second call — should be skipped

        mock_span.add_event.assert_called_once()


class TestBridgeHandlerSingleton:
    """Test singleton bridge handler."""

    def test_get_bridge_handler_returns_singleton(self):
        h1 = get_bridge_handler()
        h2 = get_bridge_handler()
        assert h1 is h2

    def test_get_bridge_handler_is_span_event_handler(self):
        handler = get_bridge_handler()
        assert isinstance(handler, SpanEventLoggingHandler)
