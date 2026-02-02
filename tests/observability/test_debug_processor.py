"""Tests for LocalDebugSpanProcessor."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_pipeline_core.observability import LocalDebugSpanProcessor, LocalTraceWriter, TraceDebugConfig


@pytest.fixture
def config(tmp_path: Path) -> TraceDebugConfig:
    """Create test configuration."""
    return TraceDebugConfig(path=tmp_path)


@pytest.fixture
def writer(config: TraceDebugConfig) -> LocalTraceWriter:
    """Create test LocalTraceWriter."""
    return LocalTraceWriter(config)


@pytest.fixture
def processor(writer: LocalTraceWriter) -> LocalDebugSpanProcessor:
    """Create test LocalDebugSpanProcessor."""
    return LocalDebugSpanProcessor(writer)


class TestLocalDebugSpanProcessor:
    """Tests for LocalDebugSpanProcessor."""

    def test_on_start_calls_writer(self, processor: LocalDebugSpanProcessor, writer: LocalTraceWriter) -> None:
        """Test on_start calls writer.on_span_start."""
        # Create mock span
        mock_span = MagicMock()
        mock_span.context.trace_id = 0xABCDEF123456
        mock_span.context.span_id = 0x123456789ABC
        mock_span.name = "test_span"
        mock_span.parent = None

        # Call on_start
        processor.on_start(mock_span, None)

        # Verify span directory was created
        trace_dirs = list(writer._config.path.iterdir())
        assert len(trace_dirs) == 1

    def test_on_end_queues_job(self, processor: LocalDebugSpanProcessor, writer: LocalTraceWriter) -> None:
        """Test on_end queues write job."""
        # First start a span
        mock_span_start = MagicMock()
        mock_span_start.context.trace_id = 0xABCDEF123456
        mock_span_start.context.span_id = 0x123456789ABC
        mock_span_start.name = "test_span"
        mock_span_start.parent = None
        processor.on_start(mock_span_start, None)

        # Create mock ReadableSpan for on_end
        mock_span_end = MagicMock()
        mock_span_end.context.trace_id = 0xABCDEF123456
        mock_span_end.context.span_id = 0x123456789ABC
        mock_span_end.name = "test_span"
        mock_span_end.parent = None
        mock_span_end.attributes = {"key": "value"}
        mock_span_end.events = []
        mock_span_end.status.status_code.name = "OK"
        mock_span_end.status.description = None
        mock_span_end.start_time = 1000000000
        mock_span_end.end_time = 2000000000

        # Patch StatusCode comparison
        with patch("ai_pipeline_core.observability._debug._processor.StatusCode") as mock_status:
            mock_status.OK = mock_span_end.status.status_code
            processor.on_end(mock_span_end)

        # Job should be queued (check queue is not empty)
        assert not writer._queue.empty()

    def test_on_start_handles_exception_gracefully(self, processor: LocalDebugSpanProcessor) -> None:
        """Test on_start doesn't raise exceptions."""
        # Pass None - should not raise
        processor.on_start(None, None)  # type: ignore

        # Pass invalid span - should not raise
        mock_span = MagicMock()
        mock_span.context = None
        processor.on_start(mock_span, None)

    def test_on_end_handles_exception_gracefully(self, processor: LocalDebugSpanProcessor) -> None:
        """Test on_end doesn't raise exceptions."""
        # Pass None - should not raise
        processor.on_end(None)  # type: ignore

        # Pass invalid span - should not raise
        mock_span = MagicMock()
        mock_span.context = None
        processor.on_end(mock_span)

    def test_shutdown_calls_writer_shutdown(self, processor: LocalDebugSpanProcessor, writer: LocalTraceWriter) -> None:
        """Test shutdown propagates to writer."""
        processor.shutdown()
        assert writer._shutdown is True

    def test_force_flush_returns_true(self, processor: LocalDebugSpanProcessor) -> None:
        """Test force_flush returns True."""
        assert processor.force_flush() is True

    def test_parent_span_id_extraction(self, processor: LocalDebugSpanProcessor) -> None:
        """Test parent span ID is correctly extracted."""
        mock_span = MagicMock()
        mock_span.parent = MagicMock()
        mock_span.parent.span_id = 0xDEADBEEF

        parent_id = processor._get_parent_span_id(mock_span)
        assert parent_id == "00000000deadbeef"

    def test_parent_span_id_none_when_no_parent(self, processor: LocalDebugSpanProcessor) -> None:
        """Test parent span ID is None when no parent."""
        mock_span = MagicMock()
        mock_span.parent = None

        parent_id = processor._get_parent_span_id(mock_span)
        assert parent_id is None
