"""Tests for TrackingService."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

clickhouse_connect = pytest.importorskip("clickhouse_connect")

from datetime import UTC

from ai_pipeline_core.observability._tracking._models import (
    RunStatus,
    SpanType,
)
from ai_pipeline_core.observability._tracking._service import TrackingService


def _make_service() -> tuple[TrackingService, MagicMock]:
    """Create a TrackingService with mocked ClickHouseClient."""
    mock_client = MagicMock()
    with patch("ai_pipeline_core.observability._tracking._service.ClickHouseWriter") as mock_writer_cls:
        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer
        service = TrackingService(mock_client)
    return service, mock_writer


class TestRunContext:
    """Test run context management."""

    def test_set_and_clear_run_context(self):
        service, _ = _make_service()
        run_id = uuid4()
        service.set_run_context(execution_id=run_id, run_id="proj", flow_name="flow")
        assert service._execution_id == run_id
        assert service._run_id == "proj"
        service.clear_run_context()
        assert service._execution_id is None
        assert service._run_id == ""

    def test_clear_clears_caches(self):
        service, _ = _make_service()
        service.clear_run_context()
        assert service._run_id == ""


class TestRunTracking:
    """Test pipeline run tracking."""

    def test_track_run_start(self):
        service, mock_writer = _make_service()
        run_id = uuid4()
        service.set_run_context(execution_id=run_id, run_id="proj", flow_name="flow")
        service.track_run_start(execution_id=run_id, run_id="proj", flow_name="flow")
        assert mock_writer.write.called
        args = mock_writer.write.call_args
        assert args[0][0] == "pipeline_runs"
        assert args[0][1][0].status == RunStatus.RUNNING

    def test_track_run_end_uses_stored_start_time(self):
        service, mock_writer = _make_service()
        run_id = uuid4()
        service.set_run_context(execution_id=run_id, run_id="proj", flow_name="flow")
        service.track_run_start(execution_id=run_id, run_id="proj", flow_name="flow")
        start_time = service._run_start_time
        service.track_run_end(execution_id=run_id, status=RunStatus.COMPLETED)
        end_call = mock_writer.write.call_args_list[-1]
        row = end_call[0][1][0]
        assert row.start_time == start_time
        assert row.end_time is not None


class TestSpanTracking:
    """Test span tracking."""

    def test_no_tracking_without_run_context(self):
        service, mock_writer = _make_service()
        from datetime import datetime

        now = datetime.now(UTC)
        service.track_span_end(
            span_id="abc",
            trace_id="0" * 32,
            parent_span_id=None,
            name="task",
            span_type=SpanType.TASK,
            status="completed",
            start_time=now,
            end_time=now,
            duration_ms=0,
        )
        assert not mock_writer.write.called


class TestDocSummaryConfig:
    """TrackingService does not expose document summary configuration."""

    def test_no_doc_summary_properties(self):
        service, _ = _make_service()
        assert not hasattr(service, "doc_summary_enabled")
        assert not hasattr(service, "doc_summary_min_chars")


# ---------------------------------------------------------------------------
# Extended tracking tests for coverage
# ---------------------------------------------------------------------------


class TestSpanStartTracking:
    def test_tracks_span_start_with_context(self):
        service, mock_writer = _make_service()
        run_id = uuid4()
        service.set_run_context(execution_id=run_id, run_id="proj", flow_name="flow")
        service.track_span_start(
            span_id="abc",
            trace_id="0" * 32,
            parent_span_id=None,
            name="task",
            span_type=SpanType.TASK,
        )
        assert mock_writer.write.called
        args = mock_writer.write.call_args
        assert args[0][0] == "tracked_spans"
        row = args[0][1][0]
        assert row.status == "running"
        assert row.span_type == SpanType.TASK

    def test_no_start_tracking_without_context(self):
        service, mock_writer = _make_service()
        service.track_span_start(
            span_id="abc",
            trace_id="0" * 32,
            parent_span_id=None,
            name="task",
            span_type=SpanType.TASK,
        )
        mock_writer.write.assert_not_called()


class TestSpanEndFull:
    def test_tracks_span_end_with_all_fields(self):
        from datetime import datetime

        service, mock_writer = _make_service()
        run_id = uuid4()
        service.set_run_context(execution_id=run_id, run_id="proj", flow_name="flow")
        now = datetime.now(UTC)
        service.track_span_end(
            span_id="abc",
            trace_id="0" * 32,
            parent_span_id="parent1",
            name="task",
            span_type=SpanType.TASK,
            status="completed",
            start_time=now,
            end_time=now,
            duration_ms=100,
            cost=0.01,
            tokens_input=50,
            tokens_output=25,
            llm_model="gpt-4",
            input_document_sha256s=["sha1"],
            output_document_sha256s=["sha2"],
        )
        assert mock_writer.write.called
        row = mock_writer.write.call_args[0][1][0]
        assert row.cost == 0.01
        assert row.tokens_input == 50
        assert row.input_document_sha256s == ("sha1",)
        assert row.output_document_sha256s == ("sha2",)


class TestSpanEventsTracking:
    def test_tracks_span_events(self):
        from datetime import datetime

        service, mock_writer = _make_service()
        run_id = uuid4()
        service.set_run_context(execution_id=run_id, run_id="proj", flow_name="flow")
        now = datetime.now(UTC)
        events = [("event1", now, {"key": "val"}, "INFO")]
        service.track_span_events(span_id="abc", events=events)
        assert mock_writer.write.called
        args = mock_writer.write.call_args
        assert args[0][0] == "span_events"

    def test_no_events_tracking_without_context(self):
        from datetime import datetime

        service, mock_writer = _make_service()
        now = datetime.now(UTC)
        service.track_span_events(span_id="abc", events=[("e", now, {}, None)])
        mock_writer.write.assert_not_called()

    def test_empty_events_noop(self):
        service, mock_writer = _make_service()
        run_id = uuid4()
        service.set_run_context(execution_id=run_id, run_id="proj", flow_name="flow")
        service.track_span_events(span_id="abc", events=[])
        mock_writer.write.assert_not_called()


class TestDocumentEventTracking:
    def test_tracks_document_event(self):
        from ai_pipeline_core.observability._tracking._models import DocumentEventType

        service, mock_writer = _make_service()
        run_id = uuid4()
        service.set_run_context(execution_id=run_id, run_id="proj", flow_name="flow")
        service.track_document_event(
            document_sha256="sha1",
            span_id="span1",
            event_type=DocumentEventType.STORE_SAVED,
        )
        assert mock_writer.write.called
        args = mock_writer.write.call_args
        assert args[0][0] == "document_events"

    def test_no_document_event_without_context(self):
        from ai_pipeline_core.observability._tracking._models import DocumentEventType

        service, mock_writer = _make_service()
        service.track_document_event(
            document_sha256="sha1",
            span_id="span1",
            event_type=DocumentEventType.STORE_SAVED,
        )
        mock_writer.write.assert_not_called()


class TestVersionManagement:
    def test_versions_monotonically_increase(self):
        service, _ = _make_service()
        v1 = service._next_version()
        v2 = service._next_version()
        v3 = service._next_version()
        assert v1 < v2 < v3


class TestRunEndExtended:
    def test_track_run_end_with_metadata(self):
        service, mock_writer = _make_service()
        run_id = uuid4()
        service.set_run_context(execution_id=run_id, run_id="proj", flow_name="flow")
        service.track_run_start(execution_id=run_id, run_id="proj", flow_name="flow")
        service.track_run_end(
            execution_id=run_id,
            status=RunStatus.COMPLETED,
            total_cost=1.5,
            total_tokens=1000,
            metadata={"key": "value"},
        )
        end_call = mock_writer.write.call_args_list[-1]
        row = end_call[0][1][0]
        assert row.total_cost == 1.5
        assert row.total_tokens == 1000
        assert '"key"' in row.metadata

    def test_track_run_end_without_start(self):
        service, mock_writer = _make_service()
        run_id = uuid4()
        service.set_run_context(execution_id=run_id, run_id="proj", flow_name="flow")
        service.track_run_end(execution_id=run_id, status=RunStatus.FAILED)
        end_call = mock_writer.write.call_args
        row = end_call[0][1][0]
        assert row.status == RunStatus.FAILED


class TestFlushAndShutdown:
    def test_flush_delegates(self):
        service, mock_writer = _make_service()
        run_id = uuid4()
        service.set_run_context(execution_id=run_id, run_id="proj", flow_name="flow")
        service.flush()
        mock_writer.flush.assert_called_once()
        assert service._execution_id is None

    def test_shutdown_delegates(self):
        service, mock_writer = _make_service()
        run_id = uuid4()
        service.set_run_context(execution_id=run_id, run_id="proj", flow_name="flow")
        service.shutdown()
        mock_writer.shutdown.assert_called_once()
        assert service._execution_id is None
