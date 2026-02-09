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
        service.set_run_context(run_id=run_id, project_name="proj", flow_name="flow")
        assert service._run_id == run_id
        assert service._project_name == "proj"
        service.clear_run_context()
        assert service._run_id is None
        assert service._project_name == ""

    def test_clear_clears_caches(self):
        service, _ = _make_service()
        service.clear_run_context()
        assert service._project_name == ""


class TestRunTracking:
    """Test pipeline run tracking."""

    def test_track_run_start(self):
        service, mock_writer = _make_service()
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="proj", flow_name="flow")
        service.track_run_start(run_id=run_id, project_name="proj", flow_name="flow")
        assert mock_writer.write.called
        args = mock_writer.write.call_args
        assert args[0][0] == "pipeline_runs"
        assert args[0][1][0].status == RunStatus.RUNNING

    def test_track_run_end_uses_stored_start_time(self):
        service, mock_writer = _make_service()
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="proj", flow_name="flow")
        service.track_run_start(run_id=run_id, project_name="proj", flow_name="flow")
        start_time = service._run_start_time
        service.track_run_end(run_id=run_id, status=RunStatus.COMPLETED)
        end_call = mock_writer.write.call_args_list[-1]
        row = end_call[0][1][0]
        assert row.start_time == start_time
        assert row.end_time is not None


class TestSpanTracking:
    """Test span tracking."""

    def test_track_span_end_caches_row(self):
        service, mock_writer = _make_service()
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="proj", flow_name="flow")
        from datetime import datetime

        now = datetime.now(UTC)
        service.track_span_end(
            span_id="abc123",
            trace_id="0" * 32,
            parent_span_id=None,
            name="my_task",
            span_type=SpanType.TASK,
            status="completed",
            start_time=now,
            end_time=now,
            duration_ms=100,
        )
        assert "abc123" in service._span_cache

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


class TestSpanSummaryUpdate:
    """Test build_span_summary_update method."""

    def test_build_span_summary_update(self):
        service, _ = _make_service()
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="proj", flow_name="flow")
        from datetime import datetime

        now = datetime.now(UTC)
        service.track_span_end(
            span_id="span1",
            trace_id="0" * 32,
            parent_span_id=None,
            name="task",
            span_type=SpanType.TASK,
            status="completed",
            start_time=now,
            end_time=now,
            duration_ms=50,
        )
        updated = service.build_span_summary_update("span1", "Did analysis")
        assert updated is not None
        assert updated.user_summary == "Did analysis"
        assert updated.version > 1

    def test_build_span_summary_update_missing(self):
        service, _ = _make_service()
        result = service.build_span_summary_update("nonexistent", "summary")
        assert result is None

    def test_no_build_doc_summary_update(self):
        """TrackingService does not have a build_doc_summary_update method."""
        service, _ = _make_service()
        assert not hasattr(service, "build_doc_summary_update")


class TestDocSummaryConfig:
    """TrackingService does not expose document summary configuration."""

    def test_no_doc_summary_properties(self):
        service, _ = _make_service()
        assert not hasattr(service, "doc_summary_enabled")
        assert not hasattr(service, "doc_summary_min_chars")
