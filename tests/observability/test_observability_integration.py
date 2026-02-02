"""Tests for observability integration changes: store events, run_scope, protocol unification, content deprecation."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

clickhouse_connect = pytest.importorskip("clickhouse_connect")

from datetime import UTC

from ai_pipeline_core.observability._tracking._models import (
    DocumentEventType,
    PipelineRunRow,
    RunStatus,
)
from ai_pipeline_core.observability._tracking._service import TrackingService


def _make_service() -> tuple[TrackingService, MagicMock]:
    """Create a TrackingService with mocked ClickHouseClient and writer."""
    mock_client = MagicMock()
    with patch("ai_pipeline_core.observability._tracking._service.ClickHouseWriter") as mock_writer_cls:
        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer
        service = TrackingService(mock_client)
    return service, mock_writer


class TestNewEventTypes:
    """Test STORE_SAVED and STORE_SAVE_FAILED enum values exist."""

    def test_store_saved_exists(self):
        assert DocumentEventType.STORE_SAVED == "store_saved"

    def test_store_save_failed_exists(self):
        assert DocumentEventType.STORE_SAVE_FAILED == "store_save_failed"

    def test_store_events_are_subset_of_all_events(self):
        all_values = {e.value for e in DocumentEventType}
        assert "store_saved" in all_values
        assert "store_save_failed" in all_values


class TestRunScopeOnModels:
    """Test run_scope field on PipelineRunRow."""

    def test_pipeline_run_row_default_run_scope(self):
        from datetime import datetime

        row = PipelineRunRow(
            run_id=uuid4(),
            project_name="proj",
            flow_name="flow",
            status=RunStatus.RUNNING,
            start_time=datetime.now(UTC),
        )
        assert row.run_scope == ""

    def test_pipeline_run_row_with_run_scope(self):
        from datetime import datetime

        row = PipelineRunRow(
            run_id=uuid4(),
            project_name="proj",
            flow_name="flow",
            run_scope="my-project",
            status=RunStatus.RUNNING,
            start_time=datetime.now(UTC),
        )
        assert row.run_scope == "my-project"


class TestRunScopePropagation:
    """Test run_scope is stored and propagated through TrackingService."""

    def test_set_run_context_stores_run_scope(self):
        service, _ = _make_service()
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="proj", flow_name="flow", run_scope="my-scope")
        assert service._run_scope == "my-scope"

    def test_set_run_context_default_run_scope(self):
        service, _ = _make_service()
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="proj", flow_name="flow")
        assert service._run_scope == ""

    def test_clear_run_context_resets_run_scope(self):
        service, _ = _make_service()
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="proj", flow_name="flow", run_scope="my-scope")
        service.clear_run_context()
        assert service._run_scope == ""

    def test_track_run_start_includes_run_scope(self):
        service, mock_writer = _make_service()
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="proj", flow_name="flow", run_scope="my-scope")
        service.track_run_start(run_id=run_id, project_name="proj", flow_name="flow", run_scope="my-scope")
        row = mock_writer.write.call_args[0][1][0]
        assert row.run_scope == "my-scope"

    def test_track_run_start_defaults_to_empty_run_scope_when_not_passed(self):
        service, mock_writer = _make_service()
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="proj", flow_name="flow", run_scope="stored-scope")
        service.track_run_start(run_id=run_id, project_name="proj", flow_name="flow")
        row = mock_writer.write.call_args[0][1][0]
        assert row.run_scope == ""

    def test_track_run_end_includes_run_scope(self):
        service, mock_writer = _make_service()
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="proj", flow_name="flow", run_scope="end-scope")
        service.track_run_start(run_id=run_id, project_name="proj", flow_name="flow", run_scope="end-scope")
        service.track_run_end(run_id=run_id, status=RunStatus.COMPLETED)
        end_call = mock_writer.write.call_args_list[-1]
        row = end_call[0][1][0]
        assert row.run_scope == "end-scope"


class TestStoreEventEmission:
    """Test _emit_store_events helper function."""

    def test_emit_store_saved_events(self):
        from ai_pipeline_core.pipeline.decorators import _emit_store_events

        mock_doc1 = MagicMock()
        mock_doc1.sha256 = "hash1"
        mock_doc2 = MagicMock()
        mock_doc2.sha256 = "hash2"

        mock_service = MagicMock()
        with (
            patch("ai_pipeline_core.pipeline.decorators.get_tracking_service", return_value=mock_service),
            patch("ai_pipeline_core.pipeline.decorators.get_current_span_id", return_value="span123"),
        ):
            _emit_store_events([mock_doc1, mock_doc2], DocumentEventType.STORE_SAVED)

        assert mock_service.track_document_event.call_count == 2
        first_call = mock_service.track_document_event.call_args_list[0][1]
        assert first_call["document_sha256"] == "hash1"
        assert first_call["event_type"] == DocumentEventType.STORE_SAVED
        assert first_call["span_id"] == "span123"

    def test_emit_store_events_noop_without_service(self):
        from ai_pipeline_core.pipeline.decorators import _emit_store_events

        mock_doc = MagicMock()
        mock_doc.sha256 = "hash1"

        with patch("ai_pipeline_core.pipeline.decorators.get_tracking_service", return_value=None):
            # Should not raise
            _emit_store_events([mock_doc], DocumentEventType.STORE_SAVED)

    def test_emit_store_events_swallows_exceptions(self):
        from ai_pipeline_core.pipeline.decorators import _emit_store_events

        mock_doc = MagicMock()
        mock_doc.sha256 = "hash1"

        with patch("ai_pipeline_core.pipeline.decorators.get_tracking_service", side_effect=RuntimeError("boom")):
            # Should not raise
            _emit_store_events([mock_doc], DocumentEventType.STORE_SAVED)


class TestTrackDocumentEvent:
    """Test track_document_event with new event types."""

    def test_track_store_saved_event(self):
        service, mock_writer = _make_service()
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="proj", flow_name="flow")
        service.track_document_event(
            document_sha256="hash1",
            span_id="span1",
            event_type=DocumentEventType.STORE_SAVED,
        )
        event_call = mock_writer.write.call_args
        assert event_call[0][0] == "document_events"
        row = event_call[0][1][0]
        assert row.event_type == DocumentEventType.STORE_SAVED

    def test_track_store_save_failed_event(self):
        service, mock_writer = _make_service()
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="proj", flow_name="flow")
        service.track_document_event(
            document_sha256="hash1",
            span_id="span1",
            event_type=DocumentEventType.STORE_SAVE_FAILED,
        )
        event_call = mock_writer.write.call_args
        row = event_call[0][1][0]
        assert row.event_type == DocumentEventType.STORE_SAVE_FAILED


class TestProtocolUnification:
    """Test that TrackingService satisfies the unified TrackingServiceProtocol."""

    def test_service_satisfies_protocol(self):

        service, _ = _make_service()
        # Structural type check: verify all protocol methods exist
        assert hasattr(service, "set_run_context")
        assert hasattr(service, "track_run_start")
        assert hasattr(service, "track_run_end")
        assert hasattr(service, "clear_run_context")
        assert hasattr(service, "track_document_event")
        assert hasattr(service, "schedule_summary")

    def test_get_tracking_service_returns_protocol_compatible(self):
        """Verify _get_tracking_service returns without type: ignore."""
        from ai_pipeline_core.observability._document_tracking import _get_tracking_service

        # When no service is initialized, returns None
        result = _get_tracking_service()
        assert result is None


class TestClickHouseClientDDL:
    """Test DDL in ClickHouseClient."""

    def test_ensure_tables_idempotent(self):
        from ai_pipeline_core.observability._tracking._client import ClickHouseClient

        client = ClickHouseClient(host="localhost")
        mock_ch_client = MagicMock()
        client._client = mock_ch_client

        client.ensure_tables()
        call_count_first = mock_ch_client.command.call_count

        # Second call should be a no-op
        client.ensure_tables()
        assert mock_ch_client.command.call_count == call_count_first

    def test_create_table_ddl_includes_run_scope(self):
        from ai_pipeline_core.observability._tracking._client import _CREATE_TABLES_SQL

        pipeline_runs_ddl = _CREATE_TABLES_SQL[0]
        assert "run_scope" in pipeline_runs_ddl
