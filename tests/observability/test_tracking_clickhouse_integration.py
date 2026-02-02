"""Integration tests for observability tracking with real ClickHouse.

Exercises TrackingService → ClickHouseWriter → ClickHouseClient → ClickHouse
with a real testcontainers-managed ClickHouse instance. No mocking.

Run with: pytest -m clickhouse tests/observability/test_tracking_clickhouse_integration.py
"""

from datetime import UTC, datetime
from typing import Any
from collections.abc import Generator
from uuid import uuid4

import pytest

clickhouse_connect = pytest.importorskip("clickhouse_connect")
testcontainers_clickhouse = pytest.importorskip("testcontainers.clickhouse")

from testcontainers.clickhouse import ClickHouseContainer

from ai_pipeline_core.observability._tracking._client import ClickHouseClient
from ai_pipeline_core.observability._tracking._models import (
    TABLE_DOCUMENT_EVENTS,
    TABLE_PIPELINE_RUNS,
    TABLE_SPAN_EVENTS,
    TABLE_TRACKED_SPANS,
    DocumentEventType,
    RunStatus,
    SpanType,
)
from ai_pipeline_core.observability._tracking._service import TrackingService

pytestmark = pytest.mark.clickhouse


@pytest.fixture(scope="module")
def clickhouse_container():
    """Start a ClickHouse container for the test module."""
    with ClickHouseContainer() as container:
        yield container


def _make_client(container: ClickHouseContainer) -> ClickHouseClient:
    """Create a ClickHouseClient from container."""
    return ClickHouseClient(
        host=container.get_container_host_ip(),
        port=int(container.get_exposed_port(8123)),
        database=container.dbname,
        username=container.username,
        password=container.password,
        secure=False,
    )


@pytest.fixture
def client(clickhouse_container: ClickHouseContainer) -> ClickHouseClient:
    """Create and connect a ClickHouseClient."""
    c = _make_client(clickhouse_container)
    c.connect()
    c.ensure_tables()
    return c


@pytest.fixture
def service(clickhouse_container: ClickHouseContainer) -> Generator[TrackingService, None, None]:
    """Create a TrackingService backed by real ClickHouse."""
    c = _make_client(clickhouse_container)
    svc = TrackingService(c, span_summary_fn=None)
    yield svc
    svc.shutdown(timeout=5.0)


def _query(client: ClickHouseClient, sql: str) -> list[tuple[Any, ...]]:
    """Run a query and return rows."""
    assert client._client is not None
    return client._client.query(sql).result_rows


def _wait_for_writer(service: TrackingService, timeout: float = 5.0) -> None:
    """Shutdown the writer to flush all pending batches."""
    service._writer.shutdown(timeout=timeout)


class TestTableCreation:
    """Verify tables are created in a real ClickHouse instance."""

    def test_all_tables_exist(self, client):
        rows = _query(client, "SHOW TABLES")
        table_names = {r[0] for r in rows}
        assert TABLE_PIPELINE_RUNS in table_names
        assert TABLE_TRACKED_SPANS in table_names
        assert TABLE_DOCUMENT_EVENTS in table_names
        assert TABLE_SPAN_EVENTS in table_names

    def test_pipeline_runs_has_run_scope_column(self, client):
        rows = _query(client, f"DESCRIBE TABLE {TABLE_PIPELINE_RUNS}")
        column_names = {r[0] for r in rows}
        assert "run_scope" in column_names

    def test_ensure_tables_is_idempotent(self, client):
        # Second call should not raise
        client.ensure_tables()


class TestRunTracking:
    """Test pipeline run lifecycle persisted to ClickHouse."""

    def test_run_start_persisted(self, service, client):
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="test-project", flow_name="test-flow", run_scope="scope-a")
        service.track_run_start(run_id=run_id, project_name="test-project", flow_name="test-flow", run_scope="scope-a")
        _wait_for_writer(service)

        rows = _query(client, f"SELECT run_id, project_name, flow_name, status, run_scope FROM {TABLE_PIPELINE_RUNS} FINAL WHERE run_id = '{run_id}'")
        assert len(rows) == 1
        assert rows[0][1] == "test-project"
        assert rows[0][2] == "test-flow"
        assert rows[0][3] == "running"
        assert rows[0][4] == "scope-a"

    def test_run_end_updates_status(self, service, client):
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="p", flow_name="f")
        service.track_run_start(run_id=run_id, project_name="p", flow_name="f")
        service.track_run_end(run_id=run_id, status=RunStatus.COMPLETED, total_cost=1.23, total_tokens=5000)
        _wait_for_writer(service)

        rows = _query(client, f"SELECT status, total_cost, total_tokens, end_time FROM {TABLE_PIPELINE_RUNS} FINAL WHERE run_id = '{run_id}'")
        assert len(rows) == 1
        assert rows[0][0] == "completed"
        assert abs(rows[0][1] - 1.23) < 0.01
        assert rows[0][2] == 5000
        assert rows[0][3] is not None  # end_time populated

    def test_run_failed_status(self, service, client):
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="p", flow_name="f")
        service.track_run_start(run_id=run_id, project_name="p", flow_name="f")
        service.track_run_end(run_id=run_id, status=RunStatus.FAILED)
        _wait_for_writer(service)

        rows = _query(client, f"SELECT status FROM {TABLE_PIPELINE_RUNS} FINAL WHERE run_id = '{run_id}'")
        assert rows[0][0] == "failed"

    def test_run_metadata_stored_as_json(self, service, client):
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="p", flow_name="f")
        service.track_run_start(run_id=run_id, project_name="p", flow_name="f")
        service.track_run_end(run_id=run_id, status=RunStatus.COMPLETED, metadata={"key": "value", "count": 42})
        _wait_for_writer(service)

        rows = _query(client, f"SELECT metadata FROM {TABLE_PIPELINE_RUNS} FINAL WHERE run_id = '{run_id}'")
        import json

        meta = json.loads(rows[0][0])
        assert meta["key"] == "value"
        assert meta["count"] == 42


class TestSpanTracking:
    """Test span lifecycle persisted to ClickHouse."""

    def test_span_start_and_end(self, service, client):
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="p", flow_name="f")
        service.track_run_start(run_id=run_id, project_name="p", flow_name="f")

        now = datetime.now(UTC)
        service.track_span_start(
            span_id="span001",
            trace_id="trace001",
            parent_span_id=None,
            name="root_task",
            span_type=SpanType.TASK,
        )
        service.track_span_end(
            span_id="span001",
            trace_id="trace001",
            parent_span_id=None,
            name="root_task",
            span_type=SpanType.TASK,
            status="completed",
            start_time=now,
            end_time=now,
            duration_ms=150,
            cost=0.005,
            tokens_input=1000,
            tokens_output=200,
            llm_model="gpt-5.1",
        )
        _wait_for_writer(service)

        rows = _query(
            client,
            f"SELECT name, span_type, status, duration_ms, cost, tokens_input, tokens_output, llm_model"
            f" FROM {TABLE_TRACKED_SPANS} FINAL WHERE span_id = 'span001'",
        )
        assert len(rows) == 1
        assert rows[0][0] == "root_task"
        assert rows[0][1] == "task"
        assert rows[0][2] == "completed"
        assert rows[0][3] == 150
        assert abs(rows[0][4] - 0.005) < 0.0001
        assert rows[0][5] == 1000
        assert rows[0][6] == 200
        assert rows[0][7] == "gpt-5.1"

    def test_parent_child_span_relationship(self, service, client):
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="p", flow_name="f")
        service.track_run_start(run_id=run_id, project_name="p", flow_name="f")

        now = datetime.now(UTC)
        # Parent span
        service.track_span_start(span_id="parent01", trace_id="t1", parent_span_id=None, name="parent", span_type=SpanType.FLOW)
        # Child span
        service.track_span_start(span_id="child01", trace_id="t1", parent_span_id="parent01", name="child", span_type=SpanType.TASK)
        service.track_span_end(
            span_id="child01",
            trace_id="t1",
            parent_span_id="parent01",
            name="child",
            span_type=SpanType.TASK,
            status="completed",
            start_time=now,
            end_time=now,
            duration_ms=50,
        )
        service.track_span_end(
            span_id="parent01",
            trace_id="t1",
            parent_span_id=None,
            name="parent",
            span_type=SpanType.FLOW,
            status="completed",
            start_time=now,
            end_time=now,
            duration_ms=100,
        )
        _wait_for_writer(service)

        rows = _query(client, f"SELECT span_id, parent_span_id FROM {TABLE_TRACKED_SPANS} FINAL WHERE trace_id = 't1' ORDER BY span_id")
        assert len(rows) == 2
        child_row = next(r for r in rows if r[0] == "child01")
        assert child_row[1] == "parent01"

    def test_user_summary_on_span(self, service, client):
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="p", flow_name="f")
        service.track_run_start(run_id=run_id, project_name="p", flow_name="f")

        now = datetime.now(UTC)
        service.track_span_end(
            span_id="summary01",
            trace_id="ts1",
            parent_span_id=None,
            name="summarized_task",
            span_type=SpanType.TASK,
            status="completed",
            start_time=now,
            end_time=now,
            duration_ms=100,
            user_summary="Processed 42 documents successfully.",
            user_visible=True,
            user_label="document_processor",
        )
        _wait_for_writer(service)

        rows = _query(client, f"SELECT user_summary, user_visible, user_label FROM {TABLE_TRACKED_SPANS} FINAL WHERE span_id = 'summary01'")
        assert len(rows) == 1
        assert rows[0][0] == "Processed 42 documents successfully."
        assert rows[0][1] is True
        assert rows[0][2] == "document_processor"

    def test_document_sha256_arrays_on_span(self, service, client):
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="p", flow_name="f")
        service.track_run_start(run_id=run_id, project_name="p", flow_name="f")

        now = datetime.now(UTC)
        service.track_span_end(
            span_id="docspan01",
            trace_id="td1",
            parent_span_id=None,
            name="doc_task",
            span_type=SpanType.TASK,
            status="completed",
            start_time=now,
            end_time=now,
            duration_ms=50,
            input_document_sha256s=["sha_in_1", "sha_in_2"],
            output_document_sha256s=["sha_out_1"],
        )
        _wait_for_writer(service)

        rows = _query(client, f"SELECT input_document_sha256s, output_document_sha256s FROM {TABLE_TRACKED_SPANS} FINAL WHERE span_id = 'docspan01'")
        assert len(rows) == 1
        assert sorted(rows[0][0]) == ["sha_in_1", "sha_in_2"]
        assert rows[0][1] == ["sha_out_1"]


class TestDocumentEvents:
    """Test document event lifecycle in ClickHouse."""

    def test_multiple_document_events(self, service, client):
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="p", flow_name="f")

        service.track_document_event(document_sha256="multi01", span_id="s1", event_type=DocumentEventType.TASK_INPUT)
        service.track_document_event(document_sha256="multi01", span_id="s2", event_type=DocumentEventType.TASK_OUTPUT)
        service.track_document_event(document_sha256="multi01", span_id="s3", event_type=DocumentEventType.STORE_SAVED)
        _wait_for_writer(service)

        rows = _query(client, f"SELECT event_type FROM {TABLE_DOCUMENT_EVENTS} WHERE document_sha256 = 'multi01' AND run_id = '{run_id}' ORDER BY timestamp")
        event_types = [r[0] for r in rows]
        assert "task_input" in event_types
        assert "task_output" in event_types
        assert "store_saved" in event_types


class TestSpanEvents:
    """Test span events (logging bridge, etc.) persisted to ClickHouse."""

    def test_span_events_persisted(self, service, client):
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="p", flow_name="f")
        service.track_run_start(run_id=run_id, project_name="p", flow_name="f")

        now = datetime.now(UTC)
        events = [
            ("log", now, {"log.message": "Processing started"}, "INFO"),
            ("log", now, {"log.message": "Processing complete"}, "INFO"),
            ("error", now, {"log.message": "Something went wrong"}, "ERROR"),
        ]
        service.track_span_events(span_id="se01", events=events)
        _wait_for_writer(service)

        rows = _query(client, f"SELECT name, level FROM {TABLE_SPAN_EVENTS} WHERE span_id = 'se01' AND run_id = '{run_id}' ORDER BY timestamp")
        assert len(rows) == 3
        assert rows[0][0] == "log"
        assert rows[0][1] == "INFO"
        assert rows[2][0] == "error"
        assert rows[2][1] == "ERROR"


class TestReplacingMergeTree:
    """Test ReplacingMergeTree deduplication for runs and spans."""

    def test_run_version_dedup(self, service, client):
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="p", flow_name="f")
        service.track_run_start(run_id=run_id, project_name="p", flow_name="f")
        service.track_run_end(run_id=run_id, status=RunStatus.COMPLETED)
        _wait_for_writer(service)

        # With FINAL, should see only the latest version
        rows = _query(client, f"SELECT status, version FROM {TABLE_PIPELINE_RUNS} FINAL WHERE run_id = '{run_id}'")
        assert len(rows) == 1
        assert rows[0][0] == "completed"
        assert rows[0][1] == 2  # version 1 = start, version 2 = end

    def test_span_version_dedup(self, service, client):
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="p", flow_name="f")
        service.track_run_start(run_id=run_id, project_name="p", flow_name="f")

        now = datetime.now(UTC)
        service.track_span_start(span_id="vspan01", trace_id="vt1", parent_span_id=None, name="task", span_type=SpanType.TASK)
        service.track_span_end(
            span_id="vspan01",
            trace_id="vt1",
            parent_span_id=None,
            name="task",
            span_type=SpanType.TASK,
            status="completed",
            start_time=now,
            end_time=now,
            duration_ms=100,
        )
        _wait_for_writer(service)

        rows = _query(client, f"SELECT status, version FROM {TABLE_TRACKED_SPANS} FINAL WHERE span_id = 'vspan01' AND run_id = '{run_id}'")
        assert len(rows) == 1
        assert rows[0][0] == "completed"
        assert rows[0][1] == 2


class TestSummaryUpdate:
    """Test span summary update via build_span_summary_update."""

    def test_summary_update_persisted(self, service, client):
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="p", flow_name="f")
        service.track_run_start(run_id=run_id, project_name="p", flow_name="f")

        now = datetime.now(UTC)
        service.track_span_end(
            span_id="sumspan01",
            trace_id="st1",
            parent_span_id=None,
            name="summarizable_task",
            span_type=SpanType.TASK,
            status="completed",
            start_time=now,
            end_time=now,
            duration_ms=200,
        )
        _wait_for_writer(service)

        # Simulate what the writer would do for a summary job
        updated_row = service.build_span_summary_update("sumspan01", "Task completed: processed 100 items.")
        assert updated_row is not None
        assert updated_row.user_summary == "Task completed: processed 100 items."

        # Insert the updated row directly
        client.update_span(updated_row)

        rows = _query(client, f"SELECT user_summary, version FROM {TABLE_TRACKED_SPANS} FINAL WHERE span_id = 'sumspan01' AND run_id = '{run_id}'")
        assert len(rows) == 1
        assert rows[0][0] == "Task completed: processed 100 items."
        assert rows[0][1] > 1  # summary version > original span end version


class TestNoRunContextGuards:
    """Test that tracking calls are no-ops without run context."""

    def test_span_start_noop_without_context(self, service, client):
        # No set_run_context called
        service.track_span_start(span_id="norun01", trace_id="nr1", parent_span_id=None, name="orphan", span_type=SpanType.TASK)
        _wait_for_writer(service)

        rows = _query(client, f"SELECT count() FROM {TABLE_TRACKED_SPANS} WHERE span_id = 'norun01'")
        assert rows[0][0] == 0


class TestConcurrentSpans:
    """Test concurrent span tracking integrity."""

    def test_multiple_spans_same_run(self, service, client):
        run_id = uuid4()
        service.set_run_context(run_id=run_id, project_name="p", flow_name="f")
        service.track_run_start(run_id=run_id, project_name="p", flow_name="f")

        now = datetime.now(UTC)
        span_count = 20
        for i in range(span_count):
            sid = f"concurrent_{i:03d}"
            service.track_span_start(span_id=sid, trace_id="ct1", parent_span_id=None, name=f"task_{i}", span_type=SpanType.TASK)
            service.track_span_end(
                span_id=sid,
                trace_id="ct1",
                parent_span_id=None,
                name=f"task_{i}",
                span_type=SpanType.TASK,
                status="completed",
                start_time=now,
                end_time=now,
                duration_ms=10,
            )
        _wait_for_writer(service)

        rows = _query(client, f"SELECT count() FROM {TABLE_TRACKED_SPANS} FINAL WHERE run_id = '{run_id}' AND trace_id = 'ct1'")
        assert rows[0][0] == span_count
