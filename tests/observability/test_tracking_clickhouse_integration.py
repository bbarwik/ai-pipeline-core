"""Integration tests for observability tracking with real ClickHouse.

Exercises ClickHouseBackend -> ClickHouseWriter -> ClickHouse
with a real testcontainers-managed ClickHouse instance. No mocking.

Run with: pytest -m clickhouse tests/observability/test_tracking_clickhouse_integration.py
"""

from collections.abc import Generator
from typing import Any
from uuid import uuid4

import pytest

clickhouse_connect = pytest.importorskip("clickhouse_connect")
testcontainers_clickhouse = pytest.importorskip("testcontainers.clickhouse")

from testcontainers.clickhouse import ClickHouseContainer

from ai_pipeline_core.observability._tracking._backend import ClickHouseBackend
from ai_pipeline_core.observability._tracking._models import (
    TABLE_PIPELINE_RUNS,
    TABLE_PIPELINE_SPANS,
    RunStatus,
)
from ai_pipeline_core.observability._tracking._writer import ClickHouseWriter

pytestmark = pytest.mark.clickhouse


@pytest.fixture(scope="module")
def clickhouse_container():
    """Start a ClickHouse container for the test module."""
    with ClickHouseContainer() as container:
        yield container


def _make_writer(container: ClickHouseContainer) -> ClickHouseWriter:
    """Create a ClickHouseWriter from container."""
    return ClickHouseWriter(
        host=container.get_container_host_ip(),
        port=int(container.get_exposed_port(8123)),
        database=container.dbname,
        username=container.username,
        password=container.password,
        secure=False,
    )


@pytest.fixture
def backend(clickhouse_container: ClickHouseContainer) -> Generator[ClickHouseBackend, None, None]:
    """Create a ClickHouseBackend backed by real ClickHouse."""
    writer = _make_writer(clickhouse_container)
    writer.start()
    b = ClickHouseBackend(writer)
    yield b
    writer.shutdown(timeout=5.0)


def _query(clickhouse_container: ClickHouseContainer, sql: str) -> list[tuple[Any, ...]]:
    """Run a query directly against ClickHouse."""
    client = clickhouse_connect.get_client(
        host=clickhouse_container.get_container_host_ip(),
        port=int(clickhouse_container.get_exposed_port(8123)),
        database=clickhouse_container.dbname,
        username=clickhouse_container.username,
        password=clickhouse_container.password,
    )
    return client.query(sql).result_rows


class TestTableCreation:
    """Verify tables are created in a real ClickHouse instance."""

    def test_both_tables_exist(self, backend, clickhouse_container):
        backend.flush(timeout=5.0)
        rows = _query(clickhouse_container, "SHOW TABLES")
        table_names = {r[0] for r in rows}
        assert TABLE_PIPELINE_RUNS in table_names
        assert TABLE_PIPELINE_SPANS in table_names


class TestRunTracking:
    """Test pipeline run lifecycle persisted to ClickHouse."""

    def test_run_start_persisted(self, backend, clickhouse_container):
        run_uuid = uuid4()
        backend.track_run_start(
            execution_id=run_uuid,
            run_id="test-project",
            flow_name="test-flow",
            run_scope="scope-a",
        )
        backend.flush(timeout=5.0)

        rows = _query(
            clickhouse_container,
            f"SELECT execution_id, run_id, flow_name, status, run_scope FROM {TABLE_PIPELINE_RUNS} FINAL WHERE execution_id = '{run_uuid}'",
        )
        assert len(rows) == 1
        assert rows[0][1] == "test-project"
        assert rows[0][2] == "test-flow"
        assert rows[0][3] == "running"
        assert rows[0][4] == "scope-a"

    def test_run_end_updates_status(self, backend, clickhouse_container):
        run_uuid = uuid4()
        start_time = backend.track_run_start(execution_id=run_uuid, run_id="p", flow_name="f")
        backend.track_run_end(
            execution_id=run_uuid,
            run_id="p",
            flow_name="f",
            status=RunStatus.COMPLETED,
            start_time=start_time,
        )
        backend.flush(timeout=5.0)

        rows = _query(
            clickhouse_container,
            f"SELECT status, end_time FROM {TABLE_PIPELINE_RUNS} FINAL WHERE execution_id = '{run_uuid}'",
        )
        assert len(rows) == 1
        assert rows[0][0] == "completed"
        assert rows[0][1] is not None


class TestParentChildLineage:
    """Test parent-child run lineage columns in pipeline_runs."""

    def test_parent_execution_id_persisted(self, backend, clickhouse_container):
        parent_uid = uuid4()
        child_uid = uuid4()
        backend.track_run_start(
            execution_id=child_uid,
            run_id="child-run",
            flow_name="child-flow",
            parent_execution_id=parent_uid,
            parent_span_id="abc123def456",
        )
        backend.flush(timeout=5.0)

        rows = _query(
            clickhouse_container,
            f"SELECT parent_execution_id, parent_span_id FROM {TABLE_PIPELINE_RUNS} FINAL WHERE execution_id = '{child_uid}'",
        )
        assert len(rows) == 1
        assert rows[0][0] == parent_uid
        assert rows[0][1] == "abc123def456"

    def test_parent_execution_id_null_by_default(self, backend, clickhouse_container):
        run_uid = uuid4()
        backend.track_run_start(execution_id=run_uid, run_id="standalone", flow_name="f")
        backend.flush(timeout=5.0)

        rows = _query(
            clickhouse_container,
            f"SELECT parent_execution_id, parent_span_id FROM {TABLE_PIPELINE_RUNS} FINAL WHERE execution_id = '{run_uid}'",
        )
        assert len(rows) == 1
        assert rows[0][0] is None
        assert rows[0][1] is None

    def test_query_children_by_parent_execution_id(self, backend, clickhouse_container):
        parent_uid = uuid4()
        child1_uid = uuid4()
        child2_uid = uuid4()
        backend.track_run_start(execution_id=parent_uid, run_id="parent", flow_name="p")
        backend.track_run_start(
            execution_id=child1_uid,
            run_id="child-1",
            flow_name="c",
            parent_execution_id=parent_uid,
        )
        backend.track_run_start(
            execution_id=child2_uid,
            run_id="child-2",
            flow_name="c",
            parent_execution_id=parent_uid,
        )
        backend.flush(timeout=5.0)

        rows = _query(
            clickhouse_container,
            f"SELECT execution_id FROM {TABLE_PIPELINE_RUNS} FINAL WHERE parent_execution_id = '{parent_uid}'",
        )
        child_ids = {r[0] for r in rows}
        assert child_ids == {child1_uid, child2_uid}
