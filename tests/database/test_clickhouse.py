"""Tests for ClickHouseDatabase backend.

Auto-skips when ClickHouse is not available (no CLICKHOUSE_HOST env var).
"""

import os
from types import SimpleNamespace
from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from clickhouse_connect.driver.summary import QuerySummary

from ai_pipeline_core.database import BlobRecord, DocumentRecord, ExecutionLog, ExecutionNode, NodeKind, NodeStatus

pytestmark = pytest.mark.clickhouse

CLICKHOUSE_NOT_AVAILABLE = pytest.mark.skipif(
    not os.environ.get("CLICKHOUSE_HOST"),
    reason="ClickHouse not available (CLICKHOUSE_HOST not set)",
)


def _get_db():
    """Create a ClickHouseDatabase connected to the test instance."""
    from ai_pipeline_core.database._clickhouse import ClickHouseDatabase
    from ai_pipeline_core.settings import Settings

    return ClickHouseDatabase(Settings())


def _make_node(**kwargs: object) -> ExecutionNode:
    did = kwargs.pop("_deployment_id", None) or uuid4()
    rid = kwargs.pop("_root_deployment_id", None) or did
    defaults: dict[str, object] = {
        "node_id": uuid4(),
        "node_kind": NodeKind.TASK,
        "deployment_id": did,
        "root_deployment_id": rid,
        "run_id": f"test-run-{uuid4().hex[:8]}",
        "run_scope": f"test-run/scope-{uuid4().hex[:8]}",
        "deployment_name": "test-pipeline",
        "name": "TestTask",
        "sequence_no": 0,
    }
    defaults.update(kwargs)
    return ExecutionNode(**defaults)  # type: ignore[arg-type]


def _make_document(**kwargs: object) -> DocumentRecord:
    defaults: dict[str, object] = {
        "document_sha256": f"sha_{uuid4().hex[:8]}",
        "content_sha256": f"content_{uuid4().hex[:8]}",
        "deployment_id": uuid4(),
        "producing_node_id": uuid4(),
        "document_type": "TestDocument",
        "name": "test.md",
    }
    defaults.update(kwargs)
    return DocumentRecord(**defaults)  # type: ignore[arg-type]


def _make_log(**kwargs: object) -> ExecutionLog:
    deployment_id = kwargs.pop("_deployment_id", None) or uuid4()
    defaults: dict[str, object] = {
        "node_id": uuid4(),
        "deployment_id": deployment_id,
        "root_deployment_id": deployment_id,
        "flow_id": None,
        "task_id": None,
        "timestamp": datetime.now(UTC),
        "sequence_no": 0,
        "level": "INFO",
        "category": "framework",
        "logger_name": "ai_pipeline_core.tests",
        "message": "test log",
    }
    defaults.update(kwargs)
    return ExecutionLog(**defaults)  # type: ignore[arg-type]


@CLICKHOUSE_NOT_AVAILABLE
class TestClickHouseNodeOperations:
    async def test_insert_and_get(self) -> None:
        db = _get_db()
        try:
            node = _make_node(node_kind=NodeKind.DEPLOYMENT)
            await db.insert_node(node)
            result = await db.get_node(node.node_id)
            assert result is not None
            assert result.node_id == node.node_id
            assert result.name == node.name
        finally:
            await db.shutdown()

    async def test_update_node(self) -> None:
        db = _get_db()
        try:
            node = _make_node(node_kind=NodeKind.DEPLOYMENT, status=NodeStatus.RUNNING)
            await db.insert_node(node)
            await db.update_node(node.node_id, status=NodeStatus.COMPLETED, ended_at=datetime.now(UTC))
            result = await db.get_node(node.node_id)
            assert result is not None
            assert result.status == NodeStatus.COMPLETED
            assert result.version == 2
        finally:
            await db.shutdown()

    async def test_get_children(self) -> None:
        db = _get_db()
        try:
            deploy_id = uuid4()
            run_id = f"test-{uuid4().hex[:8]}"
            deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id, run_id=run_id)
            child = _make_node(node_kind=NodeKind.FLOW, _deployment_id=deploy_id, parent_node_id=deploy.node_id, sequence_no=1, run_id=run_id)
            await db.insert_node(deploy)
            await db.insert_node(child)
            children = await db.get_children(deploy.node_id)
            assert len(children) >= 1
            child_ids = {c.node_id for c in children}
            assert child.node_id in child_ids
        finally:
            await db.shutdown()

    async def test_get_deployment_by_run_id(self) -> None:
        db = _get_db()
        try:
            run_id = f"unique-run-{uuid4().hex[:8]}"
            deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, run_id=run_id)
            await db.insert_node(deploy)
            result = await db.get_deployment_by_run_id(run_id)
            assert result is not None
            assert result.node_kind == NodeKind.DEPLOYMENT
        finally:
            await db.shutdown()


class TestClickHouseUpdateNodeRetries:
    def _make_stub_database(self):
        from ai_pipeline_core.database._clickhouse import ClickHouseDatabase

        database = ClickHouseDatabase.__new__(ClickHouseDatabase)
        database._ddl = []
        database._cb = SimpleNamespace(client=MagicMock())
        database._ensure_tables = lambda: None
        database._execute_sync = lambda fn: fn()
        return database

    def test_update_node_retries_when_first_write_conflicts(self) -> None:
        database = self._make_stub_database()
        node_id = uuid4()
        initial = _make_node(node_id=node_id, status=NodeStatus.RUNNING, version=1)
        retried = _make_node(node_id=node_id, status=NodeStatus.RUNNING, version=2)
        completed = _make_node(node_id=node_id, status=NodeStatus.COMPLETED, version=3)
        database._get_node_sync = MagicMock(side_effect=[initial, retried, completed])
        database._cb.client.command.side_effect = [
            QuerySummary({"written_rows": "0"}),
            QuerySummary({"written_rows": "1"}),
        ]

        database._update_node_sync(node_id, {"status": NodeStatus.COMPLETED})

        assert database._get_node_sync.call_count == 3
        assert database._cb.client.command.call_count == 2

    def test_update_node_missing_node_raises_key_error(self) -> None:
        database = self._make_stub_database()
        node_id = uuid4()
        database._get_node_sync = MagicMock(return_value=None)

        with pytest.raises(KeyError, match=str(node_id)):
            database._update_node_sync(node_id, {"status": NodeStatus.COMPLETED})

    def test_update_node_retries_when_successful_write_is_immediately_overwritten(self) -> None:
        database = self._make_stub_database()
        node_id = uuid4()
        ended_at = datetime.now(UTC)
        initial = _make_node(node_id=node_id, status=NodeStatus.RUNNING, version=1, ended_at=None, error_message="")
        overwritten = _make_node(node_id=node_id, status=NodeStatus.RUNNING, version=2, ended_at=None, error_message="other writer")
        merged = _make_node(node_id=node_id, status=NodeStatus.RUNNING, version=3, ended_at=ended_at, error_message="other writer")
        database._get_node_sync = MagicMock(side_effect=[initial, overwritten, overwritten, merged])
        database._cb.client.command.side_effect = [
            QuerySummary({"written_rows": "1"}),
            QuerySummary({"written_rows": "1"}),
        ]

        database._update_node_sync(node_id, {"ended_at": ended_at})

        assert database._cb.client.command.call_count == 2
        assert database._get_node_sync.call_count == 4


@CLICKHOUSE_NOT_AVAILABLE
class TestClickHouseDocumentOperations:
    async def test_save_and_get_document(self) -> None:
        db = _get_db()
        try:
            sha = f"doc_{uuid4().hex[:8]}"
            doc = _make_document(document_sha256=sha, run_scope="prod/clickhouse")
            await db.save_document(doc)
            result = await db.get_document(sha)
            assert result is not None
            assert result.document_sha256 == sha
            assert result.run_scope == "prod/clickhouse"
        finally:
            await db.shutdown()

    async def test_save_and_get_blob(self) -> None:
        db = _get_db()
        try:
            sha = f"blob_{uuid4().hex[:8]}"
            blob = BlobRecord(content_sha256=sha, content=b"hello", size_bytes=5)
            await db.save_blob(blob)
            result = await db.get_blob(sha)
            assert result is not None
            assert result.content == b"hello"
        finally:
            await db.shutdown()

    async def test_check_existing_documents(self) -> None:
        db = _get_db()
        try:
            sha = f"exists_{uuid4().hex[:8]}"
            doc = _make_document(document_sha256=sha)
            await db.save_document(doc)
            result = await db.check_existing_documents([sha, "nonexistent"])
            assert sha in result
            assert "nonexistent" not in result
        finally:
            await db.shutdown()


@CLICKHOUSE_NOT_AVAILABLE
class TestClickHouseTableCreation:
    async def test_ensure_tables_idempotent(self) -> None:
        """Verify ensure_tables can be called multiple times without error."""
        db = _get_db()
        try:
            db._ensure_tables()
            db._ensure_tables()
        finally:
            await db.shutdown()


@CLICKHOUSE_NOT_AVAILABLE
class TestClickHouseExecutionLogs:
    async def test_save_and_get_node_logs(self) -> None:
        db = _get_db()
        try:
            deployment_id = uuid4()
            node_id = uuid4()
            await db.save_logs_batch([
                _make_log(node_id=node_id, _deployment_id=deployment_id, sequence_no=0, category="lifecycle"),
                _make_log(node_id=node_id, _deployment_id=deployment_id, sequence_no=1, level="ERROR", category="framework"),
            ])
            logs = await db.get_node_logs(node_id)
            assert [log.sequence_no for log in logs] == [0, 1]
        finally:
            await db.shutdown()

    async def test_get_deployment_logs_filters_category(self) -> None:
        db = _get_db()
        try:
            deployment_id = uuid4()
            await db.save_logs_batch([
                _make_log(_deployment_id=deployment_id, category="framework"),
                _make_log(_deployment_id=deployment_id, category="dependency", level="WARNING"),
            ])
            logs = await db.get_deployment_logs(deployment_id, category="dependency")
            assert len(logs) == 1
            assert logs[0].level == "WARNING"
        finally:
            await db.shutdown()


@CLICKHOUSE_NOT_AVAILABLE
class TestClickHousePhase15Queries:
    async def test_phase15_document_queries_and_deployment_listing(self) -> None:
        db = _get_db()
        try:
            deployment_id = uuid4()
            deployment = _make_node(
                node_kind=NodeKind.DEPLOYMENT,
                _deployment_id=deployment_id,
                name="clickhouse-deployment",
                status=NodeStatus.COMPLETED,
                started_at=datetime(2024, 1, 2, tzinfo=UTC),
            )
            older_deployment = _make_node(
                node_kind=NodeKind.DEPLOYMENT,
                name="older-deployment",
                status=NodeStatus.COMPLETED,
                started_at=datetime(2024, 1, 1, tzinfo=UTC),
            )
            await db.insert_node(deployment)
            await db.insert_node(older_deployment)
            await db.insert_node(
                _make_node(
                    node_kind=NodeKind.CONVERSATION_TURN,
                    _deployment_id=deployment_id,
                    cost_usd=1.25,
                    tokens_input=10,
                    tokens_output=5,
                )
            )

            root = _make_document(
                document_sha256=f"root_{uuid4().hex[:8]}",
                deployment_id=deployment_id,
                producing_node_id=None,
                name="report.md",
                run_scope="scope/a",
                created_at=datetime(2024, 1, 1, tzinfo=UTC),
            )
            child = _make_document(
                document_sha256=f"child_{uuid4().hex[:8]}",
                deployment_id=deployment_id,
                producing_node_id=None,
                name="report.md",
                run_scope="scope/a",
                created_at=datetime(2024, 1, 2, tzinfo=UTC),
                derived_from=(root.document_sha256,),
            )
            triggered = _make_document(
                document_sha256=f"triggered_{uuid4().hex[:8]}",
                deployment_id=deployment_id,
                producing_node_id=None,
                name="notes.md",
                run_scope="scope/b",
                created_at=datetime(2024, 1, 3, tzinfo=UTC),
                triggered_by=(root.document_sha256,),
            )
            await db.save_document_batch([root, child, triggered])

            by_name = await db.find_document_by_name("report.md")
            assert by_name is not None
            assert by_name.document_sha256 == child.document_sha256

            ancestry = await db.get_document_ancestry(child.document_sha256)
            assert root.document_sha256 in ancestry

            origin_docs = await db.find_documents_by_origin(root.document_sha256)
            assert {doc.document_sha256 for doc in origin_docs} >= {child.document_sha256, triggered.document_sha256}

            scopes = await db.list_run_scopes(limit=10)
            assert [scope.run_scope for scope in scopes][:2] == ["scope/b", "scope/a"]

            search = await db.search_documents(name="report", document_type=None, run_scope="scope/a", limit=10, offset=0)
            assert {doc.document_sha256 for doc in search} >= {root.document_sha256, child.document_sha256}

            cost, tokens = await db.get_deployment_cost_totals(deployment_id)
            assert cost == 1.25
            assert tokens == 15

            scope_docs = await db.get_documents_by_run_scope("scope/a")
            assert {doc.document_sha256 for doc in scope_docs} >= {root.document_sha256, child.document_sha256}

            deployments = await db.list_deployments(limit=2, status="completed")
            assert deployments[0].started_at >= deployments[1].started_at
        finally:
            await db.shutdown()
