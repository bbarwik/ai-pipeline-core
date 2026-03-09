"""Tests for FilesystemDatabase backend."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from ai_pipeline_core.database import BlobRecord, DocumentRecord, ExecutionLog, ExecutionNode, NodeKind, NodeStatus
from ai_pipeline_core.database._filesystem import FilesystemDatabase, _sanitize_dir_name


# --- Helpers ---


def _make_node(
    *,
    deployment_id: None = None,
    root_deployment_id: None = None,
    **kwargs: object,
) -> ExecutionNode:
    did = kwargs.pop("_deployment_id", None) or uuid4()
    rid = kwargs.pop("_root_deployment_id", None) or did
    defaults: dict[str, object] = {
        "node_id": uuid4(),
        "node_kind": NodeKind.TASK,
        "deployment_id": did,
        "root_deployment_id": rid,
        "run_id": "test-run",
        "run_scope": "test-run/scope",
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


def _make_blob(content: bytes = b"test content", **kwargs: object) -> BlobRecord:
    defaults: dict[str, object] = {
        "content_sha256": f"blob_{uuid4().hex[:8]}",
        "content": content,
        "size_bytes": len(content),
    }
    defaults.update(kwargs)
    return BlobRecord(**defaults)  # type: ignore[arg-type]


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


# --- Sanitization ---


class TestSanitizeDirName:
    def test_alphanumeric_passes_through(self) -> None:
        assert _sanitize_dir_name("MyFlow") == "MyFlow"

    def test_special_chars_replaced(self) -> None:
        # Special chars replaced with _, trailing _ stripped
        result = _sanitize_dir_name("My Flow!@#")
        assert result.startswith("My_Flow")
        assert "!" not in result and "@" not in result

    def test_truncates_long_names(self) -> None:
        long_name = "A" * 200
        result = _sanitize_dir_name(long_name)
        assert len(result) <= 100

    def test_empty_becomes_unnamed(self) -> None:
        assert _sanitize_dir_name("") == "unnamed"


# --- Node operations ---


class TestNodeInsertAndRetrieve:
    async def test_insert_and_get(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        node = _make_node(node_kind=NodeKind.DEPLOYMENT)
        await db.insert_node(node)
        result = await db.get_node(node.node_id)
        assert result is not None
        assert result.node_id == node.node_id
        assert result.name == node.name
        assert result.node_kind == NodeKind.DEPLOYMENT

    async def test_get_nonexistent_returns_none(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        assert await db.get_node(uuid4()) is None

    async def test_deployment_creates_run_dir(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        node = _make_node(node_kind=NodeKind.DEPLOYMENT, run_id="my-run-123", deployment_name="research-pipeline")
        await db.insert_node(node)
        runs_dir = tmp_path / "runs"
        assert runs_dir.exists()
        run_dirs = list(runs_dir.iterdir())
        assert len(run_dirs) == 1
        assert "research-pipeline" in run_dirs[0].name
        assert "my-run-" in run_dirs[0].name

    async def test_flow_creates_numbered_dir(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deploy_id = uuid4()
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id)
        flow = _make_node(
            node_kind=NodeKind.FLOW,
            _deployment_id=deploy_id,
            parent_node_id=deploy.node_id,
            name="CreatePlanFlow",
            sequence_no=1,
            run_id=deploy.run_id,
        )
        await db.insert_node(deploy)
        await db.insert_node(flow)

        result = await db.get_node(flow.node_id)
        assert result is not None
        assert result.name == "CreatePlanFlow"

    async def test_task_nested_under_flow(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deploy_id = uuid4()
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id)
        flow = _make_node(node_kind=NodeKind.FLOW, _deployment_id=deploy_id, parent_node_id=deploy.node_id, sequence_no=1, run_id=deploy.run_id)
        task = _make_node(
            node_kind=NodeKind.TASK,
            _deployment_id=deploy_id,
            parent_node_id=flow.node_id,
            sequence_no=1,
            name="AnalyzeTask",
            run_id=deploy.run_id,
        )
        await db.insert_node(deploy)
        await db.insert_node(flow)
        await db.insert_node(task)

        result = await db.get_node(task.node_id)
        assert result is not None
        assert result.name == "AnalyzeTask"


class TestUpdateNode:
    async def test_update_status(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        node = _make_node(node_kind=NodeKind.DEPLOYMENT, status=NodeStatus.RUNNING)
        await db.insert_node(node)
        await db.update_node(node.node_id, status=NodeStatus.COMPLETED, ended_at=datetime.now(UTC))
        result = await db.get_node(node.node_id)
        assert result is not None
        assert result.status == NodeStatus.COMPLETED
        assert result.ended_at is not None

    async def test_update_preserves_other_fields(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        node = _make_node(node_kind=NodeKind.DEPLOYMENT, name="keep_this")
        await db.insert_node(node)
        await db.update_node(node.node_id, status=NodeStatus.FAILED)
        result = await db.get_node(node.node_id)
        assert result is not None
        assert result.name == "keep_this"

    async def test_update_increments_version(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        node = _make_node(node_kind=NodeKind.DEPLOYMENT)
        await db.insert_node(node)
        await db.update_node(node.node_id, status=NodeStatus.COMPLETED)
        result = await db.get_node(node.node_id)
        assert result is not None
        assert result.version == 2

    async def test_update_nonexistent_raises(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        with pytest.raises(KeyError):
            await db.update_node(uuid4(), status=NodeStatus.COMPLETED)


class TestGetChildren:
    async def test_returns_children_sorted(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deploy_id = uuid4()
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id)
        child1 = _make_node(
            node_kind=NodeKind.FLOW,
            _deployment_id=deploy_id,
            parent_node_id=deploy.node_id,
            sequence_no=2,
            name="second",
            run_id=deploy.run_id,
        )
        child2 = _make_node(
            node_kind=NodeKind.FLOW,
            _deployment_id=deploy_id,
            parent_node_id=deploy.node_id,
            sequence_no=1,
            name="first",
            run_id=deploy.run_id,
        )
        await db.insert_node(deploy)
        await db.insert_node(child1)
        await db.insert_node(child2)

        children = await db.get_children(deploy.node_id)
        assert len(children) == 2
        assert children[0].name == "first"
        assert children[1].name == "second"

    async def test_empty_for_no_children(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        assert await db.get_children(uuid4()) == []


class TestGetDeploymentTree:
    async def test_returns_all_nodes_in_deployment(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deploy_id = uuid4()
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id, sequence_no=0)
        flow = _make_node(node_kind=NodeKind.FLOW, _deployment_id=deploy_id, parent_node_id=deploy.node_id, sequence_no=1, run_id=deploy.run_id)
        other = _make_node(node_kind=NodeKind.DEPLOYMENT, run_id="other-run")  # different deployment

        await db.insert_node(deploy)
        await db.insert_node(flow)
        await db.insert_node(other)

        tree = await db.get_deployment_tree(deploy_id)
        assert len(tree) == 2
        tree_ids = {n.node_id for n in tree}
        assert deploy.node_id in tree_ids
        assert flow.node_id in tree_ids
        assert other.node_id not in tree_ids


class TestGetDeploymentByRunId:
    async def test_finds_deployment(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, run_id="my-run-123")
        await db.insert_node(deploy)
        result = await db.get_deployment_by_run_id("my-run-123")
        assert result is not None
        assert result.node_kind == NodeKind.DEPLOYMENT

    async def test_returns_none_for_unknown(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        assert await db.get_deployment_by_run_id("nonexistent") is None


class TestGetDeploymentByRunScope:
    async def test_finds_by_scope(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, run_scope="prod/my-pipeline")
        await db.insert_node(deploy)
        result = await db.get_deployment_by_run_scope("prod/my-pipeline")
        assert result is not None
        assert result.node_kind == NodeKind.DEPLOYMENT


# --- Document operations ---


class TestDocuments:
    async def test_save_and_get(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        # Need a deployment node first for document path resolution
        deploy_id = uuid4()
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id)
        await db.insert_node(deploy)

        doc = _make_document(document_sha256="doc1", deployment_id=deploy_id, producing_node_id=None, run_scope="prod/example")
        await db.save_document(doc)
        result = await db.get_document("doc1")
        assert result is not None
        assert result.document_sha256 == "doc1"
        assert result.run_scope == "prod/example"

    async def test_get_nonexistent_returns_none(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        assert await db.get_document("nonexistent") is None

    async def test_save_batch(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deploy_id = uuid4()
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id)
        await db.insert_node(deploy)

        docs = [_make_document(document_sha256=f"batch_{i}", deployment_id=deploy_id, producing_node_id=None) for i in range(3)]
        await db.save_document_batch(docs)
        for i in range(3):
            assert await db.get_document(f"batch_{i}") is not None

    async def test_check_existing(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deploy_id = uuid4()
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id)
        await db.insert_node(deploy)

        await db.save_document(_make_document(document_sha256="exists_1", deployment_id=deploy_id, producing_node_id=None))
        result = await db.check_existing_documents(["exists_1", "missing"])
        assert result == {"exists_1"}

    async def test_update_summary(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deploy_id = uuid4()
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id)
        await db.insert_node(deploy)

        doc = _make_document(document_sha256="sum_doc", deployment_id=deploy_id, producing_node_id=None)
        await db.save_document(doc)
        await db.update_document_summary("sum_doc", "A brief summary")
        result = await db.get_document("sum_doc")
        assert result is not None
        assert result.summary == "A brief summary"
        assert result.version == 2

    async def test_find_by_source(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deploy_id = uuid4()
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id)
        await db.insert_node(deploy)

        doc = _make_document(document_sha256="derived", deployment_id=deploy_id, producing_node_id=None, derived_from=("source_sha",))
        await db.save_document(doc)
        results = await db.find_documents_by_source("source_sha")
        assert len(results) == 1
        assert results[0].document_sha256 == "derived"


class TestDocumentsByNode:
    async def test_task_produced_documents(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deploy_id = uuid4()
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id)
        task = _make_node(node_kind=NodeKind.TASK, _deployment_id=deploy_id, parent_node_id=deploy.node_id, run_id=deploy.run_id)
        await db.insert_node(deploy)
        await db.insert_node(task)

        doc = _make_document(document_sha256="task_doc", deployment_id=deploy_id, producing_node_id=task.node_id)
        await db.save_document(doc)
        results = await db.get_documents_by_node(task.node_id)
        assert len(results) == 1
        assert results[0].document_sha256 == "task_doc"


class TestPhase15DocumentQueries:
    async def test_find_document_by_name_returns_latest_match(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deploy_id = uuid4()
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id)
        await db.insert_node(deploy)
        await db.save_document_batch([
            _make_document(
                document_sha256="doc_old",
                deployment_id=deploy_id,
                producing_node_id=None,
                name="report.md",
                created_at=datetime(2024, 1, 1, tzinfo=UTC),
            ),
            _make_document(
                document_sha256="doc_new",
                deployment_id=deploy_id,
                producing_node_id=None,
                name="report.md",
                created_at=datetime(2024, 1, 2, tzinfo=UTC),
            ),
        ])

        result = await db.find_document_by_name("report.md")

        assert result is not None
        assert result.document_sha256 == "doc_new"

    async def test_get_document_ancestry_traverses_links(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deploy_id = uuid4()
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id)
        await db.insert_node(deploy)
        await db.save_document_batch([
            _make_document(document_sha256="root", deployment_id=deploy_id, producing_node_id=None),
            _make_document(document_sha256="parent", deployment_id=deploy_id, producing_node_id=None, derived_from=("root",)),
            _make_document(document_sha256="child", deployment_id=deploy_id, producing_node_id=None, derived_from=("parent",), triggered_by=("root",)),
        ])

        ancestry = await db.get_document_ancestry("child")

        assert set(ancestry) == {"parent", "root"}

    async def test_find_documents_by_origin_checks_both_arrays(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deploy_id = uuid4()
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id)
        await db.insert_node(deploy)
        await db.save_document_batch([
            _make_document(document_sha256="derived", deployment_id=deploy_id, producing_node_id=None, derived_from=("origin",)),
            _make_document(document_sha256="triggered", deployment_id=deploy_id, producing_node_id=None, triggered_by=("origin",)),
        ])

        results = await db.find_documents_by_origin("origin")

        assert {doc.document_sha256 for doc in results} == {"derived", "triggered"}

    async def test_list_run_scopes_aggregates_non_empty_scopes(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deploy_id = uuid4()
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id)
        await db.insert_node(deploy)
        await db.save_document_batch([
            _make_document(
                document_sha256="a",
                deployment_id=deploy_id,
                producing_node_id=None,
                run_scope="scope/a",
                created_at=datetime(2024, 1, 1, tzinfo=UTC),
            ),
            _make_document(
                document_sha256="b",
                deployment_id=deploy_id,
                producing_node_id=None,
                run_scope="scope/a",
                created_at=datetime(2024, 1, 2, tzinfo=UTC),
            ),
            _make_document(
                document_sha256="c",
                deployment_id=deploy_id,
                producing_node_id=None,
                run_scope="scope/b",
                created_at=datetime(2024, 1, 3, tzinfo=UTC),
            ),
            _make_document(document_sha256="d", deployment_id=deploy_id, producing_node_id=None, run_scope=""),
        ])

        scopes = await db.list_run_scopes(limit=10)

        assert [scope.run_scope for scope in scopes] == ["scope/b", "scope/a"]
        assert scopes[1].document_count == 2

    async def test_search_documents_filters_and_paginates(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deploy_id = uuid4()
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id)
        await db.insert_node(deploy)
        await db.save_document_batch([
            _make_document(
                document_sha256="a",
                deployment_id=deploy_id,
                producing_node_id=None,
                name="alpha.txt",
                document_type="TypeA",
                run_scope="scope/a",
                created_at=datetime(2024, 1, 1, tzinfo=UTC),
            ),
            _make_document(
                document_sha256="b",
                deployment_id=deploy_id,
                producing_node_id=None,
                name="beta.txt",
                document_type="TypeA",
                run_scope="scope/a",
                created_at=datetime(2024, 1, 2, tzinfo=UTC),
            ),
            _make_document(
                document_sha256="c",
                deployment_id=deploy_id,
                producing_node_id=None,
                name="beta-notes.txt",
                document_type="TypeB",
                run_scope="scope/b",
                created_at=datetime(2024, 1, 3, tzinfo=UTC),
            ),
        ])

        results = await db.search_documents(name="beta", document_type="TypeA", run_scope="scope/a", limit=1, offset=0)

        assert [doc.document_sha256 for doc in results] == ["b"]

    async def test_get_deployment_cost_totals_sums_turn_metrics(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deployment_id = uuid4()
        await db.insert_node(_make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deployment_id))
        await db.insert_node(
            _make_node(
                node_kind=NodeKind.CONVERSATION_TURN,
                _deployment_id=deployment_id,
                sequence_no=0,
                cost_usd=1.5,
                tokens_input=10,
                tokens_output=5,
            )
        )
        await db.insert_node(
            _make_node(
                node_kind=NodeKind.CONVERSATION_TURN,
                _deployment_id=deployment_id,
                sequence_no=1,
                cost_usd=2.0,
                tokens_input=3,
                tokens_output=7,
            )
        )
        await db.insert_node(_make_node(node_kind=NodeKind.TASK, _deployment_id=deployment_id, cost_usd=99.0, tokens_input=99, tokens_output=99))

        cost, tokens = await db.get_deployment_cost_totals(deployment_id)

        assert cost == 3.5
        assert tokens == 25

    async def test_get_documents_by_run_scope_returns_matching_documents(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deploy_id = uuid4()
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id)
        await db.insert_node(deploy)
        await db.save_document_batch([
            _make_document(document_sha256="a", deployment_id=deploy_id, producing_node_id=None, run_scope="scope/a"),
            _make_document(document_sha256="b", deployment_id=deploy_id, producing_node_id=None, run_scope="scope/a"),
            _make_document(document_sha256="c", deployment_id=deploy_id, producing_node_id=None, run_scope="scope/b"),
        ])

        results = await db.get_documents_by_run_scope("scope/a")

        assert {doc.document_sha256 for doc in results} == {"a", "b"}

    async def test_list_deployments_filters_and_orders_by_started_at(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        older = _make_node(
            node_kind=NodeKind.DEPLOYMENT,
            name="older",
            run_id="older-run",
            status=NodeStatus.COMPLETED,
            started_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        newer = _make_node(
            node_kind=NodeKind.DEPLOYMENT,
            name="newer",
            run_id="newer-run",
            status=NodeStatus.COMPLETED,
            started_at=datetime(2024, 1, 2, tzinfo=UTC),
        )
        failed = _make_node(
            node_kind=NodeKind.DEPLOYMENT,
            name="failed",
            run_id="failed-run",
            status=NodeStatus.FAILED,
            started_at=datetime(2024, 1, 3, tzinfo=UTC),
        )
        await db.insert_node(older)
        await db.insert_node(newer)
        await db.insert_node(failed)

        deployments = await db.list_deployments(limit=2, status="completed")

        assert [node.name for node in deployments] == ["newer", "older"]

    async def test_deployment_includes_root_inputs(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deploy_id = uuid4()
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id)
        await db.insert_node(deploy)

        root_doc = _make_document(document_sha256="root_in", deployment_id=deploy_id, producing_node_id=None)
        await db.save_document(root_doc)

        results = await db.get_documents_by_node(deploy.node_id)
        sha_set = {d.document_sha256 for d in results}
        assert "root_in" in sha_set


class TestExecutionLogs:
    async def test_save_logs_batch_writes_root_logs_jsonl(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deployment_id = uuid4()
        await db.save_logs_batch([
            _make_log(_deployment_id=deployment_id, sequence_no=0),
            _make_log(_deployment_id=deployment_id, sequence_no=1, level="ERROR", category="application"),
        ])

        logs_path = tmp_path / "logs.jsonl"
        assert logs_path.exists()
        assert len(logs_path.read_text(encoding="utf-8").strip().splitlines()) == 2

    async def test_get_node_logs_and_deployment_logs(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deployment_id = uuid4()
        node_id = uuid4()
        await db.save_logs_batch([
            _make_log(node_id=node_id, _deployment_id=deployment_id, sequence_no=0, category="lifecycle"),
            _make_log(node_id=node_id, _deployment_id=deployment_id, sequence_no=1, level="ERROR", category="framework"),
            _make_log(_deployment_id=uuid4(), sequence_no=0, level="WARNING", category="dependency"),
        ])

        node_logs = await db.get_node_logs(node_id)
        assert [log.sequence_no for log in node_logs] == [0, 1]

        framework_logs = await db.get_deployment_logs(deployment_id, category="framework")
        assert len(framework_logs) == 1
        assert framework_logs[0].level == "ERROR"

    async def test_concurrent_instances_append_without_corrupting_logs_jsonl(self, tmp_path: Path) -> None:
        first = FilesystemDatabase(tmp_path)
        second = FilesystemDatabase(tmp_path)
        deployment_id = uuid4()

        await asyncio.gather(
            first.save_logs_batch([_make_log(_deployment_id=deployment_id, sequence_no=index, message=f"first-{index}") for index in range(25)]),
            second.save_logs_batch([_make_log(_deployment_id=deployment_id, sequence_no=index, message=f"second-{index}") for index in range(25)]),
        )

        logs_path = tmp_path / "logs.jsonl"
        lines = logs_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 50
        assert all(json.loads(line)["message"] for line in lines)


# --- Blob operations ---


class TestBlobs:
    async def test_save_and_get(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        blob = _make_blob(content=b"hello world", content_sha256="blob1")
        await db.save_blob(blob)
        result = await db.get_blob("blob1")
        assert result is not None
        assert result.content == b"hello world"

    async def test_get_nonexistent_returns_none(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        assert await db.get_blob("nonexistent") is None

    async def test_blob_path_structure(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        blob = _make_blob(content=b"data", content_sha256="abcdef1234567890")
        await db.save_blob(blob)
        expected_path = tmp_path / "blobs" / "ab" / "abcdef1234567890"
        assert expected_path.exists()
        assert expected_path.read_bytes() == b"data"

    async def test_save_batch(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        blobs = [_make_blob(content=f"data_{i}".encode(), content_sha256=f"bb_{i}") for i in range(3)]
        await db.save_blob_batch(blobs)
        for i in range(3):
            result = await db.get_blob(f"bb_{i}")
            assert result is not None
            assert result.content == f"data_{i}".encode()

    async def test_get_blobs_batch(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        blobs = [_make_blob(content=f"data_{i}".encode(), content_sha256=f"gb_{i}") for i in range(2)]
        await db.save_blob_batch(blobs)
        result = await db.get_blobs_batch(["gb_0", "gb_1", "gb_missing"])
        assert len(result) == 2
        assert "gb_missing" not in result


# --- Cache completion ---


class TestCachedCompletion:
    async def test_finds_completed_node(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        node = _make_node(
            node_kind=NodeKind.DEPLOYMENT,
            cache_key="task:abc",
            status=NodeStatus.COMPLETED,
            ended_at=datetime.now(UTC),
        )
        await db.insert_node(node)
        result = await db.get_cached_completion("task:abc")
        assert result is not None
        assert result.node_id == node.node_id

    async def test_returns_none_for_unknown_key(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        assert await db.get_cached_completion("nonexistent") is None

    async def test_respects_max_age(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        old_time = datetime.now(UTC) - timedelta(hours=2)
        node = _make_node(
            node_kind=NodeKind.DEPLOYMENT,
            cache_key="old_key",
            status=NodeStatus.COMPLETED,
            ended_at=old_time,
        )
        await db.insert_node(node)
        assert await db.get_cached_completion("old_key", max_age=timedelta(hours=3)) is not None
        assert await db.get_cached_completion("old_key", max_age=timedelta(hours=1)) is None


# --- Index rebuild ---


class TestIndexRebuild:
    async def test_rebuild_on_new_instance(self, tmp_path: Path) -> None:
        """Verify that creating a new FilesystemDatabase from an existing directory rebuilds indexes."""
        db1 = FilesystemDatabase(tmp_path)
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, run_id="rebuild-test")
        await db1.insert_node(deploy)

        # Create new instance from same directory
        db2 = FilesystemDatabase(tmp_path)
        result = await db2.get_node(deploy.node_id)
        assert result is not None
        assert result.run_id == "rebuild-test"

    async def test_rebuild_documents(self, tmp_path: Path) -> None:
        db1 = FilesystemDatabase(tmp_path)
        deploy_id = uuid4()
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id)
        await db1.insert_node(deploy)
        doc = _make_document(document_sha256="rebuild_doc", deployment_id=deploy_id, producing_node_id=None)
        await db1.save_document(doc)

        db2 = FilesystemDatabase(tmp_path)
        result = await db2.get_document("rebuild_doc")
        assert result is not None

    async def test_rebuild_children_index(self, tmp_path: Path) -> None:
        db1 = FilesystemDatabase(tmp_path)
        deploy_id = uuid4()
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id)
        flow = _make_node(node_kind=NodeKind.FLOW, _deployment_id=deploy_id, parent_node_id=deploy.node_id, sequence_no=1, run_id=deploy.run_id)
        await db1.insert_node(deploy)
        await db1.insert_node(flow)

        db2 = FilesystemDatabase(tmp_path)
        children = await db2.get_children(deploy.node_id)
        assert len(children) == 1
        assert children[0].node_id == flow.node_id

    async def test_rebuild_cache_index(self, tmp_path: Path) -> None:
        db1 = FilesystemDatabase(tmp_path)
        node = _make_node(
            node_kind=NodeKind.DEPLOYMENT,
            cache_key="rebuild_cache",
            status=NodeStatus.COMPLETED,
            ended_at=datetime.now(UTC),
        )
        await db1.insert_node(node)

        db2 = FilesystemDatabase(tmp_path)
        result = await db2.get_cached_completion("rebuild_cache")
        assert result is not None

    async def test_rebuild_cache_index_returns_newest_matching_completion(self, tmp_path: Path) -> None:
        db1 = FilesystemDatabase(tmp_path)
        deployment_id = uuid4()
        deployment = _make_node(
            node_id=deployment_id,
            node_kind=NodeKind.DEPLOYMENT,
            _deployment_id=deployment_id,
            _root_deployment_id=deployment_id,
            run_id="rebuild-shared-cache",
            name="rebuild-shared-cache",
            status=NodeStatus.COMPLETED,
            cache_key="shared-cache",
            ended_at=datetime(2025, 6, 1, tzinfo=UTC),
        )
        flow = _make_node(
            node_kind=NodeKind.FLOW,
            _deployment_id=deployment_id,
            _root_deployment_id=deployment_id,
            parent_node_id=deployment_id,
            run_id=deployment.run_id,
            name="SharedCacheFlow",
            sequence_no=1,
            status=NodeStatus.COMPLETED,
            ended_at=datetime(2025, 6, 1, tzinfo=UTC),
        )
        newest = _make_node(
            node_kind=NodeKind.TASK,
            _deployment_id=deployment_id,
            _root_deployment_id=deployment_id,
            parent_node_id=flow.node_id,
            flow_id=flow.node_id,
            run_id=deployment.run_id,
            name="NewestTask",
            sequence_no=1,
            cache_key="shared-cache",
            status=NodeStatus.COMPLETED,
            ended_at=datetime(2025, 6, 1, tzinfo=UTC),
        )
        older = _make_node(
            node_kind=NodeKind.TASK,
            _deployment_id=deployment_id,
            _root_deployment_id=deployment_id,
            parent_node_id=flow.node_id,
            flow_id=flow.node_id,
            run_id=deployment.run_id,
            name="OlderTask",
            sequence_no=2,
            cache_key="shared-cache",
            status=NodeStatus.COMPLETED,
            ended_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        for node in (deployment, flow, newest, older):
            await db1.insert_node(node)

        db2 = FilesystemDatabase(tmp_path)
        result = await db2.get_cached_completion("shared-cache")
        assert result is not None
        assert result.node_id == newest.node_id


# --- Flush/shutdown ---


class TestFlushShutdown:
    async def test_flush_is_noop(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        await db.flush()

    async def test_shutdown_is_noop(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        await db.shutdown()


# --- All document SHAs for deployment ---


class TestGetAllDocumentShasForDeployment:
    async def test_collects_shas_from_nodes(self, tmp_path: Path) -> None:
        db = FilesystemDatabase(tmp_path)
        deploy_id = uuid4()
        deploy = _make_node(
            node_kind=NodeKind.DEPLOYMENT,
            _deployment_id=deploy_id,
            input_document_shas=("sha_a", "sha_b"),
            output_document_shas=("sha_c",),
        )
        await db.insert_node(deploy)
        shas = await db.get_all_document_shas_for_deployment(deploy_id)
        assert shas == {"sha_a", "sha_b", "sha_c"}
