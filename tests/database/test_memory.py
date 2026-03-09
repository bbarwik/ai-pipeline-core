"""Comprehensive tests for MemoryDatabase backend."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from ai_pipeline_core.database import BlobRecord, DocumentRecord, ExecutionLog, ExecutionNode, MemoryDatabase, NodeKind, NodeStatus


# --- Helpers ---


def _make_node(
    *,
    deployment_id: None = None,
    root_deployment_id: None = None,
    **kwargs: object,
) -> ExecutionNode:
    """Create an ExecutionNode with sensible defaults."""
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


def _make_blob(**kwargs: object) -> BlobRecord:
    content = kwargs.pop("content", b"test content")  # type: ignore[arg-type]
    defaults: dict[str, object] = {
        "content_sha256": f"blob_{uuid4().hex[:8]}",
        "content": content,
        "size_bytes": len(content),  # type: ignore[arg-type]
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


# --- Node tests ---


class TestInsertAndRetrieveNodes:
    @pytest.mark.asyncio
    async def test_insert_and_get(self) -> None:
        db = MemoryDatabase()
        node = _make_node()
        await db.insert_node(node)
        result = await db.get_node(node.node_id)
        assert result is not None
        assert result.node_id == node.node_id
        assert result.name == node.name

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self) -> None:
        db = MemoryDatabase()
        assert await db.get_node(uuid4()) is None

    @pytest.mark.asyncio
    async def test_insert_overwrites(self) -> None:
        db = MemoryDatabase()
        node = _make_node(name="original")
        await db.insert_node(node)
        updated = ExecutionNode(
            node_id=node.node_id,
            node_kind=node.node_kind,
            deployment_id=node.deployment_id,
            root_deployment_id=node.root_deployment_id,
            parent_node_id=node.parent_node_id,
            run_id=node.run_id,
            run_scope=node.run_scope,
            deployment_name=node.deployment_name,
            name="updated",
            sequence_no=node.sequence_no,
        )
        await db.insert_node(updated)
        result = await db.get_node(node.node_id)
        assert result is not None
        assert result.name == "updated"


class TestUpdateNode:
    @pytest.mark.asyncio
    async def test_update_status(self) -> None:
        db = MemoryDatabase()
        node = _make_node(status=NodeStatus.RUNNING)
        await db.insert_node(node)
        await db.update_node(node.node_id, status=NodeStatus.COMPLETED, ended_at=datetime.now(UTC))
        result = await db.get_node(node.node_id)
        assert result is not None
        assert result.status == NodeStatus.COMPLETED
        assert result.ended_at is not None

    @pytest.mark.asyncio
    async def test_update_preserves_other_fields(self) -> None:
        db = MemoryDatabase()
        node = _make_node(name="keep_this", cost_usd=1.5)
        await db.insert_node(node)
        await db.update_node(node.node_id, status=NodeStatus.FAILED)
        result = await db.get_node(node.node_id)
        assert result is not None
        assert result.name == "keep_this"
        assert result.cost_usd == 1.5

    @pytest.mark.asyncio
    async def test_update_nonexistent_raises(self) -> None:
        db = MemoryDatabase()
        with pytest.raises(KeyError):
            await db.update_node(uuid4(), status=NodeStatus.COMPLETED)

    @pytest.mark.asyncio
    async def test_update_sets_updated_at(self) -> None:
        db = MemoryDatabase()
        node = _make_node()
        await db.insert_node(node)
        original_updated_at = node.updated_at
        await db.update_node(node.node_id, status=NodeStatus.COMPLETED)
        result = await db.get_node(node.node_id)
        assert result is not None
        assert result.updated_at >= original_updated_at

    @pytest.mark.asyncio
    async def test_update_increments_version(self) -> None:
        db = MemoryDatabase()
        node = _make_node()
        await db.insert_node(node)
        await db.update_node(node.node_id, status=NodeStatus.COMPLETED)
        result = await db.get_node(node.node_id)
        assert result is not None
        assert result.version == 2


class TestGetChildren:
    @pytest.mark.asyncio
    async def test_returns_children_sorted_by_sequence(self) -> None:
        db = MemoryDatabase()
        parent_id = uuid4()
        deploy_id = uuid4()
        child1 = _make_node(parent_node_id=parent_id, sequence_no=2, name="second", _deployment_id=deploy_id)
        child2 = _make_node(parent_node_id=parent_id, sequence_no=1, name="first", _deployment_id=deploy_id)
        child3 = _make_node(parent_node_id=parent_id, sequence_no=3, name="third", _deployment_id=deploy_id)
        for child in [child1, child2, child3]:
            await db.insert_node(child)

        children = await db.get_children(parent_id)
        assert len(children) == 3
        assert children[0].name == "first"
        assert children[1].name == "second"
        assert children[2].name == "third"

    @pytest.mark.asyncio
    async def test_empty_for_no_children(self) -> None:
        db = MemoryDatabase()
        assert await db.get_children(uuid4()) == []


class TestGetDeploymentTree:
    @pytest.mark.asyncio
    async def test_returns_all_nodes_in_deployment(self) -> None:
        db = MemoryDatabase()
        deploy_id = uuid4()
        deploy_node = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id, sequence_no=0)
        flow_node = _make_node(node_kind=NodeKind.FLOW, _deployment_id=deploy_id, parent_node_id=deploy_node.node_id, sequence_no=1)
        task_node = _make_node(node_kind=NodeKind.TASK, _deployment_id=deploy_id, parent_node_id=flow_node.node_id, sequence_no=2)
        other_node = _make_node()  # different deployment

        for node in [deploy_node, flow_node, task_node, other_node]:
            await db.insert_node(node)

        tree = await db.get_deployment_tree(deploy_id)
        assert len(tree) == 3
        tree_ids = {n.node_id for n in tree}
        assert deploy_node.node_id in tree_ids
        assert flow_node.node_id in tree_ids
        assert task_node.node_id in tree_ids
        assert other_node.node_id not in tree_ids


class TestGetDeploymentByRunId:
    @pytest.mark.asyncio
    async def test_finds_deployment_node(self) -> None:
        db = MemoryDatabase()
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, run_id="my-run-123")
        task = _make_node(node_kind=NodeKind.TASK, run_id="my-run-123")
        await db.insert_node(deploy)
        await db.insert_node(task)

        result = await db.get_deployment_by_run_id("my-run-123")
        assert result is not None
        assert result.node_kind == NodeKind.DEPLOYMENT

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown(self) -> None:
        db = MemoryDatabase()
        assert await db.get_deployment_by_run_id("nonexistent") is None


# --- Document tests ---


class TestDocuments:
    @pytest.mark.asyncio
    async def test_save_and_get(self) -> None:
        db = MemoryDatabase()
        doc = _make_document(document_sha256="doc1", run_scope="test/run-scope")
        await db.save_document(doc)
        result = await db.get_document("doc1")
        assert result is not None
        assert result.document_sha256 == "doc1"
        assert result.run_scope == "test/run-scope"

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self) -> None:
        db = MemoryDatabase()
        assert await db.get_document("nonexistent") is None

    @pytest.mark.asyncio
    async def test_save_batch(self) -> None:
        db = MemoryDatabase()
        docs = [_make_document(document_sha256=f"batch_{i}") for i in range(3)]
        await db.save_document_batch(docs)
        for i in range(3):
            result = await db.get_document(f"batch_{i}")
            assert result is not None

    @pytest.mark.asyncio
    async def test_get_documents_batch(self) -> None:
        db = MemoryDatabase()
        docs = [_make_document(document_sha256=f"b_{i}") for i in range(3)]
        await db.save_document_batch(docs)
        result = await db.get_documents_batch(["b_0", "b_1", "b_2", "b_missing"])
        assert len(result) == 3
        assert "b_missing" not in result

    @pytest.mark.asyncio
    async def test_check_existing_documents(self) -> None:
        db = MemoryDatabase()
        await db.save_document(_make_document(document_sha256="exists_1"))
        await db.save_document(_make_document(document_sha256="exists_2"))
        result = await db.check_existing_documents(["exists_1", "exists_2", "missing"])
        assert result == {"exists_1", "exists_2"}

    @pytest.mark.asyncio
    async def test_update_document_summary(self) -> None:
        db = MemoryDatabase()
        doc = _make_document(document_sha256="sum_doc")
        await db.save_document(doc)
        await db.update_document_summary("sum_doc", "A brief summary")
        result = await db.get_document("sum_doc")
        assert result is not None
        assert result.summary == "A brief summary"
        assert result.version == 2

    @pytest.mark.asyncio
    async def test_update_summary_nonexistent_is_noop(self) -> None:
        db = MemoryDatabase()
        await db.update_document_summary("nonexistent", "summary")
        # No error raised

    @pytest.mark.asyncio
    async def test_find_documents_by_source(self) -> None:
        db = MemoryDatabase()
        doc = _make_document(document_sha256="derived", derived_from=("source_sha",))
        await db.save_document(doc)
        results = await db.find_documents_by_source("source_sha")
        assert len(results) == 1
        assert results[0].document_sha256 == "derived"

    @pytest.mark.asyncio
    async def test_find_documents_by_source_no_match(self) -> None:
        db = MemoryDatabase()
        results = await db.find_documents_by_source("nonexistent")
        assert results == []


class TestDocumentsByDeployment:
    @pytest.mark.asyncio
    async def test_returns_documents_for_deployment(self) -> None:
        db = MemoryDatabase()
        deploy_id = uuid4()
        doc1 = _make_document(document_sha256="d1", deployment_id=deploy_id)
        doc2 = _make_document(document_sha256="d2", deployment_id=deploy_id)
        doc3 = _make_document(document_sha256="d3")  # different deployment
        await db.save_document_batch([doc1, doc2, doc3])

        results = await db.get_documents_by_deployment(deploy_id)
        assert len(results) == 2
        sha_set = {d.document_sha256 for d in results}
        assert sha_set == {"d1", "d2"}


class TestPhase15DocumentQueries:
    @pytest.mark.asyncio
    async def test_find_document_by_name_returns_latest_match(self) -> None:
        db = MemoryDatabase()
        older = _make_document(document_sha256="doc_old", name="report.md", created_at=datetime(2024, 1, 1, tzinfo=UTC))
        newer = _make_document(document_sha256="doc_new", name="report.md", created_at=datetime(2024, 1, 2, tzinfo=UTC))
        await db.save_document_batch([older, newer])

        result = await db.find_document_by_name("report.md")

        assert result is not None
        assert result.document_sha256 == "doc_new"

    @pytest.mark.asyncio
    async def test_get_document_ancestry_traverses_derived_and_triggered_links(self) -> None:
        db = MemoryDatabase()
        root = _make_document(document_sha256="root")
        parent = _make_document(document_sha256="parent", derived_from=("root",))
        child = _make_document(document_sha256="child", derived_from=("parent",), triggered_by=("root",))
        await db.save_document_batch([root, parent, child])

        ancestry = await db.get_document_ancestry("child")

        assert set(ancestry) == {"parent", "root"}

    @pytest.mark.asyncio
    async def test_find_documents_by_origin_checks_both_arrays(self) -> None:
        db = MemoryDatabase()
        derived = _make_document(document_sha256="derived", derived_from=("origin",))
        triggered = _make_document(document_sha256="triggered", triggered_by=("origin",))
        await db.save_document_batch([derived, triggered])

        results = await db.find_documents_by_origin("origin")

        assert {doc.document_sha256 for doc in results} == {"derived", "triggered"}

    @pytest.mark.asyncio
    async def test_list_run_scopes_aggregates_non_empty_scopes(self) -> None:
        db = MemoryDatabase()
        await db.save_document_batch([
            _make_document(document_sha256="a", run_scope="scope/a", created_at=datetime(2024, 1, 1, tzinfo=UTC)),
            _make_document(document_sha256="b", run_scope="scope/a", created_at=datetime(2024, 1, 2, tzinfo=UTC)),
            _make_document(document_sha256="c", run_scope="scope/b", created_at=datetime(2024, 1, 3, tzinfo=UTC)),
            _make_document(document_sha256="d", run_scope=""),
        ])

        scopes = await db.list_run_scopes(limit=10)

        assert [scope.run_scope for scope in scopes] == ["scope/b", "scope/a"]
        assert scopes[1].document_count == 2

    @pytest.mark.asyncio
    async def test_search_documents_filters_and_paginates(self) -> None:
        db = MemoryDatabase()
        await db.save_document_batch([
            _make_document(document_sha256="a", name="alpha.txt", document_type="TypeA", run_scope="scope/a", created_at=datetime(2024, 1, 1, tzinfo=UTC)),
            _make_document(document_sha256="b", name="beta.txt", document_type="TypeA", run_scope="scope/a", created_at=datetime(2024, 1, 2, tzinfo=UTC)),
            _make_document(
                document_sha256="c",
                name="beta-notes.txt",
                document_type="TypeB",
                run_scope="scope/b",
                created_at=datetime(2024, 1, 3, tzinfo=UTC),
            ),
        ])

        results = await db.search_documents(name="beta", document_type="TypeA", run_scope="scope/a", limit=1, offset=0)

        assert [doc.document_sha256 for doc in results] == ["b"]

    @pytest.mark.asyncio
    async def test_get_deployment_cost_totals_sums_turn_metrics(self) -> None:
        db = MemoryDatabase()
        deployment_id = uuid4()
        await db.insert_node(_make_node(node_kind=NodeKind.CONVERSATION_TURN, _deployment_id=deployment_id, cost_usd=1.5, tokens_input=10, tokens_output=5))
        await db.insert_node(_make_node(node_kind=NodeKind.CONVERSATION_TURN, _deployment_id=deployment_id, cost_usd=2.0, tokens_input=3, tokens_output=7))
        await db.insert_node(_make_node(node_kind=NodeKind.TASK, _deployment_id=deployment_id, cost_usd=99.0, tokens_input=99, tokens_output=99))

        cost, tokens = await db.get_deployment_cost_totals(deployment_id)

        assert cost == 3.5
        assert tokens == 25

    @pytest.mark.asyncio
    async def test_get_documents_by_run_scope_returns_matching_documents(self) -> None:
        db = MemoryDatabase()
        await db.save_document_batch([
            _make_document(document_sha256="a", run_scope="scope/a"),
            _make_document(document_sha256="b", run_scope="scope/a"),
            _make_document(document_sha256="c", run_scope="scope/b"),
        ])

        results = await db.get_documents_by_run_scope("scope/a")

        assert {doc.document_sha256 for doc in results} == {"a", "b"}

    @pytest.mark.asyncio
    async def test_list_deployments_filters_and_orders_by_started_at(self) -> None:
        db = MemoryDatabase()
        older = _make_node(node_kind=NodeKind.DEPLOYMENT, name="older", status=NodeStatus.COMPLETED, started_at=datetime(2024, 1, 1, tzinfo=UTC))
        newer = _make_node(node_kind=NodeKind.DEPLOYMENT, name="newer", status=NodeStatus.COMPLETED, started_at=datetime(2024, 1, 2, tzinfo=UTC))
        failed = _make_node(node_kind=NodeKind.DEPLOYMENT, name="failed", status=NodeStatus.FAILED, started_at=datetime(2024, 1, 3, tzinfo=UTC))
        await db.insert_node(older)
        await db.insert_node(newer)
        await db.insert_node(failed)

        deployments = await db.list_deployments(limit=2, status="completed")

        assert [node.name for node in deployments] == ["newer", "older"]


class TestExecutionLogs:
    @pytest.mark.asyncio
    async def test_save_logs_batch_and_get_node_logs(self) -> None:
        db = MemoryDatabase()
        node_id = uuid4()
        deployment_id = uuid4()
        logs = [
            _make_log(node_id=node_id, _deployment_id=deployment_id, sequence_no=0, level="INFO", category="lifecycle"),
            _make_log(node_id=node_id, _deployment_id=deployment_id, sequence_no=1, level="ERROR", category="framework"),
            _make_log(_deployment_id=deployment_id, sequence_no=0, level="WARNING", category="dependency"),
        ]
        await db.save_logs_batch(logs)

        node_logs = await db.get_node_logs(node_id)
        assert [log.sequence_no for log in node_logs] == [0, 1]

        error_logs = await db.get_node_logs(node_id, level="ERROR")
        assert len(error_logs) == 1
        assert error_logs[0].category == "framework"

    @pytest.mark.asyncio
    async def test_get_deployment_logs_filters_level_and_category(self) -> None:
        db = MemoryDatabase()
        deployment_id = uuid4()
        other_deployment_id = uuid4()
        await db.save_logs_batch([
            _make_log(_deployment_id=deployment_id, level="INFO", category="lifecycle"),
            _make_log(_deployment_id=deployment_id, level="WARNING", category="dependency"),
            _make_log(_deployment_id=other_deployment_id, level="ERROR", category="framework"),
        ])

        deployment_logs = await db.get_deployment_logs(deployment_id)
        assert len(deployment_logs) == 2

        dependency_logs = await db.get_deployment_logs(deployment_id, category="dependency")
        assert len(dependency_logs) == 1
        assert dependency_logs[0].level == "WARNING"


class TestDocumentsByNode:
    @pytest.mark.asyncio
    async def test_returns_documents_produced_by_node(self) -> None:
        db = MemoryDatabase()
        node_id = uuid4()
        doc = _make_document(document_sha256="nd1", producing_node_id=node_id)
        other_doc = _make_document(document_sha256="nd2")
        await db.save_document_batch([doc, other_doc])

        results = await db.get_documents_by_node(node_id)
        assert len(results) == 1
        assert results[0].document_sha256 == "nd1"

    @pytest.mark.asyncio
    async def test_deployment_node_includes_root_inputs(self) -> None:
        db = MemoryDatabase()
        deploy_id = uuid4()
        deploy_node = _make_node(node_kind=NodeKind.DEPLOYMENT, _deployment_id=deploy_id)
        await db.insert_node(deploy_node)

        # Root input (no producing node)
        root_doc = _make_document(document_sha256="root_in", producing_node_id=None, deployment_id=deploy_id)
        # Task-produced doc
        task_doc = _make_document(document_sha256="task_out", producing_node_id=deploy_node.node_id, deployment_id=deploy_id)
        await db.save_document_batch([root_doc, task_doc])

        results = await db.get_documents_by_node(deploy_node.node_id)
        sha_set = {d.document_sha256 for d in results}
        assert "root_in" in sha_set
        assert "task_out" in sha_set


class TestGetAllDocumentShasForDeployment:
    @pytest.mark.asyncio
    async def test_collects_all_sha256s_from_nodes(self) -> None:
        db = MemoryDatabase()
        deploy_id = uuid4()
        node1 = _make_node(
            _deployment_id=deploy_id,
            input_document_shas=("sha_a", "sha_b"),
            output_document_shas=("sha_c",),
            context_document_shas=(),
        )
        node2 = _make_node(
            _deployment_id=deploy_id,
            input_document_shas=("sha_b",),  # duplicate
            output_document_shas=(),
            context_document_shas=("sha_d", "sha_e"),
        )
        # Different deployment - should be excluded
        other_node = _make_node(
            input_document_shas=("sha_f",),
        )
        await db.insert_node(node1)
        await db.insert_node(node2)
        await db.insert_node(other_node)

        shas = await db.get_all_document_shas_for_deployment(deploy_id)
        assert shas == {"sha_a", "sha_b", "sha_c", "sha_d", "sha_e"}

    @pytest.mark.asyncio
    async def test_empty_for_unknown_deployment(self) -> None:
        db = MemoryDatabase()
        shas = await db.get_all_document_shas_for_deployment(uuid4())
        assert shas == set()


# --- Blob tests ---


class TestBlobs:
    @pytest.mark.asyncio
    async def test_save_and_get(self) -> None:
        db = MemoryDatabase()
        blob = _make_blob(content_sha256="blob1", content=b"hello")
        await db.save_blob(blob)
        result = await db.get_blob("blob1")
        assert result is not None
        assert result.content == b"hello"

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self) -> None:
        db = MemoryDatabase()
        assert await db.get_blob("nonexistent") is None

    @pytest.mark.asyncio
    async def test_save_batch(self) -> None:
        db = MemoryDatabase()
        blobs = [_make_blob(content_sha256=f"bb_{i}") for i in range(3)]
        await db.save_blob_batch(blobs)
        for i in range(3):
            assert await db.get_blob(f"bb_{i}") is not None

    @pytest.mark.asyncio
    async def test_get_blobs_batch(self) -> None:
        db = MemoryDatabase()
        blobs = [_make_blob(content_sha256=f"gb_{i}") for i in range(2)]
        await db.save_blob_batch(blobs)
        result = await db.get_blobs_batch(["gb_0", "gb_1", "gb_missing"])
        assert len(result) == 2
        assert "gb_missing" not in result


# --- Cache completion tests ---


class TestCachedCompletion:
    @pytest.mark.asyncio
    async def test_finds_completed_node_by_cache_key(self) -> None:
        db = MemoryDatabase()
        node = _make_node(
            cache_key="task:fingerprint:abc",
            status=NodeStatus.COMPLETED,
            ended_at=datetime.now(UTC),
        )
        await db.insert_node(node)
        result = await db.get_cached_completion("task:fingerprint:abc")
        assert result is not None
        assert result.node_id == node.node_id

    @pytest.mark.asyncio
    async def test_ignores_running_nodes(self) -> None:
        db = MemoryDatabase()
        node = _make_node(cache_key="running_key", status=NodeStatus.RUNNING)
        await db.insert_node(node)
        assert await db.get_cached_completion("running_key") is None

    @pytest.mark.asyncio
    async def test_ignores_failed_nodes(self) -> None:
        db = MemoryDatabase()
        node = _make_node(cache_key="failed_key", status=NodeStatus.FAILED, ended_at=datetime.now(UTC))
        await db.insert_node(node)
        assert await db.get_cached_completion("failed_key") is None

    @pytest.mark.asyncio
    async def test_respects_max_age(self) -> None:
        db = MemoryDatabase()
        old_time = datetime.now(UTC) - timedelta(hours=2)
        node = _make_node(
            cache_key="old_key",
            status=NodeStatus.COMPLETED,
            ended_at=old_time,
        )
        await db.insert_node(node)
        # Within max_age: should find
        assert await db.get_cached_completion("old_key", max_age=timedelta(hours=3)) is not None
        # Outside max_age: should not find
        assert await db.get_cached_completion("old_key", max_age=timedelta(hours=1)) is None

    @pytest.mark.asyncio
    async def test_no_max_age_returns_any_completed(self) -> None:
        db = MemoryDatabase()
        old_time = datetime.now(UTC) - timedelta(days=30)
        node = _make_node(
            cache_key="ancient_key",
            status=NodeStatus.COMPLETED,
            ended_at=old_time,
        )
        await db.insert_node(node)
        assert await db.get_cached_completion("ancient_key") is not None

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown_key(self) -> None:
        db = MemoryDatabase()
        assert await db.get_cached_completion("nonexistent") is None


# --- Flush/shutdown ---


class TestFlushShutdown:
    @pytest.mark.asyncio
    async def test_flush_is_noop(self) -> None:
        db = MemoryDatabase()
        await db.flush()

    @pytest.mark.asyncio
    async def test_shutdown_is_noop(self) -> None:
        db = MemoryDatabase()
        await db.shutdown()


class TestUpdateNodeExplicitUpdatedAt:
    @pytest.mark.asyncio
    async def test_explicit_updated_at_is_preserved(self) -> None:
        db = MemoryDatabase()
        node = _make_node()
        await db.insert_node(node)
        explicit_time = datetime(2025, 1, 1, tzinfo=UTC)
        await db.update_node(node.node_id, updated_at=explicit_time)
        result = await db.get_node(node.node_id)
        assert result is not None
        assert result.updated_at == explicit_time


class TestIdempotentSaveDocument:
    @pytest.mark.asyncio
    async def test_save_same_sha_overwrites_without_error(self) -> None:
        db = MemoryDatabase()
        doc1 = _make_document(document_sha256="same_sha", name="original")
        await db.save_document(doc1)
        doc2 = _make_document(document_sha256="same_sha", name="replacement")
        await db.save_document(doc2)
        result = await db.get_document("same_sha")
        assert result is not None
        assert result.name == "replacement"


class TestGetCachedCompletionNewest:
    @pytest.mark.asyncio
    async def test_returns_newest_when_multiple_match(self) -> None:
        db = MemoryDatabase()
        old_time = datetime(2025, 1, 1, tzinfo=UTC)
        new_time = datetime(2025, 6, 1, tzinfo=UTC)
        old_node = _make_node(
            cache_key="shared_key",
            status=NodeStatus.COMPLETED,
            ended_at=old_time,
        )
        new_node = _make_node(
            cache_key="shared_key",
            status=NodeStatus.COMPLETED,
            ended_at=new_time,
        )
        await db.insert_node(old_node)
        await db.insert_node(new_node)
        result = await db.get_cached_completion("shared_key")
        assert result is not None
        assert result.node_id == new_node.node_id

    @pytest.mark.asyncio
    async def test_max_age_skips_nodes_without_ended_at(self) -> None:
        db = MemoryDatabase()
        node = _make_node(
            cache_key="no_end",
            status=NodeStatus.COMPLETED,
            ended_at=None,
        )
        await db.insert_node(node)
        assert await db.get_cached_completion("no_end", max_age=timedelta(hours=1)) is None
        # Without max_age, it should still be found
        assert await db.get_cached_completion("no_end") is not None


class TestGetDeploymentByRunScope:
    @pytest.mark.asyncio
    async def test_finds_deployment_by_scope(self) -> None:
        db = MemoryDatabase()
        deploy = _make_node(node_kind=NodeKind.DEPLOYMENT, run_scope="prod/my-pipeline")
        task = _make_node(node_kind=NodeKind.TASK, run_scope="prod/my-pipeline")
        await db.insert_node(deploy)
        await db.insert_node(task)
        result = await db.get_deployment_by_run_scope("prod/my-pipeline")
        assert result is not None
        assert result.node_kind == NodeKind.DEPLOYMENT

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown_scope(self) -> None:
        db = MemoryDatabase()
        assert await db.get_deployment_by_run_scope("nonexistent") is None
