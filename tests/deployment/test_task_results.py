"""Database node operation tests.

Covers: insert_node, query_node, update_node, status transitions,
node isolation, and hierarchy queries.
"""

from uuid import uuid4

import pytest

from ai_pipeline_core.database import NULL_PARENT, ExecutionNode, MemoryDatabase, NodeKind, NodeStatus


def _make_node(
    *,
    node_kind: NodeKind = NodeKind.TASK,
    status: NodeStatus = NodeStatus.RUNNING,
    name: str = "test-node",
    parent_node_id=NULL_PARENT,
    deployment_id=None,
    sequence_no: int = 0,
) -> ExecutionNode:
    """Create an ExecutionNode with sensible defaults for testing."""
    dep_id = deployment_id or uuid4()
    return ExecutionNode(
        node_id=uuid4(),
        node_kind=node_kind,
        deployment_id=dep_id,
        root_deployment_id=dep_id,
        run_id=f"run-{uuid4().hex[:8]}",
        run_scope=f"scope-{uuid4().hex[:8]}",
        deployment_name="test-deployment",
        name=name,
        sequence_no=sequence_no,
        parent_node_id=parent_node_id,
        status=status,
    )


class TestInsertAndQueryNode:
    """MemoryDatabase insert_node and get_node operations."""

    async def test_insert_and_retrieve(self) -> None:
        db = MemoryDatabase()
        node = _make_node(name="my-task")
        await db.insert_node(node)
        result = await db.get_node(node.node_id)
        assert result is not None
        assert result.node_id == node.node_id
        assert result.name == "my-task"

    async def test_get_nonexistent_returns_none(self) -> None:
        db = MemoryDatabase()
        result = await db.get_node(uuid4())
        assert result is None

    async def test_multiple_nodes_independent(self) -> None:
        db = MemoryDatabase()
        node_a = _make_node(name="task-a")
        node_b = _make_node(name="task-b")
        await db.insert_node(node_a)
        await db.insert_node(node_b)

        result_a = await db.get_node(node_a.node_id)
        result_b = await db.get_node(node_b.node_id)
        assert result_a is not None and result_a.name == "task-a"
        assert result_b is not None and result_b.name == "task-b"

    async def test_insert_preserves_all_fields(self) -> None:
        db = MemoryDatabase()
        node = _make_node(
            node_kind=NodeKind.FLOW,
            status=NodeStatus.RUNNING,
            name="full-field-test",
            sequence_no=7,
        )
        await db.insert_node(node)
        result = await db.get_node(node.node_id)
        assert result is not None
        assert result.node_kind == NodeKind.FLOW
        assert result.status == NodeStatus.RUNNING
        assert result.sequence_no == 7
        assert result.deployment_id == node.deployment_id
        assert result.run_id == node.run_id

    async def test_get_children_returns_child_nodes(self) -> None:
        db = MemoryDatabase()
        dep_id = uuid4()
        parent = _make_node(node_kind=NodeKind.FLOW, name="parent", deployment_id=dep_id)
        child_1 = _make_node(name="child-1", parent_node_id=parent.node_id, deployment_id=dep_id, sequence_no=1)
        child_2 = _make_node(name="child-2", parent_node_id=parent.node_id, deployment_id=dep_id, sequence_no=2)
        await db.insert_node(parent)
        await db.insert_node(child_1)
        await db.insert_node(child_2)

        children = await db.get_children(parent.node_id)
        assert len(children) == 2
        assert children[0].name == "child-1"
        assert children[1].name == "child-2"

    async def test_get_children_empty_when_no_children(self) -> None:
        db = MemoryDatabase()
        node = _make_node(name="leaf")
        await db.insert_node(node)

        children = await db.get_children(node.node_id)
        assert children == []


class TestUpdateNode:
    """MemoryDatabase update_node operations."""

    async def test_update_status(self) -> None:
        db = MemoryDatabase()
        node = _make_node(status=NodeStatus.RUNNING)
        await db.insert_node(node)

        await db.update_node(node.node_id, status=NodeStatus.COMPLETED)
        result = await db.get_node(node.node_id)
        assert result is not None
        assert result.status == NodeStatus.COMPLETED

    async def test_update_sets_updated_at(self) -> None:
        db = MemoryDatabase()
        node = _make_node()
        await db.insert_node(node)
        original_updated = node.updated_at

        await db.update_node(node.node_id, status=NodeStatus.COMPLETED)
        result = await db.get_node(node.node_id)
        assert result is not None
        assert result.updated_at >= original_updated

    async def test_update_nonexistent_raises(self) -> None:
        db = MemoryDatabase()
        with pytest.raises(KeyError):
            await db.update_node(uuid4(), status=NodeStatus.FAILED)

    async def test_update_error_fields(self) -> None:
        db = MemoryDatabase()
        node = _make_node(status=NodeStatus.RUNNING)
        await db.insert_node(node)

        await db.update_node(
            node.node_id,
            status=NodeStatus.FAILED,
            error_type="ValueError",
            error_message="something went wrong",
        )
        result = await db.get_node(node.node_id)
        assert result is not None
        assert result.status == NodeStatus.FAILED
        assert result.error_type == "ValueError"
        assert result.error_message == "something went wrong"

    async def test_update_preserves_unchanged_fields(self) -> None:
        db = MemoryDatabase()
        node = _make_node(name="keep-this", node_kind=NodeKind.TASK)
        await db.insert_node(node)

        await db.update_node(node.node_id, status=NodeStatus.COMPLETED)
        result = await db.get_node(node.node_id)
        assert result is not None
        assert result.name == "keep-this"
        assert result.node_kind == NodeKind.TASK
        assert result.run_id == node.run_id

    async def test_update_does_not_affect_other_nodes(self) -> None:
        db = MemoryDatabase()
        node_a = _make_node(name="node-a")
        node_b = _make_node(name="node-b")
        await db.insert_node(node_a)
        await db.insert_node(node_b)

        await db.update_node(node_a.node_id, status=NodeStatus.FAILED)
        result_b = await db.get_node(node_b.node_id)
        assert result_b is not None
        assert result_b.status == NodeStatus.RUNNING


class TestNodeStatusTransitions:
    """Node status transitions via update_node."""

    async def test_running_to_completed(self) -> None:
        db = MemoryDatabase()
        node = _make_node(status=NodeStatus.RUNNING)
        await db.insert_node(node)

        await db.update_node(node.node_id, status=NodeStatus.COMPLETED)
        result = await db.get_node(node.node_id)
        assert result is not None
        assert result.status == NodeStatus.COMPLETED

    async def test_running_to_failed(self) -> None:
        db = MemoryDatabase()
        node = _make_node(status=NodeStatus.RUNNING)
        await db.insert_node(node)

        await db.update_node(node.node_id, status=NodeStatus.FAILED)
        result = await db.get_node(node.node_id)
        assert result is not None
        assert result.status == NodeStatus.FAILED

    async def test_running_to_skipped(self) -> None:
        db = MemoryDatabase()
        node = _make_node(status=NodeStatus.RUNNING)
        await db.insert_node(node)

        await db.update_node(node.node_id, status=NodeStatus.SKIPPED)
        result = await db.get_node(node.node_id)
        assert result is not None
        assert result.status == NodeStatus.SKIPPED

    async def test_insert_with_cached_status(self) -> None:
        db = MemoryDatabase()
        node = _make_node(status=NodeStatus.CACHED)
        await db.insert_node(node)

        result = await db.get_node(node.node_id)
        assert result is not None
        assert result.status == NodeStatus.CACHED

    async def test_all_status_values_accepted(self) -> None:
        db = MemoryDatabase()
        for status in NodeStatus:
            node = _make_node(status=status, name=f"node-{status.value}")
            await db.insert_node(node)
            result = await db.get_node(node.node_id)
            assert result is not None
            assert result.status == status


class TestDatabaseIsolation:
    """Separate MemoryDatabase instances are independent."""

    async def test_separate_databases_independent(self) -> None:
        db1 = MemoryDatabase()
        db2 = MemoryDatabase()
        node = _make_node(name="only-in-db1")
        await db1.insert_node(node)

        result = await db2.get_node(node.node_id)
        assert result is None

    async def test_shutdown_is_noop(self) -> None:
        db = MemoryDatabase()
        node = _make_node()
        await db.insert_node(node)
        await db.shutdown()

        result = await db.get_node(node.node_id)
        assert result is not None

    async def test_flush_is_noop(self) -> None:
        db = MemoryDatabase()
        node = _make_node()
        await db.insert_node(node)
        await db.flush()

        result = await db.get_node(node.node_id)
        assert result is not None
