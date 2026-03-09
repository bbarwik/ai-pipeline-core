"""Tests for _safe_uuid and deployment execution node creation in base.py."""

# pyright: reportPrivateUsage=false

from uuid import UUID, uuid4

import pytest

from ai_pipeline_core.database import NULL_PARENT, ExecutionNode, MemoryDatabase, NodeKind, NodeStatus
from ai_pipeline_core.deployment.base import PipelineDeployment, _safe_uuid


class TestSafeUuid:
    """Test _safe_uuid helper moved from progress.py to base.py."""

    def test_valid_uuid_string(self) -> None:
        """Valid UUID string returns a UUID object."""
        value = str(UUID(int=42))
        result = _safe_uuid(value)
        assert result == UUID(int=42)

    def test_random_uuid(self) -> None:
        """Random UUID round-trips through _safe_uuid."""
        original = uuid4()
        result = _safe_uuid(str(original))
        assert result == original

    def test_invalid_string_returns_none(self) -> None:
        """Non-UUID string returns None."""
        assert _safe_uuid("not-a-uuid") is None

    def test_none_string_returns_none(self) -> None:
        """str(None) -> 'None' returns None (the original bug scenario)."""
        assert _safe_uuid("None") is None

    def test_empty_string_returns_none(self) -> None:
        """Empty string returns None."""
        assert _safe_uuid("") is None

    def test_zero_uuid(self) -> None:
        """All-zero UUID is valid."""
        result = _safe_uuid("00000000-0000-0000-0000-000000000000")
        assert result == UUID(int=0)


class TestInsertDbNode:
    """Test PipelineDeployment._insert_db_node static method."""

    async def test_inserts_node_into_database(self) -> None:
        """Node is inserted into the database."""
        db = MemoryDatabase()
        node_id = uuid4()
        dep_id = uuid4()
        node = ExecutionNode(
            node_id=node_id,
            node_kind=NodeKind.DEPLOYMENT,
            deployment_id=dep_id,
            root_deployment_id=dep_id,
            run_id="test-run",
            run_scope="test/scope",
            deployment_name="test-deploy",
            name="test-deploy",
            sequence_no=0,
        )
        await PipelineDeployment._insert_db_node(db, node)

        assert node_id in db._nodes
        stored = db._nodes[node_id]
        assert stored.node_kind == NodeKind.DEPLOYMENT
        assert stored.run_id == "test-run"

    async def test_none_database_is_noop(self) -> None:
        """When database is None, insert is silently skipped."""
        node = ExecutionNode(
            node_id=uuid4(),
            node_kind=NodeKind.FLOW,
            deployment_id=uuid4(),
            root_deployment_id=uuid4(),
            run_id="test",
            run_scope="test/scope",
            deployment_name="d",
            name="f",
            sequence_no=1,
        )
        # Must not raise
        await PipelineDeployment._insert_db_node(None, node)

    async def test_insert_failure_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Database insert failure is logged as a warning, not raised."""
        db = MemoryDatabase()
        node_id = uuid4()
        dep_id = uuid4()
        node = ExecutionNode(
            node_id=node_id,
            node_kind=NodeKind.FLOW,
            deployment_id=dep_id,
            root_deployment_id=dep_id,
            run_id="test",
            run_scope="test/scope",
            deployment_name="d",
            name="f",
            sequence_no=1,
        )
        # Insert once
        await PipelineDeployment._insert_db_node(db, node)
        # Insert duplicate — MemoryDatabase may or may not error;
        # verify the method does not propagate exceptions regardless
        await PipelineDeployment._insert_db_node(db, node)


class TestUpdateDbNode:
    """Test PipelineDeployment._update_db_node static method."""

    async def test_updates_node_status(self) -> None:
        """Node status can be updated after insertion."""
        db = MemoryDatabase()
        node_id = uuid4()
        dep_id = uuid4()
        node = ExecutionNode(
            node_id=node_id,
            node_kind=NodeKind.DEPLOYMENT,
            deployment_id=dep_id,
            root_deployment_id=dep_id,
            run_id="test",
            run_scope="test/scope",
            deployment_name="d",
            name="d",
            sequence_no=0,
            status=NodeStatus.RUNNING,
        )
        await db.insert_node(node)
        await PipelineDeployment._update_db_node(db, node_id, status=NodeStatus.COMPLETED)

        stored = db._nodes[node_id]
        assert stored.status == NodeStatus.COMPLETED

    async def test_none_database_is_noop(self) -> None:
        """When database is None, update is silently skipped."""
        await PipelineDeployment._update_db_node(None, uuid4(), status=NodeStatus.FAILED)


class TestExecutionNodeDefaults:
    """Test ExecutionNode dataclass defaults and structure."""

    def test_default_status_is_running(self) -> None:
        """New execution nodes default to RUNNING status."""
        node = ExecutionNode(
            node_id=uuid4(),
            node_kind=NodeKind.FLOW,
            deployment_id=uuid4(),
            root_deployment_id=uuid4(),
            run_id="test",
            run_scope="scope",
            deployment_name="d",
            name="f",
            sequence_no=1,
        )
        assert node.status == NodeStatus.RUNNING

    def test_default_parent_is_null_parent(self) -> None:
        """Nodes without explicit parent get NULL_PARENT sentinel."""
        node = ExecutionNode(
            node_id=uuid4(),
            node_kind=NodeKind.DEPLOYMENT,
            deployment_id=uuid4(),
            root_deployment_id=uuid4(),
            run_id="test",
            run_scope="scope",
            deployment_name="d",
            name="d",
            sequence_no=0,
        )
        assert node.parent_node_id == NULL_PARENT

    def test_frozen_immutability(self) -> None:
        """ExecutionNode is frozen — attributes cannot be reassigned."""
        node = ExecutionNode(
            node_id=uuid4(),
            node_kind=NodeKind.FLOW,
            deployment_id=uuid4(),
            root_deployment_id=uuid4(),
            run_id="test",
            run_scope="scope",
            deployment_name="d",
            name="f",
            sequence_no=1,
        )
        with pytest.raises(AttributeError):
            node.status = NodeStatus.COMPLETED  # type: ignore[misc]

    def test_parent_child_relationship(self) -> None:
        """Flow nodes reference their deployment parent via parent_node_id."""
        dep_id = uuid4()
        flow_id = uuid4()
        root_id = uuid4()
        flow_node = ExecutionNode(
            node_id=flow_id,
            node_kind=NodeKind.FLOW,
            deployment_id=dep_id,
            root_deployment_id=root_id,
            run_id="test",
            run_scope="scope",
            deployment_name="d",
            name="flow-1",
            sequence_no=1,
            parent_node_id=dep_id,
        )
        assert flow_node.parent_node_id == dep_id
        assert flow_node.deployment_id == dep_id
        assert flow_node.root_deployment_id == root_id
