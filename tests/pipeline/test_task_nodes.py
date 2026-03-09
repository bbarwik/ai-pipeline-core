"""Tests for task execution node creation in the database execution DAG.

Verifies that PipelineTask._execute_invocation creates:
- A RUNNING task node on start
- Updates to COMPLETED with input/output shas on success
- Updates to FAILED with error info on failure
- Document persistence via dual-write to new database
- Conversation turn nodes from captured ConversationTurnData
"""

# pyright: reportPrivateUsage=false

import asyncio
import logging
from types import MappingProxyType
from uuid import UUID, uuid4

import pytest

from ai_pipeline_core.database import MemoryDatabase, NodeKind, NodeStatus
from ai_pipeline_core.deployment._types import _NoopPublisher
from ai_pipeline_core.documents import Document
from ai_pipeline_core.logging import ExecutionLogBuffer, ExecutionLogHandler
from ai_pipeline_core.pipeline import PipelineTask
from ai_pipeline_core.pipeline._execution_context import (
    ExecutionContext,
    FlowFrame,
    reset_execution_context,
    set_execution_context,
)
from ai_pipeline_core.pipeline.limits import _SharedStatus


# --- Test document types ---


class _NodeInDoc(Document):
    """Input document for task node tests."""


class _NodeOutDoc(Document):
    """Output document for task node tests."""


# --- Test tasks ---


class _SimpleTask(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[_NodeInDoc, ...]) -> tuple[_NodeOutDoc, ...]:
        return (_NodeOutDoc.derive(from_documents=tuple(documents), name="out.txt", content="output"),)


class _FailingTask(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[_NodeInDoc, ...]) -> tuple[_NodeOutDoc, ...]:
        raise ValueError("deliberate task failure")


class _LongFailingTask(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[_NodeInDoc, ...]) -> tuple[_NodeOutDoc, ...]:
        raise ValueError("x" * 4000)


class _CancelledTask(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[_NodeInDoc, ...]) -> tuple[_NodeOutDoc, ...]:
        raise asyncio.CancelledError("task cancelled")


class _EmptyResultTask(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[_NodeInDoc, ...]) -> tuple[_NodeOutDoc, ...]:
        return ()


class _NestedChildTask(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[_NodeInDoc, ...]) -> tuple[_NodeOutDoc, ...]:
        return (_NodeOutDoc.derive(from_documents=tuple(documents), name="child-out.txt", content="child"),)


class _NestedParentTask(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[_NodeInDoc, ...]) -> tuple[_NodeOutDoc, ...]:
        return await _NestedChildTask.run(documents)


# --- Helpers ---


def _make_input() -> _NodeInDoc:
    return _NodeInDoc(name="input.txt", content=b"test-input")


def _make_flow_frame() -> FlowFrame:
    return FlowFrame(
        name="test-flow",
        flow_class_name="TestFlow",
        step=1,
        total_steps=1,
        flow_minutes=(1.0,),
        completed_minutes=0.0,
        flow_params={},
    )


def _make_context_with_db(
    db: MemoryDatabase,
    *,
    deployment_id: UUID | None = None,
    flow_node_id: UUID | None = None,
    log_buffer: ExecutionLogBuffer | None = None,
) -> ExecutionContext:
    dep_id = deployment_id or uuid4()
    ctx = ExecutionContext(
        run_id="test-run",
        run_scope="test-run/scope",
        execution_id=None,
        publisher=_NoopPublisher(),
        limits=MappingProxyType({}),
        limits_status=_SharedStatus(),
        database=db,
        deployment_id=dep_id,
        root_deployment_id=dep_id,
        deployment_name="test-pipeline",
        flow_frame=_make_flow_frame(),
        current_node_id=flow_node_id or uuid4(),
        log_buffer=log_buffer,
    )
    return ctx


# --- Tests ---


@pytest.fixture
def db() -> MemoryDatabase:
    return MemoryDatabase()


class TestTaskNodeCreation:
    """Test that task execution creates task nodes in the database."""

    @pytest.mark.asyncio
    async def test_successful_task_creates_completed_node(self, db: MemoryDatabase) -> None:
        ctx = _make_context_with_db(db)
        token = set_execution_context(ctx)
        try:
            await _SimpleTask.run((_make_input(),))
        finally:
            reset_execution_context(token)

        task_nodes = [n for n in db._nodes.values() if n.node_kind == NodeKind.TASK]
        assert len(task_nodes) == 1
        node = task_nodes[0]
        assert node.status == NodeStatus.COMPLETED
        assert node.task_class == "_SimpleTask"
        assert node.deployment_id == ctx.deployment_id
        assert node.ended_at is not None

    @pytest.mark.asyncio
    async def test_task_node_has_correct_parent(self, db: MemoryDatabase) -> None:
        flow_node_id = uuid4()
        ctx = _make_context_with_db(db, flow_node_id=flow_node_id)
        token = set_execution_context(ctx)
        try:
            await _SimpleTask.run((_make_input(),))
        finally:
            reset_execution_context(token)

        task_node = [n for n in db._nodes.values() if n.node_kind == NodeKind.TASK][0]
        assert task_node.parent_node_id == flow_node_id

    @pytest.mark.asyncio
    async def test_task_node_has_input_output_shas(self, db: MemoryDatabase) -> None:
        ctx = _make_context_with_db(db)
        token = set_execution_context(ctx)
        input_doc = _make_input()
        try:
            result = await _SimpleTask.run((input_doc,))
        finally:
            reset_execution_context(token)

        task_node = [n for n in db._nodes.values() if n.node_kind == NodeKind.TASK][0]
        assert input_doc.sha256 in task_node.input_document_shas
        assert len(task_node.output_document_shas) == 1
        assert result[0].sha256 in task_node.output_document_shas

    @pytest.mark.asyncio
    async def test_empty_result_has_no_output_shas(self, db: MemoryDatabase) -> None:
        ctx = _make_context_with_db(db)
        token = set_execution_context(ctx)
        try:
            await _EmptyResultTask.run((_make_input(),))
        finally:
            reset_execution_context(token)

        task_node = [n for n in db._nodes.values() if n.node_kind == NodeKind.TASK][0]
        assert task_node.output_document_shas == ()

    @pytest.mark.asyncio
    async def test_task_node_payload_includes_log_summary(self, db: MemoryDatabase) -> None:
        root_logger = logging.getLogger()
        handler = next((item for item in root_logger.handlers if isinstance(item, ExecutionLogHandler)), None)
        added_handler = False
        if handler is None:
            handler = ExecutionLogHandler()
            root_logger.addHandler(handler)
            added_handler = True
        buffer = ExecutionLogBuffer()
        ctx = _make_context_with_db(db, log_buffer=buffer)
        token = set_execution_context(ctx)
        try:
            await _SimpleTask.run((_make_input(),))
        finally:
            reset_execution_context(token)
            if added_handler:
                root_logger.removeHandler(handler)

        task_node = [n for n in db._nodes.values() if n.node_kind == NodeKind.TASK][0]
        assert task_node.payload["log_summary"]["total"] == 2
        assert task_node.payload["log_summary"]["errors"] == 0

    @pytest.mark.asyncio
    async def test_task_completion_prunes_log_summary_state(self, db: MemoryDatabase) -> None:
        root_logger = logging.getLogger()
        handler = next((item for item in root_logger.handlers if isinstance(item, ExecutionLogHandler)), None)
        added_handler = False
        if handler is None:
            handler = ExecutionLogHandler()
            root_logger.addHandler(handler)
            added_handler = True
        buffer = ExecutionLogBuffer()
        ctx = _make_context_with_db(db, log_buffer=buffer)
        token = set_execution_context(ctx)
        try:
            await _SimpleTask.run((_make_input(),))
        finally:
            reset_execution_context(token)
            if added_handler:
                root_logger.removeHandler(handler)

        task_node = [n for n in db._nodes.values() if n.node_kind == NodeKind.TASK][0]
        assert buffer.get_summary(task_node.node_id) == {
            "total": 0,
            "warnings": 0,
            "errors": 0,
            "last_error": "",
        }


class TestTaskNodeFailure:
    """Test that failed tasks create FAILED nodes with error info."""

    @pytest.mark.asyncio
    async def test_failed_task_creates_failed_node(self, db: MemoryDatabase) -> None:
        ctx = _make_context_with_db(db)
        token = set_execution_context(ctx)
        try:
            with pytest.raises(ValueError, match="deliberate task failure"):
                await _FailingTask.run((_make_input(),))
        finally:
            reset_execution_context(token)

        task_nodes = [n for n in db._nodes.values() if n.node_kind == NodeKind.TASK]
        assert len(task_nodes) == 1
        node = task_nodes[0]
        assert node.status == NodeStatus.FAILED
        assert node.error_type == "ValueError"
        assert "deliberate task failure" in node.error_message
        assert node.ended_at is not None

    @pytest.mark.asyncio
    async def test_cancelled_task_creates_failed_node(self, db: MemoryDatabase) -> None:
        ctx = _make_context_with_db(db)
        token = set_execution_context(ctx)
        try:
            with pytest.raises(asyncio.CancelledError):
                await _CancelledTask.run((_make_input(),))
        finally:
            reset_execution_context(token)

        task_nodes = [n for n in db._nodes.values() if n.node_kind == NodeKind.TASK]
        assert len(task_nodes) == 1
        node = task_nodes[0]
        assert node.status == NodeStatus.FAILED
        assert node.error_type == "CancelledError"
        assert node.ended_at is not None

    @pytest.mark.asyncio
    async def test_failed_task_preserves_full_error_message(self, db: MemoryDatabase) -> None:
        ctx = _make_context_with_db(db)
        token = set_execution_context(ctx)
        try:
            with pytest.raises(ValueError, match=r"x{10}"):
                await _LongFailingTask.run((_make_input(),))
        finally:
            reset_execution_context(token)

        task_node = [n for n in db._nodes.values() if n.node_kind == NodeKind.TASK][0]
        assert task_node.status == NodeStatus.FAILED
        assert task_node.error_message == "x" * 4000


class TestTaskNodeHierarchy:
    """Test that nested tasks create proper parent-child relationships."""

    @pytest.mark.asyncio
    async def test_nested_tasks_have_parent_child(self, db: MemoryDatabase) -> None:
        ctx = _make_context_with_db(db)
        token = set_execution_context(ctx)
        try:
            await _NestedParentTask.run((_make_input(),))
        finally:
            reset_execution_context(token)

        task_nodes = [n for n in db._nodes.values() if n.node_kind == NodeKind.TASK]
        assert len(task_nodes) == 2

        parent_node = next(n for n in task_nodes if n.task_class == "_NestedParentTask")
        child_node = next(n for n in task_nodes if n.task_class == "_NestedChildTask")

        # Child's parent is the parent task node
        assert child_node.parent_node_id == parent_node.node_id


class TestDocumentDualWrite:
    """Test that documents are persisted to both old store and new database."""

    @pytest.mark.asyncio
    async def test_documents_persisted_to_database(self, db: MemoryDatabase) -> None:
        ctx = _make_context_with_db(db)
        token = set_execution_context(ctx)
        try:
            await _SimpleTask.run((_make_input(),))
        finally:
            reset_execution_context(token)

        # Check document records were created
        assert len(db._documents) > 0

        # Check blob records were created
        assert len(db._blobs) > 0

    @pytest.mark.asyncio
    async def test_persisted_documents_include_execution_run_scope(self, db: MemoryDatabase) -> None:
        ctx = _make_context_with_db(db)
        token = set_execution_context(ctx)
        try:
            await _SimpleTask.run((_make_input(),))
        finally:
            reset_execution_context(token)

        assert db._documents
        assert {doc.run_scope for doc in db._documents.values()} == {ctx.run_scope}


class TestNoDatabase:
    """Test that tasks work without a database (graceful degradation)."""

    @pytest.mark.asyncio
    async def test_task_works_without_database(self) -> None:
        """When no database is on the context, tasks run normally."""
        ctx = ExecutionContext(
            run_id="test-run",
            run_scope="test-run/scope",
            execution_id=None,
            publisher=_NoopPublisher(),
            limits=MappingProxyType({}),
            limits_status=_SharedStatus(),
            database=None,
            deployment_id=None,
        )
        token = set_execution_context(ctx)
        try:
            result = await _SimpleTask.run((_make_input(),))
            assert len(result) == 1
        finally:
            reset_execution_context(token)
