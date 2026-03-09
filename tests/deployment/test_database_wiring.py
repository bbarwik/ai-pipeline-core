"""Tests for database execution DAG wiring in PipelineDeployment.

Verifies that deployment.run() creates execution nodes in the database:
- Deployment node on start
- Flow nodes for each flow (started, completed, failed, skipped, cached)
- Deployment node updated on completion/failure
- _run_with_context() accepts pre-allocated IDs
"""

# pyright: reportPrivateUsage=false

import asyncio
from collections.abc import Sequence
from typing import Any
from uuid import UUID, uuid4

import pytest
from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness

from ai_pipeline_core import DeploymentResult, Document, FlowOptions, PipelineDeployment
from ai_pipeline_core.database import MemoryDatabase, NodeKind, NodeStatus
from ai_pipeline_core.pipeline import PipelineFlow, PipelineTask


# --- Test document types ---


class WireInputDoc(Document):
    pass


class WireMiddleDoc(Document):
    pass


class WireOutputDoc(Document):
    pass


# --- Test tasks ---


class WireToMiddleTask(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[WireInputDoc, ...]) -> tuple[WireMiddleDoc, ...]:
        return (WireMiddleDoc.derive(from_documents=(documents[0],), name="middle.txt", content="m"),)


class WireToOutputTask(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[WireMiddleDoc, ...]) -> tuple[WireOutputDoc, ...]:
        return (WireOutputDoc.derive(from_documents=(documents[0],), name="output.txt", content="o"),)


class WireFailingTask(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[WireMiddleDoc, ...]) -> tuple[WireOutputDoc, ...]:
        raise RuntimeError("deliberate test failure")


# --- Test flows ---


class WireFlowOne(PipelineFlow):
    async def run(self, documents: tuple[WireInputDoc, ...], options: FlowOptions) -> tuple[WireMiddleDoc, ...]:
        return await WireToMiddleTask.run(documents)


class WireFlowTwo(PipelineFlow):
    async def run(self, documents: tuple[WireMiddleDoc, ...], options: FlowOptions) -> tuple[WireOutputDoc, ...]:
        return await WireToOutputTask.run(documents)


class WireFailingFlowTwo(PipelineFlow):
    async def run(self, documents: tuple[WireMiddleDoc, ...], options: FlowOptions) -> tuple[WireOutputDoc, ...]:
        return await WireFailingTask.run(documents)


# --- Test result ---


class WireResult(DeploymentResult):
    doc_count: int = 0


# --- Test deployments ---


class WireTwoStageDeployment(PipelineDeployment[FlowOptions, WireResult]):
    def build_flows(self, options: FlowOptions) -> Sequence[PipelineFlow]:
        return [WireFlowOne(), WireFlowTwo()]

    @staticmethod
    def build_result(run_id: str, documents: tuple[Document, ...], options: FlowOptions) -> WireResult:
        return WireResult(success=True, doc_count=len(documents))


class WireFailingDeployment(PipelineDeployment[FlowOptions, WireResult]):
    def build_flows(self, options: FlowOptions) -> Sequence[PipelineFlow]:
        return [WireFlowOne(), WireFailingFlowTwo()]

    @staticmethod
    def build_result(run_id: str, documents: tuple[Document, ...], options: FlowOptions) -> WireResult:
        return WireResult(success=False)


class WireSingleFlowDeployment(PipelineDeployment[FlowOptions, WireResult]):
    def build_flows(self, options: FlowOptions) -> Sequence[PipelineFlow]:
        return [WireFlowOne()]

    @staticmethod
    def build_result(run_id: str, documents: tuple[Document, ...], options: FlowOptions) -> WireResult:
        return WireResult(success=True, doc_count=len(documents))


class DuplicateMiddleFlowA(PipelineFlow):
    async def run(self, documents: tuple[WireInputDoc, ...], options: FlowOptions) -> tuple[WireMiddleDoc, ...]:
        return (WireMiddleDoc.derive(from_documents=(documents[0],), name="middle.txt", content="same"),)


class DuplicateMiddleFlowB(PipelineFlow):
    async def run(self, documents: tuple[WireInputDoc, ...], options: FlowOptions) -> tuple[WireMiddleDoc, ...]:
        return (WireMiddleDoc.derive(from_documents=(documents[0],), name="middle.txt", content="same"),)


class CountMiddleDocsFlow(PipelineFlow):
    async def run(self, documents: tuple[WireMiddleDoc, ...], options: FlowOptions) -> tuple[WireOutputDoc, ...]:
        return (WireOutputDoc.derive(from_documents=tuple(documents), name="count.txt", content=str(len(documents))),)


class DedupingDeployment(PipelineDeployment[FlowOptions, WireResult]):
    def build_flows(self, options: FlowOptions) -> Sequence[PipelineFlow]:
        return [DuplicateMiddleFlowA(), DuplicateMiddleFlowB(), CountMiddleDocsFlow()]

    @staticmethod
    def build_result(run_id: str, documents: tuple[Document, ...], options: FlowOptions) -> WireResult:
        outputs = [doc for doc in documents if isinstance(doc, WireOutputDoc)]
        return WireResult(success=True, doc_count=int(outputs[0].text))


# --- Helpers ---


def _make_input_doc() -> WireInputDoc:
    return WireInputDoc.create_root(name="input.txt", content="test", reason="wiring test")


async def _run_with_db(
    deployment: PipelineDeployment[Any, Any],
    database: MemoryDatabase,
    *,
    run_id: str = "wire-test",
    docs: list[Document] | None = None,
    deployment_node_id: UUID | None = None,
    root_deployment_id: UUID | None = None,
) -> Any:
    """Run deployment with injected MemoryDatabase."""
    if docs is None:
        docs = [_make_input_doc()]
    node_id = deployment_node_id or uuid4()
    root_id = root_deployment_id or node_id
    return await deployment._run_with_context(
        run_id,
        docs,
        FlowOptions(),
        deployment_node_id=node_id,
        root_deployment_id=root_id,
        database=database,
    )


# --- Tests ---


@pytest.fixture
def db() -> MemoryDatabase:
    return MemoryDatabase()


class TestDeploymentNodeCreation:
    """Test that run() creates a deployment ExecutionNode."""

    def test_deployment_node_created(self, db: MemoryDatabase) -> None:
        deployment = WireTwoStageDeployment()
        with prefect_test_harness(), disable_run_logger():
            asyncio.run(_run_with_db(deployment, db))

        nodes = list(db._nodes.values())
        deployment_nodes = [n for n in nodes if n.node_kind == NodeKind.DEPLOYMENT]
        assert len(deployment_nodes) == 1

        dep_node = deployment_nodes[0]
        assert dep_node.status == NodeStatus.COMPLETED
        assert dep_node.deployment_name == "wire-two-stage-deployment"
        assert dep_node.run_id == "wire-test"
        assert dep_node.ended_at is not None

    def test_deployment_node_has_flow_plan(self, db: MemoryDatabase) -> None:
        deployment = WireTwoStageDeployment()
        with prefect_test_harness(), disable_run_logger():
            asyncio.run(_run_with_db(deployment, db))

        dep_node = [n for n in db._nodes.values() if n.node_kind == NodeKind.DEPLOYMENT][0]
        assert "flow_plan" in dep_node.payload
        assert len(dep_node.payload["flow_plan"]) == 2

    def test_deployment_node_has_result_on_completion(self, db: MemoryDatabase) -> None:
        deployment = WireTwoStageDeployment()
        with prefect_test_harness(), disable_run_logger():
            asyncio.run(_run_with_db(deployment, db))

        dep_node = [n for n in db._nodes.values() if n.node_kind == NodeKind.DEPLOYMENT][0]
        assert "result" in dep_node.payload
        assert dep_node.payload["result"]["success"] is True

    def test_downstream_flows_receive_deduplicated_documents(self, db: MemoryDatabase) -> None:
        deployment = DedupingDeployment()
        with prefect_test_harness(), disable_run_logger():
            result = asyncio.run(_run_with_db(deployment, db))

        assert result.doc_count == 1

    def test_root_input_documents_persist_with_run_scope(self, db: MemoryDatabase) -> None:
        deployment = WireSingleFlowDeployment()
        with prefect_test_harness(), disable_run_logger():
            asyncio.run(_run_with_db(deployment, db))

        deployment_node = next(n for n in db._nodes.values() if n.node_kind == NodeKind.DEPLOYMENT)
        root_documents = [doc for doc in db._documents.values() if doc.producing_node_id is None]

        assert root_documents
        assert {doc.run_scope for doc in root_documents} == {deployment_node.run_scope}


class TestFlowNodeCreation:
    """Test that flow execution creates flow nodes."""

    def test_flow_nodes_created_as_children(self, db: MemoryDatabase) -> None:
        deployment = WireTwoStageDeployment()
        with prefect_test_harness(), disable_run_logger():
            asyncio.run(_run_with_db(deployment, db))

        nodes = list(db._nodes.values())
        flow_nodes = sorted(
            [n for n in nodes if n.node_kind == NodeKind.FLOW],
            key=lambda n: n.sequence_no,
        )
        assert len(flow_nodes) == 2

        dep_node = [n for n in nodes if n.node_kind == NodeKind.DEPLOYMENT][0]
        for fn in flow_nodes:
            assert fn.parent_node_id == dep_node.node_id
            assert fn.deployment_id == dep_node.deployment_id

    def test_flow_nodes_have_correct_sequence(self, db: MemoryDatabase) -> None:
        deployment = WireTwoStageDeployment()
        with prefect_test_harness(), disable_run_logger():
            asyncio.run(_run_with_db(deployment, db))

        flow_nodes = sorted(
            [n for n in db._nodes.values() if n.node_kind == NodeKind.FLOW],
            key=lambda n: n.sequence_no,
        )
        assert flow_nodes[0].sequence_no == 1
        assert flow_nodes[1].sequence_no == 2
        assert flow_nodes[0].name == "WireFlowOne"
        assert flow_nodes[1].name == "WireFlowTwo"

    def test_completed_flows_have_status_and_timing(self, db: MemoryDatabase) -> None:
        deployment = WireTwoStageDeployment()
        with prefect_test_harness(), disable_run_logger():
            asyncio.run(_run_with_db(deployment, db))

        flow_nodes = [n for n in db._nodes.values() if n.node_kind == NodeKind.FLOW]
        for fn in flow_nodes:
            assert fn.status == NodeStatus.COMPLETED
            assert fn.ended_at is not None
            assert fn.started_at <= fn.ended_at


class TestFlowFailure:
    """Test that flow failure updates the flow node with error info."""

    def test_failed_flow_has_error_info(self, db: MemoryDatabase) -> None:
        deployment = WireFailingDeployment()
        with prefect_test_harness(), disable_run_logger():
            with pytest.raises(RuntimeError, match="deliberate test failure"):
                asyncio.run(_run_with_db(deployment, db))

        flow_nodes = sorted(
            [n for n in db._nodes.values() if n.node_kind == NodeKind.FLOW],
            key=lambda n: n.sequence_no,
        )
        # First flow completed successfully
        assert flow_nodes[0].status == NodeStatus.COMPLETED

        # Second flow failed
        assert flow_nodes[1].status == NodeStatus.FAILED
        assert flow_nodes[1].error_type == "RuntimeError"
        assert "deliberate test failure" in flow_nodes[1].error_message

    def test_deployment_node_fails_when_flow_fails(self, db: MemoryDatabase) -> None:
        deployment = WireFailingDeployment()
        with prefect_test_harness(), disable_run_logger():
            with pytest.raises(RuntimeError):
                asyncio.run(_run_with_db(deployment, db))

        dep_node = [n for n in db._nodes.values() if n.node_kind == NodeKind.DEPLOYMENT][0]
        assert dep_node.status == NodeStatus.FAILED
        assert dep_node.error_type == "RuntimeError"
        assert dep_node.ended_at is not None


class TestRunWithContext:
    """Test that _run_with_context() accepts pre-allocated IDs."""

    def test_pre_allocated_ids(self, db: MemoryDatabase) -> None:
        deployment = WireSingleFlowDeployment()
        pre_id = uuid4()
        root_id = uuid4()
        with prefect_test_harness(), disable_run_logger():
            asyncio.run(
                _run_with_db(
                    deployment,
                    db,
                    deployment_node_id=pre_id,
                    root_deployment_id=root_id,
                )
            )

        dep_node = [n for n in db._nodes.values() if n.node_kind == NodeKind.DEPLOYMENT][0]
        assert dep_node.node_id == pre_id
        assert dep_node.deployment_id == pre_id
        assert dep_node.root_deployment_id == root_id

    def test_flow_nodes_inherit_deployment_ids(self, db: MemoryDatabase) -> None:
        deployment = WireSingleFlowDeployment()
        pre_id = uuid4()
        root_id = uuid4()
        with prefect_test_harness(), disable_run_logger():
            asyncio.run(
                _run_with_db(
                    deployment,
                    db,
                    deployment_node_id=pre_id,
                    root_deployment_id=root_id,
                )
            )

        flow_nodes = [n for n in db._nodes.values() if n.node_kind == NodeKind.FLOW]
        assert len(flow_nodes) == 1
        assert flow_nodes[0].deployment_id == pre_id
        assert flow_nodes[0].root_deployment_id == root_id


class TestExecutionContextFields:
    """Test that ExecutionContext carries database fields."""

    def test_execution_context_has_database_fields(self, db: MemoryDatabase) -> None:
        from ai_pipeline_core.pipeline._execution_context import ExecutionContext
        from ai_pipeline_core.deployment._types import _NoopPublisher
        from ai_pipeline_core.pipeline.limits import _SharedStatus

        dep_id = uuid4()
        root_id = uuid4()
        ctx = ExecutionContext(
            run_id="test",
            run_scope="test/scope",
            execution_id=None,
            publisher=_NoopPublisher(),
            limits={},
            limits_status=_SharedStatus(),
            database=db,
            deployment_id=dep_id,
            root_deployment_id=root_id,
            deployment_name="test-deploy",
        )
        assert ctx.database is db
        assert ctx.deployment_id == dep_id
        assert ctx.root_deployment_id == root_id
        assert ctx.deployment_name == "test-deploy"

    def test_with_node_returns_copy(self) -> None:
        from ai_pipeline_core.pipeline._execution_context import ExecutionContext
        from ai_pipeline_core.deployment._types import _NoopPublisher
        from ai_pipeline_core.pipeline.limits import _SharedStatus

        ctx = ExecutionContext(
            run_id="test",
            run_scope="test/scope",
            execution_id=None,
            publisher=_NoopPublisher(),
            limits={},
            limits_status=_SharedStatus(),
        )
        node_id = uuid4()
        new_ctx = ctx.with_node(node_id)
        assert new_ctx.current_node_id == node_id
        assert ctx.current_node_id is None  # original unchanged


class TestDatabaseFactoryFromSettings:
    """Test create_database_from_settings."""

    def test_memory_backend_when_no_config(self) -> None:
        from ai_pipeline_core.database import create_database_from_settings
        from ai_pipeline_core.settings import Settings

        s = Settings(clickhouse_host="", openai_base_url="x", openai_api_key="x")
        db = create_database_from_settings(s)
        assert type(db).__name__ == "MemoryDatabase"

    def test_filesystem_backend_when_base_path(self, tmp_path: Any) -> None:
        from ai_pipeline_core.database import create_database_from_settings
        from ai_pipeline_core.settings import Settings

        s = Settings(clickhouse_host="", openai_base_url="x", openai_api_key="x")
        db = create_database_from_settings(s, base_path=tmp_path)
        assert type(db).__name__ == "FilesystemDatabase"
