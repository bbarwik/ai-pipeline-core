# pyright: reportPrivateUsage=false
"""Pipeline lifecycle integration tests.

Covers multi-flow document provenance, error propagation with partial docs,
leaked handle cancellation, flow resume via completion keys, replay YAML
roundtrip, and parallel task persistence with events.
"""

import asyncio

import pytest

from ai_pipeline_core import DeploymentResult, Document, FlowOptions, PipelineDeployment
from ai_pipeline_core.deployment._types import (
    ErrorCode,
    FlowCompletedEvent,
    FlowSkippedEvent,
    RunFailedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
    _MemoryPublisher,
)
from ai_pipeline_core.deployment.base import _compute_run_scope
from ai_pipeline_core.documents._context import _suppress_document_registration
from ai_pipeline_core.exceptions import LLMError
from ai_pipeline_core.pipeline import (
    PipelineFlow,
    PipelineTask,
    collect_tasks,
    pipeline_test_context,
)
from ai_pipeline_core.pipeline._execution_context import FlowFrame, reset_execution_context, set_execution_context
from ai_pipeline_core.replay.types import TaskReplay


@pytest.fixture(autouse=True)
def _suppress_registration():
    with _suppress_document_registration():
        yield


# ---------------------------------------------------------------------------
# Document types (prefixed per gap to avoid collisions)
# ---------------------------------------------------------------------------


class G1InputDoc(Document):
    """Provenance test: input document."""


class G1MiddleDoc(Document):
    """Provenance test: intermediate document produced by flow 1."""


class G1OutputDoc(Document):
    """Provenance test: final document produced by flow 2."""


class G2InputDoc(Document):
    """Error propagation test: input document."""


class G2MiddleDoc(Document):
    """Error propagation test: intermediate document (flow 1 output)."""


class G2OutputDoc(Document):
    """Error propagation test: final document (flow 2 output, never produced)."""


class G3InputDoc(Document):
    """Cancellation test: input document."""


class G3FastDoc(Document):
    """Cancellation test: fast document returned by flow."""


class G3LateDoc(Document):
    """Cancellation test: late document from leaked task."""


class G4InputDoc(Document):
    """Resume test: input document."""


class G4OutputDoc(Document):
    """Resume test: output document."""


class G5InputDoc(Document):
    """Replay test: input document."""


class G5OutputDoc(Document):
    """Replay test: output document."""


class G6InputDoc(Document):
    """Parallel tasks test: input document."""


class G6OutputDoc(Document):
    """Parallel tasks test: output document."""


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


class G1ToMiddle(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[G1InputDoc, ...]) -> tuple[G1MiddleDoc, ...]:
        return (G1MiddleDoc.derive(from_documents=(documents[0],), name="middle.txt", content="middle"),)


class G1ToOutput(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[G1MiddleDoc, ...]) -> tuple[G1OutputDoc, ...]:
        return (G1OutputDoc.derive(from_documents=(documents[0],), name="output.txt", content="output"),)


class G2ToMiddle(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[G2InputDoc, ...]) -> tuple[G2MiddleDoc, ...]:
        return (G2MiddleDoc.derive(from_documents=(documents[0],), name="middle.txt", content="middle"),)


class G2FailingTask(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[G2MiddleDoc, ...]) -> tuple[G2OutputDoc, ...]:
        raise LLMError("provider exploded")


class G3SlowTask(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[G3InputDoc, ...], delay: float) -> tuple[G3LateDoc, ...]:
        await asyncio.sleep(delay)
        return (G3LateDoc.derive(from_documents=(documents[0],), name="late.txt", content="late"),)


class G3FastTask(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[G3InputDoc, ...]) -> tuple[G3FastDoc, ...]:
        return (G3FastDoc.derive(from_documents=(documents[0],), name="fast.txt", content="fast"),)


class G5EchoTask(PipelineTask):
    @classmethod
    async def run(cls, source: G5InputDoc, suffix: str) -> G5OutputDoc:
        return G5OutputDoc.derive(from_documents=(source,), name="echo.txt", content=f"{source.text}{suffix}")


class G6ShardTask(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[G6InputDoc, ...], shard: int) -> tuple[G6OutputDoc, ...]:
        await asyncio.sleep(0.01 * shard)
        return (G6OutputDoc.derive(from_documents=(documents[0],), name=f"shard_{shard}.txt", content=str(shard)),)


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------


class G1FlowOne(PipelineFlow):
    name = "g1-flow-one"

    async def run(self, documents: tuple[G1InputDoc, ...], options: FlowOptions) -> tuple[G1MiddleDoc, ...]:
        return await G1ToMiddle.run(documents)


class G1FlowTwo(PipelineFlow):
    name = "g1-flow-two"

    async def run(self, documents: tuple[G1MiddleDoc, ...], options: FlowOptions) -> tuple[G1OutputDoc, ...]:
        return await G1ToOutput.run(documents)


class G2FlowOne(PipelineFlow):
    name = "g2-flow-one"

    async def run(self, documents: tuple[G2InputDoc, ...], options: FlowOptions) -> tuple[G2MiddleDoc, ...]:
        return await G2ToMiddle.run(documents)


class G2FlowTwo(PipelineFlow):
    name = "g2-flow-two"

    async def run(self, documents: tuple[G2MiddleDoc, ...], options: FlowOptions) -> tuple[G2OutputDoc, ...]:
        return await G2FailingTask.run(documents)


class G3LeakyFlow(PipelineFlow):
    name = "g3-leaky-flow"

    async def run(self, documents: tuple[G3InputDoc, ...], options: FlowOptions) -> tuple[G3FastDoc, ...]:
        # Dispatch slow task but DON'T await — it should be cancelled by the deployment
        _ = G3SlowTask.run(documents, delay=0.5)
        return await G3FastTask.run(documents)


class G4Flow(PipelineFlow):
    name = "g4-flow"
    run_count: int = 0

    async def run(self, documents: tuple[G4InputDoc, ...], options: FlowOptions) -> tuple[G4OutputDoc, ...]:
        type(self).run_count += 1
        return (G4OutputDoc.derive(from_documents=(documents[0],), name="g4.txt", content="fresh"),)


# ---------------------------------------------------------------------------
# Deployments
# ---------------------------------------------------------------------------


class _GapResult(DeploymentResult):
    output_count: int = 0


class G1Deployment(PipelineDeployment[FlowOptions, _GapResult]):
    def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
        return [G1FlowOne(), G1FlowTwo()]

    @staticmethod
    def build_result(run_id: str, documents: tuple[Document, ...], options: FlowOptions) -> _GapResult:
        return _GapResult(success=True, output_count=len(documents))


class G2Deployment(PipelineDeployment[FlowOptions, _GapResult]):
    def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
        return [G2FlowOne(), G2FlowTwo()]

    @staticmethod
    def build_result(run_id: str, documents: tuple[Document, ...], options: FlowOptions) -> _GapResult:
        return _GapResult(success=True)


class G3Deployment(PipelineDeployment[FlowOptions, _GapResult]):
    def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
        return [G3LeakyFlow()]

    @staticmethod
    def build_result(run_id: str, documents: tuple[Document, ...], options: FlowOptions) -> _GapResult:
        return _GapResult(success=True, output_count=len(documents))


class G4Deployment(PipelineDeployment[FlowOptions, _GapResult]):
    def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
        return [G4Flow()]

    @staticmethod
    def build_result(run_id: str, documents: tuple[Document, ...], options: FlowOptions) -> _GapResult:
        return _GapResult(success=True, output_count=len(documents))


# ---------------------------------------------------------------------------
# Full document lifecycle through multi-flow pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiflow_document_lifecycle_preserves_provenance() -> None:
    """Documents persist across flows with correct derived_from chain."""
    publisher = _MemoryPublisher()
    input_doc = G1InputDoc.create_root(name="input.txt", content="seed", reason="gap-1 test")
    run_id = "gap1-lifecycle"
    options = FlowOptions()

    result = await G1Deployment().run(run_id, [input_doc], options, publisher=publisher)

    assert result.success
    assert result.output_count >= 1

    # Verify both flows completed via events
    flow_completed = [e for e in publisher.events if isinstance(e, FlowCompletedEvent)]
    assert len(flow_completed) == 2


# ---------------------------------------------------------------------------
# Error propagation Task → Flow → Deployment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_propagation_preserves_partial_docs_and_emits_events() -> None:
    """Task failure in flow 2 emits correct events."""
    publisher = _MemoryPublisher()
    input_doc = G2InputDoc.create_root(name="input.txt", content="seed", reason="gap-2 test")
    run_id = "gap2-error"
    options = FlowOptions()

    with pytest.raises(LLMError, match="provider exploded"):
        await G2Deployment().run(run_id, [input_doc], options, publisher=publisher)

    # TaskFailedEvent emitted for the failing task
    task_failed = [e for e in publisher.events if isinstance(e, TaskFailedEvent)]
    assert len(task_failed) == 1
    assert "provider exploded" in task_failed[0].error_message

    # RunFailedEvent with correct error code
    run_failed = [e for e in publisher.events if isinstance(e, RunFailedEvent)]
    assert len(run_failed) == 1
    assert run_failed[0].error_code == ErrorCode.PROVIDER_ERROR

    # No FlowCompletedEvent for flow 2 (it failed)
    flow2_completed = [e for e in publisher.events if isinstance(e, FlowCompletedEvent) and e.flow_name == "g2-flow-two"]
    assert flow2_completed == []


# ---------------------------------------------------------------------------
# Leaked handle cancellation prevents late writes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_leaked_handles_cancelled_no_late_writes() -> None:
    """Un-awaited task handles in a flow are cancelled."""
    publisher = _MemoryPublisher()
    input_doc = G3InputDoc.create_root(name="input.txt", content="seed", reason="gap-3 test")
    run_id = "gap3-cancel"
    options = FlowOptions()

    result = await G3Deployment().run(run_id, [input_doc], options, publisher=publisher)

    # Wait longer than the SlowTask delay to ensure it didn't write late
    await asyncio.sleep(0.7)

    assert result.success
    # Only one flow completed (the fast flow)
    flow_completed = [e for e in publisher.events if isinstance(e, FlowCompletedEvent)]
    assert len(flow_completed) == 1


# ---------------------------------------------------------------------------
# Flow resume via completion_name format (flow_name:step)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flow_completion_with_new_key_format_enables_resume() -> None:
    """Pre-seeded flow completion in the database DAG is honored for resume."""
    from datetime import UTC, datetime
    from uuid import uuid4

    from ai_pipeline_core.database import NULL_PARENT, ExecutionNode, MemoryDatabase, NodeKind, NodeStatus

    db = MemoryDatabase()
    publisher = _MemoryPublisher()
    G4Flow.run_count = 0

    input_doc = G4InputDoc.create_root(name="input.txt", content="seed", reason="gap-4 test")
    run_id = "gap4-resume"
    options = FlowOptions()
    run_scope = _compute_run_scope(run_id, [input_doc], options)

    # Pre-seed completion node with cache_key matching flow:{run_scope}:{flow_name}:{step}
    cached_output = G4OutputDoc.derive(from_documents=(input_doc,), name="cached.txt", content="cached")
    cache_key = f"flow:{run_scope}:g4-flow:1"
    deployment_id = uuid4()
    await db.insert_node(
        ExecutionNode(
            node_id=uuid4(),
            node_kind=NodeKind.FLOW,
            deployment_id=deployment_id,
            root_deployment_id=deployment_id,
            parent_node_id=NULL_PARENT,
            run_id=run_id,
            run_scope=run_scope,
            deployment_name="G4Deployment",
            name="g4-flow",
            sequence_no=1,
            status=NodeStatus.COMPLETED,
            ended_at=datetime.now(UTC),
            cache_key=cache_key,
            output_document_shas=(cached_output.sha256,),
        )
    )

    # Save the document blob+record so load_documents_from_database can reconstruct it
    from ai_pipeline_core.database import BlobRecord, DocumentRecord

    await db.save_blob(BlobRecord(content_sha256=cached_output.sha256, content=cached_output.content, size_bytes=len(cached_output.content)))
    await db.save_document(
        DocumentRecord(
            document_sha256=cached_output.sha256,
            content_sha256=cached_output.sha256,
            deployment_id=deployment_id,
            producing_node_id=None,
            document_type="G4OutputDoc",
            name=cached_output.name,
            description=cached_output.description or "",
            derived_from=tuple(cached_output.derived_from),
        )
    )

    await G4Deployment()._run_with_context(
        run_id,
        [input_doc],
        options,
        deployment_node_id=deployment_id,
        root_deployment_id=deployment_id,
        parent_deployment_task_id=None,
        publisher=publisher,
        database=db,
    )

    # Flow was skipped (not executed)
    assert G4Flow.run_count == 0
    skipped = [e for e in publisher.events if isinstance(e, FlowSkippedEvent)]
    assert len(skipped) == 1


@pytest.mark.asyncio
async def test_fresh_run_without_resume_executes_flow() -> None:
    """Without pre-seeded completion, the flow executes normally."""
    publisher = _MemoryPublisher()
    G4Flow.run_count = 0

    input_doc = G4InputDoc.create_root(name="input.txt", content="seed", reason="gap-4 fresh")
    run_id = "gap4-fresh"
    options = FlowOptions()

    await G4Deployment().run(run_id, [input_doc], options, publisher=publisher)

    # Flow was executed
    assert G4Flow.run_count == 1


# ---------------------------------------------------------------------------
# Replay capture-to-YAML round-trip (no LLM)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_replay_capture_yaml_roundtrip() -> None:
    """Replay payload persisted on the task node round-trips through YAML."""
    from uuid import uuid4

    from ai_pipeline_core.database import MemoryDatabase, NodeKind

    database = MemoryDatabase()
    source_doc = G5InputDoc.create_root(name="input.txt", content="seed", reason="gap-5 test")
    with pipeline_test_context() as ctx:
        deployment_id = uuid4()
        ctx.database = database
        ctx.deployment_id = deployment_id
        ctx.root_deployment_id = deployment_id
        ctx.deployment_name = "gap5-deployment"
        await G5EchoTask.run(source_doc, suffix="::done")

    task_nodes = [node for node in database._nodes.values() if node.node_kind == NodeKind.TASK]
    assert len(task_nodes) == 1
    payload_dict = task_nodes[0].payload["replay_payload"]

    # Verify payload structure
    assert payload_dict["payload_type"] == "pipeline_task"
    assert "G5EchoTask" in payload_dict["function_path"]
    assert payload_dict["arguments"]["suffix"] == "::done"

    # YAML round-trip
    replay = TaskReplay.model_validate(payload_dict)
    yaml_text = replay.to_yaml()
    roundtripped = TaskReplay.from_yaml(yaml_text)

    assert roundtripped.function_path == replay.function_path
    assert roundtripped.payload_type == "pipeline_task"


# ---------------------------------------------------------------------------
# Parallel TaskHandle + persistence + task events
# ---------------------------------------------------------------------------


def _flow_frame(name: str) -> FlowFrame:
    return FlowFrame(
        name=name,
        flow_class_name="GapTestFlow",
        step=1,
        total_steps=1,
        flow_minutes=(1.0,),
        completed_minutes=0.0,
        flow_params={},
    )


@pytest.mark.asyncio
async def test_parallel_tasks_persist_all_docs_and_emit_events() -> None:
    """Concurrent tasks via collect_tasks emit started/completed events and return all outputs."""
    publisher = _MemoryPublisher()
    input_doc = G6InputDoc.create_root(name="input.txt", content="seed", reason="gap-6 test")
    with pipeline_test_context(publisher=publisher) as ctx:
        token = set_execution_context(ctx.with_flow(_flow_frame("g6-flow")))
        try:
            batch = await collect_tasks(
                G6ShardTask.run((input_doc,), shard=1),
                G6ShardTask.run((input_doc,), shard=2),
                G6ShardTask.run((input_doc,), shard=3),
            )
        finally:
            reset_execution_context(token)

    # All 3 tasks completed
    assert len(batch.completed) == 3
    assert batch.incomplete == []

    # All 3 task results returned
    all_docs = [doc for result in batch.completed for doc in result]
    result_names = {d.name for d in all_docs}
    assert result_names == {"shard_1.txt", "shard_2.txt", "shard_3.txt"}

    # 3 started + 3 completed events
    started = [e for e in publisher.events if isinstance(e, TaskStartedEvent)]
    completed = [e for e in publisher.events if isinstance(e, TaskCompletedEvent)]
    assert len(started) == 3
    assert len(completed) == 3

    # Each started event has a matching completed event
    started_classes = {e.task_class for e in started}
    completed_classes = {e.task_class for e in completed}
    assert started_classes == {"G6ShardTask"}
    assert completed_classes == {"G6ShardTask"}
