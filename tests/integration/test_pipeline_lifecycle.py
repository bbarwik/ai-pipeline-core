# pyright: reportPrivateUsage=false
"""Pipeline lifecycle integration tests.

Covers multi-flow document provenance, error propagation with partial docs,
leaked handle cancellation, flow resume via completion keys, replay YAML
roundtrip, and parallel task persistence with events.
"""

import asyncio
import json

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
from ai_pipeline_core.document_store._memory import MemoryDocumentStore
from ai_pipeline_core.document_store._protocol import set_document_store
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
    async def run(cls, documents: list[G1InputDoc]) -> list[G1MiddleDoc]:
        return [G1MiddleDoc.derive(from_documents=(documents[0],), name="middle.txt", content="middle")]


class G1ToOutput(PipelineTask):
    @classmethod
    async def run(cls, documents: list[G1MiddleDoc]) -> list[G1OutputDoc]:
        return [G1OutputDoc.derive(from_documents=(documents[0],), name="output.txt", content="output")]


class G2ToMiddle(PipelineTask):
    @classmethod
    async def run(cls, documents: list[G2InputDoc]) -> list[G2MiddleDoc]:
        return [G2MiddleDoc.derive(from_documents=(documents[0],), name="middle.txt", content="middle")]


class G2FailingTask(PipelineTask):
    @classmethod
    async def run(cls, documents: list[G2MiddleDoc]) -> list[G2OutputDoc]:
        raise LLMError("provider exploded")


class G3SlowTask(PipelineTask):
    @classmethod
    async def run(cls, documents: list[G3InputDoc], delay: float) -> list[G3LateDoc]:
        await asyncio.sleep(delay)
        return [G3LateDoc.derive(from_documents=(documents[0],), name="late.txt", content="late")]


class G3FastTask(PipelineTask):
    @classmethod
    async def run(cls, documents: list[G3InputDoc]) -> list[G3FastDoc]:
        return [G3FastDoc.derive(from_documents=(documents[0],), name="fast.txt", content="fast")]


class G5EchoTask(PipelineTask):
    @classmethod
    async def run(cls, source: G5InputDoc, suffix: str) -> G5OutputDoc:
        return G5OutputDoc.derive(from_documents=(source,), name="echo.txt", content=f"{source.text}{suffix}")


class G6ShardTask(PipelineTask):
    @classmethod
    async def run(cls, documents: list[G6InputDoc], shard: int) -> list[G6OutputDoc]:
        await asyncio.sleep(0.01 * shard)
        return [G6OutputDoc.derive(from_documents=(documents[0],), name=f"shard_{shard}.txt", content=str(shard))]


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------


class G1FlowOne(PipelineFlow):
    name = "g1-flow-one"

    async def run(self, run_id: str, documents: list[G1InputDoc], options: FlowOptions) -> list[G1MiddleDoc]:
        return await G1ToMiddle.run(documents)


class G1FlowTwo(PipelineFlow):
    name = "g1-flow-two"

    async def run(self, run_id: str, documents: list[G1MiddleDoc], options: FlowOptions) -> list[G1OutputDoc]:
        return await G1ToOutput.run(documents)


class G2FlowOne(PipelineFlow):
    name = "g2-flow-one"

    async def run(self, run_id: str, documents: list[G2InputDoc], options: FlowOptions) -> list[G2MiddleDoc]:
        return await G2ToMiddle.run(documents)


class G2FlowTwo(PipelineFlow):
    name = "g2-flow-two"

    async def run(self, run_id: str, documents: list[G2MiddleDoc], options: FlowOptions) -> list[G2OutputDoc]:
        return await G2FailingTask.run(documents)


class G3LeakyFlow(PipelineFlow):
    name = "g3-leaky-flow"

    async def run(self, run_id: str, documents: list[G3InputDoc], options: FlowOptions) -> list[G3FastDoc]:
        # Dispatch slow task but DON'T await — it should be cancelled by the deployment
        _ = G3SlowTask.run(documents, delay=0.5)
        return await G3FastTask.run(documents)


class G4Flow(PipelineFlow):
    name = "g4-flow"
    run_count: int = 0

    async def run(self, run_id: str, documents: list[G4InputDoc], options: FlowOptions) -> list[G4OutputDoc]:
        type(self).run_count += 1
        return [G4OutputDoc.derive(from_documents=(documents[0],), name="g4.txt", content="fresh")]


# ---------------------------------------------------------------------------
# Deployments
# ---------------------------------------------------------------------------


class _GapResult(DeploymentResult):
    output_count: int = 0


class G1Deployment(PipelineDeployment[FlowOptions, _GapResult]):
    def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
        return [G1FlowOne(), G1FlowTwo()]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> _GapResult:
        return _GapResult(success=True, output_count=len(documents))


class G2Deployment(PipelineDeployment[FlowOptions, _GapResult]):
    def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
        return [G2FlowOne(), G2FlowTwo()]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> _GapResult:
        return _GapResult(success=True)


class G3Deployment(PipelineDeployment[FlowOptions, _GapResult]):
    def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
        return [G3LeakyFlow()]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> _GapResult:
        return _GapResult(success=True, output_count=len(documents))


class G4Deployment(PipelineDeployment[FlowOptions, _GapResult]):
    def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
        return [G4Flow()]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> _GapResult:
        return _GapResult(success=True, output_count=len(documents))


# ---------------------------------------------------------------------------
# Full document lifecycle through multi-flow pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiflow_document_lifecycle_preserves_provenance() -> None:
    """Documents persist across flows with correct derived_from chain."""
    store = MemoryDocumentStore()
    publisher = _MemoryPublisher()
    set_document_store(store)
    try:
        input_doc = G1InputDoc.create_root(name="input.txt", content="seed", reason="gap-1 test")
        run_id = "gap1-lifecycle"
        options = FlowOptions()

        result = await G1Deployment().run(run_id, [input_doc], options, publisher=publisher)
        run_scope = _compute_run_scope(run_id, [input_doc], options)

        middle_docs = await store.load(run_scope, [G1MiddleDoc])
        output_docs = await store.load(run_scope, [G1OutputDoc])
    finally:
        set_document_store(None)
        store.shutdown()

    assert result.success
    assert result.output_count >= 1

    # Middle doc exists and is derived from input doc
    assert len(middle_docs) == 1
    assert middle_docs[0].derived_from == (input_doc.sha256,)

    # Output doc exists and is derived from middle doc (not input)
    assert len(output_docs) == 1
    assert output_docs[0].derived_from == (middle_docs[0].sha256,)

    # Flow 2 task actually received the middle doc (not the input)
    assert output_docs[0].derived_from != (input_doc.sha256,)


# ---------------------------------------------------------------------------
# Error propagation Task → Flow → Deployment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_propagation_preserves_partial_docs_and_emits_events() -> None:
    """Task failure in flow 2 preserves flow 1 docs and emits correct events."""
    store = MemoryDocumentStore()
    publisher = _MemoryPublisher()
    set_document_store(store)
    try:
        input_doc = G2InputDoc.create_root(name="input.txt", content="seed", reason="gap-2 test")
        run_id = "gap2-error"
        options = FlowOptions()
        run_scope = _compute_run_scope(run_id, [input_doc], options)

        with pytest.raises(LLMError, match="provider exploded"):
            await G2Deployment().run(run_id, [input_doc], options, publisher=publisher)

        middle_docs = await store.load(run_scope, [G2MiddleDoc])
    finally:
        set_document_store(None)
        store.shutdown()

    # Partial results: flow 1's docs are preserved in store
    assert len(middle_docs) == 1

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
    """Un-awaited task handles in a flow are cancelled, preventing late writes."""
    store = MemoryDocumentStore()
    publisher = _MemoryPublisher()
    set_document_store(store)
    try:
        input_doc = G3InputDoc.create_root(name="input.txt", content="seed", reason="gap-3 test")
        run_id = "gap3-cancel"
        options = FlowOptions()
        run_scope = _compute_run_scope(run_id, [input_doc], options)

        result = await G3Deployment().run(run_id, [input_doc], options, publisher=publisher)

        # Wait longer than the SlowTask delay to ensure it didn't write late
        await asyncio.sleep(0.7)

        fast_docs = await store.load(run_scope, [G3FastDoc])
        late_docs = await store.load(run_scope, [G3LateDoc])
    finally:
        set_document_store(None)
        store.shutdown()

    assert result.success
    # Fast docs persisted
    assert len(fast_docs) == 1
    # Late docs NOT persisted (task was cancelled)
    assert late_docs == []


# ---------------------------------------------------------------------------
# Flow resume via completion_name format (flow_name:step)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flow_completion_with_new_key_format_enables_resume() -> None:
    """Pre-seeded FlowCompletion with flow_name:step format is honored for resume."""
    store = MemoryDocumentStore()
    publisher = _MemoryPublisher()
    set_document_store(store)
    G4Flow.run_count = 0
    try:
        input_doc = G4InputDoc.create_root(name="input.txt", content="seed", reason="gap-4 test")
        run_id = "gap4-resume"
        options = FlowOptions()
        run_scope = _compute_run_scope(run_id, [input_doc], options)

        # Pre-seed completion with new format key (flow_name:step)
        cached_output = G4OutputDoc.derive(from_documents=(input_doc,), name="cached.txt", content="cached")
        await store.save_batch([input_doc, cached_output], run_scope, created_by_task="")
        await store.save_flow_completion(run_scope, "g4-flow:1", (input_doc.sha256,), (cached_output.sha256,))

        await G4Deployment().run(run_id, [input_doc], options, publisher=publisher)
    finally:
        set_document_store(None)
        store.shutdown()

    # Flow was skipped (not executed)
    assert G4Flow.run_count == 0
    skipped = [e for e in publisher.events if isinstance(e, FlowSkippedEvent)]
    assert len(skipped) == 1


@pytest.mark.asyncio
async def test_old_completion_key_format_does_not_resume() -> None:
    """FlowCompletion with old key format (just flow_name) does NOT trigger resume.

    This documents the intentional format change from flow_name to flow_name:step.
    """
    store = MemoryDocumentStore()
    publisher = _MemoryPublisher()
    set_document_store(store)
    G4Flow.run_count = 0
    try:
        input_doc = G4InputDoc.create_root(name="input.txt", content="seed", reason="gap-4 compat")
        run_id = "gap4-compat"
        options = FlowOptions()
        run_scope = _compute_run_scope(run_id, [input_doc], options)

        # Pre-seed with OLD format key (just flow_name, no step)
        cached_output = G4OutputDoc.derive(from_documents=(input_doc,), name="cached.txt", content="cached")
        await store.save_batch([input_doc, cached_output], run_scope, created_by_task="")
        await store.save_flow_completion(run_scope, "g4-flow", (input_doc.sha256,), (cached_output.sha256,))

        await G4Deployment().run(run_id, [input_doc], options, publisher=publisher)
    finally:
        set_document_store(None)
        store.shutdown()

    # Flow was re-executed (old key format not recognized)
    assert G4Flow.run_count == 1


# ---------------------------------------------------------------------------
# Replay capture-to-YAML round-trip (no LLM)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_replay_capture_yaml_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay payload captured from a real task execution round-trips through YAML."""
    captured_payloads: list[str] = []

    def capture_attrs(attrs: dict[str, object]) -> None:
        payload = attrs.get("replay.payload")
        if isinstance(payload, str):
            captured_payloads.append(payload)

    import ai_pipeline_core.pipeline._task as task_module

    monkeypatch.setattr(task_module.Laminar, "set_span_attributes", staticmethod(capture_attrs))

    source_doc = G5InputDoc.create_root(name="input.txt", content="seed", reason="gap-5 test")
    with pipeline_test_context():
        await G5EchoTask.run(source_doc, suffix="::done")

    assert len(captured_payloads) >= 1
    payload_dict = json.loads(captured_payloads[-1])

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
    """Concurrent tasks via collect_tasks persist all outputs and emit started/completed events."""
    publisher = _MemoryPublisher()
    store = MemoryDocumentStore()
    try:
        input_doc = G6InputDoc.create_root(name="input.txt", content="seed", reason="gap-6 test")
        with pipeline_test_context(store=store, publisher=publisher) as ctx:
            token = set_execution_context(ctx.with_flow(_flow_frame("g6-flow")))
            try:
                await store.save(input_doc, ctx.run_scope, created_by_task="")
                batch = await collect_tasks(
                    G6ShardTask.run([input_doc], shard=1),
                    G6ShardTask.run([input_doc], shard=2),
                    G6ShardTask.run([input_doc], shard=3),
                )
                persisted = await store.load(ctx.run_scope, [G6OutputDoc])
            finally:
                reset_execution_context(token)
    finally:
        store.shutdown()

    # All 3 tasks completed
    assert len(batch.completed) == 3
    assert batch.incomplete == []

    # All 3 documents persisted
    assert len(persisted) == 3
    persisted_names = {d.name for d in persisted}
    assert persisted_names == {"shard_1.txt", "shard_2.txt", "shard_3.txt"}

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
