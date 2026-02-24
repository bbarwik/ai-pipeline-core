"""Tests for flow resume correctness in PipelineDeployment.

Regression test: proves that partial outputs from a crashed flow must NOT
cause the resume logic to skip the flow on restart.
"""

import pytest

from ai_pipeline_core import pipeline_flow, pipeline_task
from ai_pipeline_core.deployment.base import (
    DeploymentContext,
    DeploymentResult,
    PipelineDeployment,
    _compute_run_scope,
)
from ai_pipeline_core.document_store._protocol import set_document_store
from ai_pipeline_core.document_store._memory import MemoryDocumentStore
from ai_pipeline_core.documents import Document
from ai_pipeline_core.pipeline.options import FlowOptions


class ResumeInputDoc(Document):
    """Input document for resume tests."""


class ResumeOutputDoc(Document):
    """Output document for resume tests."""


# Mutable state to control flow behavior across runs
_flow_call_count = 0
_should_crash = False


@pipeline_task
async def task_that_succeeds(inputs: list[ResumeInputDoc]) -> list[ResumeOutputDoc]:
    """Task 1: produces docs that get saved incrementally."""
    return [
        ResumeOutputDoc.create(name="out1.txt", content="output 1", derived_from=(inputs[0].sha256,)),
        ResumeOutputDoc.create(name="out2.txt", content="output 2", derived_from=(inputs[0].sha256,)),
    ]


@pipeline_task
async def task_that_crashes(docs: list[ResumeOutputDoc]) -> list[ResumeOutputDoc]:
    """Task 2: crashes before returning, so its docs are never saved."""
    if _should_crash:
        raise RuntimeError("Simulated crash in task 2")
    return [
        ResumeOutputDoc.create(name="out3.txt", content="output 3", derived_from=(docs[0].sha256,)),
        ResumeOutputDoc.create(name="out4.txt", content="output 4", derived_from=(docs[0].sha256,)),
    ]


@pipeline_task
async def produce_all_docs(inputs: list[ResumeInputDoc]) -> list[ResumeOutputDoc]:
    """Task that produces all 4 output documents in one go."""
    return [ResumeOutputDoc.create(name=f"out{i}.txt", content=f"output {i}", derived_from=(inputs[0].sha256,)) for i in range(1, 5)]


@pipeline_flow()
async def two_task_flow(
    run_id: str,
    documents: list[ResumeInputDoc],
    flow_options: FlowOptions,
) -> list[ResumeOutputDoc]:
    """Flow with 2 tasks: task 1 succeeds (docs saved), task 2 may crash."""
    global _flow_call_count
    _flow_call_count += 1
    partial = await task_that_succeeds(documents)
    remaining = await task_that_crashes(partial)
    return partial + remaining


@pipeline_flow()
async def normal_flow(
    run_id: str,
    documents: list[ResumeInputDoc],
    flow_options: FlowOptions,
) -> list[ResumeOutputDoc]:
    """Flow that always succeeds."""
    global _flow_call_count
    _flow_call_count += 1
    return await produce_all_docs(documents)


class ResumeTestOptions(FlowOptions):
    """Options for resume tests."""


class ResumeTestResult(DeploymentResult):
    """Result for resume tests."""


class CrashingDeployment(PipelineDeployment[ResumeTestOptions, ResumeTestResult]):
    """Deployment with a single two-task flow that can crash."""

    flows = [two_task_flow]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: ResumeTestOptions) -> ResumeTestResult:
        return ResumeTestResult(success=True)


class NormalDeployment(PipelineDeployment[ResumeTestOptions, ResumeTestResult]):
    """Deployment with a single flow that always succeeds."""

    flows = [normal_flow]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: ResumeTestOptions) -> ResumeTestResult:
        return ResumeTestResult(success=True)


@pytest.fixture(autouse=True)
def _reset_state():
    global _flow_call_count, _should_crash
    _flow_call_count = 0
    _should_crash = False
    yield
    _flow_call_count = 0
    _should_crash = False


@pytest.fixture
def memory_store():
    store = MemoryDocumentStore()
    set_document_store(store)
    yield store
    set_document_store(None)


class TestResumeAfterCrash:
    """Regression: partial outputs from a crashed flow must not skip re-execution."""

    @pytest.mark.asyncio
    async def test_partial_outputs_do_not_cause_false_resume(self, prefect_test_fixture, memory_store: MemoryDocumentStore):
        """Flow with 2 tasks: task 1 completes (docs saved), task 2 crashes.

        On retry, the flow must re-run because it never completed.

        Scenario:
        1. Flow runs task 1 → saves 2 ResumeOutputDoc to store
        2. Task 2 crashes → flow fails, no flow-level completion
        3. Store has partial ResumeOutputDoc documents (from task 1)
        4. On re-run, has_documents(ResumeOutputDoc) returns True
        5. BUG: resume logic skips the flow because partial outputs exist
        6. EXPECTED: flow re-runs because it never completed successfully
        """
        global _should_crash
        input_doc = ResumeInputDoc.create_root(name="input.txt", content="test input", reason="test")
        deployment = CrashingDeployment()
        ctx = DeploymentContext()
        options = ResumeTestOptions()

        # First run: task 1 succeeds (docs saved), task 2 crashes
        _should_crash = True
        with pytest.raises(RuntimeError, match="Simulated crash"):
            await deployment.run("test-project", [input_doc], options, ctx)

        assert _flow_call_count == 1

        # Verify partial outputs exist (task 1's docs were saved by @pipeline_task)
        run_scope = _compute_run_scope("test-project", [input_doc], options)
        has_partial = await memory_store.has_documents(run_scope, ResumeOutputDoc)
        assert has_partial is True, "Task 1's outputs should exist in store from the crashed run"

        # Second run: no crash this time
        _should_crash = False
        await deployment.run("test-project", [input_doc], options, ctx)

        # The flow MUST have re-executed (not skipped by false resume)
        assert _flow_call_count == 2, (
            f"Flow executed {_flow_call_count} times total — expected 2 (crash + retry). "
            "Resume logic incorrectly skipped the flow due to partial outputs from task 1."
        )


class TestResumeAfterSuccess:
    """Completed flows should be skipped on re-run."""

    @pytest.mark.asyncio
    async def test_completed_flow_is_skipped(self, prefect_test_fixture, memory_store: MemoryDocumentStore):
        """A flow that completed successfully should be skipped on second run."""
        input_doc = ResumeInputDoc.create_root(name="input.txt", content="test input", reason="test")
        deployment = NormalDeployment()
        ctx = DeploymentContext()
        options = ResumeTestOptions()

        # First run — flow executes fully
        await deployment.run("test-project", [input_doc], options, ctx)
        assert _flow_call_count == 1

        # Second run — flow should be skipped (resume from cache)
        await deployment.run("test-project", [input_doc], options, ctx)
        assert _flow_call_count == 1, f"Flow executed {_flow_call_count} times — expected 1. Completed flow should be skipped on resume."
