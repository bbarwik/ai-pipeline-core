"""Resume behavior tests for flow completion records.

Tests: cache TTL, option/input change invalidation, crash-retry resume, completed flow skip.
"""

import pytest

from ai_pipeline_core import DeploymentResult, Document, FlowOptions, PipelineDeployment, PipelineFlow, PipelineTask
from ai_pipeline_core.deployment import FlowAction, FlowDirective
from ai_pipeline_core.deployment.base import _compute_run_scope
from ai_pipeline_core.document_store._memory import MemoryDocumentStore
from ai_pipeline_core.document_store._protocol import set_document_store

from .conftest import OutputDoc, StageOne, StageTwo, _TestOptions, _TestResult


# --- Resume deployments ---


class ResumeDeployment(PipelineDeployment[_TestOptions, _TestResult]):
    def build_flows(self, options: _TestOptions):
        _ = options
        return [StageOne(), StageTwo()]

    @staticmethod
    def build_result(run_id, documents, options):
        _ = (run_id, options)
        return _TestResult(success=True, output_count=len([d for d in documents if isinstance(d, OutputDoc)]))


class SkipDeployment(ResumeDeployment):
    def plan_next_flow(self, flow_class, plan, output_documents):
        _ = (plan, output_documents)
        if flow_class is StageTwo:
            return FlowDirective(action=FlowAction.SKIP, reason="skip second")
        return FlowDirective()


@pytest.mark.asyncio
async def test_resume_uses_flow_completion_cache(input_documents):
    store = MemoryDocumentStore()
    set_document_store(store)
    try:
        deployment = ResumeDeployment()
        first = await deployment.run("resume-run", input_documents, _TestOptions())
        second = await deployment.run("resume-run", input_documents, _TestOptions())
    finally:
        set_document_store(None)
        store.shutdown()

    assert first.success and second.success


@pytest.mark.asyncio
async def test_plan_next_flow_can_skip_class(input_documents):
    store = MemoryDocumentStore()
    set_document_store(store)
    try:
        deployment = SkipDeployment()
        result = await deployment.run("skip-run", input_documents, _TestOptions())
    finally:
        set_document_store(None)
        store.shutdown()

    assert result.success is True


# --- Crash/retry resume tests ---

# Mutable state to control flow behavior across runs
_flow_call_count = 0
_should_crash = False


class ResumeInputDoc(Document):
    """Input document for resume tests."""


class ResumeOutputDoc(Document):
    """Output document for resume tests."""


class SucceedTask(PipelineTask):
    """Task 1: produces docs that get saved incrementally."""

    @classmethod
    async def run(cls, inputs: list[ResumeInputDoc]) -> list[ResumeOutputDoc]:
        return [
            ResumeOutputDoc.derive(from_documents=(inputs[0],), name="out1.txt", content="output 1"),
            ResumeOutputDoc.derive(from_documents=(inputs[0],), name="out2.txt", content="output 2"),
        ]


class CrashTask(PipelineTask):
    """Task 2: crashes before returning, so its docs are never saved."""

    @classmethod
    async def run(cls, docs: list[ResumeOutputDoc]) -> list[ResumeOutputDoc]:
        if _should_crash:
            raise RuntimeError("Simulated crash in task 2")
        return [
            ResumeOutputDoc.derive(from_documents=(docs[0],), name="out3.txt", content="output 3"),
            ResumeOutputDoc.derive(from_documents=(docs[0],), name="out4.txt", content="output 4"),
        ]


class ProduceAllTask(PipelineTask):
    """Task that produces all 4 output documents in one go."""

    @classmethod
    async def run(cls, inputs: list[ResumeInputDoc]) -> list[ResumeOutputDoc]:
        return [ResumeOutputDoc.derive(from_documents=(inputs[0],), name=f"out{i}.txt", content=f"output {i}") for i in range(1, 5)]


class CrashingFlow(PipelineFlow):
    """Flow with 2 tasks: task 1 succeeds (docs saved), task 2 may crash."""

    async def run(self, run_id: str, documents: list[ResumeInputDoc], options: FlowOptions) -> list[ResumeOutputDoc]:
        global _flow_call_count
        _flow_call_count += 1
        partial = await SucceedTask.run(documents)
        remaining = await CrashTask.run(partial)
        return partial + remaining


class NormalFlow(PipelineFlow):
    """Flow that always succeeds."""

    async def run(self, run_id: str, documents: list[ResumeInputDoc], options: FlowOptions) -> list[ResumeOutputDoc]:
        global _flow_call_count
        _flow_call_count += 1
        return await ProduceAllTask.run(documents)


class Resume_TestOptions(FlowOptions):
    """Options for resume tests."""


class Resume_TestResult(DeploymentResult):
    """Result for resume tests."""


class CrashingDeployment(PipelineDeployment[Resume_TestOptions, Resume_TestResult]):
    """Deployment with a single two-task flow that can crash."""

    def build_flows(self, options):
        return [CrashingFlow()]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: Resume_TestOptions) -> Resume_TestResult:
        return Resume_TestResult(success=True)


class NormalDeployment(PipelineDeployment[Resume_TestOptions, Resume_TestResult]):
    """Deployment with a single flow that always succeeds."""

    def build_flows(self, options):
        return [NormalFlow()]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: Resume_TestOptions) -> Resume_TestResult:
        return Resume_TestResult(success=True)


@pytest.fixture(autouse=True)
def _reset_state():
    global _flow_call_count, _should_crash
    _flow_call_count = 0
    _should_crash = False
    yield
    _flow_call_count = 0
    _should_crash = False


class TestResumeAfterCrash:
    """Regression: partial outputs from a crashed flow must not skip re-execution."""

    @pytest.mark.asyncio
    async def test_partial_outputs_do_not_cause_false_resume(self, memory_store: MemoryDocumentStore):
        """Flow with 2 tasks: task 1 completes (docs saved), task 2 crashes.

        On retry, the flow must re-run because it never completed.
        """
        global _should_crash
        input_doc = ResumeInputDoc.create_root(name="input.txt", content="test input", reason="test")
        deployment = CrashingDeployment()
        options = Resume_TestOptions()

        # First run: task 1 succeeds (docs saved), task 2 crashes
        _should_crash = True
        with pytest.raises(RuntimeError, match="Simulated crash"):
            await deployment.run("test-project", [input_doc], options)

        assert _flow_call_count == 1

        # Verify partial outputs exist (task 1's docs were saved by PipelineTask)
        run_scope = _compute_run_scope("test-project", [input_doc], options)
        has_partial = await memory_store.has_documents(run_scope, ResumeOutputDoc)
        assert has_partial is True, "Task 1's outputs should exist in store from the crashed run"

        # Second run: no crash this time
        _should_crash = False
        await deployment.run("test-project", [input_doc], options)

        # The flow MUST have re-executed (not skipped by false resume)
        assert _flow_call_count == 2, (
            f"Flow executed {_flow_call_count} times total — expected 2 (crash + retry). "
            "Resume logic incorrectly skipped the flow due to partial outputs from task 1."
        )


class TestResumeAfterSuccess:
    """Completed flows should be skipped on re-run."""

    @pytest.mark.asyncio
    async def test_completed_flow_is_skipped(self, memory_store: MemoryDocumentStore):
        """A flow that completed successfully should be skipped on second run."""
        input_doc = ResumeInputDoc.create_root(name="input.txt", content="test input", reason="test")
        deployment = NormalDeployment()
        options = Resume_TestOptions()

        # First run — flow executes fully
        await deployment.run("test-project", [input_doc], options)
        assert _flow_call_count == 1

        # Second run — flow should be skipped (resume from cache)
        await deployment.run("test-project", [input_doc], options)
        assert _flow_call_count == 1, f"Flow executed {_flow_call_count} times — expected 1. Completed flow should be skipped on resume."


class TestResumeWithDifferentOptions:
    """Different options should produce a different run_scope, bypassing cache."""

    @pytest.mark.asyncio
    async def test_different_options_bypass_cache(self, memory_store: MemoryDocumentStore):
        """Changing options produces a different run_scope, so flow re-executes."""
        input_doc = ResumeInputDoc.create_root(name="input.txt", content="test input", reason="test")

        class OptionedOptions(FlowOptions):
            flavor: str = "vanilla"

        class OptionedResult(DeploymentResult):
            pass

        class OptionedDeployment(PipelineDeployment[OptionedOptions, OptionedResult]):
            def build_flows(self, options):
                return [NormalFlow()]

            @staticmethod
            def build_result(run_id, documents, options):
                return OptionedResult(success=True)

        deployment = OptionedDeployment()

        await deployment.run("test-project", [input_doc], OptionedOptions(flavor="vanilla"))
        assert _flow_call_count == 1

        # Same options → skipped
        await deployment.run("test-project", [input_doc], OptionedOptions(flavor="vanilla"))
        assert _flow_call_count == 1

        # Different options → new scope → re-executes
        await deployment.run("test-project", [input_doc], OptionedOptions(flavor="chocolate"))
        assert _flow_call_count == 2


class TestResumeWithDifferentInputs:
    """Different input documents should produce a different run_scope."""

    @pytest.mark.asyncio
    async def test_different_inputs_bypass_cache(self, memory_store: MemoryDocumentStore):
        """Changing input documents produces a different run_scope, so flow re-executes."""
        input_doc_a = ResumeInputDoc.create_root(name="a.txt", content="input A", reason="test")
        input_doc_b = ResumeInputDoc.create_root(name="b.txt", content="input B", reason="test")

        deployment = NormalDeployment()
        options = Resume_TestOptions()

        await deployment.run("test-project", [input_doc_a], options)
        assert _flow_call_count == 1

        # Same input → skipped
        await deployment.run("test-project", [input_doc_a], options)
        assert _flow_call_count == 1

        # Different input → new scope → re-executes
        await deployment.run("test-project", [input_doc_b], options)
        assert _flow_call_count == 2
