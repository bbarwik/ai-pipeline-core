# pyright: reportPrivateUsage=false
"""Tests for plan_next_flow skip/continue logic and resolve_document_inputs provenance preservation."""

from collections.abc import Sequence

import pytest

from ai_pipeline_core import DeploymentResult, Document, FlowOptions, PipelineDeployment
from ai_pipeline_core.deployment import FlowAction, FlowDirective
from ai_pipeline_core.deployment._types import _MemoryPublisher
from ai_pipeline_core.deployment._resolve import _DocumentInput
from ai_pipeline_core.deployment._resolve import resolve_document_inputs
from ai_pipeline_core.deployment._types import FlowSkippedEvent
from ai_pipeline_core.pipeline import PipelineFlow, PipelineTask


# ---------------------------------------------------------------------------
# Document types
# ---------------------------------------------------------------------------


class PlanInputDoc(Document):
    """Input for plan_next_flow tests."""


class PlanMiddleDoc(Document):
    """Intermediate output for plan_next_flow tests."""


class PlanOutputDoc(Document):
    """Final output for plan_next_flow tests."""


class PlanResult(DeploymentResult):
    output_count: int = 0


# ---------------------------------------------------------------------------
# Tasks + Flows
# ---------------------------------------------------------------------------


class ToMiddleTask(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[PlanInputDoc, ...]) -> tuple[PlanMiddleDoc, ...]:
        return tuple(PlanMiddleDoc.derive(derived_from=(d,), name=f"mid_{d.name}", content="mid") for d in documents)


class ToOutputTask(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[PlanMiddleDoc, ...]) -> tuple[PlanOutputDoc, ...]:
        return tuple(PlanOutputDoc.derive(derived_from=(d,), name=f"out_{d.name}", content="out") for d in documents)


class ProducerFlow(PipelineFlow):
    name = "producer"

    async def run(self, documents: tuple[PlanInputDoc, ...], options: FlowOptions) -> tuple[PlanMiddleDoc, ...]:
        return await ToMiddleTask.run(documents)


class EmptyProducerFlow(PipelineFlow):
    name = "empty-producer"

    async def run(self, documents: tuple[PlanInputDoc, ...], options: FlowOptions) -> tuple[PlanMiddleDoc, ...]:
        return ()


class ConsumerFlow(PipelineFlow):
    name = "consumer"
    ran = False

    async def run(self, documents: tuple[PlanMiddleDoc, ...], options: FlowOptions) -> tuple[PlanOutputDoc, ...]:
        type(self).ran = True
        return await ToOutputTask.run(documents)


# ---------------------------------------------------------------------------
# plan_next_flow with actual previous_output_documents
# ---------------------------------------------------------------------------


class _SkipWhenEmptyDeployment(PipelineDeployment[FlowOptions, PlanResult]):
    """Skips consumer flow when producer outputs zero documents."""

    flow_retries = 0

    def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
        return [EmptyProducerFlow(), ConsumerFlow()]

    def plan_next_flow(
        self,
        flow_class: type[PipelineFlow],
        plan: Sequence[PipelineFlow],
        output_documents: tuple[Document, ...],
    ) -> FlowDirective:
        if flow_class is ConsumerFlow and not output_documents:
            return FlowDirective(action=FlowAction.SKIP, reason="no intermediate documents")
        return FlowDirective()

    @staticmethod
    def build_result(run_id: str, documents: tuple[Document, ...], options: FlowOptions) -> PlanResult:
        return PlanResult(success=True, output_count=len(documents))


class _ContinueWhenDocsExist(PipelineDeployment[FlowOptions, PlanResult]):
    """Continues consumer flow when producer produces documents."""

    flow_retries = 0

    def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
        return [ProducerFlow(), ConsumerFlow()]

    def plan_next_flow(
        self,
        flow_class: type[PipelineFlow],
        plan: Sequence[PipelineFlow],
        output_documents: tuple[Document, ...],
    ) -> FlowDirective:
        if flow_class is ConsumerFlow and not output_documents:
            return FlowDirective(action=FlowAction.SKIP, reason="no intermediate documents")
        return FlowDirective()

    @staticmethod
    def build_result(run_id: str, documents: tuple[Document, ...], options: FlowOptions) -> PlanResult:
        return PlanResult(success=True, output_count=len(documents))


@pytest.mark.asyncio
async def test_empty_flow_return_raises_type_error() -> None:
    """Flows returning empty tuple are rejected — every flow must produce output documents."""
    publisher = _MemoryPublisher()
    ConsumerFlow.ran = False
    doc = PlanInputDoc.create_root(name="in.txt", content="x", reason="gap7")
    with pytest.raises(TypeError, match="returned an empty tuple"):
        await _SkipWhenEmptyDeployment().run("gap7-skip", [doc], FlowOptions(), publisher=publisher)


@pytest.mark.asyncio
async def test_plan_next_flow_continues_when_output_docs_exist() -> None:
    """plan_next_flow receives output_documents and continues (doesn't skip)."""
    publisher = _MemoryPublisher()
    ConsumerFlow.ran = False
    doc = PlanInputDoc.create_root(name="in.txt", content="x", reason="gap7b")
    result = await _ContinueWhenDocsExist().run("gap7-continue", [doc], FlowOptions(), publisher=publisher)

    assert result.success
    assert ConsumerFlow.ran

    skipped = [e for e in publisher.events if isinstance(e, FlowSkippedEvent)]
    assert skipped == []


# ---------------------------------------------------------------------------
# resolve_document_inputs provenance preservation
# ---------------------------------------------------------------------------


class ResolveInputDoc(Document):
    """Document type for resolve test."""


@pytest.mark.asyncio
async def test_resolve_preserves_derived_from_on_input() -> None:
    """DocumentInput with derived_from preserves provenance through resolution."""
    root = ResolveInputDoc.create_root(name="root.txt", content="root", reason="test")
    inputs = [_DocumentInput(content="hello", name="x.txt", class_name="ResolveInputDoc", derived_from=(root.sha256,))]
    result = await resolve_document_inputs(inputs, [ResolveInputDoc])
    assert len(result) == 1
    assert result[0].derived_from == (root.sha256,)


@pytest.mark.asyncio
async def test_resolve_preserves_triggered_by_on_input() -> None:
    """DocumentInput with triggered_by preserves provenance through resolution."""
    trigger = ResolveInputDoc.create_root(name="trigger.txt", content="trigger", reason="test")
    inputs = [_DocumentInput(content="hello", name="x.txt", class_name="ResolveInputDoc", triggered_by=(trigger.sha256,))]
    result = await resolve_document_inputs(inputs, [ResolveInputDoc])
    assert len(result) == 1
    assert result[0].triggered_by == (trigger.sha256,)
