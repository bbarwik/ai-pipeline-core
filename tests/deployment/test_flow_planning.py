# pyright: reportPrivateUsage=false
"""Tests for plan_next_flow skip/continue logic and resolve_document_inputs provenance rejection."""

from collections.abc import Sequence

import pytest

from ai_pipeline_core import DeploymentResult, Document, FlowOptions, PipelineDeployment
from ai_pipeline_core.deployment import FlowAction, FlowDirective, _MemoryPublisher
from ai_pipeline_core.deployment._resolve import DocumentInput, resolve_document_inputs
from ai_pipeline_core.deployment._types import FlowSkippedEvent
from ai_pipeline_core.document_store._memory import MemoryDocumentStore
from ai_pipeline_core.document_store._protocol import set_document_store
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
    async def run(cls, documents: list[PlanInputDoc]) -> list[PlanMiddleDoc]:
        return [PlanMiddleDoc.derive(from_documents=(d,), name=f"mid_{d.name}", content="mid") for d in documents]


class ToOutputTask(PipelineTask):
    @classmethod
    async def run(cls, documents: list[PlanMiddleDoc]) -> list[PlanOutputDoc]:
        return [PlanOutputDoc.derive(from_documents=(d,), name=f"out_{d.name}", content="out") for d in documents]


class ProducerFlow(PipelineFlow):
    name = "producer"

    async def run(self, run_id: str, documents: list[PlanInputDoc], options: FlowOptions) -> list[PlanMiddleDoc]:
        return await ToMiddleTask.run(documents)


class EmptyProducerFlow(PipelineFlow):
    name = "empty-producer"

    async def run(self, run_id: str, documents: list[PlanInputDoc], options: FlowOptions) -> list[PlanMiddleDoc]:
        return []


class ConsumerFlow(PipelineFlow):
    name = "consumer"
    ran = False

    async def run(self, run_id: str, documents: list[PlanMiddleDoc], options: FlowOptions) -> list[PlanOutputDoc]:
        type(self).ran = True
        return await ToOutputTask.run(documents)


# ---------------------------------------------------------------------------
# plan_next_flow with actual previous_output_documents
# ---------------------------------------------------------------------------


class _SkipWhenEmptyDeployment(PipelineDeployment[FlowOptions, PlanResult]):
    """Skips consumer flow when producer outputs zero documents."""

    def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
        return [EmptyProducerFlow(), ConsumerFlow()]

    def plan_next_flow(
        self,
        flow_class: type[PipelineFlow],
        plan: Sequence[PipelineFlow],
        output_documents: list[Document],
    ) -> FlowDirective:
        if flow_class is ConsumerFlow and not output_documents:
            return FlowDirective(action=FlowAction.SKIP, reason="no intermediate documents")
        return FlowDirective()

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> PlanResult:
        return PlanResult(success=True, output_count=len(documents))


class _ContinueWhenDocsExist(PipelineDeployment[FlowOptions, PlanResult]):
    """Continues consumer flow when producer produces documents."""

    def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
        return [ProducerFlow(), ConsumerFlow()]

    def plan_next_flow(
        self,
        flow_class: type[PipelineFlow],
        plan: Sequence[PipelineFlow],
        output_documents: list[Document],
    ) -> FlowDirective:
        if flow_class is ConsumerFlow and not output_documents:
            return FlowDirective(action=FlowAction.SKIP, reason="no intermediate documents")
        return FlowDirective()

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> PlanResult:
        return PlanResult(success=True, output_count=len(documents))


@pytest.mark.asyncio
async def test_plan_next_flow_skips_when_no_output_docs() -> None:
    """plan_next_flow receives empty output_documents and skips next flow."""
    store = MemoryDocumentStore()
    publisher = _MemoryPublisher()
    set_document_store(store)
    ConsumerFlow.ran = False
    try:
        doc = PlanInputDoc.create_root(name="in.txt", content="x", reason="gap7")
        result = await _SkipWhenEmptyDeployment().run("gap7-skip", [doc], FlowOptions(), publisher=publisher)
    finally:
        set_document_store(None)
        store.shutdown()

    assert result.success
    assert not ConsumerFlow.ran

    skipped = [e for e in publisher.events if isinstance(e, FlowSkippedEvent)]
    assert len(skipped) == 1
    assert skipped[0].flow_name == "consumer"
    assert "no intermediate documents" in skipped[0].reason


@pytest.mark.asyncio
async def test_plan_next_flow_continues_when_output_docs_exist() -> None:
    """plan_next_flow receives output_documents and continues (doesn't skip)."""
    store = MemoryDocumentStore()
    publisher = _MemoryPublisher()
    set_document_store(store)
    ConsumerFlow.ran = False
    try:
        doc = PlanInputDoc.create_root(name="in.txt", content="x", reason="gap7b")
        result = await _ContinueWhenDocsExist().run("gap7-continue", [doc], FlowOptions(), publisher=publisher)
    finally:
        set_document_store(None)
        store.shutdown()

    assert result.success
    assert ConsumerFlow.ran

    skipped = [e for e in publisher.events if isinstance(e, FlowSkippedEvent)]
    assert skipped == []


# ---------------------------------------------------------------------------
# resolve_document_inputs provenance rejection
# ---------------------------------------------------------------------------


class ResolveInputDoc(Document):
    """Document type for resolve test."""


@pytest.mark.asyncio
async def test_resolve_rejects_derived_from_on_input() -> None:
    """DocumentInput with derived_from raises ValueError."""
    inputs = [DocumentInput(content="hello", name="x.txt", class_name="ResolveInputDoc", derived_from=("SOMESHA256",))]
    with pytest.raises(ValueError, match="cannot set derived_from"):
        await resolve_document_inputs(inputs, [ResolveInputDoc])


@pytest.mark.asyncio
async def test_resolve_rejects_triggered_by_on_input() -> None:
    """DocumentInput with triggered_by raises ValueError."""
    inputs = [DocumentInput(content="hello", name="x.txt", class_name="ResolveInputDoc", triggered_by=("SOMESHA256",))]
    with pytest.raises(ValueError, match="cannot set derived_from"):
        await resolve_document_inputs(inputs, [ResolveInputDoc])
