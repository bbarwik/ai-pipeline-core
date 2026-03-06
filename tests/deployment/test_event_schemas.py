"""Tests for deployment event payload schemas and field structure."""

from datetime import UTC, datetime
from typing import Any

import pytest

from ai_pipeline_core import DeploymentResult, Document, FlowOptions, PipelineDeployment
from ai_pipeline_core.deployment._types import (
    FlowCompletedEvent,
    RunCompletedEvent,
    RunStartedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
    _MemoryPublisher,
)
from ai_pipeline_core.document_store._memory import MemoryDocumentStore
from ai_pipeline_core.document_store._protocol import set_document_store
from ai_pipeline_core.pipeline import PipelineFlow


class _GapInputDoc(Document):
    """Input document for event gap regressions."""


class _GapOutputDoc(Document):
    """Output document for event gap regressions."""


class _GapResult(DeploymentResult):
    """Result model for event gap regressions."""


class _GapFlow(PipelineFlow):
    name = "gap-flow"

    async def run(self, run_id: str, documents: list[_GapInputDoc], options: FlowOptions) -> list[_GapOutputDoc]:
        _ = (run_id, options)
        return [_GapOutputDoc.derive(from_documents=documents, name="gap-out.txt", content="ok")]


class _FailingGapFlow(PipelineFlow):
    name = "failing-gap-flow"

    async def run(self, run_id: str, documents: list[_GapInputDoc], options: FlowOptions) -> list[_GapOutputDoc]:
        _ = (run_id, documents, options)
        raise RuntimeError("intentional flow failure")


class _GapDeployment(PipelineDeployment[FlowOptions, _GapResult]):
    def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
        _ = options
        return [_GapFlow()]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> _GapResult:
        _ = (run_id, documents, options)
        return _GapResult(success=True)


class _FailingGapDeployment(PipelineDeployment[FlowOptions, _GapResult]):
    def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
        _ = options
        return [_FailingGapFlow()]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> _GapResult:
        _ = (run_id, documents, options)
        return _GapResult(success=False, error="failed")


class _FakeClickHouseBackend:
    def __init__(self, total_cost: float) -> None:
        self._total_cost = total_cost

    @property
    def run_total_cost(self) -> float:
        return self._total_cost

    def track_run_start(
        self,
        *,
        execution_id: Any,
        run_id: str,
        flow_name: str,
        run_scope: str,
        parent_execution_id: Any,
        parent_span_id: str | None,
        metadata: dict[str, object] | None,
    ) -> datetime:
        _ = (execution_id, run_id, flow_name, run_scope, parent_execution_id, parent_span_id, metadata)
        return datetime.now(UTC)

    def track_run_end(
        self,
        *,
        execution_id: Any,
        run_id: str,
        flow_name: str,
        run_scope: str,
        status: Any,
        start_time: datetime,
        total_cost: float = 0.0,
        total_tokens: int = 0,
        metadata: dict[str, object] | None = None,
    ) -> None:
        _ = (execution_id, run_id, flow_name, run_scope, status, start_time, total_cost, total_tokens, metadata)

    def flush(self) -> None:
        return


def _make_input_doc() -> _GapInputDoc:
    return _GapInputDoc.create_root(name="input.txt", content="input", reason="event-gap-test")


@pytest.mark.asyncio
async def test_flow_plan_uses_flow_class_key() -> None:
    publisher = _MemoryPublisher()
    store = MemoryDocumentStore()
    set_document_store(store)
    try:
        await _GapDeployment().run("gap-run", [_make_input_doc()], FlowOptions(), publisher=publisher)
    finally:
        set_document_store(None)
        store.shutdown()

    started = [event for event in publisher.events if isinstance(event, RunStartedEvent)]
    assert len(started) == 1
    flow_plan = started[0].flow_plan
    assert len(flow_plan) == 1
    assert "flow_class" in flow_plan[0]
    assert "class" not in flow_plan[0]


@pytest.mark.asyncio
async def test_run_completed_carries_accumulated_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_backend = _FakeClickHouseBackend(total_cost=12.5)
    monkeypatch.setattr("ai_pipeline_core.deployment.base.get_clickhouse_backend", lambda: fake_backend)

    publisher = _MemoryPublisher()
    store = MemoryDocumentStore()
    set_document_store(store)
    try:
        await _GapDeployment().run("cost-run", [_make_input_doc()], FlowOptions(), publisher=publisher)
    finally:
        set_document_store(None)
        store.shutdown()

    completed = [event for event in publisher.events if isinstance(event, RunCompletedEvent)]
    assert len(completed) == 1
    assert completed[0].actual_cost == 12.5


def test_flow_failed_event_type_exists() -> None:
    from ai_pipeline_core.deployment._types import EventType, FlowFailedEvent

    assert hasattr(EventType, "FLOW_FAILED")
    event = FlowFailedEvent(run_id="r1", flow_name="a", flow_class="A", step=1, total_steps=3, error_message="boom")
    assert event.error_message == "boom"


@pytest.mark.asyncio
async def test_flow_failed_event_emitted_on_flow_exception() -> None:
    from ai_pipeline_core.deployment._types import FlowFailedEvent

    publisher = _MemoryPublisher()
    store = MemoryDocumentStore()
    set_document_store(store)
    try:
        with pytest.raises(RuntimeError, match="intentional flow failure"):
            await _FailingGapDeployment().run("flow-failed-run", [_make_input_doc()], FlowOptions(), publisher=publisher)
    finally:
        set_document_store(None)
        store.shutdown()

    flow_failed_events = [event for event in publisher.events if isinstance(event, FlowFailedEvent)]
    assert len(flow_failed_events) == 1
    assert flow_failed_events[0].flow_class == "_FailingGapFlow"
    assert flow_failed_events[0].step == 1
    assert flow_failed_events[0].error_message == "intentional flow failure"


def test_task_events_have_step() -> None:
    started = TaskStartedEvent(
        run_id="r1",
        flow_name="f",
        step=2,
        task_name="t",
        task_class="T",
        task_invocation_id="inv-1",
        parent_task=None,
        task_depth=0,
    )
    completed = TaskCompletedEvent(
        run_id="r1",
        flow_name="f",
        step=2,
        task_name="t",
        task_class="T",
        task_invocation_id="inv-1",
        parent_task=None,
        task_depth=0,
        duration_ms=10,
    )
    failed = TaskFailedEvent(
        run_id="r1",
        flow_name="f",
        step=2,
        task_name="t",
        task_class="T",
        task_invocation_id="inv-1",
        parent_task=None,
        task_depth=0,
        error_message="boom",
    )
    assert started.step == 2
    assert completed.step == 2
    assert failed.step == 2


def test_task_events_have_invocation_id() -> None:
    started = TaskStartedEvent(
        run_id="r1",
        flow_name="f",
        step=1,
        task_name="t",
        task_class="T",
        task_invocation_id="abc123",
        parent_task=None,
        task_depth=0,
    )
    completed = TaskCompletedEvent(
        run_id="r1",
        flow_name="f",
        step=1,
        task_name="t",
        task_class="T",
        task_invocation_id="abc123",
        parent_task=None,
        task_depth=0,
        duration_ms=10,
    )
    failed = TaskFailedEvent(
        run_id="r1",
        flow_name="f",
        step=1,
        task_name="t",
        task_class="T",
        task_invocation_id="abc123",
        parent_task=None,
        task_depth=0,
        error_message="boom",
    )
    assert started.task_invocation_id == "abc123"
    assert completed.task_invocation_id == "abc123"
    assert failed.task_invocation_id == "abc123"


def test_flow_completed_has_flow_class() -> None:
    event = FlowCompletedEvent(run_id="r1", flow_name="f", flow_class="F", step=1, total_steps=2, duration_ms=100)
    assert event.flow_class == "F"


def test_document_ref_has_publicly_visible() -> None:
    from ai_pipeline_core.deployment._types import DocumentRef

    ref = DocumentRef(sha256="abc", class_name="MyDoc", name="f.txt", publicly_visible=True)
    assert ref.publicly_visible is True

    ref_default = DocumentRef(sha256="abc", class_name="MyDoc", name="f.txt")
    assert ref_default.publicly_visible is False


def test_document_ref_has_provenance() -> None:
    from ai_pipeline_core.deployment._types import DocumentRef

    ref = DocumentRef(
        sha256="abc",
        class_name="MyDoc",
        name="f.txt",
        derived_from=("sha1", "https://example.com"),
        triggered_by=("sha2",),
    )
    assert ref.derived_from == ("sha1", "https://example.com")
    assert ref.triggered_by == ("sha2",)

    ref_default = DocumentRef(sha256="abc", class_name="MyDoc", name="f.txt")
    assert ref_default.derived_from == ()
    assert ref_default.triggered_by == ()


def test_document_ref_asdict_includes_new_fields() -> None:
    from dataclasses import asdict

    from ai_pipeline_core.deployment._types import DocumentRef

    ref = DocumentRef(
        sha256="abc",
        class_name="MyDoc",
        name="f.txt",
        publicly_visible=True,
        derived_from=("sha1",),
        triggered_by=("sha2",),
    )
    d = asdict(ref)
    assert d["publicly_visible"] is True
    assert d["derived_from"] == ("sha1",)
    assert d["triggered_by"] == ("sha2",)
