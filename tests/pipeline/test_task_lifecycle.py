"""Tests for PipelineTask runtime lifecycle."""

import asyncio
import contextvars
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict

from ai_pipeline_core.deployment._types import TaskCompletedEvent, TaskFailedEvent, TaskStartedEvent, _MemoryPublisher
from ai_pipeline_core.documents import Document
from ai_pipeline_core.pipeline import FlowOptions, PipelineFlow, PipelineTask, TaskHandle
from ai_pipeline_core.pipeline._execution_context import (
    FlowFrame,
    pipeline_test_context,
    reset_execution_context,
    set_execution_context,
)
from ai_pipeline_core.pipeline._task import _collect_documents


class _InDoc(Document):
    """Test input document."""


class _OutDoc(Document):
    """Test output document."""


class _VisibleOutDoc(Document):
    """Output document with publicly_visible=True."""

    publicly_visible = True


class _FlowOutDoc(Document):
    """Flow output doc for expected_tasks regression test."""


class _PassthroughTask(PipelineTask):
    @classmethod
    async def run(cls, documents: list[_InDoc]) -> list[_OutDoc]:
        return [_OutDoc(name="out.txt", content=b"output")]


class _FailingTask(PipelineTask):
    @classmethod
    async def run(cls, documents: list[_InDoc]) -> list[_OutDoc]:
        raise ValueError("task failed deliberately")


class _RetryCounterTask(PipelineTask):
    retries = 2
    retry_delay_seconds = 0
    _calls: int = 0

    @classmethod
    async def run(cls, documents: list[_InDoc]) -> list[_OutDoc]:
        _RetryCounterTask._calls += 1
        if _RetryCounterTask._calls < 3:
            raise ValueError(f"attempt {_RetryCounterTask._calls}")
        return [_OutDoc(name="retry_out.txt", content=b"ok")]


class _ExhaustedRetryTask(PipelineTask):
    retries = 1
    retry_delay_seconds = 0

    @classmethod
    async def run(cls, documents: list[_InDoc]) -> list[_OutDoc]:
        raise ValueError("always fails")


class _TimeoutTask(PipelineTask):
    timeout_seconds = 1

    @classmethod
    async def run(cls, documents: list[_InDoc]) -> list[_OutDoc]:
        await asyncio.sleep(60)
        return []


class _CacheableTask(PipelineTask):
    cacheable = True
    _run_count: int = 0

    @classmethod
    async def run(cls, documents: list[_InDoc]) -> list[_OutDoc]:
        _CacheableTask._run_count += 1
        return [_OutDoc(name="cached.txt", content=b"fresh-data")]


class _VisibleOutputTask(PipelineTask):
    @classmethod
    async def run(cls, documents: list[_InDoc]) -> list[_VisibleOutDoc]:
        return [_VisibleOutDoc.derive(from_documents=tuple(documents), name="visible.txt", content="visible")]


class _ParentTask(PipelineTask):
    """Outer task that invokes a child task."""

    @classmethod
    async def run(cls, documents: list[_InDoc]) -> list[_OutDoc]:
        return await _PassthroughTask.run(documents)


class _EmptyResultTask(PipelineTask):
    @classmethod
    async def run(cls, documents: list[_InDoc]) -> list[_OutDoc]:
        return []


class _CustomNameTask(PipelineTask):
    name = "custom-display-name"

    @classmethod
    async def run(cls, documents: list[_InDoc]) -> list[_OutDoc]:
        return [_OutDoc(name="custom-name.txt", content=b"custom")]


class _FlowWithCustomTask(PipelineFlow):
    async def run(self, run_id: str, documents: list[_InDoc], options: FlowOptions) -> list[_FlowOutDoc]:
        _ = (run_id, options)
        await _CustomNameTask.run(documents)
        return [_FlowOutDoc.derive(from_documents=documents, name="flow-out.txt", content="ok")]


def _make_flow_frame() -> FlowFrame:
    return FlowFrame(
        name="test-flow",
        flow_class_name="MockFlow",
        step=1,
        total_steps=1,
        flow_minutes=(1.0,),
        completed_minutes=0.0,
        flow_params={},
    )


def _make_input() -> _InDoc:
    return _InDoc(name="input.txt", content=b"test-input")


@pytest.mark.asyncio
async def test_task_run_returns_handle() -> None:
    with pipeline_test_context():
        handle: Any = _PassthroughTask.run([_make_input()])
        assert isinstance(handle, TaskHandle)
        result = await handle
        assert len(result) == 1
        assert isinstance(result[0], _OutDoc)


@pytest.mark.asyncio
async def test_task_run_is_directly_awaitable() -> None:
    with pipeline_test_context():
        result: list[Any] = await _PassthroughTask.run([_make_input()])
        assert len(result) == 1


def test_expected_tasks_uses_overridable_name() -> None:
    tasks = _FlowWithCustomTask.expected_tasks()
    assert "custom-display-name" in tasks
    assert "_CustomNameTask" not in tasks


@pytest.mark.asyncio
async def test_task_requires_execution_context() -> None:
    with pytest.raises(RuntimeError, match="outside pipeline execution context"):
        run_result: Any = _PassthroughTask.run([_make_input()])  # noqa: F841


@pytest.mark.asyncio
async def test_task_runtime_validates_argument_types() -> None:
    with pipeline_test_context():
        with pytest.raises(TypeError, match="invalid value for 'documents'"):
            run_result: Any = _PassthroughTask.run((_make_input(),))  # noqa: F841


@pytest.mark.asyncio
async def test_empty_result_is_valid() -> None:
    with pipeline_test_context():
        result = await _EmptyResultTask.run([_make_input()])
        assert result == []


@pytest.mark.asyncio
async def test_task_retries_until_success() -> None:
    _RetryCounterTask._calls = 0
    with pipeline_test_context():
        result = await _RetryCounterTask.run([_make_input()])
        assert len(result) == 1
        assert _RetryCounterTask._calls == 3


@pytest.mark.asyncio
async def test_task_retries_exhausted_raises() -> None:
    with pipeline_test_context():
        with pytest.raises(ValueError, match="always fails"):
            await _ExhaustedRetryTask.run([_make_input()])


@pytest.mark.asyncio
async def test_task_timeout_raises() -> None:
    with pipeline_test_context():
        with pytest.raises(TimeoutError):
            await _TimeoutTask.run([_make_input()])


@pytest.mark.asyncio
async def test_task_started_and_completed_events() -> None:
    publisher = _MemoryPublisher()
    with pipeline_test_context(publisher=publisher) as ctx:
        token = set_execution_context(ctx.with_flow(_make_flow_frame()))
        try:
            await _PassthroughTask.run([_make_input()])
        finally:
            reset_execution_context(token)

    started = [event for event in publisher.events if isinstance(event, TaskStartedEvent)]
    completed = [event for event in publisher.events if isinstance(event, TaskCompletedEvent)]
    assert len(started) == 1
    assert started[0].task_class == "_PassthroughTask"
    assert started[0].flow_name == "test-flow"
    assert started[0].step == 1
    assert started[0].task_invocation_id
    assert started[0].task_depth == 0
    assert started[0].parent_task is None
    assert len(completed) == 1
    assert completed[0].task_class == "_PassthroughTask"
    assert completed[0].step == 1
    assert completed[0].task_invocation_id == started[0].task_invocation_id
    assert completed[0].duration_ms >= 0


@pytest.mark.asyncio
async def test_task_failed_event() -> None:
    publisher = _MemoryPublisher()
    with pipeline_test_context(publisher=publisher) as ctx:
        token = set_execution_context(ctx.with_flow(_make_flow_frame()))
        try:
            with pytest.raises(ValueError, match="task failed deliberately"):
                await _FailingTask.run([_make_input()])
        finally:
            reset_execution_context(token)

    started = [event for event in publisher.events if isinstance(event, TaskStartedEvent)]
    failed = [event for event in publisher.events if isinstance(event, TaskFailedEvent)]
    assert len(started) == 1
    assert len(failed) == 1
    assert failed[0].task_class == "_FailingTask"
    assert failed[0].step == 1
    assert failed[0].task_invocation_id == started[0].task_invocation_id
    assert "task failed deliberately" in failed[0].error_message


@pytest.mark.asyncio
async def test_no_events_without_flow_frame() -> None:
    publisher = _MemoryPublisher()
    with pipeline_test_context(publisher=publisher):
        await _PassthroughTask.run([_make_input()])

    task_events = [event for event in publisher.events if isinstance(event, (TaskStartedEvent, TaskCompletedEvent, TaskFailedEvent))]
    assert task_events == []


@pytest.mark.asyncio
async def test_subtask_depth_tracking() -> None:
    publisher = _MemoryPublisher()
    with pipeline_test_context(publisher=publisher) as ctx:
        token = set_execution_context(ctx.with_flow(_make_flow_frame()))
        try:
            await _ParentTask.run([_make_input()])
        finally:
            reset_execution_context(token)

    started = [event for event in publisher.events if isinstance(event, TaskStartedEvent)]
    assert len(started) == 2

    parent_event = next(event for event in started if event.task_class == "_ParentTask")
    child_event = next(event for event in started if event.task_class == "_PassthroughTask")
    assert parent_event.task_depth == 0
    assert parent_event.parent_task is None
    assert child_event.task_depth == 1
    assert child_event.parent_task == "_ParentTask"


@pytest.mark.asyncio
async def test_cacheable_task_caches_result() -> None:
    _CacheableTask._run_count = 0
    with pipeline_test_context():
        document = _make_input()

        result1 = await _CacheableTask.run([document])
        assert _CacheableTask._run_count == 1
        assert len(result1) == 1

        result2 = await _CacheableTask.run([document])
        assert _CacheableTask._run_count == 1
        assert len(result2) == 1


@pytest.mark.asyncio
async def test_cacheable_task_different_input_reruns() -> None:
    _CacheableTask._run_count = 0
    with pipeline_test_context():
        doc1 = _InDoc(name="a.txt", content=b"aaa")
        doc2 = _InDoc(name="b.txt", content=b"bbb")

        await _CacheableTask.run([doc1])
        assert _CacheableTask._run_count == 1

        await _CacheableTask.run([doc2])
        assert _CacheableTask._run_count == 2


class _CancelTask(PipelineTask):
    @classmethod
    async def run(cls, documents: list[_InDoc]) -> list[_OutDoc]:
        raise asyncio.CancelledError()


@pytest.mark.asyncio
async def test_cancelled_error_propagates_not_returned_as_data() -> None:
    """CancelledError must be raised, not returned as list[Document]."""
    with pipeline_test_context():
        with pytest.raises(asyncio.CancelledError):
            await _CancelTask.run([_make_input()])


class _CostTask(PipelineTask):
    expected_cost: ClassVar[float | None] = 1.0
    trace_cost: ClassVar[float | None] = 2.0

    @classmethod
    async def run(cls, documents: list[_InDoc]) -> list[_OutDoc]:
        return [_OutDoc(name="out.txt", content=b"x")]


@pytest.mark.asyncio
async def test_task_cost_attributes_set_inside_traced_span(monkeypatch: pytest.MonkeyPatch) -> None:
    """expected_cost and trace_cost must be set while the task span is active."""
    in_span = contextvars.ContextVar("in_span", default=False)
    calls: list[bool] = []

    def fake_trace_decorator(fn):
        async def wrapped():
            token = in_span.set(True)
            try:
                return await fn()
            finally:
                in_span.reset(token)

        return wrapped

    monkeypatch.setattr(_CostTask, "_trace_decorator", staticmethod(fake_trace_decorator))

    def fake_set_span_attributes(_attrs):
        calls.append(in_span.get())

    def fake_set_trace_cost(_cost):
        calls.append(in_span.get())

    monkeypatch.setattr("ai_pipeline_core.pipeline._task.Laminar.set_span_attributes", fake_set_span_attributes)
    monkeypatch.setattr("ai_pipeline_core.pipeline._task.set_trace_cost", fake_set_trace_cost)

    with pipeline_test_context():
        await _CostTask.run([_make_input()])

    assert calls, "No cost-related calls recorded"
    assert all(calls), "Cost attributes were set outside the traced span"


@pytest.mark.asyncio
async def test_task_span_has_task_class_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    span_attributes_calls: list[dict[str, Any]] = []

    def fake_set_span_attributes(attrs: dict[str, Any]) -> None:
        span_attributes_calls.append(attrs)

    monkeypatch.setattr("ai_pipeline_core.pipeline._task.Laminar.set_span_attributes", fake_set_span_attributes)

    with pipeline_test_context():
        await _PassthroughTask.run([_make_input()])

    assert any("pipeline.task_class" in attrs and attrs["pipeline.task_class"] == "_PassthroughTask" for attrs in span_attributes_calls)


@pytest.mark.asyncio
async def test_task_span_has_flow_step_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    span_attributes_calls: list[dict[str, Any]] = []

    def fake_set_span_attributes(attrs: dict[str, Any]) -> None:
        span_attributes_calls.append(attrs)

    monkeypatch.setattr("ai_pipeline_core.pipeline._task.Laminar.set_span_attributes", fake_set_span_attributes)

    with pipeline_test_context() as ctx:
        token = set_execution_context(ctx.with_flow(_make_flow_frame()))
        try:
            await _PassthroughTask.run([_make_input()])
        finally:
            reset_execution_context(token)

    assert any("pipeline.flow_step" in attrs and attrs["pipeline.flow_step"] == 1 for attrs in span_attributes_calls)


def test_collect_documents_recurses_into_frozen_basemodel() -> None:
    """Documents nested in frozen BaseModel fields must be collected."""

    class NestedDoc(Document):
        """Doc inside a model."""

    class Config(BaseModel):
        model_config = ConfigDict(frozen=True)
        source: NestedDoc
        label: str

    doc = NestedDoc(name="nested.txt", content=b"data")
    config = Config(source=doc, label="test")

    collected: list[Document] = []
    _collect_documents(config, collected)
    assert len(collected) == 1
    assert collected[0].sha256 == doc.sha256


@pytest.mark.asyncio
async def test_output_refs_carry_publicly_visible() -> None:
    publisher = _MemoryPublisher()
    with pipeline_test_context(publisher=publisher) as ctx:
        token = set_execution_context(ctx.with_flow(_make_flow_frame()))
        try:
            await _VisibleOutputTask.run([_make_input()])
        finally:
            reset_execution_context(token)

    completed = [event for event in publisher.events if isinstance(event, TaskCompletedEvent)]
    assert len(completed) == 1
    assert len(completed[0].output_documents) == 1
    ref = completed[0].output_documents[0]
    assert ref.publicly_visible is True
    assert ref.class_name == "_VisibleOutDoc"


@pytest.mark.asyncio
async def test_output_refs_carry_provenance() -> None:
    publisher = _MemoryPublisher()
    with pipeline_test_context(publisher=publisher) as ctx:
        token = set_execution_context(ctx.with_flow(_make_flow_frame()))
        try:
            await _VisibleOutputTask.run([_make_input()])
        finally:
            reset_execution_context(token)

    completed = [event for event in publisher.events if isinstance(event, TaskCompletedEvent)]
    assert len(completed) == 1
    ref = completed[0].output_documents[0]
    # _VisibleOutputTask.run derives from the input doc
    assert len(ref.derived_from) > 0
    assert ref.triggered_by == ()
