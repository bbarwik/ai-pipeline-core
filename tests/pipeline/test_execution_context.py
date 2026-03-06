"""Tests for unified execution context transitions."""

from types import MappingProxyType

from ai_pipeline_core.deployment._types import _NoopPublisher
from ai_pipeline_core.document_store._memory import MemoryDocumentStore
from ai_pipeline_core.documents import RunScope
from ai_pipeline_core.pipeline._execution_context import (
    ExecutionContext,
    FlowFrame,
    TaskFrame,
    get_execution_context,
    pipeline_test_context,
    reset_execution_context,
    set_execution_context,
)
from ai_pipeline_core.pipeline.limits import _SharedStatus


def _make_ctx() -> ExecutionContext:
    return ExecutionContext(
        run_id="test-run",
        run_scope=RunScope("test-run/scope"),
        execution_id=None,
        store=MemoryDocumentStore(),
        publisher=_NoopPublisher(),
        summary_generator=None,
        limits=MappingProxyType({}),
        limits_status=_SharedStatus(),
    )


def _make_flow_frame(**overrides: object) -> FlowFrame:
    defaults: dict[str, object] = {
        "name": "f",
        "flow_class_name": "F",
        "step": 1,
        "total_steps": 2,
        "flow_minutes": (1.0, 2.0),
        "completed_minutes": 0.0,
        "flow_params": {},
    }
    defaults.update(overrides)
    return FlowFrame(**defaults)  # type: ignore[arg-type]


def test_with_flow_returns_new_context_with_flow_frame() -> None:
    ctx = _make_ctx()
    frame = _make_flow_frame()
    new_ctx = ctx.with_flow(frame)
    assert new_ctx.flow_frame is frame
    assert new_ctx is not ctx
    assert ctx.flow_frame is None  # original unchanged


def test_with_task_returns_new_context_with_task_frame() -> None:
    ctx = _make_ctx()
    task_frame = TaskFrame(task_class_name="T", task_id="t1", depth=0)
    new_ctx = ctx.with_task(task_frame)
    assert new_ctx.task_frame is task_frame
    assert new_ctx is not ctx
    assert ctx.task_frame is None


def test_with_flow_clears_task_frame() -> None:
    ctx = _make_ctx()
    ctx_with_task = ctx.with_task(TaskFrame(task_class_name="T", task_id="t1", depth=0))
    assert ctx_with_task.task_frame is not None

    ctx_with_flow = ctx_with_task.with_flow(_make_flow_frame())
    assert ctx_with_flow.flow_frame is not None
    assert ctx_with_flow.task_frame is None


def test_with_flow_preserves_other_fields() -> None:
    ctx = _make_ctx()
    frame = _make_flow_frame()
    new_ctx = ctx.with_flow(frame)
    assert new_ctx.run_id == ctx.run_id
    assert new_ctx.run_scope == ctx.run_scope
    assert new_ctx.store is ctx.store
    assert new_ctx.publisher is ctx.publisher


def test_set_and_reset_execution_context() -> None:
    original = get_execution_context()
    ctx = _make_ctx()
    token = set_execution_context(ctx)
    assert get_execution_context() is ctx
    reset_execution_context(token)
    assert get_execution_context() is original


def test_pipeline_test_context_sets_and_restores() -> None:
    before = get_execution_context()
    with pipeline_test_context(run_id="ctx-test") as ctx:
        assert get_execution_context() is ctx
        assert ctx.run_id == "ctx-test"
        assert ctx.store is not None
    assert get_execution_context() is before


def test_pipeline_test_context_with_custom_publisher() -> None:
    pub = _NoopPublisher()
    with pipeline_test_context(publisher=pub) as ctx:
        assert ctx.publisher is pub


def test_task_frame_depth_and_parent() -> None:
    parent = TaskFrame(task_class_name="Parent", task_id="p1", depth=0)
    child = TaskFrame(task_class_name="Child", task_id="c1", depth=1, parent=parent)
    assert child.parent is parent
    assert child.depth == 1
    assert parent.parent is None
