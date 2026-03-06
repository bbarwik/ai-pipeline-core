"""Unified execution context for pipeline run, flow, and task scopes.

Replaces scattered ContextVars with a single immutable context object that is
replaced (not mutated) at scope boundaries.
"""

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any
from uuid import UUID

from ai_pipeline_core.deployment._types import ResultPublisher, _NoopPublisher
from ai_pipeline_core.document_store._memory import MemoryDocumentStore
from ai_pipeline_core.document_store._protocol import DocumentStore
from ai_pipeline_core.document_store._summary_worker import SummaryGenerator
from ai_pipeline_core.documents import RunScope
from ai_pipeline_core.documents._context import TaskContext, reset_task_context, set_task_context
from ai_pipeline_core.pipeline.limits import PipelineLimit, _SharedStatus

__all__ = [
    "ExecutionContext",
    "FlowFrame",
    "TaskFrame",
    "get_execution_context",
    "pipeline_test_context",
    "reset_execution_context",
    "set_execution_context",
]


@dataclass(frozen=True, slots=True)
class TaskFrame:
    """Identity of a task invocation in a nested task hierarchy."""

    task_class_name: str
    task_id: str
    depth: int
    parent: "TaskFrame | None" = None


@dataclass(frozen=True, slots=True)
class FlowFrame:
    """Flow execution state used for progress and task event enrichment."""

    name: str
    flow_class_name: str
    step: int
    total_steps: int
    flow_minutes: tuple[float, ...]
    completed_minutes: float
    flow_params: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Immutable pipeline execution context propagated through async boundaries."""

    run_id: str
    run_scope: RunScope
    execution_id: UUID | None
    store: DocumentStore | None
    publisher: ResultPublisher
    summary_generator: SummaryGenerator | None
    limits: Mapping[str, PipelineLimit]
    limits_status: _SharedStatus
    flow_frame: FlowFrame | None = None
    task_frame: TaskFrame | None = None
    active_task_handles: set[object] = field(default_factory=set)

    def with_flow(self, flow_frame: FlowFrame) -> "ExecutionContext":
        """Return a copy with a new flow frame and cleared task frame."""
        return replace(self, flow_frame=flow_frame, task_frame=None)

    def with_task(self, task_frame: TaskFrame) -> "ExecutionContext":
        """Return a copy with a new task frame."""
        return replace(self, task_frame=task_frame)


_context: ContextVar[ExecutionContext | None] = ContextVar("pipeline_execution_context", default=None)


def get_execution_context() -> ExecutionContext | None:
    """Get the current execution context."""
    return _context.get()


def set_execution_context(ctx: ExecutionContext) -> Token[ExecutionContext | None]:
    """Set the execution context and return reset token."""
    return _context.set(ctx)


def reset_execution_context(token: Token[ExecutionContext | None]) -> None:
    """Reset execution context to the previous value."""
    _context.reset(token)


@contextmanager
def pipeline_test_context(
    run_id: str = "test-run",
    store: DocumentStore | None = None,
    publisher: ResultPublisher | None = None,
) -> Generator[ExecutionContext, None, None]:
    """Set up an execution + task context for tests without full deployment wiring.

    Yields:
        The active execution context for the test scope.
    """
    owns_store = store is None
    active_store = store or MemoryDocumentStore()
    ctx = ExecutionContext(
        run_id=run_id,
        run_scope=RunScope(f"{run_id}/test"),
        execution_id=None,
        store=active_store,
        publisher=publisher or _NoopPublisher(),
        summary_generator=None,
        limits=MappingProxyType({}),
        limits_status=_SharedStatus(),
    )
    ctx_token = set_execution_context(ctx)
    task_token = set_task_context(TaskContext(scope_kind="test", task_class_name="pipeline_test_context"))
    try:
        yield ctx
    finally:
        reset_task_context(task_token)
        reset_execution_context(ctx_token)
        if owns_store:
            active_store.shutdown()
