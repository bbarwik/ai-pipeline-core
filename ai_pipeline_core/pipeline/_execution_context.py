"""Unified execution context for pipeline run, flow, and task scopes.

Replaces scattered ContextVars with a single shared context object that is
replaced at scope boundaries while intentionally sharing mutable runtime state
such as task handles and child-sequence counters across derived contexts.
"""

import json
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, replace
from datetime import datetime
from itertools import count
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from uuid import UUID

from ai_pipeline_core.database import DatabaseWriter
from ai_pipeline_core.deployment._types import ResultPublisher, _NoopPublisher
from ai_pipeline_core.documents import RunScope
from ai_pipeline_core.documents._context import TaskContext, reset_task_context, set_task_context
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.pipeline.limits import PipelineLimit, _SharedStatus

if TYPE_CHECKING:
    from ai_pipeline_core.logging import ExecutionLogBuffer

__all__ = [
    "ConversationTurnData",
    "ExecutionContext",
    "FlowFrame",
    "TaskFrame",
    "get_conversation_turns",
    "get_execution_context",
    "get_run_id",
    "pipeline_test_context",
    "record_lifecycle_event",
    "reset_conversation_turns",
    "reset_execution_context",
    "set_conversation_turns",
    "set_execution_context",
]

logger = get_pipeline_logger(__name__)


@dataclass(frozen=True, slots=True)
class ConversationTurnData:
    """Lightweight capture of a single Conversation.send() call for database recording.

    Appended to a contextvar list by Conversation._execute_send() so the task wrapper
    can create conversation_turn execution nodes without Conversation knowing about the database.
    """

    conversation_id: str
    conversation_name: str
    model: str
    cost_usd: float
    tokens_input: int
    tokens_output: int
    tokens_cache_read: int
    tokens_reasoning: int
    prompt_content: str
    response_content: str
    reasoning_content: str
    started_at: datetime
    ended_at: datetime
    time_taken: float
    first_token_time: float
    context_document_shas: tuple[str, ...]
    model_options_json: str
    response_format_class: str
    response_id: str
    citations_json: str
    replay_payload_json: str
    status: str = "completed"
    error_type: str = ""
    error_message: str = ""


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


@dataclass(slots=True)
class ExecutionContext:
    """Pipeline execution context propagated through async boundaries.

    The wrapper object is replaced as flow/task scopes change, but some nested
    mutable state is intentionally shared between derived contexts so child tasks
    and sequence counters stay coordinated across the whole run.
    """

    run_id: str
    run_scope: RunScope
    execution_id: UUID | None
    publisher: ResultPublisher
    limits: Mapping[str, PipelineLimit]
    limits_status: _SharedStatus
    flow_frame: FlowFrame | None = None
    task_frame: TaskFrame | None = None
    active_task_handles: set[object] = field(default_factory=set)

    database: DatabaseWriter | None = None
    deployment_id: UUID | None = None
    root_deployment_id: UUID | None = None
    parent_deployment_task_id: UUID | None = None
    deployment_name: str = ""
    current_node_id: UUID | None = None
    flow_node_id: UUID | None = None
    log_buffer: "ExecutionLogBuffer | None" = None
    _child_sequence_counters: "dict[UUID, count[int]]" = field(default_factory=dict)

    def next_child_sequence(self, parent_node_id: UUID) -> int:
        """Return the next monotonic sequence number for children of the given parent."""
        if parent_node_id not in self._child_sequence_counters:
            self._child_sequence_counters[parent_node_id] = count()
        return next(self._child_sequence_counters[parent_node_id])

    def with_flow(self, flow_frame: FlowFrame) -> "ExecutionContext":
        """Return a copy with a new flow frame and cleared task frame."""
        return replace(self, flow_frame=flow_frame, task_frame=None)

    def with_task(self, task_frame: TaskFrame) -> "ExecutionContext":
        """Return a copy with a new task frame."""
        return replace(self, task_frame=task_frame)

    def with_node(self, node_id: UUID) -> "ExecutionContext":
        """Return a copy with a new current node ID."""
        return replace(self, current_node_id=node_id)


_context: ContextVar[ExecutionContext | None] = ContextVar("pipeline_execution_context", default=None)


def get_execution_context() -> ExecutionContext | None:
    """Get the current execution context."""
    return _context.get()


def get_run_id() -> str:
    """Return the current run ID from the active execution context."""
    ctx = get_execution_context()
    if ctx is None:
        msg = (
            "get_run_id() called outside execution context. "
            "This function is available inside PipelineFlow.run() and PipelineTask.run() "
            "during deployment execution. "
            "In tests, wrap your code with pipeline_test_context(run_id='...')."
        )
        raise RuntimeError(msg)
    return ctx.run_id


def set_execution_context(ctx: ExecutionContext) -> Token[ExecutionContext | None]:
    """Set the execution context and return reset token."""
    return _context.set(ctx)


def reset_execution_context(token: Token[ExecutionContext | None]) -> None:
    """Reset execution context to the previous value."""
    _context.reset(token)


# --- Conversation turn capture ---

_conversation_turns: ContextVar[list[ConversationTurnData] | None] = ContextVar("_conversation_turns", default=None)


def get_conversation_turns() -> list[ConversationTurnData] | None:
    """Get the current conversation turn accumulator, or None if not inside a task."""
    return _conversation_turns.get()


def set_conversation_turns(turns: list[ConversationTurnData]) -> Token[list[ConversationTurnData] | None]:
    """Set the conversation turn accumulator. Returns a token for restoring the previous value."""
    return _conversation_turns.set(turns)


def reset_conversation_turns(token: Token[list[ConversationTurnData] | None]) -> None:
    """Reset the conversation turn accumulator to its previous value."""
    _conversation_turns.reset(token)


def record_lifecycle_event(event_type: str, message: str, **fields: Any) -> None:
    """Emit a structured lifecycle log event for the current execution scope."""
    logger.info(
        message,
        extra={
            "lifecycle": True,
            "event_type": event_type,
            "fields_json": json.dumps(fields, default=str, sort_keys=True),
        },
    )


@contextmanager
def pipeline_test_context(
    run_id: str = "test-run",
    publisher: ResultPublisher | None = None,
) -> Generator[ExecutionContext, None, None]:
    """Set up an execution + task context for tests without full deployment wiring.

    Yields:
        The active execution context for the test scope.
    """
    ctx = ExecutionContext(
        run_id=run_id,
        run_scope=RunScope(f"{run_id}/test"),
        execution_id=None,
        publisher=publisher or _NoopPublisher(),
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
