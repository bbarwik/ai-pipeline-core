"""Low-level ContextVar declarations for run context and document lifecycle tracking.

Extracted into a separate module to break the circular dependency between
document.py and context.py (which defines TaskDocumentContext that depends on Document).
"""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

from ai_pipeline_core.documents.types import DocumentSha256, RunScope

__all__ = [
    "RunContext",
    "TaskContext",
    "_suppress_document_registration",
    "get_run_context",
    "get_task_context",
    "is_registration_suppressed",
    "reset_run_context",
    "reset_task_context",
    "set_run_context",
    "set_task_context",
]


@dataclass(frozen=True, slots=True)
class RunContext:
    """Immutable context for a pipeline run, carried via ContextVar."""

    run_scope: RunScope


_run_context: ContextVar[RunContext | None] = ContextVar("_run_context", default=None)


def get_run_context() -> RunContext | None:
    """Get the current run context, or None if not set."""
    return _run_context.get()


def set_run_context(ctx: RunContext) -> Token[RunContext | None]:
    """Set the run context. Returns a token for restoring the previous value."""
    return _run_context.set(ctx)


def reset_run_context(token: Token[RunContext | None]) -> None:
    """Reset the run context to its previous value using a token from set_run_context."""
    _run_context.reset(token)


# --- Task-level document lifecycle tracking ---


@dataclass
class TaskContext:
    """Mutable set of document SHA256s created within the current task/flow."""

    created: set[DocumentSha256] = field(default_factory=set)


_task_context: ContextVar[TaskContext | None] = ContextVar("_task_context", default=None)
_suppress_registration: ContextVar[bool] = ContextVar("_suppress_registration", default=False)


def get_task_context() -> TaskContext | None:
    """Get the current task context, or None if not inside a pipeline task/flow."""
    return _task_context.get()


def set_task_context(ctx: TaskContext) -> Token[TaskContext | None]:
    """Set the task context. Returns a token for restoring the previous value."""
    return _task_context.set(ctx)


def reset_task_context(token: Token[TaskContext | None]) -> None:
    """Reset the task context to its previous value."""
    _task_context.reset(token)


@contextmanager
def _suppress_document_registration() -> Generator[None, None, None]:
    """Suppress document registration during deserialization (store loads, from_dict, etc.)."""
    token = _suppress_registration.set(True)
    try:
        yield
    finally:
        _suppress_registration.reset(token)


def is_registration_suppressed() -> bool:
    """Check if document registration is currently suppressed."""
    return _suppress_registration.get()
