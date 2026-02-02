"""Low-level ContextVar declarations for document registration and task context.

Extracted into a separate module to break the circular dependency between
document.py (which needs suppression/task-context checks) and context.py
(which defines the full TaskDocumentContext that depends on Document).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass

# --- Run context ---


@dataclass(frozen=True, slots=True)
class RunContext:
    """Immutable context for a pipeline run, carried via ContextVar."""

    run_scope: str


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


# --- Suppression flag ---

_suppression_flag: ContextVar[bool] = ContextVar("_document_registration_suppressed", default=False)


def is_registration_suppressed() -> bool:
    """Check if document registration is currently suppressed."""
    return _suppression_flag.get()


@contextmanager
def suppress_registration() -> Iterator[None]:
    """Context manager that suppresses Document registration with TaskDocumentContext.

    Used during model_validate() and other internal Pydantic operations that
    construct intermediate Document objects that should not be tracked.
    """
    token = _suppression_flag.set(True)
    try:
        yield
    finally:
        _suppression_flag.reset(token)


# --- Task document context ContextVar ---

# Forward reference: the actual TaskDocumentContext class lives in context.py.
# Here we only manage the ContextVar holding it.

_task_context: ContextVar[object | None] = ContextVar("_task_context", default=None)


def get_task_context() -> object | None:
    """Get the current task document context, or None if not inside a pipeline task."""
    return _task_context.get()


def set_task_context(ctx: object) -> Token[object | None]:
    """Set the task document context. Returns a token for restoring the previous value."""
    return _task_context.set(ctx)


def reset_task_context(token: Token[object | None]) -> None:
    """Reset the task document context to its previous value."""
    _task_context.reset(token)
