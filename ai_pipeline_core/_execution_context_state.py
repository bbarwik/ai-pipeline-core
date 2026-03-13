"""Low-level execution context state shared across modules."""

from contextvars import ContextVar, Token
from typing import Any

__all__ = [
    "get_execution_context_state",
    "reset_execution_context_state",
    "set_execution_context_state",
]

_execution_context: ContextVar[Any | None] = ContextVar("pipeline_execution_context", default=None)


def get_execution_context_state() -> Any | None:
    """Return the raw execution context object for the current task."""
    return _execution_context.get()


def set_execution_context_state(ctx: Any) -> Token[Any | None]:
    """Bind the raw execution context object for the current task."""
    return _execution_context.set(ctx)


def reset_execution_context_state(token: Token[Any | None]) -> None:
    """Restore the previous execution context binding."""
    _execution_context.reset(token)
