"""Logging infrastructure for AI Pipeline Core.

Provides a configured stdlib logging facade and execution-log capture primitives.
Prefer get_pipeline_logger instead of logging.getLogger to ensure consistent setup.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .logging_config import LoggingConfig, get_pipeline_logger, setup_logging

if TYPE_CHECKING:
    from ._buffer import ExecutionLogBuffer
    from ._handler import ExecutionLogHandler

__all__ = [
    "ExecutionLogBuffer",
    "ExecutionLogHandler",
    "LoggingConfig",
    "get_pipeline_logger",
    "setup_logging",
]


def __getattr__(name: str) -> Any:
    """Resolve logging exports lazily to avoid import cycles during startup."""
    if name == "ExecutionLogBuffer":
        return getattr(import_module("ai_pipeline_core.logging._buffer"), name)
    if name == "ExecutionLogHandler":
        return getattr(import_module("ai_pipeline_core.logging._handler"), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
