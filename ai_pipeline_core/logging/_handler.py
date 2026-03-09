"""Root logger handler that captures execution-scoped logs for database storage."""

import json
import traceback
from datetime import UTC, datetime
from importlib import import_module
from typing import Any
from uuid import UUID

from ai_pipeline_core.database._types import ExecutionLog
from ai_pipeline_core.pipeline._execution_context import get_execution_context

__all__ = [
    "ExecutionLogHandler",
]

_logging: Any = import_module("logging")
_APPLICATION_LOG_LEVEL = _logging.INFO
_DEPENDENCY_LOG_LEVEL = _logging.WARNING
_FRAMEWORK_LOG_LEVEL = _logging.DEBUG
_SKIP_EXECUTION_LOG_ATTR = "_skip_execution_log"
_DEPENDENCY_LOGGER_PREFIXES = (
    "clickhouse_connect",
    "httpcore",
    "httpx",
    "litellm",
    "prefect",
)
_FRAMEWORK_LOGGER_PREFIX = "ai_pipeline_core"


def _safe_uuid(value: str | None) -> UUID | None:
    """Parse a UUID string, returning ``None`` for empty or invalid values."""
    if not value:
        return None
    try:
        return UUID(value)
    except ValueError:
        return None


def _matches_prefix(logger_name: str, prefix: str) -> bool:
    """Return whether a logger name matches a namespace prefix exactly or by descendant."""
    return logger_name == prefix or logger_name.startswith(f"{prefix}.")


def _classify_record(record: Any) -> tuple[str, int]:
    """Classify a log record and return the category plus minimum persisted level."""
    if getattr(record, "lifecycle", False):
        return "lifecycle", _logging.NOTSET

    if _matches_prefix(record.name, _FRAMEWORK_LOGGER_PREFIX):
        return "framework", _FRAMEWORK_LOG_LEVEL

    if any(_matches_prefix(record.name, prefix) for prefix in _DEPENDENCY_LOGGER_PREFIXES):
        return "dependency", _DEPENDENCY_LOG_LEVEL

    return "application", _APPLICATION_LOG_LEVEL


def _coerce_fields_json(record: Any) -> str:
    """Normalize structured log fields into a JSON string for persistence."""
    raw_fields = getattr(record, "fields_json", "{}")
    if isinstance(raw_fields, str):
        return raw_fields
    return json.dumps(raw_fields, default=str, sort_keys=True)


def _format_exception_text(record: Any) -> str:
    """Render ``exc_info`` into text, ignoring empty exception tuples."""
    if record.exc_info is None or record.exc_info[0] is None:
        return ""
    return "".join(traceback.format_exception(*record.exc_info))


class ExecutionLogHandler(_logging.Handler):
    """Route execution-scoped logs from the root logger into the active log buffer."""

    def emit(self, record: Any) -> None:
        """Append an execution-scoped log record to the active buffer when configured."""
        if getattr(record, _SKIP_EXECUTION_LOG_ATTR, False):
            return

        execution_ctx = get_execution_context()
        if execution_ctx is None or execution_ctx.log_buffer is None:
            return
        if execution_ctx.current_node_id is None or execution_ctx.deployment_id is None:
            return

        category, minimum_level = _classify_record(record)
        if record.levelno < minimum_level:
            return

        timestamp = datetime.fromtimestamp(record.created, tz=UTC)
        task_id = _safe_uuid(execution_ctx.task_frame.task_id) if execution_ctx.task_frame is not None else None
        root_deployment_id = execution_ctx.root_deployment_id or execution_ctx.deployment_id

        try:
            execution_ctx.log_buffer.append(
                ExecutionLog(
                    node_id=execution_ctx.current_node_id,
                    deployment_id=execution_ctx.deployment_id,
                    root_deployment_id=root_deployment_id,
                    flow_id=execution_ctx.flow_node_id,
                    task_id=task_id,
                    timestamp=timestamp,
                    sequence_no=0,
                    level=record.levelname,
                    category=category,
                    logger_name=record.name,
                    message=record.getMessage(),
                    event_type=str(getattr(record, "event_type", "")),
                    fields=_coerce_fields_json(record),
                    exception_text=_format_exception_text(record),
                )
            )
        except (AttributeError, OSError, OverflowError, TypeError, ValueError):
            self.handleError(record)
