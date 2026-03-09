"""Helper functions for pipeline deployments."""

import asyncio
import contextlib
import hashlib
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from importlib import import_module
from threading import Lock
from typing import Any
from uuid import UUID, uuid4

from ai_pipeline_core.database import Database, ExecutionLog, ExecutionNode, NodeKind, NodeStatus
from ai_pipeline_core.documents import Document, RunScope
from ai_pipeline_core.exceptions import LLMError, PipelineCoreError
from ai_pipeline_core.logging import ExecutionLogBuffer, ExecutionLogHandler, get_pipeline_logger
from ai_pipeline_core.logging._buffer import MAX_PENDING_EXECUTION_LOGS
from ai_pipeline_core.pipeline._execution_context import (
    get_execution_context,
    record_lifecycle_event,
    reset_execution_context,
    set_execution_context,
)
from ai_pipeline_core.pipeline.options import FlowOptions
from ai_pipeline_core.settings import Settings

from ._types import ErrorCode, FlowSkippedEvent, ResultPublisher, _NoopPublisher

logger = get_pipeline_logger(__name__)
_logging: Any = import_module("logging")

__all__ = [
    "MAX_RUN_ID_LENGTH",
    "SKIP_EXECUTION_LOG_ATTR",
    "_CLI_FIELDS",
    "_HANDLE_CANCEL_GRACE_SECONDS",
    "_HEARTBEAT_INTERVAL_SECONDS",
    "_MILLISECONDS_PER_SECOND",
    "_build_log_summary",
    "_classify_error",
    "_compute_run_scope",
    "_create_publisher",
    "_ensure_execution_log_handler_installed",
    "_execution_log_flush_loop",
    "_heartbeat_loop",
    "_record_terminal_flow_node",
    "class_name_to_deployment_name",
    "extract_generic_params",
    "validate_run_id",
]

_RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
MAX_RUN_ID_LENGTH = 100

# Fields added by run_cli()'s _CliOptions that should not affect fingerprints (run scope or remote run_id)
_CLI_FIELDS: frozenset[str] = frozenset({"working_directory", "run_id", "start", "end"})

_HEARTBEAT_INTERVAL_SECONDS = 30
_MILLISECONDS_PER_SECOND = 1000
_HANDLE_CANCEL_GRACE_SECONDS = 5
LOG_BUFFER_FLUSH_INTERVAL_SECONDS = 2
SKIP_EXECUTION_LOG_ATTR = "_skip_execution_log"
_execution_log_handler_lock = Lock()


def _trim_pending_execution_logs(pending_logs: list[ExecutionLog]) -> tuple[list[ExecutionLog], int]:
    """Cap pending execution logs and report how many oldest entries were dropped."""
    if len(pending_logs) <= MAX_PENDING_EXECUTION_LOGS:
        return pending_logs, 0
    dropped_count = len(pending_logs) - MAX_PENDING_EXECUTION_LOGS
    return pending_logs[dropped_count:], dropped_count


def validate_run_id(run_id: str) -> None:
    """Validate run_id: alphanumeric + underscore + hyphen, 1-100 chars.

    Must be called at deployment entry points (PipelineDeployment.run, RemoteDeployment._execute, CLI).
    """
    if not run_id:
        raise ValueError("run_id must not be empty")
    if len(run_id) > MAX_RUN_ID_LENGTH:
        raise ValueError(
            f"run_id '{run_id[:20]}...' is {len(run_id)} chars, max is {MAX_RUN_ID_LENGTH}. Shorten the base run_id before passing to the deployment."
        )
    if not _RUN_ID_PATTERN.match(run_id):
        raise ValueError(
            f"run_id '{run_id}' contains invalid characters. "
            f"Only alphanumeric characters, underscores, and hyphens are allowed (pattern: {_RUN_ID_PATTERN.pattern})."
        )


def class_name_to_deployment_name(class_name: str) -> str:
    """Convert PascalCase to kebab-case: ResearchPipeline -> research-pipeline."""
    name = re.sub(r"(?<!^)(?=[A-Z])", "-", class_name)
    return name.lower()


def extract_generic_params(cls: type, base_class: type) -> tuple[Any, ...]:
    """Extract Generic type arguments from a class's base.

    Works with any number of Generic parameters (2 for PipelineDeployment, 3 for RemoteDeployment).
    Returns () if the base class is not found in __orig_bases__.
    """
    for base in getattr(cls, "__orig_bases__", []):
        origin = getattr(base, "__origin__", None)
        if origin is base_class:
            args = getattr(base, "__args__", ())
            if args:
                return args

    return ()


def _classify_error(exc: BaseException) -> ErrorCode:
    """Map exception to ErrorCode enum value."""
    if isinstance(exc, LLMError):
        return ErrorCode.PROVIDER_ERROR
    if isinstance(exc, asyncio.CancelledError):
        return ErrorCode.CANCELLED
    if isinstance(exc, TimeoutError):
        return ErrorCode.DURATION_EXCEEDED
    if isinstance(exc, (ValueError, TypeError)):
        return ErrorCode.INVALID_INPUT
    if isinstance(exc, PipelineCoreError):
        return ErrorCode.PIPELINE_ERROR
    return ErrorCode.UNKNOWN


def _create_publisher(settings_obj: Settings, service_type: str) -> ResultPublisher:
    """Create publisher based on environment and deployment configuration.

    Returns PubSubPublisher when Pub/Sub is configured and service_type is set,
    _NoopPublisher otherwise.
    """
    if not service_type:
        return _NoopPublisher()
    if settings_obj.pubsub_project_id and settings_obj.pubsub_topic_id:
        from ._pubsub import PubSubPublisher

        return PubSubPublisher(
            project_id=settings_obj.pubsub_project_id,
            topic_id=settings_obj.pubsub_topic_id,
            service_type=service_type,
        )
    return _NoopPublisher()


def _compute_run_scope(run_id: str, documents: Sequence[Document], options: FlowOptions) -> RunScope:
    """Compute a run scope that fingerprints inputs and options.

    Different inputs or options produce a different scope, preventing
    stale cache hits when re-running with the same run_id.
    """
    exclude = set(_CLI_FIELDS & set(type(options).model_fields))
    options_json = options.model_dump_json(exclude=exclude, exclude_none=True)

    if not documents:
        fingerprint = hashlib.sha256(options_json.encode()).hexdigest()[:16]
        return RunScope(f"{run_id}:{fingerprint}")

    sha256s = sorted(doc.sha256 for doc in documents)
    fingerprint = hashlib.sha256(f"{':'.join(sha256s)}|{options_json}".encode()).hexdigest()[:16]
    return RunScope(f"{run_id}:{fingerprint}")


def _ensure_execution_log_handler_installed() -> None:
    """Install the process-wide execution log handler on the root logger once."""
    root_logger = _logging.getLogger()
    with _execution_log_handler_lock:
        if any(isinstance(handler, ExecutionLogHandler) for handler in root_logger.handlers):
            return
        root_logger.addHandler(ExecutionLogHandler())


def _build_log_summary(log_buffer: ExecutionLogBuffer | None, node_id: UUID) -> dict[str, int | str]:
    """Return lightweight log counters for the given execution node."""
    if log_buffer is None:
        return {"total": 0, "warnings": 0, "errors": 0, "last_error": ""}
    return log_buffer.get_summary(node_id)


def _consume_log_summary(log_buffer: ExecutionLogBuffer | None, node_id: UUID) -> dict[str, int | str]:
    """Return and clear lightweight log counters for a terminal execution node."""
    if log_buffer is None:
        return {"total": 0, "warnings": 0, "errors": 0, "last_error": ""}
    return log_buffer.consume_summary(node_id)


async def _flush_execution_logs_once(
    database: Database | None,
    log_buffer: ExecutionLogBuffer | None,
    pending_logs: list[ExecutionLog],
) -> list[ExecutionLog]:
    """Drain buffered logs, save them, and retain any batch that failed to persist."""
    if log_buffer is not None:
        pending_logs.extend(log_buffer.drain())
    pending_logs, dropped_from_backlog = _trim_pending_execution_logs(pending_logs)
    if dropped_from_backlog > 0:
        logger.warning(
            "Execution log backlog exceeded %d entries. Dropping %d oldest log(s) to keep memory bounded while database writes are failing.",
            MAX_PENDING_EXECUTION_LOGS,
            dropped_from_backlog,
            extra={SKIP_EXECUTION_LOG_ATTR: True},
        )
    if database is None or not pending_logs:
        return pending_logs
    try:
        await database.save_logs_batch(pending_logs)
    except Exception as exc:
        logger.warning(
            "Execution log flush failed. The framework will retry on the next flush cycle. Error: %s",
            exc,
            extra={SKIP_EXECUTION_LOG_ATTR: True},
        )
        return pending_logs
    if log_buffer is not None:
        dropped_from_buffer = log_buffer.consume_dropped_count()
        if dropped_from_buffer > 0:
            logger.warning(
                "Execution log buffer exceeded %d entries. Dropped %d oldest log(s) before persistence. "
                "Increase database reliability or flush logs more frequently.",
                MAX_PENDING_EXECUTION_LOGS,
                dropped_from_buffer,
                extra={SKIP_EXECUTION_LOG_ATTR: True},
            )
    return []


async def _execution_log_flush_loop(
    database: Database | None,
    log_buffer: ExecutionLogBuffer | None,
    flush_event: asyncio.Event,
) -> None:
    """Flush buffered execution logs on a timer or when the buffer reaches capacity."""
    pending_logs: list[ExecutionLog] = []
    try:
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(flush_event.wait(), timeout=LOG_BUFFER_FLUSH_INTERVAL_SECONDS)
            flush_event.clear()
            pending_logs = await _flush_execution_logs_once(database, log_buffer, pending_logs)
    except asyncio.CancelledError:
        pending_logs = await _flush_execution_logs_once(database, log_buffer, pending_logs)
        if pending_logs:
            logger.warning(
                "Execution log flush stopped with %d pending log(s). The database write path is still failing; inspect earlier log flush warnings.",
                len(pending_logs),
                extra={SKIP_EXECUTION_LOG_ATTR: True},
            )
        raise


async def _record_terminal_flow_node(
    *,
    insert_node: Callable[[Database | None, ExecutionNode], Awaitable[None]],
    update_node: Callable[..., Awaitable[None]],
    database: Database | None,
    publisher: ResultPublisher,
    deployment_name: str,
    deployment_node_id: UUID,
    root_deployment_id: UUID,
    run_id: str,
    run_scope: RunScope,
    flow_name: str,
    flow_class_name: str,
    step: int,
    total_steps: int,
    root_id_str: str,
    parent_task_id_str: str | None,
    status: NodeStatus,
    publish_reason: str,
    log_buffer: ExecutionLogBuffer | None,
    lifecycle_event: str,
    lifecycle_message: str,
    lifecycle_fields: Mapping[str, Any],
    payload: Mapping[str, Any] | None = None,
    output_document_shas: tuple[str, ...] = (),
) -> None:
    """Insert a terminal flow node, emit lifecycle logs, and persist its log summary."""
    node_id = uuid4()
    node_payload = dict(payload or {})
    await insert_node(
        database,
        ExecutionNode(
            node_id=node_id,
            node_kind=NodeKind.FLOW,
            deployment_id=deployment_node_id,
            root_deployment_id=root_deployment_id,
            parent_node_id=deployment_node_id,
            run_id=run_id,
            run_scope=run_scope,
            deployment_name=deployment_name,
            name=flow_name,
            sequence_no=step,
            flow_class=flow_class_name,
            status=status,
            ended_at=datetime.now(UTC),
            output_document_shas=output_document_shas,
            payload=node_payload,
        ),
    )
    await publisher.publish_flow_skipped(
        FlowSkippedEvent(
            run_id=run_id,
            node_id=str(node_id),
            root_deployment_id=root_id_str,
            parent_deployment_task_id=parent_task_id_str or "",
            flow_name=flow_name,
            step=step,
            total_steps=total_steps,
            reason=publish_reason,
        )
    )

    current_exec_ctx = get_execution_context()
    if current_exec_ctx is None:
        return

    node_exec_ctx = replace(
        current_exec_ctx,
        current_node_id=node_id,
        flow_node_id=node_id,
    )
    execution_token = set_execution_context(node_exec_ctx)
    try:
        record_lifecycle_event(lifecycle_event, lifecycle_message, **lifecycle_fields)
        await update_node(
            database,
            node_id,
            payload={
                **node_payload,
                "log_summary": _consume_log_summary(log_buffer, node_id),
            },
        )
    finally:
        reset_execution_context(execution_token)


async def _heartbeat_loop(publisher: ResultPublisher, run_id: str) -> None:
    """Publish heartbeat signals at regular intervals until cancelled."""
    while True:
        await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)
        try:
            await publisher.publish_heartbeat(run_id)
        except Exception as e:
            logger.warning("Heartbeat publish failed: %s", e)
