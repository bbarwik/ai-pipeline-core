"""Debug execution helpers for running individual tasks and flows with filesystem persistence.

Provides ``run_task_debug()`` and ``run_flow_debug()`` for executing single pipeline
components into a filesystem-backed runtime context. Results (spans, documents, blobs,
logs) are persisted to the output directory and can be inspected with ``ai-trace show``.
"""

import asyncio
import contextlib
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import UUID, uuid7

from ai_pipeline_core.database import SpanKind
from ai_pipeline_core.database.filesystem._backend import FilesystemDatabase
from ai_pipeline_core.database.filesystem.overlay import create_debug_sink
from ai_pipeline_core.deployment._types import _NoopPublisher
from ai_pipeline_core.documents import Document
from ai_pipeline_core.logger._buffer import ExecutionLogBuffer
from ai_pipeline_core.pipeline._execution_context import (
    ExecutionContext,
    FlowFrame,
    TaskContext,
    get_execution_context,
    set_execution_context,
    set_task_context,
)
from ai_pipeline_core.pipeline._flow import PipelineFlow
from ai_pipeline_core.pipeline._runtime_sinks import build_runtime_sinks
from ai_pipeline_core.pipeline._task import PipelineTask
from ai_pipeline_core.pipeline._track_span import track_span
from ai_pipeline_core.pipeline.limits import _LimitsState, _set_limits_state, _SharedStatus
from ai_pipeline_core.pipeline.options import FlowOptions
from ai_pipeline_core.settings import settings

logger = logging.getLogger(__name__)

__all__ = ["DebugRunResult", "run_flow_debug", "run_task_debug"]

_RUN_ID_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _make_debug_run_id(name: str) -> str:
    """Generate a sanitized debug run_id from a task/flow name."""
    sanitized = _RUN_ID_SANITIZE_RE.sub("-", name).strip("-")[:40]
    if not sanitized:
        sanitized = "debug"
    return f"debug-{sanitized}-{str(uuid7())[:8]}"


@dataclass(frozen=True, slots=True)
class DebugRunResult:
    """Result from a debug execution with metadata for post-run inspection."""

    output_documents: tuple[Document, ...]
    output_dir: Path
    run_id: str
    root_deployment_id: UUID


async def run_task_debug(
    task_cls: type[PipelineTask],
    /,
    *task_args: Any,
    output_dir: Path,
    run_id: str | None = None,
    database: FilesystemDatabase | None = None,
    **task_kwargs: Any,
) -> DebugRunResult:
    """Execute a single PipelineTask into a filesystem-backed debug context.

    Creates a full ExecutionContext with database, sinks, and span tracking.
    Results (spans, documents, blobs, logs) are persisted to output_dir.

    Call using the same arguments as the task's ``run()`` method (minus ``cls``)::

        result = await run_task_debug(MyTask, (doc,), output_dir=Path("debug_out"))
    """
    owns_db = database is None
    if owns_db:
        database = create_debug_sink(output_dir)
    assert database is not None

    deployment_id = uuid7()
    resolved_run_id = run_id or _make_debug_run_id(task_cls.name)
    effective_output_dir = output_dir if owns_db else database.base_path

    try:
        result = await _run_debug(
            database=database,
            deployment_id=deployment_id,
            run_id=resolved_run_id,
            name=f"debug-{task_cls.name}",
            execute_fn=lambda: task_cls.run(*task_args, **task_kwargs),
        )
        return DebugRunResult(output_documents=result, output_dir=effective_output_dir, run_id=resolved_run_id, root_deployment_id=deployment_id)
    finally:
        if owns_db:
            await _safe_shutdown(database)


async def run_flow_debug(
    flow: PipelineFlow,
    *,
    documents: Sequence[Document],
    options: FlowOptions,
    output_dir: Path,
    run_id: str | None = None,
    database: FilesystemDatabase | None = None,
) -> DebugRunResult:
    """Execute a single PipelineFlow into a filesystem-backed debug context.

    Creates a full ExecutionContext with deployment and flow span tracking.
    Results (spans, documents, blobs, logs) are persisted to output_dir.
    """
    owns_db = database is None
    if owns_db:
        database = create_debug_sink(output_dir)
    assert database is not None

    deployment_id = uuid7()
    flow_cls = type(flow)
    resolved_run_id = run_id or _make_debug_run_id(flow.name)
    effective_output_dir = output_dir if owns_db else database.base_path

    async def _execute() -> tuple[Document, ...]:
        from ai_pipeline_core.deployment._deployment_runtime import _execute_flow_with_context  # noqa: PLC0415 — deferred to avoid circular import

        ctx = get_execution_context()
        assert ctx is not None
        flow_span_id = uuid7()
        flow_frame = FlowFrame(
            name=flow.name,
            flow_class_name=flow_cls.__name__,
            step=1,
            total_steps=1,
            flow_minutes=(flow_cls.estimated_minutes,),
            completed_minutes=0.0,
            flow_params=flow.get_params(),
        )
        flow_ctx = ctx.with_flow(flow_frame).with_span(flow_span_id, parent_span_id=deployment_id)
        active_handles_before: set[object] = set(ctx.active_task_handles)

        return await _execute_flow_with_context(
            flow_instance=flow,
            flow_class=flow_cls,
            flow_name=flow.name,
            current_docs=list(documents),
            options=options,
            flow_exec_ctx=flow_ctx,
            current_exec_ctx=ctx,
            active_handles_before=active_handles_before,
            database=database,
            publisher=_NoopPublisher(),
            deployment_span_id=deployment_id,
            run_id=resolved_run_id,
            flow_span_id=flow_span_id,
            flow_cache_key="",
            flow_options_payload=options.model_dump(mode="json", exclude_none=True),
            expected_tasks=flow_cls.expected_tasks(),
            step=1,
            total_steps=1,
            root_id_str=str(deployment_id),
            parent_task_id_str=None,
        )

    try:
        result = await _run_debug(
            database=database,
            deployment_id=deployment_id,
            run_id=resolved_run_id,
            name=f"debug-{flow.name}",
            execute_fn=_execute,
        )
        return DebugRunResult(output_documents=result, output_dir=effective_output_dir, run_id=resolved_run_id, root_deployment_id=deployment_id)
    finally:
        if owns_db:
            await _safe_shutdown(database)


async def _run_debug(
    *,
    database: FilesystemDatabase,
    deployment_id: UUID,
    run_id: str,
    name: str,
    execute_fn: Callable[[], Awaitable[tuple[Document, ...]]],
) -> tuple[Document, ...]:
    """Shared execution wrapper: sets up context, sinks, log buffer, runs, cleans up."""
    from ai_pipeline_core.deployment._helpers import (  # noqa: PLC0415 — deferred to avoid circular import
        _cancel_dispatched_handles,
        _ensure_execution_log_handler_installed,
        _log_flush_loop,
    )

    sinks = build_runtime_sinks(database=database, settings_obj=settings)
    _ensure_execution_log_handler_installed()
    flush_event = asyncio.Event()
    event_loop = asyncio.get_running_loop()

    def _request_flush() -> None:
        event_loop.call_soon_threadsafe(flush_event.set)

    log_buffer = ExecutionLogBuffer(request_flush=_request_flush)

    ctx = ExecutionContext(
        run_id=run_id,
        execution_id=uuid7(),
        publisher=_NoopPublisher(),
        limits=MappingProxyType({}),
        limits_status=_SharedStatus(),
        database=database,
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        deployment_name=name,
        span_id=deployment_id,
        current_span_id=deployment_id,
        log_buffer=log_buffer,
        sinks=sinks,
    )

    log_flush_task = asyncio.create_task(_log_flush_loop(database, log_buffer, flush_event))
    limits_scope = contextlib.ExitStack()
    limits_scope.enter_context(_set_limits_state(_LimitsState(limits=MappingProxyType({}), status=_SharedStatus())))
    baseline_handles: set[object] = set()

    try:
        with limits_scope, set_execution_context(ctx), set_task_context(TaskContext(scope_kind="debug", task_class_name=name)):
            async with track_span(SpanKind.DEPLOYMENT, name, "", sinks=sinks, span_id=deployment_id, db=database):
                baseline_handles = set(ctx.active_task_handles)
                result = await execute_fn()
                if not isinstance(result, tuple):
                    result = (result,) if isinstance(result, Document) else ()
                return tuple(result)
    finally:
        await _cancel_dispatched_handles(ctx.active_task_handles, baseline_handles=baseline_handles)
        log_flush_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await log_flush_task


async def _safe_shutdown(database: FilesystemDatabase) -> None:
    """Flush and shutdown database, suppressing errors."""
    try:
        await database.flush()
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Debug database flush failed: %s", exc)
    try:
        await database.shutdown()
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Debug database shutdown failed: %s", exc)
