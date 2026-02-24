"""Intra-flow progress tracking with Prefect label updates and publisher delivery."""

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from uuid import UUID

from prefect import get_client

from ai_pipeline_core.logging import get_pipeline_logger

from ._types import ProgressEvent, ResultPublisher
from .contract import FlowStatus

logger = get_pipeline_logger(__name__)


def _on_publish_done(task: asyncio.Task[None]) -> None:
    """Log exceptions from fire-and-forget publish tasks."""
    if not task.cancelled() and (exc := task.exception()) is not None:
        logger.warning("Progress publish failed: %s", exc)


def _safe_uuid(value: str) -> UUID | None:
    """Parse a UUID string, returning None if invalid."""
    try:
        return UUID(value)
    except (ValueError, AttributeError):
        return None


@dataclass(frozen=True, slots=True)
class _ProgressContext:
    """Internal context holding state for progress calculation, label updates, and publisher delivery."""

    run_id: str
    flow_run_id: str
    flow_name: str
    step: int
    total_steps: int
    total_minutes: float
    completed_minutes: float
    current_flow_minutes: float
    publisher: ResultPublisher | None = None


_context: ContextVar[_ProgressContext | None] = ContextVar("progress_context", default=None)


def _compute_weighted_progress(
    completed_minutes: float,
    current_flow_minutes: float,
    fraction: float,
    total_minutes: float,
) -> float:
    """Compute overall weighted progress as a fraction in [0.0, 1.0]."""
    if total_minutes > 0:
        overall = (completed_minutes + current_flow_minutes * fraction) / total_minutes
    else:
        overall = fraction
    return round(max(0.0, min(1.0, overall)), 4)


def _build_progress_labels(
    *,
    step: int,
    total_steps: int,
    flow_name: str,
    status: str,
    progress: float,
    step_progress: float,
    message: str,
) -> dict[str, str | int | float]:
    """Build the label dict for Prefect flow run label updates."""
    return {
        "progress.step": step,
        "progress.total_steps": total_steps,
        "progress.flow_name": flow_name,
        "progress.status": status,
        "progress.progress": progress,
        "progress.step_progress": step_progress,
        "progress.message": message,
    }


async def _emit_progress(
    *,
    flow_run_id: str,
    step: int,
    total_steps: int,
    flow_name: str,
    status: FlowStatus,
    progress: float,
    step_progress: float,
    message: str = "",
) -> None:
    """Update Prefect flow run labels with current progress."""
    run_uuid = _safe_uuid(flow_run_id) if flow_run_id else None

    if run_uuid is not None:
        try:
            labels = _build_progress_labels(
                step=step,
                total_steps=total_steps,
                flow_name=flow_name,
                status=status,
                progress=progress,
                step_progress=step_progress,
                message=message,
            )
            async with get_client() as client:
                await client.update_flow_run_labels(flow_run_id=run_uuid, labels=labels)
        except Exception as e:
            logger.warning("Progress label update failed: %s", e)


async def progress_update(fraction: float, message: str = "") -> None:
    """Report intra-flow progress (0.0-1.0). No-op without context.

    Publishes a ProgressEvent via the publisher and updates Prefect flow run
    labels (if flow_run_id available) so poll consumers see progress and
    staleness detection stays current.
    """
    ctx = _context.get()
    if ctx is None:
        return

    fraction = max(0.0, min(1.0, fraction))
    overall = _compute_weighted_progress(ctx.completed_minutes, ctx.current_flow_minutes, fraction, ctx.total_minutes)
    step_progress = round(fraction, 4)

    # Fire-and-forget progress event publish to avoid blocking flow execution
    if ctx.publisher is not None:
        event = ProgressEvent(
            run_id=ctx.run_id,
            flow_run_id=ctx.flow_run_id,
            flow_name=ctx.flow_name,
            step=ctx.step,
            total_steps=ctx.total_steps,
            progress=overall,
            step_progress=step_progress,
            status=FlowStatus.PROGRESS,
            message=message,
        )
        task = asyncio.create_task(ctx.publisher.publish_progress(event))
        task.add_done_callback(_on_publish_done)

    await _emit_progress(
        flow_run_id=ctx.flow_run_id,
        step=ctx.step,
        total_steps=ctx.total_steps,
        flow_name=ctx.flow_name,
        status=FlowStatus.PROGRESS,
        progress=overall,
        step_progress=step_progress,
        message=message,
    )


@contextmanager
def _flow_context(
    run_id: str,
    flow_run_id: str,
    flow_name: str,
    step: int,
    *,
    total_steps: int,
    flow_minutes: tuple[float, ...],
    completed_minutes: float,
    publisher: ResultPublisher | None = None,
) -> Generator[None, None, None]:
    """Set up progress context for a flow. Framework internal use."""
    current_flow_minutes = flow_minutes[step - 1] if step <= len(flow_minutes) else 1.0
    total_minutes = sum(flow_minutes) if flow_minutes else current_flow_minutes
    ctx = _ProgressContext(
        run_id=run_id,
        flow_run_id=flow_run_id,
        flow_name=flow_name,
        step=step,
        total_steps=total_steps,
        total_minutes=total_minutes,
        completed_minutes=completed_minutes,
        current_flow_minutes=current_flow_minutes,
        publisher=publisher,
    )
    token = _context.set(ctx)
    try:
        yield
    finally:
        _context.reset(token)


__all__ = [
    "_flow_context",
    "progress_update",
]
