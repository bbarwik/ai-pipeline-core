"""Intra-flow progress tracking with webhook delivery and Prefect label updates."""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from prefect import get_client

from ai_pipeline_core.logging import get_pipeline_logger

from ._helpers import send_webhook
from .contract import ProgressRun

logger = get_pipeline_logger(__name__)

_ZERO_UUID = UUID(int=0)


def _safe_uuid(value: str) -> UUID | None:
    """Parse a UUID string, returning None if invalid."""
    try:
        return UUID(value)
    except (ValueError, AttributeError):
        return None


@dataclass(frozen=True, slots=True)
class ProgressContext:
    """Internal context holding state for progress calculation and webhook delivery."""

    webhook_url: str
    project_name: str
    flow_run_id: str
    flow_name: str
    step: int
    total_steps: int
    total_minutes: float
    completed_minutes: float
    current_flow_minutes: float


_context: ContextVar[ProgressContext | None] = ContextVar("progress_context", default=None)

_PROGRESS_WEBHOOK_MAX_RETRIES = 1
_PROGRESS_WEBHOOK_RETRY_DELAY = 0.0


async def update(fraction: float, message: str = "") -> None:
    """Report intra-flow progress (0.0-1.0). No-op without context.

    Sends webhook payload (if webhook_url configured) AND updates Prefect
    flow run labels (if flow_run_id available) so both push and poll consumers
    see progress, and staleness detection stays current.
    """
    ctx = _context.get()
    if ctx is None:
        return

    fraction = max(0.0, min(1.0, fraction))

    if ctx.total_minutes > 0:
        overall = (ctx.completed_minutes + ctx.current_flow_minutes * fraction) / ctx.total_minutes
    else:
        overall = fraction
    overall = round(max(0.0, min(1.0, overall)), 4)
    step_progress = round(fraction, 4)

    run_uuid = _safe_uuid(ctx.flow_run_id) if ctx.flow_run_id else None

    if ctx.webhook_url:
        payload = ProgressRun(
            flow_run_id=run_uuid or _ZERO_UUID,
            project_name=ctx.project_name,
            state="RUNNING",
            timestamp=datetime.now(UTC),
            step=ctx.step,
            total_steps=ctx.total_steps,
            flow_name=ctx.flow_name,
            status="progress",
            progress=overall,
            step_progress=step_progress,
            message=message,
        )
        try:
            await send_webhook(ctx.webhook_url, payload, _PROGRESS_WEBHOOK_MAX_RETRIES, _PROGRESS_WEBHOOK_RETRY_DELAY)
        except Exception as e:
            logger.warning("Progress webhook failed: %s", e)

    if run_uuid is not None:
        try:
            async with get_client() as client:
                await client.update_flow_run_labels(
                    flow_run_id=run_uuid,
                    labels={
                        "progress.step": ctx.step,
                        "progress.total_steps": ctx.total_steps,
                        "progress.flow_name": ctx.flow_name,
                        "progress.status": "progress",
                        "progress.progress": overall,
                        "progress.step_progress": step_progress,
                        "progress.message": message,
                    },
                )
        except Exception as e:
            logger.warning("Progress label update failed: %s", e)


@contextmanager
def flow_context(
    webhook_url: str,
    project_name: str,
    flow_run_id: str,
    flow_name: str,
    step: int,
    *,
    total_steps: int,
    flow_minutes: tuple[float, ...],
    completed_minutes: float,
) -> Generator[None, None, None]:
    """Set up progress context for a flow. Framework internal use."""
    current_flow_minutes = flow_minutes[step - 1] if step <= len(flow_minutes) else 1.0
    total_minutes = sum(flow_minutes) if flow_minutes else current_flow_minutes
    ctx = ProgressContext(
        webhook_url=webhook_url,
        project_name=project_name,
        flow_run_id=flow_run_id,
        flow_name=flow_name,
        step=step,
        total_steps=total_steps,
        total_minutes=total_minutes,
        completed_minutes=completed_minutes,
        current_flow_minutes=current_flow_minutes,
    )
    token = _context.set(ctx)
    try:
        yield
    finally:
        _context.reset(token)


__all__ = ["ProgressContext", "flow_context", "update"]
