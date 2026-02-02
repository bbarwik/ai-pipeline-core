"""Intra-flow progress tracking with order-preserving webhook delivery."""

import asyncio
import contextlib
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from ai_pipeline_core.logging import get_pipeline_logger

from .contract import ProgressRun
from .helpers import send_webhook

logger = get_pipeline_logger(__name__)


@dataclass(frozen=True, slots=True)
class ProgressContext:
    """Internal context holding state for progress calculation and webhook delivery."""

    webhook_url: str
    project_name: str
    run_id: str
    flow_run_id: str
    flow_name: str
    step: int
    total_steps: int
    total_minutes: float
    completed_minutes: float
    current_flow_minutes: float
    queue: asyncio.Queue[ProgressRun | None]


_context: ContextVar[ProgressContext | None] = ContextVar("progress_context", default=None)


async def update(fraction: float, message: str = "") -> None:
    """Report intra-flow progress (0.0-1.0). No-op without context."""
    ctx = _context.get()
    if ctx is None or not ctx.webhook_url:
        return

    fraction = max(0.0, min(1.0, fraction))

    if ctx.total_minutes > 0:
        overall = (ctx.completed_minutes + ctx.current_flow_minutes * fraction) / ctx.total_minutes
    else:
        overall = fraction
    overall = round(max(0.0, min(1.0, overall)), 4)

    payload = ProgressRun(
        flow_run_id=UUID(ctx.flow_run_id) if ctx.flow_run_id else UUID(int=0),
        project_name=ctx.project_name,
        state="RUNNING",
        timestamp=datetime.now(UTC),
        step=ctx.step,
        total_steps=ctx.total_steps,
        flow_name=ctx.flow_name,
        status="progress",
        progress=overall,
        step_progress=round(fraction, 4),
        message=message,
    )

    ctx.queue.put_nowait(payload)


async def webhook_worker(
    queue: asyncio.Queue[ProgressRun | None],
    webhook_url: str,
    max_retries: int = 3,
    retry_delay: float = 10.0,
) -> None:
    """Process webhooks sequentially with retries, preserving order."""
    while True:
        payload = await queue.get()
        if payload is None:
            queue.task_done()
            break

        with contextlib.suppress(Exception):
            await send_webhook(webhook_url, payload, max_retries, retry_delay)

        queue.task_done()


@contextmanager
def flow_context(  # noqa: PLR0917
    webhook_url: str,
    project_name: str,
    run_id: str,
    flow_run_id: str,
    flow_name: str,
    step: int,
    total_steps: int,
    flow_minutes: tuple[float, ...],
    completed_minutes: float,
    queue: asyncio.Queue[ProgressRun | None],
) -> Generator[None, None, None]:
    """Set up progress context for a flow. Framework internal use."""
    current_flow_minutes = flow_minutes[step - 1] if step <= len(flow_minutes) else 1.0
    total_minutes = sum(flow_minutes) if flow_minutes else current_flow_minutes
    ctx = ProgressContext(
        webhook_url=webhook_url,
        project_name=project_name,
        run_id=run_id,
        flow_run_id=flow_run_id,
        flow_name=flow_name,
        step=step,
        total_steps=total_steps,
        total_minutes=total_minutes,
        completed_minutes=completed_minutes,
        current_flow_minutes=current_flow_minutes,
        queue=queue,
    )
    token = _context.set(ctx)
    try:
        yield
    finally:
        _context.reset(token)


__all__ = ["ProgressContext", "flow_context", "update", "webhook_worker"]
