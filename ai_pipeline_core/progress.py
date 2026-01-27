"""@public Intra-flow progress tracking with order-preserving webhook delivery."""

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from ai_pipeline_core.deployment.contract import ProgressRun
from ai_pipeline_core.logging import get_pipeline_logger

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
    weights: tuple[float, ...]
    completed_weight: float
    current_flow_weight: float
    queue: asyncio.Queue[ProgressRun | None]


_context: ContextVar[ProgressContext | None] = ContextVar("progress_context", default=None)


async def update(fraction: float, message: str = "") -> None:
    """@public Report intra-flow progress (0.0-1.0). No-op without context."""
    ctx = _context.get()
    if ctx is None or not ctx.webhook_url:
        return

    fraction = max(0.0, min(1.0, fraction))

    total_weight = sum(ctx.weights)
    if total_weight > 0:
        overall = (ctx.completed_weight + ctx.current_flow_weight * fraction) / total_weight
    else:
        overall = fraction
    overall = round(max(0.0, min(1.0, overall)), 4)

    payload = ProgressRun(
        flow_run_id=UUID(ctx.flow_run_id) if ctx.flow_run_id else UUID(int=0),
        project_name=ctx.project_name,
        state="RUNNING",
        timestamp=datetime.now(timezone.utc),
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
    from ai_pipeline_core.deployment.helpers import send_webhook  # noqa: PLC0415

    while True:
        payload = await queue.get()
        if payload is None:
            queue.task_done()
            break

        try:
            await send_webhook(webhook_url, payload, max_retries, retry_delay)
        except Exception:
            pass  # Already logged in send_webhook

        queue.task_done()


@contextmanager
def flow_context(
    webhook_url: str,
    project_name: str,
    run_id: str,
    flow_run_id: str,
    flow_name: str,
    step: int,
    total_steps: int,
    weights: tuple[float, ...],
    completed_weight: float,
    queue: asyncio.Queue[ProgressRun | None],
) -> Generator[None, None, None]:
    """Set up progress context for a flow. Framework internal use."""
    current_flow_weight = weights[step - 1] if step <= len(weights) else 1.0
    ctx = ProgressContext(
        webhook_url=webhook_url,
        project_name=project_name,
        run_id=run_id,
        flow_run_id=flow_run_id,
        flow_name=flow_name,
        step=step,
        total_steps=total_steps,
        weights=weights,
        completed_weight=completed_weight,
        current_flow_weight=current_flow_weight,
        queue=queue,
    )
    token = _context.set(ctx)
    try:
        yield
    finally:
        _context.reset(token)


__all__ = ["update", "webhook_worker", "flow_context", "ProgressContext"]
