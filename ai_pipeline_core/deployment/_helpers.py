"""Helper functions for pipeline deployments."""

import asyncio
import re
from typing import Any, Literal, TypedDict

import httpx

from ai_pipeline_core.deployment.contract import CompletedRun, FailedRun, ProgressRun
from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)


class StatusPayload(TypedDict):
    """Webhook payload for Prefect state transitions (sub-flow level)."""

    type: Literal["status"]
    flow_run_id: str
    project_name: str
    step: int
    total_steps: int
    flow_name: str
    state: str
    state_name: str
    timestamp: str


def class_name_to_deployment_name(class_name: str) -> str:
    """Convert PascalCase to kebab-case: ResearchPipeline -> research-pipeline."""
    name = re.sub(r"(?<!^)(?=[A-Z])", "-", class_name)
    return name.lower()


def extract_generic_params(cls: type, base_class: type) -> tuple[Any, ...]:
    """Extract Generic type arguments from a class's base.

    Works with any number of Generic parameters (2 for PipelineDeployment, 3 for RemoteDeployment).
    Returns (None, None) if the base class is not found in __orig_bases__.
    """
    for base in getattr(cls, "__orig_bases__", []):
        origin = getattr(base, "__origin__", None)
        if origin is base_class:
            args = getattr(base, "__args__", ())
            if args:
                return args

    return (None, None)


async def send_webhook(
    url: str,
    payload: ProgressRun | CompletedRun | FailedRun,
    max_retries: int = 3,
    retry_delay: float = 10.0,
) -> None:
    """Send webhook with retries. Uses a single httpx client across retry attempts."""
    data: dict[str, Any] = payload.model_dump(mode="json")
    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(max_retries):
            try:
                response = await client.post(url, json=data, follow_redirects=True)
                response.raise_for_status()
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning("Webhook retry %d/%d: %s", attempt + 1, max_retries, e)
                    await asyncio.sleep(retry_delay)
                else:
                    logger.exception("Webhook failed after %d attempts", max_retries)
                    raise
