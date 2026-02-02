"""Helper functions for pipeline deployments."""

import asyncio
import re
from typing import Any, Literal, TypedDict

import httpx

from ai_pipeline_core.deployment.contract import CompletedRun, FailedRun, ProgressRun
from ai_pipeline_core.documents import Document
from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)


class DownloadedDocument(Document):
    """Concrete document for downloaded content."""


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


def extract_generic_params(cls: type, base_class: type) -> tuple[type | None, type | None]:
    """Extract TOptions and TResult from a generic base class's args."""
    for base in getattr(cls, "__orig_bases__", []):
        origin = getattr(base, "__origin__", None)
        if origin is base_class:
            args = getattr(base, "__args__", ())
            if len(args) == 2:
                return args[0], args[1]

    return None, None


async def download_documents(urls: list[str]) -> list[Document]:
    """Download documents from URLs."""
    documents: list[Document] = []
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        for url in urls:
            response = await client.get(url)
            response.raise_for_status()
            filename = url.split("/")[-1].split("?")[0] or "document"
            documents.append(DownloadedDocument(name=filename, content=response.content))
    return documents


async def upload_documents(documents: list[Document], url_mapping: dict[str, str]) -> None:
    """Upload documents to their mapped URLs."""
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        for doc in documents:
            if doc.name in url_mapping:
                response = await client.put(
                    url_mapping[doc.name],
                    content=doc.content,
                    headers={"Content-Type": doc.mime_type},
                )
                response.raise_for_status()


async def send_webhook(
    url: str,
    payload: ProgressRun | CompletedRun | FailedRun,
    max_retries: int = 3,
    retry_delay: float = 10.0,
) -> None:
    """Send webhook with retries."""
    data: dict[str, Any] = payload.model_dump(mode="json")
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(url, json=data, follow_redirects=True)
                response.raise_for_status()
            return
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Webhook retry {attempt + 1}/{max_retries}: {e}")
                await asyncio.sleep(retry_delay)
            else:
                logger.exception(f"Webhook failed after {max_retries} attempts")
                raise
