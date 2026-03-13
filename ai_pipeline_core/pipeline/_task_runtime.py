"""Private runtime support for PipelineTask execution."""

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from pydantic import BaseModel

from ai_pipeline_core.database import BlobRecord
from ai_pipeline_core.database._documents import document_to_blobs, document_to_record
from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents._context import DocumentSha256
from ai_pipeline_core.logger import get_pipeline_logger

__all__ = [
    "_TaskRunSpec",
    "_attach_task_attempt",
    "_class_name",
    "_collect_documents",
    "_get_task_attempt",
    "_input_documents",
    "_maybe_with_timeout",
    "_next_span_version",
    "_ordered_unique_document_types",
    "_persist_documents_to_database",
]

logger = get_pipeline_logger(__name__)

_DATABASE_EXCEPTIONS = (Exception,)
_TASK_ATTEMPT_ATTRIBUTE = "_pipeline_task_attempt"
_CONVERSATION_MODULE = "ai_pipeline_core.llm.conversation"
_CONVERSATION_CLASS = "Conversation"


def _next_span_version(previous_version: int | None = None) -> int:
    """Return a monotonic version for append-only span writes."""
    candidate = time.time_ns()
    if previous_version is None:
        return candidate
    if candidate > previous_version:
        return candidate
    return previous_version + 1


def _attach_task_attempt(exc: BaseException, attempt: int) -> BaseException:
    setattr(exc, _TASK_ATTEMPT_ATTRIBUTE, attempt)
    return exc


def _get_task_attempt(exc: BaseException) -> int:
    attempt = getattr(exc, _TASK_ATTEMPT_ATTRIBUTE, 0)
    return attempt if isinstance(attempt, int) else 0


async def _persist_documents_to_database(
    documents: Sequence[Document],
    database: Any,
) -> None:
    """Persist content-addressed documents and blobs for span-era backends."""
    if database is None:
        return

    document_records = [document_to_record(doc) for doc in documents]
    seen_content_sha256s: set[str] = set()
    blob_records: list[BlobRecord] = []
    for doc in documents:
        for blob in document_to_blobs(doc):
            if blob.content_sha256 not in seen_content_sha256s:
                seen_content_sha256s.add(blob.content_sha256)
                blob_records.append(blob)

    try:
        if blob_records:
            await database.save_blob_batch(blob_records)
        if document_records:
            await database.save_document_batch(document_records)
    except _DATABASE_EXCEPTIONS as exc:
        logger.warning("Database document persistence failed: %s", exc)


@dataclass(frozen=True, slots=True)
class _TaskRunSpec:
    """Validated task run metadata stored on the task class."""

    user_run: Callable[..., Awaitable[Any]]
    signature: inspect.Signature
    hints: Mapping[str, Any]
    input_document_types: tuple[type[Document], ...]
    output_document_types: tuple[type[Document], ...]


def _class_name(value: Any) -> str:
    return getattr(value, "__name__", str(value))


def _ordered_unique_document_types(document_types: list[type[Document]]) -> tuple[type[Document], ...]:
    ordered: dict[str, type[Document]] = {}
    for document_type in document_types:
        ordered.setdefault(document_type.__name__, document_type)
    return tuple(ordered.values())


async def _maybe_with_timeout[T](timeout_seconds: int | None, call: Callable[[], Awaitable[T]]) -> T:
    if timeout_seconds is None:
        return await call()
    async with asyncio.timeout(timeout_seconds):
        return await call()


def _collect_documents(value: Any, collected_docs: list[Document]) -> None:
    """Collect Documents nested in supported task input values."""
    if isinstance(value, Document):
        collected_docs.append(cast(Document[Any], value))
        return
    if _is_conversation_like(value):
        for document in value.context:
            _collect_documents(document, collected_docs)
        for message in value.messages:
            _collect_documents(message, collected_docs)
        return
    if isinstance(value, (list, tuple)):
        for item in cast(Sequence[Any], value):
            _collect_documents(item, collected_docs)
        return
    if isinstance(value, dict):
        for item in cast(dict[Any, Any], value).values():
            _collect_documents(item, collected_docs)
        return
    if isinstance(value, BaseModel):
        for field_name in type(value).model_fields:
            _collect_documents(getattr(value, field_name), collected_docs)


def _is_conversation_like(value: Any) -> bool:
    value_type = type(value)
    return (
        value_type.__module__ == _CONVERSATION_MODULE
        and value_type.__name__ == _CONVERSATION_CLASS
        and hasattr(value, "context")
        and hasattr(value, "messages")
    )


def _input_documents(arguments: Mapping[str, Any]) -> tuple[Document, ...]:
    """Flatten Document inputs from task arguments while preserving order."""
    collected_docs: list[Document] = []
    for value in arguments.values():
        _collect_documents(value, collected_docs)
    deduped: dict[DocumentSha256, Document] = {}
    for document in collected_docs:
        deduped.setdefault(document.sha256, document)
    return tuple(deduped.values())
