"""Document input resolution for pipeline deployments.

Provides typed input/output models and the resolver that converts
DocumentInput (inline content or URL references) into typed Documents.
"""

import asyncio
import ipaddress
import re
import socket
from typing import Any, ClassVar, Self
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, model_validator

from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents._hashing import compute_content_sha256
from ai_pipeline_core.documents.attachment import Attachment
from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)

_ALLOWED_SCHEMES = re.compile(r"^(https?|gs)://")
_DOWNLOAD_TIMEOUT = 120
_MAX_CONCURRENT_DOWNLOADS = 10


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class AttachmentInput(BaseModel):
    """Attachment provided to a deployment — inline content or a URL reference."""

    name: str = ""
    description: str | None = None
    content: str | None = None
    url: str = ""

    _STRIP_KEYS: ClassVar[frozenset[str]] = frozenset({"mime_type", "size"})

    @model_validator(mode="before")
    @classmethod
    def _strip_serialize_metadata(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k not in cls._STRIP_KEYS}
        return data

    @model_validator(mode="after")
    def _check_mode(self) -> Self:
        if self.url and self.content is not None:
            raise ValueError("AttachmentInput cannot have both 'url' and 'content'")
        if not self.url and self.content is None:
            raise ValueError("AttachmentInput must have either 'url' or 'content'")
        return self

    model_config = ConfigDict(frozen=True, extra="forbid")


class DocumentInput(BaseModel):
    """Document provided to a deployment — inline content or a URL reference."""

    name: str = ""
    description: str = ""
    class_name: str = ""

    content: str | None = None
    sources: tuple[str, ...] = ()
    origins: tuple[str, ...] = ()
    attachments: tuple[AttachmentInput, ...] = ()

    url: str = ""

    _STRIP_KEYS: ClassVar[frozenset[str]] = frozenset({
        "id",
        "sha256",
        "content_sha256",
        "size",
        "mime_type",
    })

    @model_validator(mode="before")
    @classmethod
    def _strip_serialize_metadata(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k not in cls._STRIP_KEYS}
        return data

    @model_validator(mode="after")
    def _check_mode(self) -> Self:
        if self.url and self.content is not None:
            raise ValueError("DocumentInput cannot have both 'url' and 'content'")
        if not self.url and self.content is None:
            raise ValueError("DocumentInput must have either 'url' or 'content'")
        return self

    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class OutputAttachment(BaseModel):
    """Attachment metadata in deployment results. Binary content is None."""

    sha256: str
    name: str
    mime_type: str
    size: int
    description: str | None = None
    content: str | None = None

    model_config = ConfigDict(frozen=True)


class OutputDocument(BaseModel):
    """Document metadata in deployment results. Binary content is None."""

    sha256: str
    name: str
    class_name: str
    mime_type: str
    size: int
    description: str = ""
    content: str | None = None
    attachments: tuple[OutputAttachment, ...] = ()

    model_config = ConfigDict(frozen=True)


def build_output_document(doc: Document) -> OutputDocument:
    """Build an OutputDocument from a live Document object."""
    att_outputs = tuple(
        OutputAttachment(
            sha256=compute_content_sha256(att.content),
            name=att.name,
            mime_type=att.mime_type,
            size=att.size,
            description=att.description,
            content=att.text if att.is_text else None,
        )
        for att in doc.attachments
    )
    return OutputDocument(
        sha256=doc.sha256,
        name=doc.name,
        class_name=doc.__class__.__name__,
        mime_type=doc.mime_type,
        size=doc.size,
        description=doc.description or "",
        content=doc.text if doc.is_text else None,
        attachments=att_outputs,
    )


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _is_private_ip(hostname: str) -> bool:
    """Check if hostname resolves to a private/reserved IP address (SSRF protection)."""
    try:
        addr = ipaddress.ip_address(hostname)
        return addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved
    except ValueError:
        pass
    try:
        resolved = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        return any(
            ipaddress.ip_address(addr[4][0]).is_private
            or ipaddress.ip_address(addr[4][0]).is_loopback
            or ipaddress.ip_address(addr[4][0]).is_link_local
            or ipaddress.ip_address(addr[4][0]).is_reserved
            for addr in resolved
        )
    except (socket.gaierror, ValueError, OSError):
        return False


def _validate_url(url: str) -> None:
    """Validate URL scheme and block private/reserved IP ranges (SSRF protection)."""
    if not _ALLOWED_SCHEMES.match(url):
        raise ValueError(f"Only http://, https://, and gs:// URLs are supported, got: {url}")
    if url.startswith("gs://"):
        return
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if _is_private_ip(hostname):
        raise ValueError(f"URL points to a private/reserved IP address (blocked for security): {hostname}")


def _derive_name(url: str, content_disposition: str | None) -> str:
    """Derive a filename from Content-Disposition or URL path."""
    if content_disposition:
        match = re.search(r'filename="?([^";\s]+)"?', content_disposition)
        if match:
            return match.group(1)
    path_part = url.split("?", maxsplit=1)[0].rsplit("/", maxsplit=1)[-1]
    if path_part and path_part != "/":
        return path_part
    return ""


async def _fetch_http(url: str, client: httpx.AsyncClient) -> tuple[bytes, str | None]:
    """Fetch content via HTTP with streaming and size enforcement.

    Returns (content_bytes, content_disposition_header).
    """
    async with client.stream("GET", url) as response:
        response.raise_for_status()
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes(chunk_size=65536):
            total += len(chunk)
            if total > Document.MAX_CONTENT_SIZE:
                raise ValueError(f"Download exceeds {Document.MAX_CONTENT_SIZE // (1024 * 1024)}MB limit: {url}")
            chunks.append(chunk)
        return b"".join(chunks), response.headers.get("content-disposition")


async def _fetch_gcs(url: str) -> bytes:
    """Fetch content from GCS with size enforcement."""
    try:
        from google.cloud import storage  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("google-cloud-storage required for gs:// URLs. Install: pip install google-cloud-storage") from None

    # Parse gs://bucket/path
    parts = url.removeprefix("gs://").split("/", maxsplit=1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"Invalid GCS URL format: {url}")
    bucket_name, blob_path = parts

    def _download() -> bytes:
        from ai_pipeline_core.settings import settings

        if settings.gcs_service_account_file:
            client = storage.Client.from_service_account_json(settings.gcs_service_account_file)
        else:
            client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_path)
        blob.reload()
        if blob.size is not None and blob.size > Document.MAX_CONTENT_SIZE:
            raise ValueError(f"GCS blob exceeds {Document.MAX_CONTENT_SIZE // (1024 * 1024)}MB limit: {url}")
        content = blob.download_as_bytes()
        if len(content) > Document.MAX_CONTENT_SIZE:
            raise ValueError(f"Downloaded content exceeds {Document.MAX_CONTENT_SIZE // (1024 * 1024)}MB limit: {url}")
        return content

    return await asyncio.to_thread(_download)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


async def resolve_document_inputs(
    inputs: list[DocumentInput],
    known_types: list[type[Document]],
    start_step_input_types: list[type[Document]] | None = None,
) -> list[Document]:
    """Resolve DocumentInput list into typed Documents.

    Handles both inline content and URL references. URL references are fetched
    in parallel with bounded concurrency.

    Args:
        inputs: List of DocumentInput from the deployment parameters.
        known_types: All document types from all flows (for explicit class_name matching).
        start_step_input_types: Input types from the start-step flow (for class_name inference).
    """
    if not inputs:
        return []

    type_map = {t.__name__: t for t in known_types}
    inference_types = {t.__name__: t for t in (start_step_input_types or [])}

    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_DOWNLOADS)

    async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT, follow_redirects=False) as client:

        async def _resolve_attachment(att_input: AttachmentInput) -> Attachment:
            if att_input.content is not None:
                name = att_input.name
                if not name:
                    raise ValueError("AttachmentInput with inline content must have a name")
                return Attachment(name=name, content=att_input.content, description=att_input.description)  # pyright: ignore[reportArgumentType]

            # URL attachment
            _validate_url(att_input.url)
            async with semaphore:
                if att_input.url.startswith("gs://"):
                    content = await _fetch_gcs(att_input.url)
                    disposition = None
                else:
                    content, disposition = await _fetch_http(att_input.url, client)

            name = att_input.name or _derive_name(att_input.url, disposition)
            if not name:
                raise ValueError(f"Cannot derive attachment name from URL: {att_input.url}")
            return Attachment(name=name, content=content, description=att_input.description)

        async def _resolve_one(doc_input: DocumentInput) -> Document:
            # Resolve class_name
            class_name = doc_input.class_name
            if not class_name:
                if len(inference_types) == 1:
                    class_name = next(iter(inference_types.keys()))
                elif len(inference_types) == 0:
                    raise ValueError("No input document types discoverable from flows; 'class_name' must be specified")
                else:
                    available = sorted(inference_types.keys())
                    raise ValueError(f"Multiple input types available ({', '.join(available)}); 'class_name' must be specified")

            doc_type = type_map.get(class_name)
            if doc_type is None:
                available = sorted(type_map.keys())
                raise ValueError(f"Unknown class_name '{class_name}'. Available: {', '.join(available)}")

            # Resolve attachments
            attachments: tuple[Attachment, ...] = ()
            if doc_input.attachments:
                att_list = await asyncio.gather(*[_resolve_attachment(a) for a in doc_input.attachments])
                attachments = tuple(att_list)

            if doc_input.content is not None:
                # Inline document — content is str (plain text or data URI), validator converts to bytes
                return doc_type(
                    name=doc_input.name,
                    content=doc_input.content,  # pyright: ignore[reportArgumentType]
                    description=doc_input.description or None,
                    sources=doc_input.sources or None,
                    origins=doc_input.origins or None,
                    attachments=attachments or None,
                )

            # URL document
            _validate_url(doc_input.url)
            async with semaphore:
                if doc_input.url.startswith("gs://"):
                    content_bytes = await _fetch_gcs(doc_input.url)
                    disposition = None
                else:
                    content_bytes, disposition = await _fetch_http(doc_input.url, client)

            name = doc_input.name or _derive_name(doc_input.url, disposition)
            if not name:
                raise ValueError(f"Cannot derive document name from URL: {doc_input.url}")

            return doc_type(
                name=name,
                content=content_bytes,
                description=doc_input.description or None,
                sources=doc_input.sources or None,
                origins=doc_input.origins or None,
                attachments=attachments or None,
            )

        results = await asyncio.gather(*[_resolve_one(inp) for inp in inputs], return_exceptions=True)
        errors = [(i, r) for i, r in enumerate(results) if isinstance(r, BaseException)]
        if errors:
            msgs = [f"  input[{i}]: {type(e).__name__}: {e}" for i, e in errors]
            raise ValueError(f"Failed to resolve {len(errors)}/{len(inputs)} document inputs:\n" + "\n".join(msgs))
        return [r for r in results if isinstance(r, Document)]


__all__ = [
    "AttachmentInput",
    "DocumentInput",
    "OutputAttachment",
    "OutputDocument",
    "build_output_document",
    "resolve_document_inputs",
]
