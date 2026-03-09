"""Shared fixtures and helpers for replay tests."""

from collections.abc import Generator
import hashlib
import struct
import zlib
from enum import StrEnum
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from ai_pipeline_core.database import BlobRecord, DatabaseWriter, DocumentRecord, MemoryDatabase
from ai_pipeline_core.documents import Attachment, Document
from ai_pipeline_core.documents._context import _suppress_document_registration
from ai_pipeline_core.pipeline.options import FlowOptions


@pytest.fixture(autouse=True)
def _suppress_registration() -> Generator[None, None, None]:
    with _suppress_document_registration():
        yield


class ReplayTextDocument(Document):
    """Text document used by replay tests."""


class ReplayBinaryDocument(Document):
    """Binary document used by replay tests."""


class ReplayAttachmentDocument(Document):
    """Document with attachments used by replay tests."""


class ReplayResultDocument(Document):
    """Output document returned from replay test tasks."""


class ReplayMode(StrEnum):
    """Enum argument used in task replay serialization tests."""

    FAST = "fast"
    DEEP = "deep"


class ReplayArgsModel(BaseModel):
    """Frozen BaseModel argument used in task replay serialization tests."""

    model_config = ConfigDict(frozen=True)

    max_items: int
    label: str


class ReplayFlowOptions(FlowOptions):
    """Flow options used by replay integration tests."""

    replay_label: str = "baseline"
    replay_mode: str = "fast"


def make_test_png_bytes() -> bytes:
    """Return a valid minimal 1x1 red PNG."""
    png_header = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)

    raw_data = zlib.compress(b"\x00\xff\x00\x00")
    idat_crc = zlib.crc32(b"IDAT" + raw_data) & 0xFFFFFFFF
    idat = struct.pack(">I", len(raw_data)) + b"IDAT" + raw_data + struct.pack(">I", idat_crc)

    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    return png_header + ihdr + idat + iend


def doc_ref_dict(doc: Document) -> dict[str, str]:
    """Build a $doc_ref reference dict from a Document instance."""
    return {
        "$doc_ref": doc.sha256,
        "class_name": doc.__class__.__name__,
        "name": doc.name,
    }


def _blob_record(content: bytes) -> BlobRecord:
    sha256 = hashlib.sha256(content).hexdigest()
    return BlobRecord(content_sha256=sha256, content=content, size_bytes=len(content))


async def store_document_in_database(
    database: DatabaseWriter,
    doc: Document,
    *,
    deployment_id: UUID | None = None,
    producing_node_id: UUID | None = None,
) -> DocumentRecord:
    """Persist a Document and its blobs into a database backend."""
    target_deployment_id = deployment_id or uuid4()
    main_blob = _blob_record(doc.content)
    attachment_blobs = tuple(_blob_record(attachment.content) for attachment in doc.attachments)

    record = DocumentRecord(
        document_sha256=doc.sha256,
        content_sha256=main_blob.content_sha256,
        deployment_id=target_deployment_id,
        producing_node_id=producing_node_id,
        document_type=type(doc).__name__,
        name=doc.name,
        description=doc.description or "",
        mime_type=doc.mime_type,
        size_bytes=len(doc.content),
        derived_from=doc.derived_from,
        triggered_by=doc.triggered_by,
        attachment_names=tuple(attachment.name for attachment in doc.attachments),
        attachment_descriptions=tuple(attachment.description or "" for attachment in doc.attachments),
        attachment_sha256s=tuple(blob.content_sha256 for blob in attachment_blobs),
        attachment_mime_types=tuple(attachment.mime_type for attachment in doc.attachments),
        attachment_sizes=tuple(len(attachment.content) for attachment in doc.attachments),
    )

    await database.save_document(record)
    await database.save_blob(main_blob)
    if attachment_blobs:
        await database.save_blob_batch(list(attachment_blobs))
    return record


@pytest.fixture
def memory_database() -> MemoryDatabase:
    """Empty MemoryDatabase used by replay tests."""
    return MemoryDatabase()


@pytest.fixture
def sample_text_doc() -> ReplayTextDocument:
    """A simple text document."""
    return ReplayTextDocument(
        name="notes.txt",
        content=b"Replay fixture text document with some content for testing.",
        description="Text fixture for replay tests",
    )


@pytest.fixture
def sample_binary_doc() -> ReplayBinaryDocument:
    """A minimal PNG image document."""
    return ReplayBinaryDocument(
        name="screen.png",
        content=make_test_png_bytes(),
        description="Binary fixture for replay tests",
    )


@pytest.fixture
def sample_attachment_doc() -> ReplayAttachmentDocument:
    """A document with two attachments (text + image)."""
    return ReplayAttachmentDocument(
        name="bundle.md",
        content=b"# Bundle\nMain content for replay fixture.",
        description="Attachment fixture for replay tests",
        attachments=(
            Attachment(
                name="details.txt",
                content="Attachment text content",
                description="Text attachment",
            ),
            Attachment(
                name="preview.png",
                content=make_test_png_bytes(),
                description="Image attachment",
            ),
        ),
    )


@pytest.fixture
def sample_documents(
    sample_text_doc: ReplayTextDocument,
    sample_binary_doc: ReplayBinaryDocument,
    sample_attachment_doc: ReplayAttachmentDocument,
) -> dict[str, Document]:
    """All sample documents keyed by type."""
    return {
        "text": sample_text_doc,
        "binary": sample_binary_doc,
        "attachment": sample_attachment_doc,
    }


@pytest.fixture
async def stored_text_document(memory_database: MemoryDatabase, sample_text_doc: ReplayTextDocument) -> ReplayTextDocument:
    """Persist the sample text document to the memory database."""
    await store_document_in_database(memory_database, sample_text_doc)
    return sample_text_doc


@pytest.fixture
async def stored_binary_document(memory_database: MemoryDatabase, sample_binary_doc: ReplayBinaryDocument) -> ReplayBinaryDocument:
    """Persist the sample binary document to the memory database."""
    await store_document_in_database(memory_database, sample_binary_doc)
    return sample_binary_doc


@pytest.fixture
async def stored_attachment_document(memory_database: MemoryDatabase, sample_attachment_doc: ReplayAttachmentDocument) -> ReplayAttachmentDocument:
    """Persist the sample attachment document to the memory database."""
    await store_document_in_database(memory_database, sample_attachment_doc)
    return sample_attachment_doc
