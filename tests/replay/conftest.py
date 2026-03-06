"""Shared fixtures and test types for replay tests."""

import struct
import zlib
from enum import StrEnum
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from ai_pipeline_core.documents import Attachment, Document
from ai_pipeline_core.documents._context import _suppress_document_registration
from ai_pipeline_core.pipeline.options import FlowOptions


@pytest.fixture(autouse=True)
def _suppress_registration():
    with _suppress_document_registration():
        yield


# ---------------------------------------------------------------------------
# Document types for replay tests
# ---------------------------------------------------------------------------


class ReplayTextDocument(Document):
    """Text document used by replay tests."""


class ReplayBinaryDocument(Document):
    """Binary (image) document used by replay tests."""


class ReplayAttachmentDocument(Document):
    """Document with attachments used by replay tests."""


class ReplayResultDocument(Document):
    """Output document returned from replay test tasks."""


# ---------------------------------------------------------------------------
# BaseModel / Enum types for argument serialization tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store_base(tmp_path: Path) -> Path:
    """Base path for LocalDocumentStore-based replay tests."""
    return tmp_path / "output"


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
async def populated_store(
    store_base: Path,
    sample_documents: dict[str, Document],
) -> Path:
    """Write all sample documents into a LocalDocumentStore and return store_base."""
    from ai_pipeline_core.document_store._local import LocalDocumentStore
    from ai_pipeline_core.documents import RunScope

    store = LocalDocumentStore(base_path=store_base)
    for doc in sample_documents.values():
        await store.save(doc, RunScope("replay/test"))
    return store_base
