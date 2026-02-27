"""Tests demonstrating the public read-only DocumentReader API.

These tests use @pytest.mark.ai_docs to appear in the generated .ai-docs guide.
They show how application code interacts with the document store using only
the public API: DocumentReader, get_document_store.
"""

from collections.abc import Generator

import pytest

from ai_pipeline_core import Document, DocumentReader, get_document_store
from ai_pipeline_core.document_store._memory import MemoryDocumentStore
from ai_pipeline_core.document_store._protocol import set_document_store
from ai_pipeline_core.documents import DocumentSha256, RunScope


class ReportDoc(Document):
    """Example document type for tests."""


class SourceDoc(Document):
    """Example source document type."""


@pytest.fixture
def populated_store() -> Generator[MemoryDocumentStore, None, None]:
    """Create a store pre-populated with documents for read tests."""
    store = MemoryDocumentStore()
    set_document_store(store)
    yield store
    set_document_store(None)


async def _seed(store: MemoryDocumentStore) -> tuple[ReportDoc, SourceDoc, SourceDoc]:
    """Seed the store with a report derived from two sources."""
    src_a = SourceDoc.create(name="source_a.md", content="Alpha data", derived_from=("https://example.com/a",))
    src_b = SourceDoc.create(name="source_b.md", content="Beta data", derived_from=("https://example.com/b",))
    report = ReportDoc.create(name="report.md", content="Final report", derived_from=(src_a.sha256, src_b.sha256))
    await store.save_batch([src_a, src_b, report], RunScope("project/run-1"))
    return report, src_a, src_b


@pytest.mark.ai_docs
@pytest.mark.asyncio
async def test_get_document_store_returns_reader(populated_store: MemoryDocumentStore):
    """get_document_store() returns a DocumentReader for read-only access."""
    reader = get_document_store()
    assert reader is not None
    assert isinstance(reader, DocumentReader)


@pytest.mark.ai_docs
@pytest.mark.asyncio
async def test_load_documents_by_type(populated_store: MemoryDocumentStore):
    """Load all documents of specific types from a run scope."""
    report, src_a, src_b = await _seed(populated_store)
    reader = get_document_store()

    reports = await reader.load(RunScope("project/run-1"), [ReportDoc])
    assert len(reports) == 1
    assert reports[0].name == "report.md"

    sources = await reader.load(RunScope("project/run-1"), [SourceDoc])
    assert len(sources) == 2

    all_docs = await reader.load(RunScope("project/run-1"), [ReportDoc, SourceDoc])
    assert len(all_docs) == 3


@pytest.mark.ai_docs
@pytest.mark.asyncio
async def test_has_documents_check(populated_store: MemoryDocumentStore):
    """Check whether documents of a given type exist in a scope."""
    await _seed(populated_store)
    reader = get_document_store()

    assert await reader.has_documents(RunScope("project/run-1"), ReportDoc) is True
    assert await reader.has_documents(RunScope("project/run-1"), SourceDoc) is True
    assert await reader.has_documents(RunScope("nonexistent"), ReportDoc) is False


@pytest.mark.ai_docs
@pytest.mark.asyncio
async def test_check_existing_sha256s(populated_store: MemoryDocumentStore):
    """Check which SHA256 hashes exist in the store."""
    report, *_ = await _seed(populated_store)
    reader = get_document_store()

    existing = await reader.check_existing([report.sha256, DocumentSha256("NONEXISTENT" * 4 + "AAAA")])
    assert report.sha256 in existing
    assert len(existing) == 1


@pytest.mark.ai_docs
@pytest.mark.asyncio
async def test_load_by_sha256s(populated_store: MemoryDocumentStore):
    """Batch-load full documents by their SHA256 hashes."""
    report, src_a, src_b = await _seed(populated_store)
    reader = get_document_store()

    result = await reader.load_by_sha256s([src_a.sha256, src_b.sha256], SourceDoc, RunScope("project/run-1"))
    assert len(result) == 2
    assert result[src_a.sha256].name == "source_a.md"
    assert result[src_b.sha256].name == "source_b.md"


@pytest.mark.ai_docs
@pytest.mark.asyncio
async def test_load_scope_metadata(populated_store: MemoryDocumentStore):
    """Load lightweight metadata for all documents in a scope."""
    report, src_a, src_b = await _seed(populated_store)
    reader = get_document_store()

    nodes = await reader.load_scope_metadata(RunScope("project/run-1"))
    assert len(nodes) == 3
    sha_set = {n.sha256 for n in nodes}
    assert report.sha256 in sha_set

    report_node = next(n for n in nodes if n.sha256 == report.sha256)
    assert report_node.class_name == "ReportDoc"
    assert report_node.name == "report.md"
    assert src_a.sha256 in report_node.derived_from


@pytest.mark.ai_docs
@pytest.mark.asyncio
async def test_load_nodes_by_sha256s(populated_store: MemoryDocumentStore):
    """Batch-load lightweight metadata nodes by SHA256 — works across scopes."""
    report, src_a, _ = await _seed(populated_store)
    reader = get_document_store()

    nodes = await reader.load_nodes_by_sha256s([report.sha256, src_a.sha256])
    assert len(nodes) == 2
    assert nodes[report.sha256].name == "report.md"
    assert nodes[src_a.sha256].class_name == "SourceDoc"
