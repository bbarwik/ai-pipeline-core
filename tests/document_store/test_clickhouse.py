"""Tests for ClickHouseDocumentStore using testcontainers.

Marked with @pytest.mark.clickhouse — spins up a ClickHouse Docker container.
Run with: pytest -m clickhouse
"""

import time

import pytest

clickhouse_connect = pytest.importorskip("clickhouse_connect")
testcontainers_clickhouse = pytest.importorskip("testcontainers.clickhouse")

from testcontainers.clickhouse import ClickHouseContainer

from ai_pipeline_core.document_store import DocumentNode, walk_provenance
from ai_pipeline_core.document_store.clickhouse import (
    TABLE_DOCUMENT_CONTENT,
    TABLE_DOCUMENT_INDEX,
    TABLE_RUN_DOCUMENTS,
    _DDL_CONTENT,
    _DDL_INDEX,
    _DDL_RUN_DOCUMENTS,
    _FAILURE_THRESHOLD,
    _MAX_BUFFER_SIZE,
    ClickHouseDocumentStore,
)
from ai_pipeline_core.documents import Attachment, Document
from ai_pipeline_core.documents._hashing import compute_content_sha256
from ai_pipeline_core.settings import settings

pytestmark = pytest.mark.clickhouse

HAS_API_KEYS = bool(settings.openai_api_key and settings.openai_base_url)


class CHDocA(Document):
    pass


class CHDocB(Document):
    pass


@pytest.fixture(scope="module")
def clickhouse_container():
    """Start a ClickHouse container for the test module."""
    with ClickHouseContainer() as container:
        yield container


@pytest.fixture
def store(clickhouse_container: ClickHouseContainer) -> ClickHouseDocumentStore:
    """Create a ClickHouseDocumentStore connected to the test container."""
    return ClickHouseDocumentStore(
        host=clickhouse_container.get_container_host_ip(),
        port=int(clickhouse_container.get_exposed_port(8123)),
        database=clickhouse_container.dbname,
        username=clickhouse_container.username,
        password=clickhouse_container.password,
        secure=False,
    )


def _make(cls: type[Document], name: str, content: str = "test content", **kwargs: object) -> Document:
    return cls.create(name=name, content=content, **kwargs)


async def _optimize_tables(store: ClickHouseDocumentStore) -> None:
    """Force merge of ReplacingMergeTree tables so FINAL queries see latest version."""
    await store._run(lambda: store._client.command(f"OPTIMIZE TABLE {TABLE_DOCUMENT_INDEX} FINAL"))
    await store._run(lambda: store._client.command(f"OPTIMIZE TABLE {TABLE_RUN_DOCUMENTS} FINAL"))


async def _query_count(store: ClickHouseDocumentStore, table: str, where: str = "1=1") -> int:
    """Run SELECT count() on a table with optional WHERE clause."""
    result = await store._run(lambda: store._client.query(f"SELECT count() FROM {table} FINAL WHERE {where}").result_rows[0][0])
    return result


# --- DDL & Init ---


class TestDDL:
    def test_ddl_statements_are_valid_sql(self):
        assert "ReplacingMergeTree" in _DDL_CONTENT
        assert "ReplacingMergeTree" in _DDL_INDEX
        assert "ReplacingMergeTree" in _DDL_RUN_DOCUMENTS
        assert "content_sha256" in _DDL_CONTENT
        assert "document_sha256" in _DDL_INDEX
        assert "run_scope" in _DDL_RUN_DOCUMENTS
        assert "stored_at" in _DDL_CONTENT
        assert "stored_at" in _DDL_INDEX
        assert "bloom_filter" not in _DDL_CONTENT
        assert "bloom_filter" not in _DDL_INDEX
        assert "bloom_filter" not in _DDL_RUN_DOCUMENTS
        assert "ZSTD" in _DDL_INDEX

    async def test_tables_created_in_clickhouse(self, store: ClickHouseDocumentStore):
        """Verify DDL actually creates tables with correct engines."""
        # Trigger table creation
        await store.load("__ddl_test__", [CHDocA])

        for table in [TABLE_DOCUMENT_CONTENT, TABLE_DOCUMENT_INDEX, TABLE_RUN_DOCUMENTS]:
            result = await store._run(lambda t=table: store._client.query(f"SELECT engine FROM system.tables WHERE name = '{t}'").result_rows)
            assert len(result) == 1
            assert result[0][0] == "ReplacingMergeTree"


class TestInit:
    def test_store_init_does_not_connect(self):
        s = ClickHouseDocumentStore(host="nonexistent.invalid", port=9999)
        assert s._client is None
        assert s._tables_initialized is False

    def test_circuit_breaker_initial_state(self):
        s = ClickHouseDocumentStore(host="nonexistent.invalid")
        assert s._circuit_open is False
        assert s._consecutive_failures == 0
        assert len(s._buffer) == 0


# --- Save & Load ---


class TestSaveAndLoad:
    async def test_save_and_load_single(self, store: ClickHouseDocumentStore):
        doc = _make(CHDocA, "report.txt", "hello world")
        await store.save(doc, "test-save-load")
        loaded = await store.load("test-save-load", [CHDocA])
        assert len(loaded) == 1
        assert loaded[0].name == "report.txt"
        assert loaded[0].content == b"hello world"

    async def test_save_batch_and_load(self, store: ClickHouseDocumentStore):
        docs = [_make(CHDocA, "a.txt", "aaa"), _make(CHDocA, "b.txt", "bbb")]
        await store.save_batch(docs, "test-batch")
        loaded = await store.load("test-batch", [CHDocA])
        assert len(loaded) == 2
        assert {d.name for d in loaded} == {"a.txt", "b.txt"}

    async def test_load_filters_by_type(self, store: ClickHouseDocumentStore):
        await store.save(_make(CHDocA, "a.txt", "aaa"), "test-filter")
        await store.save(_make(CHDocB, "b.txt", "bbb"), "test-filter")

        assert len(await store.load("test-filter", [CHDocA])) == 1
        assert len(await store.load("test-filter", [CHDocB])) == 1
        assert len(await store.load("test-filter", [CHDocA, CHDocB])) == 2

    async def test_save_idempotent(self, store: ClickHouseDocumentStore):
        doc = _make(CHDocA, "dup.txt", "same content")
        await store.save(doc, "test-idempotent")
        await store.save(doc, "test-idempotent")
        loaded = await store.load("test-idempotent", [CHDocA])
        assert len(loaded) == 1

    async def test_scopes_are_isolated(self, store: ClickHouseDocumentStore):
        await store.save(_make(CHDocA, "s1.txt", "scope1"), "scope-1")
        await store.save(_make(CHDocA, "s2.txt", "scope2"), "scope-2")
        assert len(await store.load("scope-1", [CHDocA])) == 1
        assert len(await store.load("scope-2", [CHDocA])) == 1

    async def test_load_empty_scope(self, store: ClickHouseDocumentStore):
        assert await store.load("nonexistent-scope", [CHDocA]) == []

    async def test_save_and_load_preserves_metadata(self, store: ClickHouseDocumentStore):
        """Verify description, sources, and origins round-trip through ClickHouse."""
        # Create a real document to use its sha256 as an origin
        origin_doc = _make(CHDocA, "origin.txt", "origin content")
        doc = _make(
            CHDocA,
            "meta.txt",
            "metadata test",
            description="A test document",
            sources=("https://example.com",),
            origins=(origin_doc.sha256,),
        )
        await store.save(doc, "test-metadata")
        loaded = await store.load("test-metadata", [CHDocA])
        assert len(loaded) == 1
        assert loaded[0].description == "A test document"
        assert "https://example.com" in loaded[0].sources
        assert loaded[0].origins == (origin_doc.sha256,)

    async def test_save_and_load_empty_content(self, store: ClickHouseDocumentStore):
        doc = CHDocA(name="empty.txt", content=b"")
        await store.save(doc, "test-empty-content")
        loaded = await store.load("test-empty-content", [CHDocA])
        assert len(loaded) == 1
        assert loaded[0].content == b""

    async def test_save_and_load_special_characters(self, store: ClickHouseDocumentStore):
        doc = _make(CHDocA, "résumé — final (2).txt", "Ünïcödé content: 日本語 🎉")
        await store.save(doc, "test-special-chars")
        loaded = await store.load("test-special-chars", [CHDocA])
        assert len(loaded) == 1
        assert loaded[0].name == "résumé — final (2).txt"
        assert "日本語" in loaded[0].content.decode()

    async def test_save_and_load_large_document(self, store: ClickHouseDocumentStore):
        """Verify ZSTD compression handles large content correctly."""
        large_content = "x" * 1_000_000  # 1MB
        doc = _make(CHDocA, "large.bin", large_content)
        await store.save(doc, "test-large-doc")
        loaded = await store.load("test-large-doc", [CHDocA])
        assert len(loaded) == 1
        assert len(loaded[0].content) == 1_000_000

    async def test_load_no_matching_types_in_populated_scope(self, store: ClickHouseDocumentStore):
        """Save CHDocA but load CHDocB — should return empty."""
        await store.save(_make(CHDocA, "only-a.txt", "aaa"), "test-type-mismatch")
        loaded = await store.load("test-type-mismatch", [CHDocB])
        assert loaded == []

    async def test_save_batch_mixed_types(self, store: ClickHouseDocumentStore):
        docs = [_make(CHDocA, "batch-a.txt", "aaa"), _make(CHDocB, "batch-b.txt", "bbb")]
        await store.save_batch(docs, "test-batch-mixed")
        assert len(await store.load("test-batch-mixed", [CHDocA])) == 1
        assert len(await store.load("test-batch-mixed", [CHDocB])) == 1
        assert len(await store.load("test-batch-mixed", [CHDocA, CHDocB])) == 2


# --- Has Documents ---


class TestHasDocuments:
    async def test_returns_false_when_empty(self, store: ClickHouseDocumentStore):
        assert await store.has_documents("empty-scope", CHDocA) is False

    async def test_returns_true_when_present(self, store: ClickHouseDocumentStore):
        await store.save(_make(CHDocA, "exists.txt"), "test-has")
        assert await store.has_documents("test-has", CHDocA) is True

    async def test_returns_false_for_wrong_type(self, store: ClickHouseDocumentStore):
        await store.save(_make(CHDocA, "wrong.txt"), "test-has-type")
        assert await store.has_documents("test-has-type", CHDocB) is False

    async def test_returns_false_for_wrong_scope(self, store: ClickHouseDocumentStore):
        await store.save(_make(CHDocA, "scoped.txt"), "test-has-scope-a")
        assert await store.has_documents("test-has-scope-b", CHDocA) is False


# --- Check Existing ---


class TestCheckExisting:
    async def test_returns_matching_hashes(self, store: ClickHouseDocumentStore):
        doc = _make(CHDocA, "hash.txt", "hash content")
        await store.save(doc, "test-check")
        result = await store.check_existing([doc.sha256, "0" * 64])
        assert doc.sha256 in result
        assert "0" * 64 not in result

    async def test_returns_empty_for_no_matches(self, store: ClickHouseDocumentStore):
        assert await store.check_existing(["0" * 64]) == set()

    async def test_returns_empty_for_empty_input(self, store: ClickHouseDocumentStore):
        """Exercises the early-return guard in _check_existing_sync."""
        assert await store.check_existing([]) == set()


# --- Attachments ---


class TestAttachments:
    async def test_save_and_load_with_attachments(self, store: ClickHouseDocumentStore):
        doc = CHDocA(
            name="with-att.txt",
            content=b"main content",
            attachments=(
                Attachment(name="att1.txt", content=b"attachment 1"),
                Attachment(name="att2.txt", content=b"attachment 2", description="second"),
            ),
        )
        await store.save(doc, "test-attachments")
        loaded = await store.load("test-attachments", [CHDocA])
        assert len(loaded) == 1
        assert len(loaded[0].attachments) == 2
        assert {a.name for a in loaded[0].attachments} == {"att1.txt", "att2.txt"}

    async def test_attachment_description_nullable_roundtrip(self, store: ClickHouseDocumentStore):
        """Verify None vs non-empty description survives the round-trip."""
        doc = CHDocA(
            name="att-desc.txt",
            content=b"body",
            attachments=(
                Attachment(name="no-desc.txt", content=b"aaa"),
                Attachment(name="with-desc.txt", content=b"bbb", description="has description"),
            ),
        )
        await store.save(doc, "test-att-desc")
        loaded = await store.load("test-att-desc", [CHDocA])
        atts = {a.name: a for a in loaded[0].attachments}
        assert atts["no-desc.txt"].description is None
        assert atts["with-desc.txt"].description == "has description"

    async def test_attachment_content_roundtrip(self, store: ClickHouseDocumentStore):
        """Verify attachment content bytes survive the round-trip."""
        doc = CHDocA(
            name="att-content.txt",
            content=b"main",
            attachments=(
                Attachment(name="data.csv", content=b"col1,col2\n1,2\n3,4"),
                Attachment(name="text.txt", content=b"hello attachment"),
            ),
        )
        await store.save(doc, "test-att-content")
        loaded = await store.load("test-att-content", [CHDocA])
        atts = {a.name: a for a in loaded[0].attachments}
        assert atts["data.csv"].content == b"col1,col2\n1,2\n3,4"
        assert atts["text.txt"].content == b"hello attachment"

    async def test_document_with_and_without_attachments(self, store: ClickHouseDocumentStore):
        """Load docs with and without attachments in the same scope."""
        doc_with = CHDocA(
            name="has-att.txt",
            content=b"with att",
            attachments=(Attachment(name="a.txt", content=b"att"),),
        )
        doc_without = CHDocA(name="no-att.txt", content=b"without att")
        await store.save(doc_with, "test-mixed-att")
        await store.save(doc_without, "test-mixed-att")
        loaded = await store.load("test-mixed-att", [CHDocA])
        assert len(loaded) == 2
        by_name = {d.name: d for d in loaded}
        assert len(by_name["has-att.txt"].attachments) == 1
        assert len(by_name["no-att.txt"].attachments) == 0


# --- Summaries ---


class TestSummaries:
    async def test_update_and_load_summary(self, store: ClickHouseDocumentStore):
        doc = _make(CHDocA, "summary-doc.txt", "content for summary test")
        await store.save(doc, "test-summary")
        await store.update_summary(doc.sha256, "A test document about summary functionality.")
        await _optimize_tables(store)
        summaries = await store.load_summaries([doc.sha256])
        assert doc.sha256 in summaries
        assert "summary" in summaries[doc.sha256].lower()

    async def test_load_summaries_returns_empty_for_no_summary(self, store: ClickHouseDocumentStore):
        doc = _make(CHDocA, "no-summary.txt", "no summary here")
        await store.save(doc, "test-no-summary")
        assert await store.load_summaries([doc.sha256]) == {}

    async def test_load_summaries_multiple_documents(self, store: ClickHouseDocumentStore):
        doc1 = _make(CHDocA, "multi1.txt", "first document for multi test")
        doc2 = _make(CHDocA, "multi2.txt", "second document for multi test")
        doc3 = _make(CHDocA, "multi3.txt", "third document for multi test")
        await store.save_batch([doc1, doc2, doc3], "test-multi-summary")
        await store.update_summary(doc1.sha256, "Summary of first document.")
        await store.update_summary(doc3.sha256, "Summary of third document.")
        await _optimize_tables(store)
        summaries = await store.load_summaries([doc1.sha256, doc2.sha256, doc3.sha256])
        assert len(summaries) == 2
        assert doc1.sha256 in summaries
        assert doc2.sha256 not in summaries
        assert doc3.sha256 in summaries

    async def test_summary_overwrite(self, store: ClickHouseDocumentStore):
        """Second update_summary replaces the first."""
        doc = _make(CHDocA, "overwrite.txt", "content for overwrite test")
        await store.save(doc, "test-summary-overwrite")
        await store.update_summary(doc.sha256, "First summary.")
        await _optimize_tables(store)
        await store.update_summary(doc.sha256, "Second summary.")
        await _optimize_tables(store)
        summaries = await store.load_summaries([doc.sha256])
        assert summaries[doc.sha256] == "Second summary."

    async def test_load_summaries_nonexistent_sha256s(self, store: ClickHouseDocumentStore):
        assert await store.load_summaries(["0" * 64, "1" * 64]) == {}

    async def test_load_summaries_empty_list(self, store: ClickHouseDocumentStore):
        """Exercises the early-return guard in _load_summaries_sync."""
        assert await store.load_summaries([]) == {}


# --- Content Deduplication ---


class TestBinaryContentRoundtrip:
    """Test that binary content (PDFs, images) survives save/load round-trip."""

    async def test_pdf_content_roundtrip(self, store: ClickHouseDocumentStore):
        """PDF binary content must be identical after load."""
        # Minimal valid PDF (binary content with non-UTF8 bytes)
        pdf_content = (
            b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
            b"xref\n0 3\n0000000000 65535 f \n"
            b"trailer<</Size 3/Root 1 0 R>>\nstartxref\n101\n%%EOF"
        )
        doc = CHDocA(name="test.pdf", content=pdf_content)

        await store.save(doc, "test-pdf-binary")
        loaded = await store.load("test-pdf-binary", [CHDocA])

        assert len(loaded) == 1
        assert loaded[0].content == pdf_content, f"PDF content mismatch: expected {len(pdf_content)} bytes, got {len(loaded[0].content)} bytes"

    async def test_png_content_roundtrip(self, store: ClickHouseDocumentStore):
        """PNG binary content must be identical after load."""
        # Minimal valid 1x1 transparent PNG
        png_content = bytes([
            0x89,
            0x50,
            0x4E,
            0x47,
            0x0D,
            0x0A,
            0x1A,
            0x0A,  # PNG signature
            0x00,
            0x00,
            0x00,
            0x0D,
            0x49,
            0x48,
            0x44,
            0x52,  # IHDR chunk
            0x00,
            0x00,
            0x00,
            0x01,
            0x00,
            0x00,
            0x00,
            0x01,  # 1x1
            0x08,
            0x06,
            0x00,
            0x00,
            0x00,
            0x1F,
            0x15,
            0xC4,  # 8-bit RGBA
            0x89,
            0x00,
            0x00,
            0x00,
            0x0A,
            0x49,
            0x44,
            0x41,  # IDAT chunk
            0x54,
            0x78,
            0x9C,
            0x63,
            0x00,
            0x01,
            0x00,
            0x00,
            0x05,
            0x00,
            0x01,
            0x0D,
            0x0A,
            0x2D,
            0xB4,
            0x00,
            0x00,
            0x00,
            0x00,
            0x49,
            0x45,
            0x4E,
            0x44,
            0xAE,  # IEND chunk
            0x42,
            0x60,
            0x82,
        ])
        doc = CHDocA(name="test.png", content=png_content)

        await store.save(doc, "test-png-binary")
        loaded = await store.load("test-png-binary", [CHDocA])

        assert len(loaded) == 1
        assert loaded[0].content == png_content, f"PNG content mismatch: expected {len(png_content)} bytes, got {len(loaded[0].content)} bytes"

    async def test_arbitrary_binary_content_roundtrip(self, store: ClickHouseDocumentStore):
        """Arbitrary binary bytes (all 256 values) must survive round-trip."""
        # Content with all possible byte values
        binary_content = bytes(range(256)) * 10
        doc = CHDocA(name="binary.bin", content=binary_content)

        await store.save(doc, "test-arbitrary-binary")
        loaded = await store.load("test-arbitrary-binary", [CHDocA])

        assert len(loaded) == 1
        assert loaded[0].content == binary_content, f"Binary content mismatch: expected {len(binary_content)} bytes, got {len(loaded[0].content)} bytes"

    async def test_binary_attachment_roundtrip(self, store: ClickHouseDocumentStore):
        """Binary attachment content must be identical after load."""
        png_attachment = bytes([
            0x89,
            0x50,
            0x4E,
            0x47,
            0x0D,
            0x0A,
            0x1A,
            0x0A,
            0x00,
            0x00,
            0x00,
            0x0D,
            0x49,
            0x48,
            0x44,
            0x52,
            0x00,
            0x00,
            0x00,
            0x01,
            0x00,
            0x00,
            0x00,
            0x01,
            0x08,
            0x06,
            0x00,
            0x00,
            0x00,
            0x1F,
            0x15,
            0xC4,
            0x89,
            0x00,
            0x00,
            0x00,
            0x0A,
            0x49,
            0x44,
            0x41,
            0x54,
            0x78,
            0x9C,
            0x63,
            0x00,
            0x01,
            0x00,
            0x00,
            0x05,
            0x00,
            0x01,
            0x0D,
            0x0A,
            0x2D,
            0xB4,
            0x00,
            0x00,
            0x00,
            0x00,
            0x49,
            0x45,
            0x4E,
            0x44,
            0xAE,
            0x42,
            0x60,
            0x82,
        ])
        doc = CHDocA(
            name="doc-with-binary-att.txt",
            content=b"text content",
            attachments=(Attachment(name="screenshot.png", content=png_attachment),),
        )

        await store.save(doc, "test-binary-attachment")
        loaded = await store.load("test-binary-attachment", [CHDocA])

        assert len(loaded) == 1
        assert len(loaded[0].attachments) == 1
        assert loaded[0].attachments[0].content == png_attachment, "PNG attachment content mismatch"

    async def test_mixed_text_and_binary_attachments(self, store: ClickHouseDocumentStore):
        """Document with both text and binary attachments."""
        binary_att = bytes(range(256))
        text_att = b"plain text attachment"

        doc = CHDocA(
            name="mixed.txt",
            content=b"main content",
            attachments=(
                Attachment(name="data.bin", content=binary_att),
                Attachment(name="notes.txt", content=text_att),
            ),
        )

        await store.save(doc, "test-mixed-attachments")
        loaded = await store.load("test-mixed-attachments", [CHDocA])

        assert len(loaded) == 1
        atts = {a.name: a for a in loaded[0].attachments}
        assert atts["data.bin"].content == binary_att, "Binary attachment content mismatch"
        assert atts["notes.txt"].content == text_att, "Text attachment content mismatch"

    async def test_binary_document_with_binary_attachment(self, store: ClickHouseDocumentStore):
        """Both document content and attachment are binary."""
        pdf_content = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\ntest binary content with non-utf8 bytes: \x80\x81\x82\xff"
        png_attachment = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, *range(200)])

        doc = CHDocA(
            name="report.pdf",
            content=pdf_content,
            attachments=(Attachment(name="figure.png", content=png_attachment),),
        )

        await store.save(doc, "test-all-binary")
        loaded = await store.load("test-all-binary", [CHDocA])

        assert len(loaded) == 1
        assert loaded[0].content == pdf_content, f"PDF content mismatch: {loaded[0].content[:50]}..."
        assert loaded[0].attachments[0].content == png_attachment, "PNG attachment mismatch"


class TestContentDeduplication:
    async def test_same_content_different_names_single_content_row(self, store: ClickHouseDocumentStore):
        """Two documents with identical content should produce only one row in document_content."""
        content = "deduplicated content body"
        doc1 = _make(CHDocA, "dedup-1.txt", content)
        doc2 = _make(CHDocA, "dedup-2.txt", content)
        await store.save(doc1, "test-dedup")
        await store.save(doc2, "test-dedup")

        content_sha = compute_content_sha256(content.encode())
        count = await _query_count(store, TABLE_DOCUMENT_CONTENT, f"content_sha256 = '{content_sha}'")
        assert count == 1

        loaded = await store.load("test-dedup", [CHDocA])
        assert len(loaded) == 2
        assert {d.name for d in loaded} == {"dedup-1.txt", "dedup-2.txt"}

    async def test_same_content_across_scopes(self, store: ClickHouseDocumentStore):
        """Same content in different scopes: one content row, both scopes loadable."""
        content = "shared across scopes"
        doc1 = _make(CHDocA, "cross-1.txt", content)
        doc2 = _make(CHDocA, "cross-2.txt", content)
        await store.save(doc1, "test-dedup-scope-a")
        await store.save(doc2, "test-dedup-scope-b")

        content_sha = compute_content_sha256(content.encode())
        count = await _query_count(store, TABLE_DOCUMENT_CONTENT, f"content_sha256 = '{content_sha}'")
        assert count == 1

        assert len(await store.load("test-dedup-scope-a", [CHDocA])) == 1
        assert len(await store.load("test-dedup-scope-b", [CHDocA])) == 1

    async def test_shared_attachment_content_deduplication(self, store: ClickHouseDocumentStore):
        """Two documents sharing the same attachment content produce one content row for that attachment."""
        shared_att_content = b"shared attachment body"
        doc1 = CHDocA(name="att-dedup-1.txt", content=b"body1", attachments=(Attachment(name="shared.txt", content=shared_att_content),))
        doc2 = CHDocA(name="att-dedup-2.txt", content=b"body2", attachments=(Attachment(name="shared.txt", content=shared_att_content),))
        await store.save(doc1, "test-att-dedup")
        await store.save(doc2, "test-att-dedup")

        att_sha = compute_content_sha256(shared_att_content)
        count = await _query_count(store, TABLE_DOCUMENT_CONTENT, f"content_sha256 = '{att_sha}'")
        assert count == 1

        loaded = await store.load("test-att-dedup", [CHDocA])
        assert len(loaded) == 2
        for doc in loaded:
            assert len(doc.attachments) == 1
            assert doc.attachments[0].content == shared_att_content


# --- ReplacingMergeTree Idempotency ---


class TestReplacingMergeTree:
    async def test_duplicate_save_single_row_after_optimize(self, store: ClickHouseDocumentStore):
        """Save same document 5 times, OPTIMIZE, verify single physical row in index."""
        doc = _make(CHDocA, "rmt.txt", "replacing merge tree test")
        for _ in range(5):
            await store.save(doc, "test-rmt")

        await store._run(lambda: store._client.command(f"OPTIMIZE TABLE {TABLE_DOCUMENT_INDEX} FINAL"))

        count = await store._run(lambda: store._client.query(f"SELECT count() FROM {TABLE_DOCUMENT_INDEX} WHERE name = 'rmt.txt'").result_rows[0][0])
        assert count == 1


# --- Circuit Breaker ---


class TestCircuitBreaker:
    async def test_save_buffers_when_disconnected(self):
        """Save to an unreachable store should buffer, not raise."""
        bad_store = ClickHouseDocumentStore(host="127.0.0.1", port=1, secure=False)
        doc = _make(CHDocA, "buffered.txt", "will be buffered")
        await bad_store.save(doc, "test-buffer")
        assert len(bad_store._buffer) == 1
        assert bad_store._buffer[0].document.name == "buffered.txt"

    async def test_circuit_opens_after_threshold_failures(self):
        """Circuit breaker opens after _FAILURE_THRESHOLD consecutive failures."""
        bad_store = ClickHouseDocumentStore(host="127.0.0.1", port=1, secure=False)
        for i in range(_FAILURE_THRESHOLD):
            await bad_store.save(_make(CHDocA, f"fail-{i}.txt", "fail"), "test-cb")
        assert bad_store._circuit_open is True
        assert bad_store._consecutive_failures >= _FAILURE_THRESHOLD

    async def test_buffer_respects_max_size(self):
        """Buffer should not grow beyond _MAX_BUFFER_SIZE (deque maxlen)."""
        bad_store = ClickHouseDocumentStore(host="127.0.0.1", port=1, secure=False)
        # Force circuit open to skip connection attempts
        bad_store._circuit_open = True
        bad_store._last_reconnect_attempt = time.monotonic()  # prevent reconnect

        for i in range(_MAX_BUFFER_SIZE + 100):
            bad_store._buffer.append(
                __import__("ai_pipeline_core.document_store.clickhouse", fromlist=["_BufferedWrite"])._BufferedWrite(
                    document=_make(CHDocA, f"buf-{i}.txt"), run_scope="test"
                )
            )
        assert len(bad_store._buffer) == _MAX_BUFFER_SIZE

    async def test_buffered_writes_flushed_on_reconnect(self, store: ClickHouseDocumentStore):
        """Documents buffered during outage are flushed when connection recovers."""
        doc1 = _make(CHDocA, "pre-outage.txt", "before outage")
        doc2 = _make(CHDocA, "during-outage.txt", "during outage")

        # Save first document normally
        await store.save(doc1, "test-flush")

        # Manually buffer a write (simulating outage)
        from ai_pipeline_core.document_store.clickhouse import _BufferedWrite

        store._buffer.append(_BufferedWrite(document=doc2, run_scope="test-flush"))
        store._circuit_open = True

        # Trigger reconnect by resetting state and saving
        store._circuit_open = False
        store._consecutive_failures = 0
        await store._run(store._flush_buffer)

        loaded = await store.load("test-flush", [CHDocA])
        names = {d.name for d in loaded}
        assert "pre-outage.txt" in names
        assert "during-outage.txt" in names


# --- Load by SHA256 ---


class TestLoadBySha256s:
    async def test_batch_lookup_text(self, store: ClickHouseDocumentStore):
        doc = _make(CHDocA, "lookup.txt", "point lookup content")
        await store.save(doc, "test-load-sha")
        result = await store.load_by_sha256s([doc.sha256], CHDocA, "test-load-sha")
        assert doc.sha256 in result
        loaded = result[doc.sha256]
        assert loaded.name == "lookup.txt"
        assert loaded.content == b"point lookup content"
        assert isinstance(loaded, CHDocA)

    async def test_batch_lookup_binary_with_attachments(self, store: ClickHouseDocumentStore):
        binary_content = bytes(range(256))
        att = Attachment(name="data.bin", content=b"\x89PNG" + b"\x00" * 50, description="binary att")
        doc = CHDocA(name="binary.bin", content=binary_content, attachments=(att,))
        await store.save(doc, "test-load-sha-binary")
        result = await store.load_by_sha256s([doc.sha256], CHDocA, "test-load-sha-binary")
        assert doc.sha256 in result
        loaded = result[doc.sha256]
        assert loaded.content == binary_content
        assert len(loaded.attachments) == 1
        assert loaded.attachments[0].content == att.content

    async def test_returns_empty_for_not_found(self, store: ClickHouseDocumentStore):
        assert await store.load_by_sha256s(["NONEXISTENT" * 4 + "AAAA"], CHDocA, "test-load-sha-none") == {}

    async def test_class_name_not_enforced(self, store: ClickHouseDocumentStore):
        """document_type is a construction hint, not a filter — doc is found regardless of stored type."""
        doc = _make(CHDocA, "typed.txt", "typed content")
        await store.save(doc, "test-load-sha-type")
        result = await store.load_by_sha256s([doc.sha256], CHDocB, "test-load-sha-type")
        assert doc.sha256 in result
        assert isinstance(result[doc.sha256], CHDocB)

    async def test_returns_empty_for_wrong_scope(self, store: ClickHouseDocumentStore):
        doc = _make(CHDocA, "scoped.txt", "scoped content")
        await store.save(doc, "test-load-sha-scope-a")
        assert await store.load_by_sha256s([doc.sha256], CHDocA, "test-load-sha-scope-b") == {}

    async def test_preserves_metadata(self, store: ClickHouseDocumentStore):
        origin = _make(CHDocA, "origin.txt", "origin")
        doc = _make(
            CHDocA,
            "meta.txt",
            "metadata",
            description="A doc",
            sources=("https://example.com",),
            origins=(origin.sha256,),
        )
        await store.save(doc, "test-load-sha-meta")
        result = await store.load_by_sha256s([doc.sha256], CHDocA, "test-load-sha-meta")
        assert doc.sha256 in result
        loaded = result[doc.sha256]
        assert loaded.description == "A doc"
        assert "https://example.com" in loaded.sources
        assert loaded.origins == (origin.sha256,)

    async def test_cross_scope_lookup_without_run_scope(self, store: ClickHouseDocumentStore):
        """When run_scope=None, searches across all scopes."""
        doc = _make(CHDocA, "cross-scope.txt", "cross scope lookup content")
        await store.save(doc, "test-load-sha-cross")
        result = await store.load_by_sha256s([doc.sha256], CHDocA)
        assert doc.sha256 in result
        loaded = result[doc.sha256]
        assert loaded.sha256 == doc.sha256
        assert loaded.name == "cross-scope.txt"

    async def test_multiple_sha256s(self, store: ClickHouseDocumentStore):
        doc1 = _make(CHDocA, "batch1.txt", "batch content 1")
        doc2 = _make(CHDocA, "batch2.txt", "batch content 2")
        await store.save(doc1, "test-load-sha-batch")
        await store.save(doc2, "test-load-sha-batch")
        result = await store.load_by_sha256s([doc1.sha256, doc2.sha256], CHDocA, "test-load-sha-batch")
        assert len(result) == 2
        assert doc1.sha256 in result
        assert doc2.sha256 in result

    async def test_empty_input_returns_empty(self, store: ClickHouseDocumentStore):
        assert await store.load_by_sha256s([], CHDocA) == {}

    async def test_partial_match(self, store: ClickHouseDocumentStore):
        doc = _make(CHDocA, "partial.txt", "partial content")
        await store.save(doc, "test-load-sha-partial")
        result = await store.load_by_sha256s([doc.sha256, "NONEXISTENT" * 4 + "AAAA"], CHDocA, "test-load-sha-partial")
        assert len(result) == 1
        assert doc.sha256 in result


# --- Load Scope Metadata ---


class TestLoadScopeMetadata:
    async def test_returns_all_docs_metadata(self, store: ClickHouseDocumentStore):
        doc1 = _make(CHDocA, "meta-a.txt", "content a for metadata")
        doc2 = _make(CHDocB, "meta-b.txt", "content b for metadata")
        await store.save(doc1, "test-scope-meta")
        await store.save(doc2, "test-scope-meta")
        metadata = await store.load_scope_metadata("test-scope-meta")
        assert len(metadata) >= 2
        shas = {m.sha256 for m in metadata}
        assert doc1.sha256 in shas
        assert doc2.sha256 in shas
        for node in metadata:
            assert isinstance(node, DocumentNode)

    async def test_returns_empty_for_empty_scope(self, store: ClickHouseDocumentStore):
        metadata = await store.load_scope_metadata("nonexistent-scope-meta")
        assert metadata == []

    async def test_includes_summaries(self, store: ClickHouseDocumentStore):
        doc = _make(CHDocA, "summary-meta.txt", "content for summary meta")
        await store.save(doc, "test-scope-meta-summary")
        await store.update_summary(doc.sha256, "A test summary.")
        await _optimize_tables(store)
        metadata = await store.load_scope_metadata("test-scope-meta-summary")
        matching = [m for m in metadata if m.sha256 == doc.sha256]
        assert len(matching) == 1
        assert matching[0].summary == "A test summary."

    async def test_metadata_class_names(self, store: ClickHouseDocumentStore):
        await store.save(_make(CHDocA, "class-a.txt", "aaa class meta"), "test-scope-meta-class")
        await store.save(_make(CHDocB, "class-b.txt", "bbb class meta"), "test-scope-meta-class")
        metadata = await store.load_scope_metadata("test-scope-meta-class")
        class_names = {m.class_name for m in metadata}
        assert "CHDocA" in class_names
        assert "CHDocB" in class_names


# --- Cross-Pipeline Provenance Graph ---


class TestLoadNodesBySha256s:
    async def test_batch_lookup(self, store: ClickHouseDocumentStore):
        doc1 = _make(CHDocA, "batch-a.txt", "batch a content for nodes lookup")
        doc2 = _make(CHDocB, "batch-b.txt", "batch b content for nodes lookup")
        await store.save(doc1, "test-nodes-batch")
        await store.save(doc2, "test-nodes-batch")
        result = await store.load_nodes_by_sha256s([doc1.sha256, doc2.sha256])
        assert len(result) == 2
        assert result[doc1.sha256].name == "batch-a.txt"
        assert result[doc2.sha256].class_name == "CHDocB"

    async def test_missing_sha256s_omitted(self, store: ClickHouseDocumentStore):
        doc = _make(CHDocA, "single.txt", "single content for nodes")
        await store.save(doc, "test-nodes-missing")
        result = await store.load_nodes_by_sha256s([doc.sha256, "NONEXISTENT" * 4 + "AAAA"])
        assert len(result) == 1
        assert doc.sha256 in result

    async def test_empty_input_returns_empty(self, store: ClickHouseDocumentStore):
        assert await store.load_nodes_by_sha256s([]) == {}

    async def test_cross_scope_lookup(self, store: ClickHouseDocumentStore):
        doc1 = _make(CHDocA, "scope-a.txt", "scope a content for nodes")
        doc2 = _make(CHDocB, "scope-b.txt", "scope b content for nodes")
        await store.save(doc1, "test-nodes-scope-a")
        await store.save(doc2, "test-nodes-scope-b")
        result = await store.load_nodes_by_sha256s([doc1.sha256, doc2.sha256])
        assert len(result) == 2

    async def test_includes_metadata(self, store: ClickHouseDocumentStore):
        origin = _make(CHDocA, "origin.txt", "origin content for metadata test")
        doc = CHDocA.create(
            name="meta.txt",
            content="metadata content for test",
            description="A doc",
            sources=("https://example.com",),
            origins=(origin.sha256,),
        )
        await store.save(doc, "test-nodes-meta")
        result = await store.load_nodes_by_sha256s([doc.sha256])
        node = result[doc.sha256]
        assert isinstance(node, DocumentNode)
        assert node.description == "A doc"
        assert "https://example.com" in node.sources
        assert origin.sha256 in node.origins


class TestCrossPipelineProvenance:
    """End-to-end: store pipeline docs, use walk_provenance for cross-scope traversal."""

    async def test_walk_provenance_full_chain(self, store: ClickHouseDocumentStore):
        """Simulate ai_research → credora pipeline handoff with walk_provenance."""
        scope = "test-provenance-chain"
        task = _make(CHDocA, "research_task.md", "Research Ethereum provenance chain")
        source_a = CHDocB.create(
            name="source_a.md",
            content="Ethereum whitepaper provenance chain",
            sources=("https://ethereum.org/whitepaper",),
            origins=(task.sha256,),
        )
        source_b = CHDocB.create(
            name="source_b.md",
            content="Ethereum wiki provenance chain",
            sources=("https://en.wikipedia.org/wiki/Ethereum",),
            origins=(task.sha256,),
        )
        report = CHDocA.create(
            name="report.md",
            content="# Ethereum Research Report provenance chain",
            sources=(source_a.sha256, source_b.sha256),
            origins=(task.sha256,),
        )
        for doc in [task, source_a, source_b, report]:
            await store.save(doc, scope)

        # Walk provenance from report — no scope needed
        graph = await walk_provenance(report.sha256, store.load_nodes_by_sha256s)
        assert len(graph) == 4
        assert all(sha in graph for sha in [report.sha256, source_a.sha256, source_b.sha256, task.sha256])

        # Extract external URLs
        urls = {src for node in graph.values() for src in node.sources if "://" in src}
        assert "https://ethereum.org/whitepaper" in urls
        assert "https://en.wikipedia.org/wiki/Ethereum" in urls

    async def test_provenance_graph_with_url_and_sha256_sources(self, store: ClickHouseDocumentStore):
        """Provenance graph correctly follows SHA256 refs and ignores URL refs."""
        scope = "test-provenance-mixed"
        parent = _make(CHDocA, "parent.md", "parent content for mixed provenance")
        child = CHDocA.create(
            name="child.md",
            content="child content for mixed provenance",
            sources=(parent.sha256, "https://example.com/api"),
            origins=(),
        )
        await store.save(parent, scope)
        await store.save(child, scope)

        graph = await walk_provenance(child.sha256, store.load_nodes_by_sha256s)
        assert len(graph) == 2
        child_node = graph[child.sha256]
        assert parent.sha256 in child_node.sources
        assert "https://example.com/api" in child_node.sources


# --- End-to-End with LLM ---


@pytest.mark.integration
class TestSummariesWithLLM:
    """End-to-end: generate summaries with real LLM and persist in ClickHouse."""

    @pytest.mark.skipif(not HAS_API_KEYS, reason="OpenAI API keys not configured")
    async def test_save_generate_summary_and_persist(self, store: ClickHouseDocumentStore):
        from ai_pipeline_core.observability._summary import generate_document_summary

        doc = _make(
            CHDocA,
            "research_report.md",
            (
                "# Market Analysis Q3 2025\n\n"
                "The European AI market grew 28% year-over-year, reaching $12.4B. "
                "Key drivers include enterprise adoption of LLMs and regulatory clarity from the EU AI Act. "
                "Germany and France lead adoption with 34% and 29% growth respectively."
            ),
        )
        await store.save(doc, "test-llm-summary")

        summary = await generate_document_summary(name=doc.name, excerpt=doc.content.decode())
        assert summary, "LLM should produce a non-empty summary"

        await store.update_summary(doc.sha256, summary)
        await _optimize_tables(store)

        loaded = await store.load_summaries([doc.sha256])
        assert doc.sha256 in loaded
        assert loaded[doc.sha256] == summary

    @pytest.mark.skipif(not HAS_API_KEYS, reason="OpenAI API keys not configured")
    async def test_llm_summaries_multiple_documents(self, store: ClickHouseDocumentStore):
        """Generate and persist LLM summaries for multiple documents, verify all round-trip."""
        from ai_pipeline_core.observability._summary import generate_document_summary

        docs = [
            _make(CHDocA, "code_review.md", "Python function for sorting a list of dictionaries by multiple keys."),
            _make(CHDocA, "meeting_notes.txt", "Discussed Q4 roadmap. Agreed to prioritize mobile app launch."),
        ]
        await store.save_batch(docs, "test-llm-multi")

        summaries: dict[str, str] = {}
        for doc in docs:
            s = await generate_document_summary(name=doc.name, excerpt=doc.content.decode())
            assert s, f"LLM summary empty for {doc.name}"
            summaries[doc.sha256] = s
            await store.update_summary(doc.sha256, s)

        await _optimize_tables(store)

        loaded = await store.load_summaries([d.sha256 for d in docs])
        assert len(loaded) == 2
        for doc in docs:
            assert loaded[doc.sha256] == summaries[doc.sha256]
