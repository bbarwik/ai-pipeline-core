"""Tests for DualDocumentStore."""

import pytest

from ai_pipeline_core.document_store import DocumentNode, DocumentStore, set_document_store, walk_provenance
from ai_pipeline_core.document_store._dual_store import DualDocumentStore
from ai_pipeline_core.document_store.memory import MemoryDocumentStore
from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents._hashing import compute_document_sha256


class DualReportDoc(Document):
    pass


@pytest.fixture(autouse=True)
def _reset_store():
    yield
    set_document_store(None)


def _make(name: str, content: str = "test") -> DualReportDoc:
    return DualReportDoc.create(name=name, content=content)


class TestProtocolCompliance:
    def test_satisfies_document_store_protocol(self):
        dual = DualDocumentStore(primary=MemoryDocumentStore(), secondary=MemoryDocumentStore())
        assert isinstance(dual, DocumentStore)


class TestSaveFanOut:
    @pytest.mark.asyncio
    async def test_save_writes_to_both_stores(self):
        primary, secondary = MemoryDocumentStore(), MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=secondary)
        doc = _make("a.md", "content")
        await dual.save(doc, "run1")
        assert await primary.has_documents("run1", DualReportDoc)
        assert await secondary.has_documents("run1", DualReportDoc)

    @pytest.mark.asyncio
    async def test_save_batch_writes_to_both_stores(self):
        primary, secondary = MemoryDocumentStore(), MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=secondary)
        docs = [_make("a.md", "aaa"), _make("b.md", "bbb")]
        await dual.save_batch(docs, "run1")
        assert len(await primary.load("run1", [DualReportDoc])) == 2
        assert len(await secondary.load("run1", [DualReportDoc])) == 2


class TestReadDelegation:
    @pytest.mark.asyncio
    async def test_load_reads_primary_only(self):
        primary, secondary = MemoryDocumentStore(), MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=secondary)
        doc = _make("a.md", "content")
        await secondary.save(doc, "run1")
        loaded = await dual.load("run1", [DualReportDoc])
        assert loaded == []

    @pytest.mark.asyncio
    async def test_has_documents_reads_primary(self):
        primary, secondary = MemoryDocumentStore(), MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=secondary)
        doc = _make("a.md", "content")
        await primary.save(doc, "run1")
        assert await dual.has_documents("run1", DualReportDoc)

    @pytest.mark.asyncio
    async def test_check_existing_reads_primary(self):
        primary, secondary = MemoryDocumentStore(), MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=secondary)
        doc = _make("a.md", "content")
        sha = compute_document_sha256(doc)
        await primary.save(doc, "run1")
        assert sha in await dual.check_existing([sha])

    @pytest.mark.asyncio
    async def test_load_summaries_reads_primary(self):
        primary, secondary = MemoryDocumentStore(), MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=secondary)
        doc = _make("a.md", "content")
        sha = compute_document_sha256(doc)
        await primary.save(doc, "run1")
        await primary.update_summary(sha, "test summary")
        result = await dual.load_summaries([sha])
        assert result[sha] == "test summary"


class TestSecondaryFailure:
    @pytest.mark.asyncio
    async def test_secondary_save_failure_does_not_propagate(self):
        class FailingSave(MemoryDocumentStore):
            async def save(self, document: Document, run_scope: str) -> None:
                raise RuntimeError("disk full")

        primary = MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=FailingSave())
        doc = _make("a.md", "content")
        await dual.save(doc, "run1")  # Should not raise
        assert await primary.has_documents("run1", DualReportDoc)

    @pytest.mark.asyncio
    async def test_secondary_save_batch_failure_does_not_propagate(self):
        class FailingBatch(MemoryDocumentStore):
            async def save_batch(self, documents: list[Document], run_scope: str) -> None:
                raise RuntimeError("disk full")

        primary = MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=FailingBatch())
        docs = [_make("a.md", "content")]
        await dual.save_batch(docs, "run1")  # Should not raise
        assert len(await primary.load("run1", [DualReportDoc])) == 1

    @pytest.mark.asyncio
    async def test_secondary_update_summary_failure_does_not_propagate(self):
        class FailingSummary(MemoryDocumentStore):
            async def update_summary(self, document_sha256: str, summary: str) -> None:
                raise RuntimeError("write error")

        primary = MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=FailingSummary())
        doc = _make("a.md", "content")
        sha = compute_document_sha256(doc)
        await primary.save(doc, "run1")
        await dual.update_summary(sha, "summary")  # Should not raise
        assert (await primary.load_summaries([sha]))[sha] == "summary"

    def test_secondary_shutdown_failure_does_not_propagate(self):
        class FailingShutdown(MemoryDocumentStore):
            def shutdown(self) -> None:
                raise RuntimeError("shutdown failed")

        primary = MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=FailingShutdown())
        dual.shutdown()  # Should not raise

    def test_secondary_flush_failure_does_not_propagate(self):
        class FailingFlush(MemoryDocumentStore):
            def flush(self) -> None:
                raise RuntimeError("flush failed")

        primary = MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=FailingFlush())
        dual.flush()  # Should not raise


class TestPrimaryFailure:
    @pytest.mark.asyncio
    async def test_primary_save_failure_propagates(self):
        class FailingSave(MemoryDocumentStore):
            async def save(self, document: Document, run_scope: str) -> None:
                raise RuntimeError("primary down")

        secondary = MemoryDocumentStore()
        dual = DualDocumentStore(primary=FailingSave(), secondary=secondary)
        doc = _make("a.md", "content")
        with pytest.raises(RuntimeError, match="primary down"):
            await dual.save(doc, "run1")
        # Secondary should NOT have been called since primary failed first
        assert not await secondary.has_documents("run1", DualReportDoc)

    @pytest.mark.asyncio
    async def test_primary_update_summary_failure_still_attempts_secondary(self):
        class FailingSummary(MemoryDocumentStore):
            async def update_summary(self, document_sha256: str, summary: str) -> None:
                raise RuntimeError("primary down")

        secondary = MemoryDocumentStore()
        dual = DualDocumentStore(primary=FailingSummary(), secondary=secondary)
        doc = _make("a.md", "content")
        sha = compute_document_sha256(doc)
        await secondary.save(doc, "run1")
        with pytest.raises(RuntimeError, match="primary down"):
            await dual.update_summary(sha, "summary")
        # Secondary should still have been updated
        assert (await secondary.load_summaries([sha]))[sha] == "summary"

    def test_primary_flush_failure_still_attempts_secondary(self):
        flushed = []

        class FailingFlush(MemoryDocumentStore):
            def flush(self) -> None:
                raise RuntimeError("primary flush failed")

        class TrackingFlush(MemoryDocumentStore):
            def flush(self) -> None:
                flushed.append(True)

        dual = DualDocumentStore(primary=FailingFlush(), secondary=TrackingFlush())
        with pytest.raises(RuntimeError, match="primary flush failed"):
            dual.flush()
        assert flushed == [True]


class TestUpdateSummaryFanOut:
    @pytest.mark.asyncio
    async def test_update_summary_fans_out_to_both(self):
        primary, secondary = MemoryDocumentStore(), MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=secondary)
        doc = _make("a.md", "content")
        sha = compute_document_sha256(doc)
        await primary.save(doc, "run1")
        await secondary.save(doc, "run1")
        await dual.update_summary(sha, "summary text")
        assert (await primary.load_summaries([sha]))[sha] == "summary text"
        assert (await secondary.load_summaries([sha]))[sha] == "summary text"


class TestLoadBySha256sDelegation:
    @pytest.mark.asyncio
    async def test_delegates_to_primary(self):
        primary, secondary = MemoryDocumentStore(), MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=secondary)
        doc = _make("a.md", "content")
        await primary.save(doc, "run1")
        result = await dual.load_by_sha256s([doc.sha256], DualReportDoc, "run1")
        assert doc.sha256 in result
        assert result[doc.sha256].sha256 == doc.sha256

    @pytest.mark.asyncio
    async def test_does_not_read_secondary(self):
        primary, secondary = MemoryDocumentStore(), MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=secondary)
        doc = _make("a.md", "content")
        await secondary.save(doc, "run1")
        assert await dual.load_by_sha256s([doc.sha256], DualReportDoc, "run1") == {}

    @pytest.mark.asyncio
    async def test_cross_scope_delegates_to_primary(self):
        primary, secondary = MemoryDocumentStore(), MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=secondary)
        doc = _make("a.md", "cross scope dual")
        await primary.save(doc, "run1")
        result = await dual.load_by_sha256s([doc.sha256], DualReportDoc)
        assert doc.sha256 in result
        assert result[doc.sha256].sha256 == doc.sha256


class TestLoadScopeMetadataDelegation:
    @pytest.mark.asyncio
    async def test_delegates_to_primary(self):
        primary, secondary = MemoryDocumentStore(), MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=secondary)
        doc = _make("a.md", "content")
        await primary.save(doc, "run1")
        metadata = await dual.load_scope_metadata("run1")
        assert len(metadata) == 1
        assert isinstance(metadata[0], DocumentNode)
        assert metadata[0].sha256 == doc.sha256

    @pytest.mark.asyncio
    async def test_does_not_read_secondary(self):
        primary, secondary = MemoryDocumentStore(), MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=secondary)
        doc = _make("a.md", "content")
        await secondary.save(doc, "run1")
        assert await dual.load_scope_metadata("run1") == []


class TestLoadNodesBySha256sDelegation:
    @pytest.mark.asyncio
    async def test_delegates_to_primary(self):
        primary, secondary = MemoryDocumentStore(), MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=secondary)
        doc = _make("a.md", "content")
        await primary.save(doc, "run1")
        result = await dual.load_nodes_by_sha256s([doc.sha256])
        assert len(result) == 1
        assert doc.sha256 in result

    @pytest.mark.asyncio
    async def test_does_not_read_secondary(self):
        primary, secondary = MemoryDocumentStore(), MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=secondary)
        doc = _make("a.md", "content")
        await secondary.save(doc, "run1")
        assert await dual.load_nodes_by_sha256s([doc.sha256]) == {}

    @pytest.mark.asyncio
    async def test_walk_provenance_through_dual(self):
        """walk_provenance works through DualDocumentStore delegation."""
        primary, secondary = MemoryDocumentStore(), MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=secondary)
        parent = _make("parent.md", "parent content")
        child = DualReportDoc.create(name="child.md", content="child", sources=(parent.sha256,))
        await dual.save(parent, "run1")
        await dual.save(child, "run1")
        graph = await walk_provenance(child.sha256, dual.load_nodes_by_sha256s)
        assert len(graph) == 2
        assert child.sha256 in graph
        assert parent.sha256 in graph


class TestShutdown:
    def test_shutdown_cascades_to_both(self):
        primary, secondary = MemoryDocumentStore(), MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=secondary)
        dual.shutdown()  # No exception

    def test_flush_cascades_to_both(self):
        primary, secondary = MemoryDocumentStore(), MemoryDocumentStore()
        dual = DualDocumentStore(primary=primary, secondary=secondary)
        dual.flush()  # No exception
