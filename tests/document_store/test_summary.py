"""Tests for document store summary generation."""

import asyncio
import time

import pytest

from ai_pipeline_core.document_store.local import LocalDocumentStore
from ai_pipeline_core.document_store.memory import MemoryDocumentStore
from ai_pipeline_core.documents.document import Document


class SummaryTestDocument(Document):
    pass


async def _mock_generator(name: str, excerpt: str) -> str:
    return f"Summary of {name}"


async def _failing_generator(name: str, excerpt: str) -> str:
    raise RuntimeError("LLM unavailable")


async def _wait_for_summary(store: MemoryDocumentStore | LocalDocumentStore, run_scope: str, sha256: str, timeout: float = 5.0) -> str | None:
    """Poll for a summary to appear in the store."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        summaries = await store.load_summaries(run_scope, [sha256])
        if sha256 in summaries:
            return summaries[sha256]
        await asyncio.sleep(0.1)
    return None


class TestMemoryStoreSummary:
    @pytest.mark.asyncio
    async def test_update_and_load_summary(self):
        store = MemoryDocumentStore()
        doc = SummaryTestDocument(name="test.txt", content=b"Hello world", description="test")
        await store.save(doc, "run1")
        await store.update_summary("run1", doc.sha256, "A greeting document")
        summaries = await store.load_summaries("run1", [doc.sha256])
        assert summaries[doc.sha256] == "A greeting document"

    @pytest.mark.asyncio
    async def test_summary_not_returned_when_absent(self):
        store = MemoryDocumentStore()
        doc = SummaryTestDocument(name="test.txt", content=b"Hello world", description="test")
        await store.save(doc, "run1")
        summaries = await store.load_summaries("run1", [doc.sha256])
        assert summaries == {}

    @pytest.mark.asyncio
    async def test_worker_generates_summary(self):
        store = MemoryDocumentStore(summary_generator=_mock_generator)
        try:
            doc = SummaryTestDocument(name="report.txt", content=b"some content", description="test")
            await store.save(doc, "run1")
            summary = await _wait_for_summary(store, "run1", doc.sha256)
            assert summary == "Summary of report.txt"
        finally:
            store.shutdown()

    @pytest.mark.asyncio
    async def test_worker_generates_summary_for_empty_doc(self):
        store = MemoryDocumentStore(summary_generator=_mock_generator)
        try:
            doc = SummaryTestDocument(name="empty.txt", content=b"", description="test")
            await store.save(doc, "run1")
            summary = await _wait_for_summary(store, "run1", doc.sha256)
            assert summary == "Summary of empty.txt"
        finally:
            store.shutdown()

    @pytest.mark.asyncio
    async def test_worker_generates_summary_for_binary_doc(self):
        store = MemoryDocumentStore(summary_generator=_mock_generator)
        try:
            doc = SummaryTestDocument(name="image.bin", content=b"\x00\xff" * 100, description="test")
            await store.save(doc, "run1")
            summary = await _wait_for_summary(store, "run1", doc.sha256)
            assert summary == "Summary of image.bin"
        finally:
            store.shutdown()

    @pytest.mark.asyncio
    async def test_worker_handles_generator_failure(self):
        store = MemoryDocumentStore(summary_generator=_failing_generator)
        try:
            doc = SummaryTestDocument(name="fail.txt", content=b"some content", description="test")
            await store.save(doc, "run1")
            await asyncio.sleep(1.0)
            summaries = await store.load_summaries("run1", [doc.sha256])
            assert summaries == {}
        finally:
            store.shutdown()

    @pytest.mark.asyncio
    async def test_idempotent_save_no_requeue(self):
        call_count = 0

        async def counting_generator(name: str, excerpt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"Summary #{call_count}"

        store = MemoryDocumentStore(summary_generator=counting_generator)
        try:
            doc = SummaryTestDocument(name="repeat.txt", content=b"some content", description="test")
            await store.save(doc, "run1")
            await store.save(doc, "run1")  # second save, same doc
            await asyncio.sleep(1.0)
            assert call_count == 1
        finally:
            store.shutdown()


class TestLocalStoreSummary:
    @pytest.mark.asyncio
    async def test_update_and_load_summary(self, tmp_path):
        store = LocalDocumentStore(base_path=tmp_path)
        content = "Hello world content for testing"
        doc = SummaryTestDocument(name="test.txt", content=content.encode(), description="test")
        await store.save(doc, "run1")
        await store.update_summary("run1", doc.sha256, "A greeting document")
        summaries = await store.load_summaries("run1", [doc.sha256])
        assert summaries[doc.sha256] == "A greeting document"

    @pytest.mark.asyncio
    async def test_summary_persisted_in_meta_json(self, tmp_path):
        import json

        store = LocalDocumentStore(base_path=tmp_path)
        content = "Some content for summary test"
        doc = SummaryTestDocument(name="data.txt", content=content.encode(), description="test")
        await store.save(doc, "run1")
        await store.update_summary("run1", doc.sha256, "Data summary")

        # Verify in meta.json on disk
        meta_files = list(tmp_path.rglob("*.meta.json"))
        assert len(meta_files) == 1
        meta = json.loads(meta_files[0].read_text())
        assert meta["summary"] == "Data summary"

    @pytest.mark.asyncio
    async def test_worker_generates_summary(self, tmp_path):
        store = LocalDocumentStore(base_path=tmp_path, summary_generator=_mock_generator)
        try:
            doc = SummaryTestDocument(name="report.txt", content=b"some content", description="test")
            await store.save(doc, "run1")
            summary = await _wait_for_summary(store, "run1", doc.sha256)
            assert summary == "Summary of report.txt"
        finally:
            store.shutdown()
