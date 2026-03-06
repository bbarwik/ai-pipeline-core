"""Tests for document store summary generation."""

import asyncio
import time

import pytest

from ai_pipeline_core.document_store._summary_worker import SUMMARY_EXCERPT_CHARS, _build_excerpt
from ai_pipeline_core.document_store._local import LocalDocumentStore
from ai_pipeline_core.document_store._memory import MemoryDocumentStore
from ai_pipeline_core.documents._context import _suppress_document_registration
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.documents import RunScope


class SummaryTestDocument(Document):
    pass


@pytest.fixture(autouse=True)
def _suppress_registration():
    with _suppress_document_registration():
        yield


async def _mock_generator(name: str, excerpt: str) -> str:
    return f"Summary of {name}"


async def _failing_generator(name: str, excerpt: str) -> str:
    raise RuntimeError("LLM unavailable")


async def _wait_for_summary(store: MemoryDocumentStore | LocalDocumentStore, sha256: str, timeout: float = 5.0) -> str | None:
    """Poll for a summary to appear in the store."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        summaries = await store.load_summaries([sha256])
        if sha256 in summaries:
            return summaries[sha256]
        await asyncio.sleep(0.1)
    return None


class TestMemoryStoreSummary:
    @pytest.mark.asyncio
    async def test_update_and_load_summary(self):
        store = MemoryDocumentStore()
        doc = SummaryTestDocument(name="test.txt", content=b"Hello world", description="test")
        await store.save(doc, RunScope("run1"))
        await store.update_summary(doc.sha256, "A greeting document")
        summaries = await store.load_summaries([doc.sha256])
        assert summaries[doc.sha256] == "A greeting document"

    @pytest.mark.asyncio
    async def test_summary_not_returned_when_absent(self):
        store = MemoryDocumentStore()
        doc = SummaryTestDocument(name="test.txt", content=b"Hello world", description="test")
        await store.save(doc, RunScope("run1"))
        summaries = await store.load_summaries([doc.sha256])
        assert summaries == {}

    @pytest.mark.asyncio
    async def test_worker_generates_summary(self):
        store = MemoryDocumentStore(summary_generator=_mock_generator)
        try:
            doc = SummaryTestDocument(name="report.txt", content=b"some content", description="test")
            await store.save(doc, RunScope("run1"))
            summary = await _wait_for_summary(store, doc.sha256)
            assert summary == "Summary of report.txt"
        finally:
            store.shutdown()

    @pytest.mark.asyncio
    async def test_worker_generates_summary_for_empty_doc(self):
        store = MemoryDocumentStore(summary_generator=_mock_generator)
        try:
            doc = SummaryTestDocument(name="empty.txt", content=b"", description="test")
            await store.save(doc, RunScope("run1"))
            summary = await _wait_for_summary(store, doc.sha256)
            assert summary == "Summary of empty.txt"
        finally:
            store.shutdown()

    @pytest.mark.asyncio
    async def test_worker_generates_summary_for_binary_doc(self):
        store = MemoryDocumentStore(summary_generator=_mock_generator)
        try:
            doc = SummaryTestDocument(name="image.bin", content=b"\x00\xff" * 100, description="test")
            await store.save(doc, RunScope("run1"))
            summary = await _wait_for_summary(store, doc.sha256)
            assert summary == "Summary of image.bin"
        finally:
            store.shutdown()

    @pytest.mark.asyncio
    async def test_worker_handles_generator_failure(self):
        store = MemoryDocumentStore(summary_generator=_failing_generator)
        try:
            doc = SummaryTestDocument(name="fail.txt", content=b"some content", description="test")
            await store.save(doc, RunScope("run1"))
            await asyncio.sleep(1.0)
            summaries = await store.load_summaries([doc.sha256])
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
            await store.save(doc, RunScope("run1"))
            await store.save(doc, RunScope("run1"))  # second save, same doc
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
        await store.save(doc, RunScope("run1"))
        await store.update_summary(doc.sha256, "A greeting document")
        summaries = await store.load_summaries([doc.sha256])
        assert summaries[doc.sha256] == "A greeting document"

    @pytest.mark.asyncio
    async def test_summary_persisted_in_meta_json(self, tmp_path):
        import json

        store = LocalDocumentStore(base_path=tmp_path)
        content = "Some content for summary test"
        doc = SummaryTestDocument(name="data.txt", content=content.encode(), description="test")
        await store.save(doc, RunScope("run1"))
        await store.update_summary(doc.sha256, "Data summary")

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
            await store.save(doc, RunScope("run1"))
            summary = await _wait_for_summary(store, doc.sha256)
            assert summary == "Summary of report.txt"
        finally:
            store.shutdown()


class TestBuildExcerpt:
    def test_short_text_included_in_full(self):
        doc = SummaryTestDocument(name="short.txt", content=b"Hello world", description="A greeting")
        excerpt = _build_excerpt(doc)
        assert "<name>short.txt</name>" in excerpt
        assert "<class>SummaryTestDocument</class>" in excerpt
        assert "<description>A greeting</description>" in excerpt
        assert "<content>" in excerpt
        assert "Hello world" in excerpt
        assert "</content>" in excerpt
        assert "</document>" in excerpt
        assert "truncated" not in excerpt

    def test_no_description_omitted(self):
        doc = SummaryTestDocument(name="short.txt", content=b"Hello world")
        excerpt = _build_excerpt(doc)
        assert "<class>SummaryTestDocument</class>" in excerpt
        assert "<description>" not in excerpt

    def test_long_text_has_truncation_markers(self):
        text = "A" * 10_000 + "B" * 30_000 + "C" * 10_000
        doc = SummaryTestDocument(name="long.txt", content=text.encode(), description="Long doc")
        excerpt = _build_excerpt(doc)
        assert "truncated" in excerpt
        assert "<description>Long doc</description>" in excerpt
        assert "<content>" in excerpt
        assert "</content>" in excerpt

    def test_long_text_contains_start_and_end_content(self):
        text = "START_MARKER_" + "x" * 50_000 + "END_MARKER_HERE"
        doc = SummaryTestDocument(name="long.txt", content=text.encode())
        excerpt = _build_excerpt(doc)
        assert "START_MARKER_" in excerpt
        assert "END_MARKER_HERE" in excerpt

    def test_truncation_marker_shows_char_count(self):
        total = 60_000
        text = "x" * total
        doc = SummaryTestDocument(name="big.txt", content=text.encode())
        excerpt = _build_excerpt(doc)
        assert "chars truncated" in excerpt

    def test_binary_document(self):
        doc = SummaryTestDocument(name="image.png", content=b"\x89PNG\r\n" + b"\x00" * 100)
        excerpt = _build_excerpt(doc)
        assert "<class>SummaryTestDocument</class>" in excerpt
        assert "<content>[Binary:" in excerpt
        assert "image/png" in excerpt
        assert "</document>" in excerpt

    def test_binary_document_with_description(self):
        doc = SummaryTestDocument(name="photo.jpg", content=b"\xff\xd8\xff" + b"\x00" * 100, description="Product photo")
        excerpt = _build_excerpt(doc)
        assert "<description>Product photo</description>" in excerpt
        assert "[Binary:" in excerpt

    def test_exactly_at_threshold_not_split(self):
        text = "x" * SUMMARY_EXCERPT_CHARS
        doc = SummaryTestDocument(name="exact.txt", content=text.encode())
        excerpt = _build_excerpt(doc)
        assert "truncated" not in excerpt

    def test_one_over_threshold_is_split(self):
        text = "x" * (SUMMARY_EXCERPT_CHARS + 1)
        doc = SummaryTestDocument(name="over.txt", content=text.encode())
        excerpt = _build_excerpt(doc)
        assert "chars truncated" in excerpt

    def test_document_name_always_in_excerpt(self):
        doc = SummaryTestDocument(name="report.md", content=b"content")
        excerpt = _build_excerpt(doc)
        assert "<name>report.md</name>" in excerpt
