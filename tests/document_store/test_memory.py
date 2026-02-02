"""Tests for MemoryDocumentStore."""

import pytest

from ai_pipeline_core.document_store import DocumentStore
from ai_pipeline_core.document_store.memory import MemoryDocumentStore
from ai_pipeline_core.documents import Document


class DocA(Document):
    pass


class DocB(Document):
    pass


@pytest.fixture
def store() -> MemoryDocumentStore:
    return MemoryDocumentStore()


def _make(cls: type[Document], name: str, content: str = "test") -> Document:
    return cls.create(name=name, content=content)


class TestProtocolCompliance:
    def test_satisfies_document_store_protocol(self):
        store = MemoryDocumentStore()
        assert isinstance(store, DocumentStore)


class TestSaveAndLoad:
    @pytest.mark.asyncio
    async def test_save_and_load_single(self, store: MemoryDocumentStore):
        doc = _make(DocA, "report.txt", "hello world")
        await store.save(doc, "run1")
        loaded = await store.load("run1", [DocA])
        assert len(loaded) == 1
        assert loaded[0].name == "report.txt"
        assert loaded[0].content == b"hello world"

    @pytest.mark.asyncio
    async def test_save_batch(self, store: MemoryDocumentStore):
        docs = [_make(DocA, "a.txt", "aaa"), _make(DocA, "b.txt", "bbb")]
        await store.save_batch(docs, "run1")
        loaded = await store.load("run1", [DocA])
        assert len(loaded) == 2

    @pytest.mark.asyncio
    async def test_load_filters_by_type(self, store: MemoryDocumentStore):
        await store.save(_make(DocA, "a.txt", "aaa"), "run1")
        await store.save(_make(DocB, "b.txt", "bbb"), "run1")

        loaded_a = await store.load("run1", [DocA])
        loaded_b = await store.load("run1", [DocB])
        loaded_both = await store.load("run1", [DocA, DocB])

        assert len(loaded_a) == 1
        assert len(loaded_b) == 1
        assert len(loaded_both) == 2

    @pytest.mark.asyncio
    async def test_load_empty_scope(self, store: MemoryDocumentStore):
        loaded = await store.load("nonexistent", [DocA])
        assert loaded == []

    @pytest.mark.asyncio
    async def test_save_idempotent(self, store: MemoryDocumentStore):
        doc = _make(DocA, "a.txt", "aaa")
        await store.save(doc, "run1")
        await store.save(doc, "run1")
        loaded = await store.load("run1", [DocA])
        assert len(loaded) == 1

    @pytest.mark.asyncio
    async def test_scopes_are_isolated(self, store: MemoryDocumentStore):
        await store.save(_make(DocA, "a.txt", "aaa"), "run1")
        await store.save(_make(DocA, "b.txt", "bbb"), "run2")

        assert len(await store.load("run1", [DocA])) == 1
        assert len(await store.load("run2", [DocA])) == 1


class TestHasDocuments:
    @pytest.mark.asyncio
    async def test_returns_false_when_empty(self, store: MemoryDocumentStore):
        assert await store.has_documents("run1", DocA) is False

    @pytest.mark.asyncio
    async def test_returns_true_when_present(self, store: MemoryDocumentStore):
        await store.save(_make(DocA, "a.txt"), "run1")
        assert await store.has_documents("run1", DocA) is True

    @pytest.mark.asyncio
    async def test_returns_false_for_wrong_type(self, store: MemoryDocumentStore):
        await store.save(_make(DocA, "a.txt"), "run1")
        assert await store.has_documents("run1", DocB) is False


class TestCheckExisting:
    @pytest.mark.asyncio
    async def test_returns_matching_hashes(self, store: MemoryDocumentStore):
        doc = _make(DocA, "a.txt", "aaa")
        await store.save(doc, "run1")
        result = await store.check_existing([doc.sha256, "NONEXISTENT" * 4 + "AAAA"])
        assert doc.sha256 in result

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_matches(self, store: MemoryDocumentStore):
        result = await store.check_existing(["NONEXISTENT" * 4 + "AAAA"])
        assert result == set()
