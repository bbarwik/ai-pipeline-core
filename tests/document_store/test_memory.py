"""Tests for MemoryDocumentStore."""

from datetime import timedelta

import pytest

from ai_pipeline_core.document_store import DocumentNode, DocumentStore, walk_provenance
from ai_pipeline_core.document_store.memory import MemoryDocumentStore
from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents._types import DocumentSha256, RunScope


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
        await store.save(doc, RunScope("run1"))
        loaded = await store.load(RunScope("run1"), [DocA])
        assert len(loaded) == 1
        assert loaded[0].name == "report.txt"
        assert loaded[0].content == b"hello world"

    @pytest.mark.asyncio
    async def test_save_batch(self, store: MemoryDocumentStore):
        docs = [_make(DocA, "a.txt", "aaa"), _make(DocA, "b.txt", "bbb")]
        await store.save_batch(docs, RunScope("run1"))
        loaded = await store.load(RunScope("run1"), [DocA])
        assert len(loaded) == 2

    @pytest.mark.asyncio
    async def test_load_filters_by_type(self, store: MemoryDocumentStore):
        await store.save(_make(DocA, "a.txt", "aaa"), RunScope("run1"))
        await store.save(_make(DocB, "b.txt", "bbb"), RunScope("run1"))

        loaded_a = await store.load(RunScope("run1"), [DocA])
        loaded_b = await store.load(RunScope("run1"), [DocB])
        loaded_both = await store.load(RunScope("run1"), [DocA, DocB])

        assert len(loaded_a) == 1
        assert len(loaded_b) == 1
        assert len(loaded_both) == 2

    @pytest.mark.asyncio
    async def test_load_empty_scope(self, store: MemoryDocumentStore):
        loaded = await store.load(RunScope("nonexistent"), [DocA])
        assert loaded == []

    @pytest.mark.asyncio
    async def test_save_idempotent(self, store: MemoryDocumentStore):
        doc = _make(DocA, "a.txt", "aaa")
        await store.save(doc, RunScope("run1"))
        await store.save(doc, RunScope("run1"))
        loaded = await store.load(RunScope("run1"), [DocA])
        assert len(loaded) == 1

    @pytest.mark.asyncio
    async def test_scopes_are_isolated(self, store: MemoryDocumentStore):
        await store.save(_make(DocA, "a.txt", "aaa"), RunScope("run1"))
        await store.save(_make(DocA, "b.txt", "bbb"), RunScope("run2"))

        assert len(await store.load(RunScope("run1"), [DocA])) == 1
        assert len(await store.load(RunScope("run2"), [DocA])) == 1


class TestHasDocuments:
    @pytest.mark.asyncio
    async def test_returns_false_when_empty(self, store: MemoryDocumentStore):
        assert await store.has_documents(RunScope("run1"), DocA) is False

    @pytest.mark.asyncio
    async def test_returns_true_when_present(self, store: MemoryDocumentStore):
        await store.save(_make(DocA, "a.txt"), RunScope("run1"))
        assert await store.has_documents(RunScope("run1"), DocA) is True

    @pytest.mark.asyncio
    async def test_returns_false_for_wrong_type(self, store: MemoryDocumentStore):
        await store.save(_make(DocA, "a.txt"), RunScope("run1"))
        assert await store.has_documents(RunScope("run1"), DocB) is False

    @pytest.mark.asyncio
    async def test_max_age_ignored_in_memory_store(self, store: MemoryDocumentStore):
        """Memory store ignores max_age — always returns True if documents exist."""
        await store.save(_make(DocA, "a.txt"), RunScope("run1"))
        assert await store.has_documents(RunScope("run1"), DocA, max_age=timedelta(seconds=0)) is True


class TestCheckExisting:
    @pytest.mark.asyncio
    async def test_returns_matching_hashes(self, store: MemoryDocumentStore):
        doc = _make(DocA, "a.txt", "aaa")
        await store.save(doc, RunScope("run1"))
        result = await store.check_existing([doc.sha256, DocumentSha256("NONEXISTENT" * 4 + "AAAA")])
        assert doc.sha256 in result

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_matches(self, store: MemoryDocumentStore):
        result = await store.check_existing([DocumentSha256("NONEXISTENT" * 4 + "AAAA")])
        assert result == set()


class TestLoadBySha256s:
    @pytest.mark.asyncio
    async def test_returns_correct_docs(self, store: MemoryDocumentStore):
        doc = _make(DocA, "a.txt", "content")
        await store.save(doc, RunScope("run1"))
        result = await store.load_by_sha256s([doc.sha256], DocA, RunScope("run1"))
        assert doc.sha256 in result
        assert result[doc.sha256].sha256 == doc.sha256
        assert isinstance(result[doc.sha256], DocA)

    @pytest.mark.asyncio
    async def test_returns_empty_for_unknown_sha(self, store: MemoryDocumentStore):
        assert await store.load_by_sha256s([DocumentSha256("NONEXISTENT" * 4 + "AAAA")], DocA, RunScope("run1")) == {}

    @pytest.mark.asyncio
    async def test_returns_empty_for_wrong_scope(self, store: MemoryDocumentStore):
        doc = _make(DocA, "a.txt", "content")
        await store.save(doc, RunScope("run1"))
        assert await store.load_by_sha256s([doc.sha256], DocA, RunScope("run2")) == {}

    @pytest.mark.asyncio
    async def test_class_name_not_enforced(self, store: MemoryDocumentStore):
        """document_type is a construction hint, not a filter — doc is returned regardless of stored type."""
        doc = _make(DocA, "a.txt", "content")
        await store.save(doc, RunScope("run1"))
        result = await store.load_by_sha256s([doc.sha256], DocB, RunScope("run1"))
        assert doc.sha256 in result

    @pytest.mark.asyncio
    async def test_return_type_matches_document_type(self, store: MemoryDocumentStore):
        """Verify the generic return type works — returned object is the requested type."""
        doc = _make(DocA, "a.txt", "content")
        await store.save(doc, RunScope("run1"))
        result: dict[DocumentSha256, DocA] = await store.load_by_sha256s([doc.sha256], DocA, RunScope("run1"))
        assert doc.sha256 in result
        assert type(result[doc.sha256]) is DocA

    @pytest.mark.asyncio
    async def test_cross_scope_lookup_without_run_scope(self, store: MemoryDocumentStore):
        """When run_scope=None, searches across all scopes."""
        doc = _make(DocA, "a.txt", "cross scope content")
        await store.save(doc, RunScope("run1"))
        result = await store.load_by_sha256s([doc.sha256], DocA)
        assert doc.sha256 in result
        assert result[doc.sha256].sha256 == doc.sha256

    @pytest.mark.asyncio
    async def test_cross_scope_class_name_not_enforced(self, store: MemoryDocumentStore):
        """Cross-scope: document_type is a construction hint, not a filter."""
        doc = _make(DocA, "a.txt", "content")
        await store.save(doc, RunScope("run1"))
        result = await store.load_by_sha256s([doc.sha256], DocB)
        assert doc.sha256 in result

    @pytest.mark.asyncio
    async def test_multiple_sha256s(self, store: MemoryDocumentStore):
        doc1 = _make(DocA, "a.txt", "content1")
        doc2 = _make(DocA, "b.txt", "content2")
        await store.save(doc1, RunScope("run1"))
        await store.save(doc2, RunScope("run1"))
        result = await store.load_by_sha256s([doc1.sha256, doc2.sha256], DocA, RunScope("run1"))
        assert len(result) == 2
        assert doc1.sha256 in result
        assert doc2.sha256 in result

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self, store: MemoryDocumentStore):
        assert await store.load_by_sha256s([], DocA) == {}

    @pytest.mark.asyncio
    async def test_partial_match(self, store: MemoryDocumentStore):
        """Only matching SHA256s are returned, missing ones are omitted."""
        doc = _make(DocA, "a.txt", "content")
        await store.save(doc, RunScope("run1"))
        result = await store.load_by_sha256s([doc.sha256, DocumentSha256("NONEXISTENT" * 4 + "AAAA")], DocA, RunScope("run1"))
        assert len(result) == 1
        assert doc.sha256 in result


class TestLoadScopeMetadata:
    @pytest.mark.asyncio
    async def test_returns_all_docs_in_scope(self, store: MemoryDocumentStore):
        doc_a = _make(DocA, "a.txt", "aaa")
        doc_b = _make(DocB, "b.txt", "bbb")
        await store.save(doc_a, RunScope("run1"))
        await store.save(doc_b, RunScope("run1"))
        metadata = await store.load_scope_metadata(RunScope("run1"))
        assert len(metadata) == 2
        shas = {m.sha256 for m in metadata}
        assert doc_a.sha256 in shas
        assert doc_b.sha256 in shas

    @pytest.mark.asyncio
    async def test_returns_empty_for_nonexistent_scope(self, store: MemoryDocumentStore):
        assert await store.load_scope_metadata(RunScope("nonexistent")) == []

    @pytest.mark.asyncio
    async def test_includes_summaries(self, store: MemoryDocumentStore):
        doc = _make(DocA, "a.txt", "content")
        await store.save(doc, RunScope("run1"))
        await store.update_summary(doc.sha256, "test summary")
        metadata = await store.load_scope_metadata(RunScope("run1"))
        assert len(metadata) == 1
        assert metadata[0].summary == "test summary"

    @pytest.mark.asyncio
    async def test_returns_document_node_instances(self, store: MemoryDocumentStore):
        doc = _make(DocA, "a.txt", "content")
        await store.save(doc, RunScope("run1"))
        metadata = await store.load_scope_metadata(RunScope("run1"))
        assert len(metadata) == 1
        node = metadata[0]
        assert isinstance(node, DocumentNode)
        assert node.sha256 == doc.sha256
        assert node.class_name == "DocA"
        assert node.name == "a.txt"

    @pytest.mark.asyncio
    async def test_metadata_has_sources_and_origins(self, store: MemoryDocumentStore):
        origin_doc = _make(DocA, "origin.txt", "origin")
        doc = DocA.create(name="child.txt", content="child", sources=("https://example.com",), origins=(origin_doc.sha256,))
        await store.save(doc, RunScope("run1"))
        metadata = await store.load_scope_metadata(RunScope("run1"))
        assert len(metadata) == 1
        assert "https://example.com" in metadata[0].sources
        assert origin_doc.sha256 in metadata[0].origins


class TestLoadNodesBySha256s:
    @pytest.mark.asyncio
    async def test_returns_found_nodes(self, store: MemoryDocumentStore):
        doc_a = _make(DocA, "a.txt", "aaa")
        doc_b = _make(DocB, "b.txt", "bbb")
        await store.save(doc_a, RunScope("run1"))
        await store.save(doc_b, RunScope("run1"))
        result = await store.load_nodes_by_sha256s([doc_a.sha256, doc_b.sha256])
        assert len(result) == 2
        assert result[doc_a.sha256].name == "a.txt"
        assert result[doc_b.sha256].name == "b.txt"

    @pytest.mark.asyncio
    async def test_missing_sha256s_omitted(self, store: MemoryDocumentStore):
        doc = _make(DocA, "a.txt", "aaa")
        await store.save(doc, RunScope("run1"))
        result = await store.load_nodes_by_sha256s([doc.sha256, DocumentSha256("NONEXISTENT" * 4 + "AAAA")])
        assert len(result) == 1
        assert doc.sha256 in result

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self, store: MemoryDocumentStore):
        assert await store.load_nodes_by_sha256s([]) == {}

    @pytest.mark.asyncio
    async def test_cross_scope_lookup(self, store: MemoryDocumentStore):
        doc1 = _make(DocA, "a.txt", "scope1 content")
        doc2 = _make(DocB, "b.txt", "scope2 content")
        await store.save(doc1, RunScope("scope1"))
        await store.save(doc2, RunScope("scope2"))
        result = await store.load_nodes_by_sha256s([doc1.sha256, doc2.sha256])
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_returns_document_node_instances(self, store: MemoryDocumentStore):
        doc = DocA.create(name="a.txt", content="test", sources=("https://example.com",))
        await store.save(doc, RunScope("run1"))
        result = await store.load_nodes_by_sha256s([doc.sha256])
        node = result[doc.sha256]
        assert isinstance(node, DocumentNode)
        assert node.class_name == "DocA"
        assert "https://example.com" in node.sources

    @pytest.mark.asyncio
    async def test_includes_summaries(self, store: MemoryDocumentStore):
        doc = _make(DocA, "a.txt", "content")
        await store.save(doc, RunScope("run1"))
        await store.update_summary(doc.sha256, "test summary")
        result = await store.load_nodes_by_sha256s([doc.sha256])
        assert result[doc.sha256].summary == "test summary"


class TestCrossPipelineProvenanceGraph:
    """End-to-end test simulating the cross-pipeline use case:
    Pipeline A stores documents with provenance → Pipeline B uses walk_provenance
    to traverse the entire chain without knowing the run scope.
    """

    @pytest.mark.asyncio
    async def test_walk_provenance_cross_pipeline(self, store: MemoryDocumentStore):
        # Pipeline A: ai_research stores documents
        task_doc = _make(DocA, "research_task.md", "What is Bitcoin?")
        source_a = DocB.create(
            name="source_a.md",
            content="Bitcoin whitepaper",
            sources=("https://bitcoin.org/whitepaper",),
            origins=(task_doc.sha256,),
        )
        source_b = DocB.create(
            name="source_b.md",
            content="Bitcoin wiki",
            sources=("https://en.wikipedia.org/wiki/Bitcoin",),
            origins=(task_doc.sha256,),
        )
        report = DocA.create(
            name="research_report.md",
            content="Bitcoin is a cryptocurrency...",
            sources=(source_a.sha256, source_b.sha256),
            origins=(task_doc.sha256,),
        )
        for doc in [task_doc, source_a, source_b, report]:
            await store.save(doc, RunScope("ai_research/bitcoin/run1"))

        # Pipeline B: credora receives the report, walks provenance without knowing scope
        graph = await walk_provenance(report.sha256, store.load_nodes_by_sha256s)
        assert report.sha256 in graph
        assert source_a.sha256 in graph
        assert source_b.sha256 in graph
        assert task_doc.sha256 in graph

        # Extract all external URLs from the provenance chain
        urls = set()
        for node in graph.values():
            for src in node.sources:
                if "://" in src:
                    urls.add(src)
        assert "https://bitcoin.org/whitepaper" in urls
        assert "https://en.wikipedia.org/wiki/Bitcoin" in urls

    @pytest.mark.asyncio
    async def test_walk_provenance_url_sources_not_followed(self, store: MemoryDocumentStore):
        """URL sources are preserved in metadata but not followed as graph edges."""
        source_doc = DocA.create(
            name="fetched.md",
            content="fetched content",
            sources=("https://example.com/data",),
        )
        report = DocA.create(
            name="report.md",
            content="report content",
            sources=(source_doc.sha256, "https://example.com/other"),
        )
        for doc in [source_doc, report]:
            await store.save(doc, RunScope("run1"))

        graph = await walk_provenance(report.sha256, store.load_nodes_by_sha256s)
        assert report.sha256 in graph
        assert source_doc.sha256 in graph
        assert "https://example.com/other" in graph[report.sha256].sources

    @pytest.mark.asyncio
    async def test_walk_provenance_returns_empty_for_unknown_root(self, store: MemoryDocumentStore):
        graph = await walk_provenance(DocumentSha256("NONEXISTENT" * 4 + "AAAA"), store.load_nodes_by_sha256s)
        assert graph == {}

    @pytest.mark.asyncio
    async def test_cross_scope_finds_doc_in_any_scope(self, store: MemoryDocumentStore):
        """When same doc exists in multiple scopes, cross-scope lookup finds it."""
        doc = _make(DocA, "shared.txt", "shared multi-scope content")
        await store.save(doc, RunScope("scope1"))
        await store.save(doc, RunScope("scope2"))
        result = await store.load_by_sha256s([doc.sha256], DocA)
        assert doc.sha256 in result
        assert result[doc.sha256].sha256 == doc.sha256

    @pytest.mark.asyncio
    async def test_cross_scope_not_found_returns_empty(self, store: MemoryDocumentStore):
        """Cross-scope lookup for nonexistent doc returns empty dict."""
        assert await store.load_by_sha256s([DocumentSha256("NONEXISTENT" * 4 + "AAAA")], DocA) == {}


class TestGlobalDedup:
    @pytest.mark.asyncio
    async def test_same_doc_two_scopes_single_global_entry(self, store: MemoryDocumentStore):
        """Same document saved to two scopes → one global entry, two run memberships."""
        doc = _make(DocA, "shared.txt", "shared content")
        await store.save(doc, RunScope("scope_a"))
        await store.save(doc, RunScope("scope_b"))
        assert len(store._documents) == 1
        assert doc.sha256 in store._run_docs.get(RunScope("scope_a"), set())
        assert doc.sha256 in store._run_docs.get(RunScope("scope_b"), set())

    @pytest.mark.asyncio
    async def test_has_documents_false_for_wrong_scope(self, store: MemoryDocumentStore):
        """has_documents returns False when doc exists globally but not in queried scope."""
        doc = _make(DocA, "a.txt", "content")
        await store.save(doc, RunScope("scope_a"))
        assert await store.has_documents(RunScope("scope_a"), DocA) is True
        assert await store.has_documents(RunScope("scope_b"), DocA) is False

    @pytest.mark.asyncio
    async def test_summary_visible_across_scopes(self, store: MemoryDocumentStore):
        """Summary set on a document is visible from any scope's metadata."""
        doc = _make(DocA, "a.txt", "content")
        await store.save(doc, RunScope("scope_a"))
        await store.save(doc, RunScope("scope_b"))
        await store.update_summary(doc.sha256, "global summary")
        meta_a = await store.load_scope_metadata(RunScope("scope_a"))
        meta_b = await store.load_scope_metadata(RunScope("scope_b"))
        assert meta_a[0].summary == "global summary"
        assert meta_b[0].summary == "global summary"
