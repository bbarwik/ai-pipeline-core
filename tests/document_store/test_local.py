"""Tests for LocalDocumentStore."""

import json
from pathlib import Path

import pytest

from ai_pipeline_core.document_store import DocumentNode, DocumentStore, create_document_store, set_document_store, walk_provenance
from ai_pipeline_core.document_store.local import LocalDocumentStore, _safe_filename
from ai_pipeline_core.documents import Attachment, Document
from ai_pipeline_core.documents._hashing import compute_document_sha256


class ReportDoc(Document):
    pass


class DataDoc(Document):
    pass


@pytest.fixture(autouse=True)
def _reset_store():
    yield
    set_document_store(None)


@pytest.fixture
def store(tmp_path: Path) -> LocalDocumentStore:
    return LocalDocumentStore(base_path=tmp_path)


def _make(cls: type[Document], name: str, content: str = "test", **kwargs) -> Document:
    return cls.create(name=name, content=content, **kwargs)


class TestProtocolCompliance:
    def test_satisfies_document_store_protocol(self, tmp_path: Path):
        store = LocalDocumentStore(base_path=tmp_path)
        assert isinstance(store, DocumentStore)


class TestSaveLoadRoundTrip:
    @pytest.mark.asyncio
    async def test_basic_round_trip(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "report.md", "# Hello\nWorld")
        await store.save(doc, "run1")
        loaded = await store.load("run1", [ReportDoc])
        assert len(loaded) == 1
        assert loaded[0].name == "report.md"
        assert loaded[0].content == b"# Hello\nWorld"
        assert isinstance(loaded[0], ReportDoc)

    @pytest.mark.asyncio
    async def test_round_trip_preserves_description(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "report.md", "content", description="Important report")
        await store.save(doc, "run1")
        loaded = await store.load("run1", [ReportDoc])
        assert loaded[0].description == "Important report"

    @pytest.mark.asyncio
    async def test_round_trip_preserves_sources(self, store: LocalDocumentStore):
        source_doc = _make(ReportDoc, "source.txt", "original")
        doc = _make(ReportDoc, "derived.txt", "derived", sources=(source_doc.sha256, "https://example.com"))
        await store.save(doc, "run1")
        loaded = await store.load("run1", [ReportDoc])
        assert loaded[0].sources == (source_doc.sha256, "https://example.com")

    @pytest.mark.asyncio
    async def test_round_trip_preserves_origins(self, store: LocalDocumentStore):
        parent = _make(ReportDoc, "parent.txt", "parent")
        doc = ReportDoc.create(name="child.txt", content="child", origins=(parent.sha256,))
        await store.save(doc, "run1")
        loaded = await store.load("run1", [ReportDoc])
        assert loaded[0].origins == (parent.sha256,)

    @pytest.mark.asyncio
    async def test_save_batch(self, store: LocalDocumentStore):
        docs = [_make(ReportDoc, "a.md", "aaa"), _make(ReportDoc, "b.md", "bbb")]
        await store.save_batch(docs, "run1")
        loaded = await store.load("run1", [ReportDoc])
        assert len(loaded) == 2
        names = {d.name for d in loaded}
        assert names == {"a.md", "b.md"}


class TestMultipleTypes:
    @pytest.mark.asyncio
    async def test_load_filters_by_type(self, store: LocalDocumentStore):
        await store.save(_make(ReportDoc, "report.md", "report"), "run1")
        await store.save(_make(DataDoc, "data.json", '{"key": "val"}'), "run1")

        reports = await store.load("run1", [ReportDoc])
        data = await store.load("run1", [DataDoc])
        both = await store.load("run1", [ReportDoc, DataDoc])

        assert len(reports) == 1
        assert len(data) == 1
        assert len(both) == 2

    @pytest.mark.asyncio
    async def test_load_empty_scope_returns_empty(self, store: LocalDocumentStore):
        loaded = await store.load("nonexistent", [ReportDoc])
        assert loaded == []


class TestAttachments:
    @pytest.mark.asyncio
    async def test_round_trip_with_attachments(self, store: LocalDocumentStore):
        att = Attachment(name="screenshot.png", content=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, description="A screenshot")
        doc = ReportDoc.create(name="report.md", content="# Report", attachments=(att,))
        await store.save(doc, "run1")
        loaded = await store.load("run1", [ReportDoc])

        assert len(loaded) == 1
        assert len(loaded[0].attachments) == 1
        assert loaded[0].attachments[0].name == "screenshot.png"
        assert loaded[0].attachments[0].content == att.content
        assert loaded[0].attachments[0].description == "A screenshot"

    @pytest.mark.asyncio
    async def test_multiple_attachments(self, store: LocalDocumentStore):
        att1 = Attachment(name="a.txt", content=b"attachment A")
        att2 = Attachment(name="b.txt", content=b"attachment B")
        doc = ReportDoc.create(name="report.md", content="# Report", attachments=(att1, att2))
        await store.save(doc, "run1")
        loaded = await store.load("run1", [ReportDoc])

        assert len(loaded[0].attachments) == 2
        att_names = {a.name for a in loaded[0].attachments}
        assert att_names == {"a.txt", "b.txt"}


class TestCrashRecovery:
    @pytest.mark.asyncio
    async def test_content_without_meta_is_ignored(self, store: LocalDocumentStore, tmp_path: Path):
        """Content file without meta.json should be skipped during load."""
        # Create a content file without corresponding meta
        canonical = ReportDoc.canonical_name()
        doc_dir = tmp_path / "run1" / canonical
        doc_dir.mkdir(parents=True)
        (doc_dir / "orphan.md").write_text("orphaned content")

        loaded = await store.load("run1", [ReportDoc])
        assert loaded == []

    @pytest.mark.asyncio
    async def test_meta_without_content_is_skipped(self, store: LocalDocumentStore, tmp_path: Path):
        """Meta file without content file should be skipped with warning."""
        canonical = ReportDoc.canonical_name()
        doc_dir = tmp_path / "run1" / canonical
        doc_dir.mkdir(parents=True)
        meta = {"document_sha256": "X" * 52, "content_sha256": "Y" * 52, "class_name": "ReportDoc"}
        (doc_dir / "missing.md.meta.json").write_text(json.dumps(meta))

        loaded = await store.load("run1", [ReportDoc])
        assert loaded == []

    @pytest.mark.asyncio
    async def test_corrupted_meta_is_skipped(self, store: LocalDocumentStore, tmp_path: Path):
        """Invalid JSON in meta file should be skipped."""
        canonical = ReportDoc.canonical_name()
        doc_dir = tmp_path / "run1" / canonical
        doc_dir.mkdir(parents=True)
        (doc_dir / "bad.md").write_text("content")
        (doc_dir / "bad.md.meta.json").write_text("not valid json{{{")

        loaded = await store.load("run1", [ReportDoc])
        assert loaded == []


class TestConcurrentAccess:
    @pytest.mark.asyncio
    async def test_idempotent_save(self, store: LocalDocumentStore):
        """Saving the same document twice is a no-op."""
        doc = _make(ReportDoc, "report.md", "content")
        await store.save(doc, "run1")
        await store.save(doc, "run1")
        loaded = await store.load("run1", [ReportDoc])
        assert len(loaded) == 1

    @pytest.mark.asyncio
    async def test_same_name_different_content_coexist(self, store: LocalDocumentStore):
        """Documents with same name but different content get separate files."""
        doc1 = _make(ReportDoc, "report.md", "version 1")
        doc2 = _make(ReportDoc, "report.md", "version 2")
        await store.save(doc1, "run1")
        await store.save(doc2, "run1")

        loaded = await store.load("run1", [ReportDoc])
        assert len(loaded) == 2
        contents = {d.content for d in loaded}
        assert contents == {b"version 1", b"version 2"}


class TestHasDocuments:
    @pytest.mark.asyncio
    async def test_returns_false_when_empty(self, store: LocalDocumentStore):
        assert await store.has_documents("run1", ReportDoc) is False

    @pytest.mark.asyncio
    async def test_returns_true_when_present(self, store: LocalDocumentStore):
        await store.save(_make(ReportDoc, "a.md"), "run1")
        assert await store.has_documents("run1", ReportDoc) is True

    @pytest.mark.asyncio
    async def test_returns_false_for_wrong_type(self, store: LocalDocumentStore):
        await store.save(_make(ReportDoc, "a.md"), "run1")
        assert await store.has_documents("run1", DataDoc) is False


class TestCheckExisting:
    @pytest.mark.asyncio
    async def test_finds_saved_document(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "a.md", "content")
        doc_sha = compute_document_sha256(doc)
        await store.save(doc, "run1")
        result = await store.check_existing([doc_sha])
        assert doc_sha in result

    @pytest.mark.asyncio
    async def test_returns_empty_for_unknown(self, store: LocalDocumentStore):
        result = await store.check_existing(["NONEXISTENT" * 4 + "AAAA"])
        assert result == set()


class TestFileLayout:
    @pytest.mark.asyncio
    async def test_creates_correct_directory_structure(self, store: LocalDocumentStore, tmp_path: Path):
        doc = _make(ReportDoc, "report.md", "# Hello")
        sha = compute_document_sha256(doc)
        safe_name = _safe_filename("report.md", sha)
        await store.save(doc, "run1")

        canonical = ReportDoc.canonical_name()
        assert (tmp_path / "run1" / canonical / safe_name).exists()
        assert (tmp_path / "run1" / canonical / f"{safe_name}.meta.json").exists()

    @pytest.mark.asyncio
    async def test_meta_json_content(self, store: LocalDocumentStore, tmp_path: Path):
        doc = _make(ReportDoc, "report.md", "content", description="desc")
        sha = compute_document_sha256(doc)
        safe_name = _safe_filename("report.md", sha)
        await store.save(doc, "run1")

        canonical = ReportDoc.canonical_name()
        meta_path = tmp_path / "run1" / canonical / f"{safe_name}.meta.json"
        meta = json.loads(meta_path.read_text())

        assert meta["name"] == "report.md"
        assert meta["class_name"] == "ReportDoc"
        assert meta["description"] == "desc"
        assert meta["document_sha256"] == sha
        assert "content_sha256" in meta
        assert isinstance(meta["sources"], list)
        assert isinstance(meta["origins"], list)

    @pytest.mark.asyncio
    async def test_attachment_directory(self, store: LocalDocumentStore, tmp_path: Path):
        att = Attachment(name="img.png", content=b"\x89PNG" + b"\x00" * 50)
        doc = ReportDoc.create(name="report.md", content="text", attachments=(att,))
        sha = compute_document_sha256(doc)
        safe_name = _safe_filename("report.md", sha)
        await store.save(doc, "run1")

        canonical = ReportDoc.canonical_name()
        att_dir = tmp_path / "run1" / canonical / f"{safe_name}.att"
        assert att_dir.is_dir()
        assert (att_dir / "img.png").exists()


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_binary_content_round_trip(self, store: LocalDocumentStore):
        """Binary content (non-UTF-8) should survive save/load."""
        binary = bytes(range(256))  # All byte values 0-255
        doc = ReportDoc(name="binary.bin", content=binary)
        await store.save(doc, "run1")
        loaded = await store.load("run1", [ReportDoc])
        assert len(loaded) == 1
        assert loaded[0].content == binary

    @pytest.mark.asyncio
    async def test_empty_content_round_trip(self, store: LocalDocumentStore):
        doc = ReportDoc(name="empty.txt", content=b"")
        await store.save(doc, "run1")
        loaded = await store.load("run1", [ReportDoc])
        assert len(loaded) == 1
        assert loaded[0].content == b""

    @pytest.mark.asyncio
    async def test_origins_empty_tuple_round_trip(self, store: LocalDocumentStore):
        """Origins=() should survive round-trip as (), not become None."""
        doc = ReportDoc.create(name="a.txt", content="test")
        assert doc.origins == ()
        await store.save(doc, "run1")
        loaded = await store.load("run1", [ReportDoc])
        assert loaded[0].origins == ()


class TestCollisionSafeFilenames:
    @pytest.mark.asyncio
    async def test_loaded_name_is_original(self, store: LocalDocumentStore):
        """Loaded document.name must be the original, not the suffixed filesystem name."""
        doc = _make(ReportDoc, "report.md", "content")
        await store.save(doc, "run1")
        loaded = await store.load("run1", [ReportDoc])
        assert loaded[0].name == "report.md"

    @pytest.mark.asyncio
    async def test_sha256_preserved_after_round_trip(self, store: LocalDocumentStore):
        """Document SHA256 must be identical after save/load (name round-trips correctly)."""
        doc = _make(ReportDoc, "report.md", "content")
        original_sha = compute_document_sha256(doc)
        await store.save(doc, "run1")
        loaded = await store.load("run1", [ReportDoc])
        assert compute_document_sha256(loaded[0]) == original_sha

    @pytest.mark.asyncio
    async def test_no_extension(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "README", "content")
        await store.save(doc, "run1")
        loaded = await store.load("run1", [ReportDoc])
        assert loaded[0].name == "README"

    @pytest.mark.asyncio
    async def test_multiple_dots_in_name(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "archive.tar.gz", "content")
        await store.save(doc, "run1")
        loaded = await store.load("run1", [ReportDoc])
        assert loaded[0].name == "archive.tar.gz"

    @pytest.mark.asyncio
    async def test_dotfile_name(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, ".gitignore", "content")
        await store.save(doc, "run1")
        loaded = await store.load("run1", [ReportDoc])
        assert loaded[0].name == ".gitignore"

    def test_path_traversal_rejected_at_document_level(self):
        """Document validation rejects path traversal before the store."""
        with pytest.raises(Exception, match="path traversal"):
            ReportDoc(name="../../../etc/passwd", content=b"evil")

    @pytest.mark.asyncio
    async def test_backward_compat_load_without_name_in_meta(self, store: LocalDocumentStore, tmp_path: Path):
        """Old meta.json files without 'name' field should fall back to filesystem name."""
        canonical = ReportDoc.canonical_name()
        doc_dir = tmp_path / "run1" / canonical
        doc_dir.mkdir(parents=True)
        (doc_dir / "old_doc.md").write_bytes(b"old content")
        meta = {
            "document_sha256": "X" * 52,
            "content_sha256": "Y" * 52,
            "class_name": "ReportDoc",
            "description": None,
            "sources": [],
            "origins": [],
            "mime_type": "text/markdown",
            "attachments": [],
        }
        (doc_dir / "old_doc.md.meta.json").write_text(json.dumps(meta))
        loaded = await store.load("run1", [ReportDoc])
        assert len(loaded) == 1
        assert loaded[0].name == "old_doc.md"


class TestSafeFilenameHelper:
    def test_standard_extension(self):
        assert _safe_filename("report.md", "A7B2C9XXXX") == "report_A7B2C9.md"

    def test_no_extension(self):
        assert _safe_filename("README", "A7B2C9XXXX") == "README_A7B2C9"

    def test_multiple_dots(self):
        assert _safe_filename("archive.tar.gz", "A7B2C9XXXX") == "archive.tar_A7B2C9.gz"

    def test_dotfile(self):
        assert _safe_filename(".gitignore", "A7B2C9XXXX") == ".gitignore_A7B2C9"


class TestFactory:
    def test_create_document_store_returns_local_when_no_clickhouse(self):
        from ai_pipeline_core.settings import Settings

        settings = Settings(clickhouse_host="")
        store = create_document_store(settings)
        assert isinstance(store, LocalDocumentStore)

    def test_create_document_store_rejects_non_settings(self):
        with pytest.raises(TypeError, match="Expected Settings"):
            create_document_store("not_settings")  # type: ignore[arg-type]


class TestLoadBySha256s:
    @pytest.mark.asyncio
    async def test_returns_correct_doc(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "report.md", "content")
        await store.save(doc, "run1")
        result = await store.load_by_sha256s([doc.sha256], ReportDoc, "run1")
        assert doc.sha256 in result
        assert result[doc.sha256].sha256 == doc.sha256
        assert result[doc.sha256].name == "report.md"
        assert isinstance(result[doc.sha256], ReportDoc)

    @pytest.mark.asyncio
    async def test_cache_hit_path(self, store: LocalDocumentStore):
        """Second load_by_sha256s call uses the meta path cache."""
        doc = _make(ReportDoc, "report.md", "content")
        await store.save(doc, "run1")
        result1 = await store.load_by_sha256s([doc.sha256], ReportDoc, "run1")
        result2 = await store.load_by_sha256s([doc.sha256], ReportDoc, "run1")
        assert doc.sha256 in result1
        assert doc.sha256 in result2
        assert result1[doc.sha256].sha256 == result2[doc.sha256].sha256

    @pytest.mark.asyncio
    async def test_cache_miss_scans_type_dir(self, store: LocalDocumentStore):
        """When cache is empty, scan type directory for the document."""
        doc = _make(ReportDoc, "report.md", "content")
        await store.save(doc, "run1")
        store._meta_path_cache.clear()
        result = await store.load_by_sha256s([doc.sha256], ReportDoc, "run1")
        assert doc.sha256 in result
        assert result[doc.sha256].sha256 == doc.sha256

    @pytest.mark.asyncio
    async def test_returns_empty_for_unknown_sha(self, store: LocalDocumentStore):
        assert await store.load_by_sha256s(["NONEXISTENT" * 4 + "AAAA"], ReportDoc, "run1") == {}

    @pytest.mark.asyncio
    async def test_class_name_not_enforced(self, store: LocalDocumentStore):
        """document_type is a construction hint, not a filter — doc is found regardless of stored type."""
        doc = _make(ReportDoc, "report.md", "content")
        await store.save(doc, "run1")
        result = await store.load_by_sha256s([doc.sha256], DataDoc, "run1")
        assert doc.sha256 in result

    @pytest.mark.asyncio
    async def test_with_attachments(self, store: LocalDocumentStore):
        att = Attachment(name="screenshot.png", content=b"\x89PNG" + b"\x00" * 50, description="A screenshot")
        doc = ReportDoc.create(name="report.md", content="# Report", attachments=(att,))
        await store.save(doc, "run1")
        result = await store.load_by_sha256s([doc.sha256], ReportDoc, "run1")
        assert doc.sha256 in result
        loaded = result[doc.sha256]
        assert len(loaded.attachments) == 1
        assert loaded.attachments[0].name == "screenshot.png"
        assert loaded.attachments[0].content == att.content

    @pytest.mark.asyncio
    async def test_returns_empty_for_nonexistent_scope(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "report.md", "content")
        await store.save(doc, "run1")
        assert await store.load_by_sha256s([doc.sha256], ReportDoc, "run2") == {}

    @pytest.mark.asyncio
    async def test_cross_scope_lookup_without_run_scope(self, store: LocalDocumentStore):
        """When run_scope=None, searches across all scope directories."""
        doc = _make(ReportDoc, "report.md", "cross scope content")
        await store.save(doc, "run1")
        result = await store.load_by_sha256s([doc.sha256], ReportDoc)
        assert doc.sha256 in result
        assert result[doc.sha256].sha256 == doc.sha256

    @pytest.mark.asyncio
    async def test_cross_scope_class_name_not_enforced(self, store: LocalDocumentStore):
        """Cross-scope: document_type is a construction hint, not a filter."""
        doc = _make(ReportDoc, "report.md", "content")
        await store.save(doc, "run1")
        result = await store.load_by_sha256s([doc.sha256], DataDoc)
        assert doc.sha256 in result

    @pytest.mark.asyncio
    async def test_multiple_sha256s(self, store: LocalDocumentStore):
        doc1 = _make(ReportDoc, "a.md", "content1")
        doc2 = _make(ReportDoc, "b.md", "content2")
        await store.save(doc1, "run1")
        await store.save(doc2, "run1")
        result = await store.load_by_sha256s([doc1.sha256, doc2.sha256], ReportDoc, "run1")
        assert len(result) == 2
        assert doc1.sha256 in result
        assert doc2.sha256 in result

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self, store: LocalDocumentStore):
        assert await store.load_by_sha256s([], ReportDoc) == {}

    @pytest.mark.asyncio
    async def test_partial_match(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "a.md", "content")
        await store.save(doc, "run1")
        result = await store.load_by_sha256s([doc.sha256, "NONEXISTENT" * 4 + "AAAA"], ReportDoc, "run1")
        assert len(result) == 1
        assert doc.sha256 in result


class TestLoadScopeMetadata:
    @pytest.mark.asyncio
    async def test_returns_all_docs_metadata(self, store: LocalDocumentStore):
        doc1 = _make(ReportDoc, "a.md", "aaa")
        doc2 = _make(DataDoc, "b.json", '{"key": "val"}')
        await store.save(doc1, "run1")
        await store.save(doc2, "run1")
        metadata = await store.load_scope_metadata("run1")
        assert len(metadata) == 2
        shas = {m.sha256 for m in metadata}
        assert doc1.sha256 in shas
        assert doc2.sha256 in shas

    @pytest.mark.asyncio
    async def test_returns_empty_for_nonexistent_scope(self, store: LocalDocumentStore):
        assert await store.load_scope_metadata("nonexistent") == []

    @pytest.mark.asyncio
    async def test_returns_document_node_instances(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "report.md", "content", description="desc")
        await store.save(doc, "run1")
        metadata = await store.load_scope_metadata("run1")
        assert len(metadata) == 1
        node = metadata[0]
        assert isinstance(node, DocumentNode)
        assert node.sha256 == doc.sha256
        assert node.class_name == "ReportDoc"
        assert node.name == "report.md"
        assert node.description == "desc"

    @pytest.mark.asyncio
    async def test_includes_summaries(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "report.md", "content")
        await store.save(doc, "run1")
        await store.update_summary(doc.sha256, "test summary")
        metadata = await store.load_scope_metadata("run1")
        assert len(metadata) == 1
        assert metadata[0].summary == "test summary"

    @pytest.mark.asyncio
    async def test_metadata_has_sources_and_origins(self, store: LocalDocumentStore):
        origin_doc = _make(ReportDoc, "origin.txt", "origin")
        doc = ReportDoc.create(
            name="child.txt",
            content="child",
            sources=("https://example.com",),
            origins=(origin_doc.sha256,),
        )
        await store.save(doc, "run1")
        metadata = await store.load_scope_metadata("run1")
        # Filter for the child doc
        child_nodes = [m for m in metadata if m.sha256 == doc.sha256]
        assert len(child_nodes) == 1
        assert "https://example.com" in child_nodes[0].sources
        assert origin_doc.sha256 in child_nodes[0].origins


class TestLoadNodesBySha256s:
    @pytest.mark.asyncio
    async def test_returns_found_nodes(self, store: LocalDocumentStore):
        doc1 = _make(ReportDoc, "a.md", "aaa")
        doc2 = _make(DataDoc, "b.json", '{"key": "val"}')
        await store.save(doc1, "run1")
        await store.save(doc2, "run1")
        result = await store.load_nodes_by_sha256s([doc1.sha256, doc2.sha256])
        assert len(result) == 2
        assert result[doc1.sha256].name == "a.md"
        assert result[doc2.sha256].name == "b.json"

    @pytest.mark.asyncio
    async def test_missing_sha256s_omitted(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "a.md", "content")
        await store.save(doc, "run1")
        result = await store.load_nodes_by_sha256s([doc.sha256, "NONEXISTENT" * 4 + "AAAA"])
        assert len(result) == 1
        assert doc.sha256 in result

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self, store: LocalDocumentStore):
        assert await store.load_nodes_by_sha256s([]) == {}

    @pytest.mark.asyncio
    async def test_cross_scope_lookup(self, store: LocalDocumentStore):
        doc1 = _make(ReportDoc, "a.md", "scope1 content")
        doc2 = _make(DataDoc, "b.json", '{"scope": 2}')
        await store.save(doc1, "scope_a")
        await store.save(doc2, "scope_b")
        result = await store.load_nodes_by_sha256s([doc1.sha256, doc2.sha256])
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_returns_document_node_instances_with_metadata(self, store: LocalDocumentStore):
        origin = _make(ReportDoc, "origin.md", "origin")
        doc = ReportDoc.create(
            name="child.md",
            content="child",
            description="A child doc",
            sources=("https://example.com",),
            origins=(origin.sha256,),
        )
        await store.save(doc, "run1")
        result = await store.load_nodes_by_sha256s([doc.sha256])
        node = result[doc.sha256]
        assert isinstance(node, DocumentNode)
        assert node.class_name == "ReportDoc"
        assert node.description == "A child doc"
        assert "https://example.com" in node.sources
        assert origin.sha256 in node.origins

    @pytest.mark.asyncio
    async def test_includes_summaries(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "a.md", "content")
        await store.save(doc, "run1")
        await store.update_summary(doc.sha256, "test summary")
        result = await store.load_nodes_by_sha256s([doc.sha256])
        assert result[doc.sha256].summary == "test summary"


class TestCrossPipelineProvenanceGraph:
    """End-to-end provenance graph on local filesystem with walk_provenance."""

    @pytest.mark.asyncio
    async def test_walk_provenance_full_chain(self, store: LocalDocumentStore):
        """Simulate ai_research → credora: walk provenance without knowing scope."""
        task = _make(ReportDoc, "task.md", "Research Bitcoin")
        source_a = DataDoc.create(
            name="src_a.json",
            content='{"url": "https://bitcoin.org"}',
            sources=("https://bitcoin.org",),
            origins=(task.sha256,),
        )
        source_b = DataDoc.create(
            name="src_b.json",
            content='{"url": "https://wiki.org"}',
            sources=("https://wiki.org",),
            origins=(task.sha256,),
        )
        report = ReportDoc.create(
            name="report.md",
            content="# Bitcoin Report",
            sources=(source_a.sha256, source_b.sha256),
            origins=(task.sha256,),
        )
        scope = "research/btc/run1"
        for doc in [task, source_a, source_b, report]:
            await store.save(doc, scope)

        # Walk provenance from report without knowing scope
        graph = await walk_provenance(report.sha256, store.load_nodes_by_sha256s)
        assert len(graph) == 4
        assert all(sha in graph for sha in [report.sha256, source_a.sha256, source_b.sha256, task.sha256])

        # Extract external URLs
        urls = {src for node in graph.values() for src in node.sources if "://" in src}
        assert "https://bitcoin.org" in urls
        assert "https://wiki.org" in urls

    @pytest.mark.asyncio
    async def test_cross_scope_multiple_scopes(self, store: LocalDocumentStore):
        """Documents in different scopes are found by cross-scope lookup."""
        doc1 = _make(ReportDoc, "doc1.md", "scope1 content")
        doc2 = _make(DataDoc, "doc2.json", '{"scope": 2}')
        await store.save(doc1, "scope_a")
        await store.save(doc2, "scope_b")

        # Find both without scope
        assert doc1.sha256 in await store.load_by_sha256s([doc1.sha256], ReportDoc)
        assert doc2.sha256 in await store.load_by_sha256s([doc2.sha256], DataDoc)

        # class_name not enforced — wrong type still finds the doc
        assert doc1.sha256 in await store.load_by_sha256s([doc1.sha256], DataDoc)


class TestLocalStoreSummaryAcrossScopes:
    @pytest.mark.asyncio
    async def test_summary_update_writes_all_scope_copies(self, store: LocalDocumentStore):
        """update_summary writes to ALL meta.json files across scopes."""
        import json

        doc = _make(ReportDoc, "shared.md", "shared content")
        await store.save(doc, "scope_a")
        await store.save(doc, "scope_b")
        await store.update_summary(doc.sha256, "global summary")

        # Verify both meta files have the summary
        meta_files = list(store.base_path.rglob("*.meta.json"))
        assert len(meta_files) == 2
        for mf in meta_files:
            meta = json.loads(mf.read_text())
            assert meta["summary"] == "global summary"

    @pytest.mark.asyncio
    async def test_summary_visible_from_load_summaries(self, store: LocalDocumentStore):
        """load_summaries returns summary for docs across all scopes."""
        doc = _make(ReportDoc, "shared.md", "shared content")
        await store.save(doc, "scope_a")
        await store.update_summary(doc.sha256, "test summary")
        summaries = await store.load_summaries([doc.sha256])
        assert summaries[doc.sha256] == "test summary"
