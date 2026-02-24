"""Tests for LocalDocumentStore."""

import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ai_pipeline_core.document_store._factory import create_document_store
from ai_pipeline_core.document_store._protocol import DocumentStore, set_document_store
from ai_pipeline_core.document_store._models import DocumentNode, walk_provenance
from ai_pipeline_core.document_store._local import LocalDocumentStore, _safe_filename, _atomic_write_text
from ai_pipeline_core.documents import Attachment, Document
from ai_pipeline_core.documents._hashing import compute_document_sha256
from ai_pipeline_core.documents.types import DocumentSha256, RunScope


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


def _make(cls: type[Document], name: str, content: str = "test", **kwargs: Any) -> Document:
    if "derived_from" in kwargs or "triggered_by" in kwargs:
        return cls.create(name=name, content=content, **kwargs)
    return cls.create_root(name=name, content=content, reason="test fixture", **kwargs)


class TestProtocolCompliance:
    def test_satisfies_document_store_protocol(self, tmp_path: Path):
        store = LocalDocumentStore(base_path=tmp_path)
        assert isinstance(store, DocumentStore)


class TestSaveLoadRoundTrip:
    @pytest.mark.asyncio
    async def test_basic_round_trip(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "report.md", "# Hello\nWorld")
        await store.save(doc, RunScope("run1"))
        loaded = await store.load(RunScope("run1"), [ReportDoc])
        assert len(loaded) == 1
        assert loaded[0].name == "report.md"
        assert loaded[0].content == b"# Hello\nWorld"
        assert isinstance(loaded[0], ReportDoc)

    @pytest.mark.asyncio
    async def test_round_trip_preserves_description(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "report.md", "content", description="Important report")
        await store.save(doc, RunScope("run1"))
        loaded = await store.load(RunScope("run1"), [ReportDoc])
        assert loaded[0].description == "Important report"

    @pytest.mark.asyncio
    async def test_round_trip_preserves_derived_from(self, store: LocalDocumentStore):
        source_doc = _make(ReportDoc, "source.txt", "original")
        doc = _make(ReportDoc, "derived.txt", "derived", derived_from=(source_doc.sha256, "https://example.com"))
        await store.save(doc, RunScope("run1"))
        loaded = await store.load(RunScope("run1"), [ReportDoc])
        assert loaded[0].derived_from == (source_doc.sha256, "https://example.com")

    @pytest.mark.asyncio
    async def test_round_trip_preserves_triggered_by(self, store: LocalDocumentStore):
        parent = _make(ReportDoc, "parent.txt", "parent")
        doc = ReportDoc.create(name="child.txt", content="child", triggered_by=(parent.sha256,))
        await store.save(doc, RunScope("run1"))
        loaded = await store.load(RunScope("run1"), [ReportDoc])
        assert loaded[0].triggered_by == (parent.sha256,)

    @pytest.mark.asyncio
    async def test_save_batch(self, store: LocalDocumentStore):
        docs = [_make(ReportDoc, "a.md", "aaa"), _make(ReportDoc, "b.md", "bbb")]
        await store.save_batch(docs, RunScope("run1"))
        loaded = await store.load(RunScope("run1"), [ReportDoc])
        assert len(loaded) == 2
        names = {d.name for d in loaded}
        assert names == {"a.md", "b.md"}


class TestMultipleTypes:
    @pytest.mark.asyncio
    async def test_load_filters_by_type(self, store: LocalDocumentStore):
        await store.save(_make(ReportDoc, "report.md", "report"), RunScope("run1"))
        await store.save(_make(DataDoc, "data.json", '{"key": "val"}'), RunScope("run1"))

        reports = await store.load(RunScope("run1"), [ReportDoc])
        data = await store.load(RunScope("run1"), [DataDoc])
        both = await store.load(RunScope("run1"), [ReportDoc, DataDoc])

        assert len(reports) == 1
        assert len(data) == 1
        assert len(both) == 2

    @pytest.mark.asyncio
    async def test_load_empty_scope_returns_empty(self, store: LocalDocumentStore):
        loaded = await store.load(RunScope("nonexistent"), [ReportDoc])
        assert loaded == []


class TestAttachments:
    @pytest.mark.asyncio
    async def test_round_trip_with_attachments(self, store: LocalDocumentStore):
        att = Attachment(name="screenshot.png", content=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, description="A screenshot")
        doc = ReportDoc.create_root(name="report.md", content="# Report", attachments=(att,), reason="test input")
        await store.save(doc, RunScope("run1"))
        loaded = await store.load(RunScope("run1"), [ReportDoc])

        assert len(loaded) == 1
        assert len(loaded[0].attachments) == 1
        assert loaded[0].attachments[0].name == "screenshot.png"
        assert loaded[0].attachments[0].content == att.content
        assert loaded[0].attachments[0].description == "A screenshot"

    @pytest.mark.asyncio
    async def test_multiple_attachments(self, store: LocalDocumentStore):
        att1 = Attachment(name="a.txt", content=b"attachment A")
        att2 = Attachment(name="b.txt", content=b"attachment B")
        doc = ReportDoc.create_root(name="report.md", content="# Report", attachments=(att1, att2), reason="test input")
        await store.save(doc, RunScope("run1"))
        loaded = await store.load(RunScope("run1"), [ReportDoc])

        assert len(loaded[0].attachments) == 2
        att_names = {a.name for a in loaded[0].attachments}
        assert att_names == {"a.txt", "b.txt"}


class TestCrashRecovery:
    @pytest.mark.asyncio
    async def test_content_without_meta_is_ignored(self, store: LocalDocumentStore, tmp_path: Path):
        """Content file without meta.json should be skipped during load."""
        # Create a content file without corresponding meta
        doc_dir = tmp_path / ReportDoc.__name__
        doc_dir.mkdir(parents=True)
        (doc_dir / "orphan.md").write_text("orphaned content")

        loaded = await store.load(RunScope("run1"), [ReportDoc])
        assert loaded == []

    @pytest.mark.asyncio
    async def test_meta_without_content_is_skipped(self, store: LocalDocumentStore, tmp_path: Path):
        """Meta file without content file should be skipped with warning."""
        doc_dir = tmp_path / ReportDoc.__name__
        doc_dir.mkdir(parents=True)
        meta = {"document_sha256": "X" * 52, "content_sha256": "Y" * 52, "class_name": "ReportDoc"}
        (doc_dir / "missing.md.meta.json").write_text(json.dumps(meta))

        loaded = await store.load(RunScope("run1"), [ReportDoc])
        assert loaded == []

    @pytest.mark.asyncio
    async def test_corrupted_meta_is_skipped(self, store: LocalDocumentStore, tmp_path: Path):
        """Invalid JSON in meta file should be skipped."""
        doc_dir = tmp_path / ReportDoc.__name__
        doc_dir.mkdir(parents=True)
        (doc_dir / "bad.md").write_text("content")
        (doc_dir / "bad.md.meta.json").write_text("not valid json{{{")

        loaded = await store.load(RunScope("run1"), [ReportDoc])
        assert loaded == []


class TestConcurrentAccess:
    @pytest.mark.asyncio
    async def test_idempotent_save(self, store: LocalDocumentStore):
        """Saving the same document twice is a no-op."""
        doc = _make(ReportDoc, "report.md", "content")
        await store.save(doc, RunScope("run1"))
        await store.save(doc, RunScope("run1"))
        loaded = await store.load(RunScope("run1"), [ReportDoc])
        assert len(loaded) == 1

    @pytest.mark.asyncio
    async def test_same_name_different_content_coexist(self, store: LocalDocumentStore):
        """Documents with same name but different content get separate files."""
        doc1 = _make(ReportDoc, "report.md", "version 1")
        doc2 = _make(ReportDoc, "report.md", "version 2")
        await store.save(doc1, RunScope("run1"))
        await store.save(doc2, RunScope("run1"))

        loaded = await store.load(RunScope("run1"), [ReportDoc])
        assert len(loaded) == 2
        contents = {d.content for d in loaded}
        assert contents == {b"version 1", b"version 2"}


class TestHasDocuments:
    @pytest.mark.asyncio
    async def test_returns_false_when_empty(self, store: LocalDocumentStore):
        assert await store.has_documents(RunScope("run1"), ReportDoc) is False

    @pytest.mark.asyncio
    async def test_returns_true_when_present(self, store: LocalDocumentStore):
        await store.save(_make(ReportDoc, "a.md"), RunScope("run1"))
        assert await store.has_documents(RunScope("run1"), ReportDoc) is True

    @pytest.mark.asyncio
    async def test_returns_false_for_wrong_type(self, store: LocalDocumentStore):
        await store.save(_make(ReportDoc, "a.md"), RunScope("run1"))
        assert await store.has_documents(RunScope("run1"), DataDoc) is False

    @pytest.mark.asyncio
    async def test_max_age_returns_true_for_recent(self, store: LocalDocumentStore):
        await store.save(_make(ReportDoc, "a.md"), RunScope("run1"))
        assert await store.has_documents(RunScope("run1"), ReportDoc, max_age=timedelta(hours=1)) is True

    @pytest.mark.asyncio
    async def test_max_age_returns_false_for_expired(self, store: LocalDocumentStore, tmp_path: Path):
        await store.save(_make(ReportDoc, "a.md"), RunScope("run1"))
        # Backdate stored_at in the meta file to 2 days ago
        for meta_path in tmp_path.rglob("*.meta.json"):
            meta = json.loads(meta_path.read_text())
            meta["stored_at"] = (datetime.now(UTC) - timedelta(days=2)).isoformat()
            meta_path.write_text(json.dumps(meta))
        assert await store.has_documents(RunScope("run1"), ReportDoc, max_age=timedelta(hours=24)) is False

    @pytest.mark.asyncio
    async def test_max_age_none_ignores_age(self, store: LocalDocumentStore, tmp_path: Path):
        await store.save(_make(ReportDoc, "a.md"), RunScope("run1"))
        # Backdate to very old
        for meta_path in tmp_path.rglob("*.meta.json"):
            meta = json.loads(meta_path.read_text())
            meta["stored_at"] = (datetime.now(UTC) - timedelta(days=365)).isoformat()
            meta_path.write_text(json.dumps(meta))
        assert await store.has_documents(RunScope("run1"), ReportDoc, max_age=None) is True


class TestCheckExisting:
    @pytest.mark.asyncio
    async def test_finds_saved_document(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "a.md", "content")
        doc_sha = compute_document_sha256(doc)
        await store.save(doc, RunScope("run1"))
        result = await store.check_existing([doc_sha])
        assert doc_sha in result

    @pytest.mark.asyncio
    async def test_returns_empty_for_unknown(self, store: LocalDocumentStore):
        result = await store.check_existing([DocumentSha256("NONEXISTENT" * 4 + "AAAA")])
        assert result == set()


class TestFileLayout:
    @pytest.mark.asyncio
    async def test_creates_correct_directory_structure(self, store: LocalDocumentStore, tmp_path: Path):
        doc = _make(ReportDoc, "report.md", "# Hello")
        sha = compute_document_sha256(doc)
        safe_name = _safe_filename("report.md", sha)
        await store.save(doc, RunScope("run1"))

        class_name = ReportDoc.__name__
        assert (tmp_path / class_name / safe_name).exists()
        assert (tmp_path / class_name / f"{safe_name}.meta.json").exists()

    @pytest.mark.asyncio
    async def test_meta_json_content(self, store: LocalDocumentStore, tmp_path: Path):
        doc = _make(ReportDoc, "report.md", "content", description="desc")
        sha = compute_document_sha256(doc)
        safe_name = _safe_filename("report.md", sha)
        await store.save(doc, RunScope("run1"))

        class_name = ReportDoc.__name__
        meta_path = tmp_path / class_name / f"{safe_name}.meta.json"
        meta = json.loads(meta_path.read_text())

        assert meta["name"] == "report.md"
        assert meta["class_name"] == "ReportDoc"
        assert meta["description"] == "desc"
        assert meta["document_sha256"] == sha
        assert "content_sha256" in meta
        assert isinstance(meta["derived_from"], list)
        assert isinstance(meta["triggered_by"], list)

    @pytest.mark.asyncio
    async def test_attachment_directory(self, store: LocalDocumentStore, tmp_path: Path):
        att = Attachment(name="img.png", content=b"\x89PNG" + b"\x00" * 50)
        doc = ReportDoc.create_root(name="report.md", content="text", attachments=(att,), reason="test input")
        sha = compute_document_sha256(doc)
        safe_name = _safe_filename("report.md", sha)
        await store.save(doc, RunScope("run1"))

        class_name = ReportDoc.__name__
        att_dir = tmp_path / class_name / f"{safe_name}.att"
        assert att_dir.is_dir()
        assert (att_dir / "img.png").exists()


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_binary_content_round_trip(self, store: LocalDocumentStore):
        """Binary content (non-UTF-8) should survive save/load."""
        binary = bytes(range(256))  # All byte values 0-255
        doc = ReportDoc(name="binary.bin", content=binary)
        await store.save(doc, RunScope("run1"))
        loaded = await store.load(RunScope("run1"), [ReportDoc])
        assert len(loaded) == 1
        assert loaded[0].content == binary

    @pytest.mark.asyncio
    async def test_empty_content_round_trip(self, store: LocalDocumentStore):
        doc = ReportDoc(name="empty.txt", content=b"")
        await store.save(doc, RunScope("run1"))
        loaded = await store.load(RunScope("run1"), [ReportDoc])
        assert len(loaded) == 1
        assert loaded[0].content == b""

    @pytest.mark.asyncio
    async def test_triggered_by_empty_tuple_round_trip(self, store: LocalDocumentStore):
        """Origins=() should survive round-trip as (), not become None."""
        doc = ReportDoc.create_root(name="a.txt", content="test", reason="test input")
        assert doc.triggered_by == ()
        await store.save(doc, RunScope("run1"))
        loaded = await store.load(RunScope("run1"), [ReportDoc])
        assert loaded[0].triggered_by == ()


class TestCollisionSafeFilenames:
    @pytest.mark.asyncio
    async def test_loaded_name_is_original(self, store: LocalDocumentStore):
        """Loaded document.name must be the original, not the suffixed filesystem name."""
        doc = _make(ReportDoc, "report.md", "content")
        await store.save(doc, RunScope("run1"))
        loaded = await store.load(RunScope("run1"), [ReportDoc])
        assert loaded[0].name == "report.md"

    @pytest.mark.asyncio
    async def test_sha256_preserved_after_round_trip(self, store: LocalDocumentStore):
        """Document SHA256 must be identical after save/load (name round-trips correctly)."""
        doc = _make(ReportDoc, "report.md", "content")
        original_sha = compute_document_sha256(doc)
        await store.save(doc, RunScope("run1"))
        loaded = await store.load(RunScope("run1"), [ReportDoc])
        assert compute_document_sha256(loaded[0]) == original_sha

    @pytest.mark.asyncio
    async def test_no_extension(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "README", "content")
        await store.save(doc, RunScope("run1"))
        loaded = await store.load(RunScope("run1"), [ReportDoc])
        assert loaded[0].name == "README"

    @pytest.mark.asyncio
    async def test_multiple_dots_in_name(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "archive.tar.gz", "content")
        await store.save(doc, RunScope("run1"))
        loaded = await store.load(RunScope("run1"), [ReportDoc])
        assert loaded[0].name == "archive.tar.gz"

    @pytest.mark.asyncio
    async def test_dotfile_name(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, ".gitignore", "content")
        await store.save(doc, RunScope("run1"))
        loaded = await store.load(RunScope("run1"), [ReportDoc])
        assert loaded[0].name == ".gitignore"

    def test_path_traversal_rejected_at_document_level(self):
        """Document validation rejects path traversal before the store."""
        with pytest.raises(Exception, match="path traversal"):
            ReportDoc(name="../../../etc/passwd", content=b"evil")

    @pytest.mark.asyncio
    async def test_backward_compat_load_without_name_in_meta(self, store: LocalDocumentStore, tmp_path: Path):
        """Old meta.json files without 'name' field should fall back to filesystem name."""
        doc_dir = tmp_path / ReportDoc.__name__
        doc_dir.mkdir(parents=True)
        (doc_dir / "old_doc.md").write_bytes(b"old content")
        meta = {
            "document_sha256": "X" * 52,
            "content_sha256": "Y" * 52,
            "class_name": "ReportDoc",
            "description": None,
            "derived_from": [],
            "triggered_by": [],
            "mime_type": "text/markdown",
            "attachments": [],
        }
        (doc_dir / "old_doc.md.meta.json").write_text(json.dumps(meta))
        loaded = await store.load(RunScope("run1"), [ReportDoc])
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
        with pytest.raises(AttributeError):
            create_document_store("not_settings")  # type: ignore[arg-type]


class TestLoadBySha256s:
    @pytest.mark.asyncio
    async def test_returns_correct_doc(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "report.md", "content")
        await store.save(doc, RunScope("run1"))
        result = await store.load_by_sha256s([doc.sha256], ReportDoc, RunScope("run1"))
        assert doc.sha256 in result
        assert result[doc.sha256].sha256 == doc.sha256
        assert result[doc.sha256].name == "report.md"
        assert isinstance(result[doc.sha256], ReportDoc)

    @pytest.mark.asyncio
    async def test_cache_hit_path(self, store: LocalDocumentStore):
        """Second load_by_sha256s call uses the meta path cache."""
        doc = _make(ReportDoc, "report.md", "content")
        await store.save(doc, RunScope("run1"))
        result1 = await store.load_by_sha256s([doc.sha256], ReportDoc, RunScope("run1"))
        result2 = await store.load_by_sha256s([doc.sha256], ReportDoc, RunScope("run1"))
        assert doc.sha256 in result1
        assert doc.sha256 in result2
        assert result1[doc.sha256].sha256 == result2[doc.sha256].sha256

    @pytest.mark.asyncio
    async def test_cache_miss_scans_type_dir(self, store: LocalDocumentStore):
        """Store can find documents by scanning type directory."""
        doc = _make(ReportDoc, "report.md", "content")
        await store.save(doc, RunScope("run1"))
        result = await store.load_by_sha256s([doc.sha256], ReportDoc, RunScope("run1"))
        assert doc.sha256 in result
        assert result[doc.sha256].sha256 == doc.sha256

    @pytest.mark.asyncio
    async def test_returns_empty_for_unknown_sha(self, store: LocalDocumentStore):
        assert await store.load_by_sha256s([DocumentSha256("NONEXISTENT" * 4 + "AAAA")], ReportDoc, RunScope("run1")) == {}

    @pytest.mark.asyncio
    async def test_class_name_not_enforced(self, store: LocalDocumentStore):
        """document_type is a construction hint, not a filter — doc is found regardless of stored type."""
        doc = _make(ReportDoc, "report.md", "content")
        await store.save(doc, RunScope("run1"))
        result = await store.load_by_sha256s([doc.sha256], DataDoc, RunScope("run1"))
        assert doc.sha256 in result

    @pytest.mark.asyncio
    async def test_with_attachments(self, store: LocalDocumentStore):
        att = Attachment(name="screenshot.png", content=b"\x89PNG" + b"\x00" * 50, description="A screenshot")
        doc = ReportDoc.create_root(name="report.md", content="# Report", attachments=(att,), reason="test input")
        await store.save(doc, RunScope("run1"))
        result = await store.load_by_sha256s([doc.sha256], ReportDoc, RunScope("run1"))
        assert doc.sha256 in result
        loaded = result[doc.sha256]
        assert len(loaded.attachments) == 1
        assert loaded.attachments[0].name == "screenshot.png"
        assert loaded.attachments[0].content == att.content

    @pytest.mark.asyncio
    async def test_scope_ignored_for_path(self, store: LocalDocumentStore):
        """run_scope is accepted but ignored — doc saved with one scope is found with another."""
        doc = _make(ReportDoc, "report.md", "content")
        await store.save(doc, RunScope("run1"))
        result = await store.load_by_sha256s([doc.sha256], ReportDoc, RunScope("run2"))
        assert doc.sha256 in result

    @pytest.mark.asyncio
    async def test_cross_scope_lookup_without_run_scope(self, store: LocalDocumentStore):
        """When run_scope=None, searches across all scope directories."""
        doc = _make(ReportDoc, "report.md", "cross scope content")
        await store.save(doc, RunScope("run1"))
        result = await store.load_by_sha256s([doc.sha256], ReportDoc)
        assert doc.sha256 in result
        assert result[doc.sha256].sha256 == doc.sha256

    @pytest.mark.asyncio
    async def test_cross_scope_class_name_not_enforced(self, store: LocalDocumentStore):
        """Cross-scope: document_type is a construction hint, not a filter."""
        doc = _make(ReportDoc, "report.md", "content")
        await store.save(doc, RunScope("run1"))
        result = await store.load_by_sha256s([doc.sha256], DataDoc)
        assert doc.sha256 in result

    @pytest.mark.asyncio
    async def test_multiple_sha256s(self, store: LocalDocumentStore):
        doc1 = _make(ReportDoc, "a.md", "content1")
        doc2 = _make(ReportDoc, "b.md", "content2")
        await store.save(doc1, RunScope("run1"))
        await store.save(doc2, RunScope("run1"))
        result = await store.load_by_sha256s([doc1.sha256, doc2.sha256], ReportDoc, RunScope("run1"))
        assert len(result) == 2
        assert doc1.sha256 in result
        assert doc2.sha256 in result

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self, store: LocalDocumentStore):
        assert await store.load_by_sha256s([], ReportDoc) == {}

    @pytest.mark.asyncio
    async def test_partial_match(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "a.md", "content")
        await store.save(doc, RunScope("run1"))
        result = await store.load_by_sha256s([doc.sha256, DocumentSha256("NONEXISTENT" * 4 + "AAAA")], ReportDoc, RunScope("run1"))
        assert len(result) == 1
        assert doc.sha256 in result


class TestLoadScopeMetadata:
    @pytest.mark.asyncio
    async def test_returns_all_docs_metadata(self, store: LocalDocumentStore):
        doc1 = _make(ReportDoc, "a.md", "aaa")
        doc2 = _make(DataDoc, "b.json", '{"key": "val"}')
        await store.save(doc1, RunScope("run1"))
        await store.save(doc2, RunScope("run1"))
        metadata = await store.load_scope_metadata(RunScope("run1"))
        assert len(metadata) == 2
        shas = {m.sha256 for m in metadata}
        assert doc1.sha256 in shas
        assert doc2.sha256 in shas

    @pytest.mark.asyncio
    async def test_returns_empty_for_nonexistent_scope(self, store: LocalDocumentStore):
        assert await store.load_scope_metadata(RunScope("nonexistent")) == []

    @pytest.mark.asyncio
    async def test_returns_document_node_instances(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "report.md", "content", description="desc")
        await store.save(doc, RunScope("run1"))
        metadata = await store.load_scope_metadata(RunScope("run1"))
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
        await store.save(doc, RunScope("run1"))
        await store.update_summary(doc.sha256, "test summary")
        metadata = await store.load_scope_metadata(RunScope("run1"))
        assert len(metadata) == 1
        assert metadata[0].summary == "test summary"

    @pytest.mark.asyncio
    async def test_metadata_has_derived_from_and_triggered_by(self, store: LocalDocumentStore):
        origin_doc = _make(ReportDoc, "origin.txt", "origin")
        doc = ReportDoc.create(
            name="child.txt",
            content="child",
            derived_from=("https://example.com",),
            triggered_by=(origin_doc.sha256,),
        )
        await store.save(doc, RunScope("run1"))
        metadata = await store.load_scope_metadata(RunScope("run1"))
        # Filter for the child doc
        child_nodes = [m for m in metadata if m.sha256 == doc.sha256]
        assert len(child_nodes) == 1
        assert "https://example.com" in child_nodes[0].derived_from
        assert origin_doc.sha256 in child_nodes[0].triggered_by


class TestLoadNodesBySha256s:
    @pytest.mark.asyncio
    async def test_returns_found_nodes(self, store: LocalDocumentStore):
        doc1 = _make(ReportDoc, "a.md", "aaa")
        doc2 = _make(DataDoc, "b.json", '{"key": "val"}')
        await store.save(doc1, RunScope("run1"))
        await store.save(doc2, RunScope("run1"))
        result = await store.load_nodes_by_sha256s([doc1.sha256, doc2.sha256])
        assert len(result) == 2
        assert result[doc1.sha256].name == "a.md"
        assert result[doc2.sha256].name == "b.json"

    @pytest.mark.asyncio
    async def test_missing_sha256s_omitted(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "a.md", "content")
        await store.save(doc, RunScope("run1"))
        result = await store.load_nodes_by_sha256s([doc.sha256, DocumentSha256("NONEXISTENT" * 4 + "AAAA")])
        assert len(result) == 1
        assert doc.sha256 in result

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self, store: LocalDocumentStore):
        assert await store.load_nodes_by_sha256s([]) == {}

    @pytest.mark.asyncio
    async def test_cross_scope_lookup(self, store: LocalDocumentStore):
        doc1 = _make(ReportDoc, "a.md", "scope1 content")
        doc2 = _make(DataDoc, "b.json", '{"scope": 2}')
        await store.save(doc1, RunScope("scope_a"))
        await store.save(doc2, RunScope("scope_b"))
        result = await store.load_nodes_by_sha256s([doc1.sha256, doc2.sha256])
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_returns_document_node_instances_with_metadata(self, store: LocalDocumentStore):
        origin = _make(ReportDoc, "origin.md", "origin")
        doc = ReportDoc.create(
            name="child.md",
            content="child",
            description="A child doc",
            derived_from=("https://example.com",),
            triggered_by=(origin.sha256,),
        )
        await store.save(doc, RunScope("run1"))
        result = await store.load_nodes_by_sha256s([doc.sha256])
        node = result[doc.sha256]
        assert isinstance(node, DocumentNode)
        assert node.class_name == "ReportDoc"
        assert node.description == "A child doc"
        assert "https://example.com" in node.derived_from
        assert origin.sha256 in node.triggered_by

    @pytest.mark.asyncio
    async def test_includes_summaries(self, store: LocalDocumentStore):
        doc = _make(ReportDoc, "a.md", "content")
        await store.save(doc, RunScope("run1"))
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
            derived_from=("https://bitcoin.org",),
            triggered_by=(task.sha256,),
        )
        source_b = DataDoc.create(
            name="src_b.json",
            content='{"url": "https://wiki.org"}',
            derived_from=("https://wiki.org",),
            triggered_by=(task.sha256,),
        )
        report = ReportDoc.create(
            name="report.md",
            content="# Bitcoin Report",
            derived_from=(source_a.sha256, source_b.sha256),
            triggered_by=(task.sha256,),
        )
        scope = RunScope("research/btc/run1")
        for doc in [task, source_a, source_b, report]:
            await store.save(doc, scope)

        # Walk provenance from report without knowing scope
        graph = await walk_provenance(report.sha256, store.load_nodes_by_sha256s)
        assert len(graph) == 4
        assert all(sha in graph for sha in [report.sha256, source_a.sha256, source_b.sha256, task.sha256])

        # Extract external URLs
        urls = {src for node in graph.values() for src in node.derived_from if "://" in src}
        assert "https://bitcoin.org" in urls
        assert "https://wiki.org" in urls

    @pytest.mark.asyncio
    async def test_cross_scope_multiple_scopes(self, store: LocalDocumentStore):
        """Documents in different scopes are found by cross-scope lookup."""
        doc1 = _make(ReportDoc, "doc1.md", "scope1 content")
        doc2 = _make(DataDoc, "doc2.json", '{"scope": 2}')
        await store.save(doc1, RunScope("scope_a"))
        await store.save(doc2, RunScope("scope_b"))

        # Find both without scope
        assert doc1.sha256 in await store.load_by_sha256s([doc1.sha256], ReportDoc)
        assert doc2.sha256 in await store.load_by_sha256s([doc2.sha256], DataDoc)

        # class_name not enforced — wrong type still finds the doc
        assert doc1.sha256 in await store.load_by_sha256s([doc1.sha256], DataDoc)


class TestLocalStoreSummary:
    @pytest.mark.asyncio
    async def test_summary_update_writes_meta(self, store: LocalDocumentStore):
        """update_summary writes to the meta.json file."""
        import json

        doc = _make(ReportDoc, "shared.md", "shared content")
        await store.save(doc, RunScope("scope_a"))
        await store.update_summary(doc.sha256, "global summary")

        meta_files = list(store.base_path.rglob("*.meta.json"))
        assert len(meta_files) == 1
        meta = json.loads(meta_files[0].read_text())
        assert meta["summary"] == "global summary"

    @pytest.mark.asyncio
    async def test_summary_visible_from_load_summaries(self, store: LocalDocumentStore):
        """load_summaries returns summary for saved doc."""
        doc = _make(ReportDoc, "shared.md", "shared content")
        await store.save(doc, RunScope("scope_a"))
        await store.update_summary(doc.sha256, "test summary")
        summaries = await store.load_summaries([doc.sha256])
        assert summaries[doc.sha256] == "test summary"


class TestFlowCompletion:
    @pytest.mark.asyncio
    async def test_save_and_get_round_trip(self, store: LocalDocumentStore):
        await store.save_flow_completion(RunScope("proj/run1"), "flow_a", ("sha1", "sha2"), ("sha3",))
        result = await store.get_flow_completion(RunScope("proj/run1"), "flow_a")
        assert result is not None
        assert result.flow_name == "flow_a"
        assert result.input_sha256s == ("sha1", "sha2")
        assert result.output_sha256s == ("sha3",)

    @pytest.mark.asyncio
    async def test_nonexistent_returns_none(self, store: LocalDocumentStore):
        result = await store.get_flow_completion(RunScope("proj/run1"), "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_overwrite_on_rerun(self, store: LocalDocumentStore):
        scope = RunScope("proj/run1")
        await store.save_flow_completion(scope, "flow_a", ("sha1",), ("sha2",))
        await store.save_flow_completion(scope, "flow_a", ("sha1",), ("sha2", "sha3"))
        result = await store.get_flow_completion(scope, "flow_a")
        assert result is not None
        assert result.output_sha256s == ("sha2", "sha3")

    @pytest.mark.asyncio
    async def test_file_layout(self, store: LocalDocumentStore):
        """Completion records are stored in .flow_completions/{scope}/ subdirectory."""
        await store.save_flow_completion(RunScope("proj/run1"), "my_flow", (), ())
        expected = store.base_path / ".flow_completions" / "proj__run1" / "my_flow.json"
        assert expected.exists()
        data = json.loads(expected.read_text())
        assert data["flow_name"] == "my_flow"

    @pytest.mark.asyncio
    async def test_max_age_filters_expired(self, store: LocalDocumentStore):
        """Expired completion records are filtered out by max_age."""
        scope = RunScope("proj/run1")
        await store.save_flow_completion(scope, "flow_a", (), ())
        # Manually backdate the stored_at in the file
        path = store.base_path / ".flow_completions" / "proj__run1" / "flow_a.json"
        data = json.loads(path.read_text())
        old_time = datetime(2020, 1, 1, tzinfo=UTC)
        data["stored_at"] = old_time.isoformat()
        path.write_text(json.dumps(data))
        # Should be expired with a short max_age
        result = await store.get_flow_completion(scope, "flow_a", max_age=timedelta(hours=1))
        assert result is None
        # Should still be found without max_age
        result = await store.get_flow_completion(scope, "flow_a")
        assert result is not None

    @pytest.mark.asyncio
    async def test_corrupt_json_returns_none(self, store: LocalDocumentStore):
        """Corrupt JSON file returns None instead of crashing."""
        scope = RunScope("proj/run1")
        await store.save_flow_completion(scope, "flow_a", (), ())
        path = store.base_path / ".flow_completions" / "proj__run1" / "flow_a.json"
        path.write_text("{invalid json")
        result = await store.get_flow_completion(scope, "flow_a")
        assert result is None

    @pytest.mark.asyncio
    async def test_different_run_scopes_are_isolated(self, store: LocalDocumentStore):
        """Flow completions under different run_scopes must not interfere.

        ClickHouse and MemoryDocumentStore both scope by (run_scope, flow_name).
        LocalDocumentStore must do the same — a completion saved for scope A
        must not be returned when querying scope B.
        """
        scope_a = RunScope("project_a/run1")
        scope_b = RunScope("project_b/run1")

        await store.save_flow_completion(scope_a, "shared_flow", ("inp_a",), ("out_a",))

        # Querying scope_a should find it
        result_a = await store.get_flow_completion(scope_a, "shared_flow")
        assert result_a is not None
        assert result_a.output_sha256s == ("out_a",)

        # Querying scope_b must NOT find scope_a's completion
        result_b = await store.get_flow_completion(scope_b, "shared_flow")
        assert result_b is None

    @pytest.mark.asyncio
    async def test_save_preserves_both_scopes(self, store: LocalDocumentStore):
        """Saving the same flow_name under two run_scopes must keep both records.

        If scope is ignored, the second save overwrites the first and the first
        scope's data is lost.
        """
        scope_a = RunScope("project_a/run1")
        scope_b = RunScope("project_b/run1")

        await store.save_flow_completion(scope_a, "flow_x", ("inp_a",), ("out_a",))
        await store.save_flow_completion(scope_b, "flow_x", ("inp_b",), ("out_b",))

        result_a = await store.get_flow_completion(scope_a, "flow_x")
        result_b = await store.get_flow_completion(scope_b, "flow_x")

        assert result_a is not None, "scope_a completion was lost after scope_b save"
        assert result_b is not None, "scope_b completion was not saved"
        assert result_a.input_sha256s == ("inp_a",)
        assert result_b.input_sha256s == ("inp_b",)

    @pytest.mark.asyncio
    async def test_cross_scope_does_not_skip_flow(self, store: LocalDocumentStore):
        """Simulate the resume-logic scenario: a flow completed in scope A must
        not cause scope B to skip it.

        This is the user-facing manifestation of the bug — resume logic calls
        get_flow_completion(scope_b, flow_name) and if it returns a result from
        scope_a, the flow is incorrectly skipped for scope_b's inputs.
        """
        scope_old = RunScope("project/old_run")
        scope_new = RunScope("project/new_run")

        # Old run completed flow_a with old inputs/outputs
        await store.save_flow_completion(scope_old, "flow_a", ("old_in",), ("old_out",))

        # New run queries flow_a — must NOT find old run's completion
        result = await store.get_flow_completion(scope_new, "flow_a")
        assert result is None, "get_flow_completion returned a completion from a different run_scope — resume logic would incorrectly skip this flow"


class TestNonAtomicWriteRace:
    """Non-atomic Path.write_text() creates a window where concurrent readers
    see truncated (empty) files. This is the root cause of the
    'Expecting value: line 1 column 1 (char 0)' warnings in production.
    """

    @pytest.mark.asyncio
    async def test_concurrent_read_during_write_sees_valid_json(self, store: LocalDocumentStore):
        """A reader must never observe empty or truncated JSON from a meta file.

        Reproduces the race: one thread writes a .meta.json file repeatedly
        while another reads it. With non-atomic write_text(), the reader sees
        an empty file during the truncation window.
        """
        scope = RunScope("proj/race_test")
        doc = _make(ReportDoc, "race.txt", content="x" * 500)
        await store.save(doc, scope)

        # Find the meta file — local store saves under {base_path}/{ClassName}/
        meta_dir = store.base_path / "ReportDoc"
        meta_paths = list(meta_dir.glob("*.meta.json"))
        assert len(meta_paths) == 1
        meta_path = meta_paths[0]

        corrupt_reads: list[str] = []
        iterations = 500
        barrier = threading.Barrier(2)

        def writer():
            barrier.wait()
            valid_json = json.dumps({"document_sha256": doc.sha256, "test": True}, indent=2)
            for _ in range(iterations):
                _atomic_write_text(meta_path, valid_json)

        def reader():
            barrier.wait()
            for _ in range(iterations):
                try:
                    text = meta_path.read_text(encoding="utf-8")
                    if not text:
                        corrupt_reads.append("empty file")
                    else:
                        json.loads(text)
                except json.JSONDecodeError as e:
                    corrupt_reads.append(str(e))
                except OSError:
                    pass  # file may not exist momentarily

        t_write = threading.Thread(target=writer)
        t_read = threading.Thread(target=reader)
        t_write.start()
        t_read.start()
        t_write.join()
        t_read.join()

        assert not corrupt_reads, (
            f"Reader observed {len(corrupt_reads)} corrupt reads during concurrent writes. "
            f"First: {corrupt_reads[0]}. "
            f"Writes must be atomic (tempfile + os.replace) to prevent truncation window."
        )
