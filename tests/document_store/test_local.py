"""Tests for LocalDocumentStore."""

import json
from pathlib import Path

import pytest

from ai_pipeline_core.document_store import DocumentStore, create_document_store, set_document_store
from ai_pipeline_core.document_store.local import LocalDocumentStore
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
    async def test_overwrite_different_content_logs_warning(self, store: LocalDocumentStore, tmp_path: Path):
        """Saving a document with same name but different content overwrites with warning."""
        doc1 = _make(ReportDoc, "report.md", "version 1")
        doc2 = _make(ReportDoc, "report.md", "version 2")
        await store.save(doc1, "run1")
        await store.save(doc2, "run1")

        loaded = await store.load("run1", [ReportDoc])
        assert len(loaded) == 1
        assert loaded[0].content == b"version 2"


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
        await store.save(doc, "run1")

        canonical = ReportDoc.canonical_name()
        assert (tmp_path / "run1" / canonical / "report.md").exists()
        assert (tmp_path / "run1" / canonical / "report.md.meta.json").exists()

    @pytest.mark.asyncio
    async def test_meta_json_content(self, store: LocalDocumentStore, tmp_path: Path):
        doc = _make(ReportDoc, "report.md", "content", description="desc")
        await store.save(doc, "run1")

        canonical = ReportDoc.canonical_name()
        meta_path = tmp_path / "run1" / canonical / "report.md.meta.json"
        meta = json.loads(meta_path.read_text())

        assert meta["class_name"] == "ReportDoc"
        assert meta["description"] == "desc"
        assert "document_sha256" in meta
        assert "content_sha256" in meta
        assert isinstance(meta["sources"], list)
        assert isinstance(meta["origins"], list)

    @pytest.mark.asyncio
    async def test_attachment_directory(self, store: LocalDocumentStore, tmp_path: Path):
        att = Attachment(name="img.png", content=b"\x89PNG" + b"\x00" * 50)
        doc = ReportDoc.create(name="report.md", content="text", attachments=(att,))
        await store.save(doc, "run1")

        canonical = ReportDoc.canonical_name()
        att_dir = tmp_path / "run1" / canonical / "report.md.att"
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


class TestFactory:
    def test_create_document_store_returns_local_when_no_clickhouse(self):
        from ai_pipeline_core.settings import Settings

        settings = Settings(clickhouse_host="")
        store = create_document_store(settings)
        assert isinstance(store, LocalDocumentStore)

    def test_create_document_store_rejects_non_settings(self):
        with pytest.raises(TypeError, match="Expected Settings"):
            create_document_store("not_settings")  # type: ignore[arg-type]
