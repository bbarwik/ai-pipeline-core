"""Document reference resolution tests for the replay module.

Tests resolve_document_ref (loading a Document from LocalDocumentStore by SHA256)
and _infer_store_base (walking up to find .trace/ parent directory).
The replay module does not exist yet — these tests will fail with ImportError
until it is implemented.
"""

import json
from pathlib import Path

import pytest

from tests.replay.conftest import (
    ReplayAttachmentDocument,
    ReplayBinaryDocument,
    ReplayTextDocument,
    doc_ref_dict,
)

from ai_pipeline_core.document_store._local import _safe_filename
from ai_pipeline_core.documents.document import _class_name_registry
from ai_pipeline_core.replay import DocumentRef
from ai_pipeline_core.replay.types import _infer_store_base
from ai_pipeline_core.replay._resolve import resolve_document_ref


class TestResolveDocumentRef:
    @pytest.mark.asyncio
    async def test_resolve_text_document(
        self,
        populated_store: Path,
        sample_text_doc: ReplayTextDocument,
    ) -> None:
        ref = DocumentRef.model_validate(doc_ref_dict(sample_text_doc))
        resolved = resolve_document_ref(ref, populated_store)
        assert isinstance(resolved, ReplayTextDocument)
        assert resolved.content == sample_text_doc.content
        assert resolved.name == sample_text_doc.name

    @pytest.mark.asyncio
    async def test_resolve_binary_document(
        self,
        populated_store: Path,
        sample_binary_doc: ReplayBinaryDocument,
    ) -> None:
        ref = DocumentRef.model_validate(doc_ref_dict(sample_binary_doc))
        resolved = resolve_document_ref(ref, populated_store)
        assert isinstance(resolved, ReplayBinaryDocument)
        assert resolved.content == sample_binary_doc.content
        assert resolved.is_image

    @pytest.mark.asyncio
    async def test_resolve_document_with_attachments(
        self,
        populated_store: Path,
        sample_attachment_doc: ReplayAttachmentDocument,
    ) -> None:
        ref = DocumentRef.model_validate(doc_ref_dict(sample_attachment_doc))
        resolved = resolve_document_ref(ref, populated_store)
        assert isinstance(resolved, ReplayAttachmentDocument)
        assert len(resolved.attachments) == 2
        att_names = {a.name for a in resolved.attachments}
        assert att_names == {"details.txt", "preview.png"}

    def test_resolve_missing_document_raises(self, store_base: Path) -> None:
        store_base.mkdir(parents=True, exist_ok=True)
        fake_sha = "Z" * 52
        ref = DocumentRef.model_validate({
            "$doc_ref": fake_sha,
            "class_name": "ReplayTextDocument",
            "name": "ghost.txt",
        })
        with pytest.raises(FileNotFoundError):
            resolve_document_ref(ref, store_base)

    @pytest.mark.asyncio
    async def test_resolve_wrong_sha256_same_prefix_raises(
        self,
        populated_store: Path,
        sample_text_doc: ReplayTextDocument,
    ) -> None:
        """Mutate SHA256 after the 6-char prefix so the filesystem file is found
        but the full SHA256 does not match any stored document."""
        real_sha = sample_text_doc.sha256
        # Keep first 6 chars (filename prefix), flip the rest
        mutated_sha = real_sha[:6] + ("A" if real_sha[6] != "A" else "B") + real_sha[7:]
        ref = DocumentRef.model_validate({
            "$doc_ref": mutated_sha,
            "class_name": "ReplayTextDocument",
            "name": "notes.txt",
        })
        with pytest.raises(FileNotFoundError):
            resolve_document_ref(ref, populated_store)

    def test_resolve_missing_class_in_registry_raises(self, store_base: Path) -> None:
        """Manually create store files with an unknown class_name and verify
        that resolution fails because the class is not in Document._class_name_registry."""
        fake_class = "NonExistentDocumentXYZ"
        class_dir = store_base / fake_class
        class_dir.mkdir(parents=True, exist_ok=True)

        fake_sha = "ABCDEF" + "0" * 46
        safe_name = _safe_filename("fake.txt", fake_sha)
        content_path = class_dir / safe_name
        meta_path = class_dir / f"{safe_name}.meta.json"

        content_path.write_bytes(b"fake content")
        meta = {
            "name": "fake.txt",
            "document_sha256": fake_sha,
            "content_sha256": "0" * 52,
            "class_name": fake_class,
            "description": "",
            "derived_from": [],
            "triggered_by": [],
            "mime_type": "text/plain",
            "attachments": [],
        }
        meta_path.write_text(json.dumps(meta))

        ref = DocumentRef.model_validate({
            "$doc_ref": fake_sha,
            "class_name": fake_class,
            "name": "fake.txt",
        })
        with pytest.raises((KeyError, FileNotFoundError, ValueError)):
            resolve_document_ref(ref, store_base)

    @pytest.mark.asyncio
    async def test_resolve_test_document_not_in_registry(
        self,
        populated_store: Path,
        sample_text_doc: ReplayTextDocument,
    ) -> None:
        """Test Document subclasses defined in test modules are excluded from
        _class_name_registry by the _is_test_module() guard.  resolve_document_ref()
        must still resolve them via __subclasses__() fallback."""
        assert "ReplayTextDocument" not in _class_name_registry

        ref = DocumentRef.model_validate(doc_ref_dict(sample_text_doc))
        resolved = resolve_document_ref(ref, populated_store)
        assert isinstance(resolved, ReplayTextDocument)
        assert resolved.content == sample_text_doc.content


class TestInferStoreBase:
    def test_from_nested_span_dir(self, tmp_path: Path) -> None:
        """Create a deeply nested path under .trace/ and verify _infer_store_base
        finds the correct store base (parent of .trace/)."""
        # Layout: tmp_path/output/.trace/spans/task_abc/span_123/
        output_dir = tmp_path / "output"
        trace_dir = output_dir / ".trace"
        nested = trace_dir / "spans" / "task_abc" / "span_123"
        nested.mkdir(parents=True)

        replay_file = nested / "conversation.replay.yaml"
        replay_file.write_text("dummy")

        result = _infer_store_base(replay_file)
        assert result == output_dir

    def test_missing_trace_raises(self, tmp_path: Path) -> None:
        """When no .trace/ directory exists in any ancestor, raise FileNotFoundError."""
        deep_path = tmp_path / "some" / "random" / "path"
        deep_path.mkdir(parents=True)
        replay_file = deep_path / "replay.yaml"
        replay_file.write_text("dummy")

        with pytest.raises(FileNotFoundError):
            _infer_store_base(replay_file)
