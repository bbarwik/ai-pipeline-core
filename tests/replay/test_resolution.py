"""Document reference resolution tests for the replay module."""

from uuid import uuid4

import pytest

from ai_pipeline_core.database import BlobRecord, DocumentRecord, MemoryDatabase
from ai_pipeline_core.documents.document import _class_name_registry
from ai_pipeline_core.replay import DocumentRef
from ai_pipeline_core.replay._resolve import resolve_document_ref
from tests.replay.conftest import (
    ReplayAttachmentDocument,
    ReplayBinaryDocument,
    ReplayTextDocument,
    doc_ref_dict,
)


class TestResolveDocumentRef:
    @pytest.mark.asyncio
    async def test_resolve_text_document(
        self,
        memory_database: MemoryDatabase,
        stored_text_document: ReplayTextDocument,
    ) -> None:
        ref = DocumentRef.model_validate(doc_ref_dict(stored_text_document))
        resolved = await resolve_document_ref(ref, memory_database)
        assert isinstance(resolved, ReplayTextDocument)
        assert resolved.content == stored_text_document.content
        assert resolved.name == stored_text_document.name

    @pytest.mark.asyncio
    async def test_resolve_binary_document(
        self,
        memory_database: MemoryDatabase,
        stored_binary_document: ReplayBinaryDocument,
    ) -> None:
        ref = DocumentRef.model_validate(doc_ref_dict(stored_binary_document))
        resolved = await resolve_document_ref(ref, memory_database)
        assert isinstance(resolved, ReplayBinaryDocument)
        assert resolved.content == stored_binary_document.content
        assert resolved.is_image

    @pytest.mark.asyncio
    async def test_resolve_document_with_attachments(
        self,
        memory_database: MemoryDatabase,
        stored_attachment_document: ReplayAttachmentDocument,
    ) -> None:
        ref = DocumentRef.model_validate(doc_ref_dict(stored_attachment_document))
        resolved = await resolve_document_ref(ref, memory_database)
        assert isinstance(resolved, ReplayAttachmentDocument)
        assert len(resolved.attachments) == 2
        assert {attachment.name for attachment in resolved.attachments} == {"details.txt", "preview.png"}

    @pytest.mark.asyncio
    async def test_resolve_missing_document_raises(self, memory_database: MemoryDatabase) -> None:
        ref = DocumentRef.model_validate({
            "$doc_ref": "Z" * 52,
            "class_name": "ReplayTextDocument",
            "name": "ghost.txt",
        })
        with pytest.raises(FileNotFoundError, match="not found in database"):
            await resolve_document_ref(ref, memory_database)

    @pytest.mark.asyncio
    async def test_resolve_missing_blob_raises(self, memory_database: MemoryDatabase) -> None:
        record = DocumentRecord(
            document_sha256="A" * 64,
            content_sha256="B" * 64,
            deployment_id=uuid4(),
            producing_node_id=None,
            document_type="ReplayTextDocument",
            name="broken.txt",
        )
        await memory_database.save_document(record)
        ref = DocumentRef.model_validate({
            "$doc_ref": record.document_sha256,
            "class_name": "ReplayTextDocument",
            "name": record.name,
        })

        with pytest.raises(FileNotFoundError, match="Blob"):
            await resolve_document_ref(ref, memory_database)

    @pytest.mark.asyncio
    async def test_resolve_missing_attachment_blob_raises(self, memory_database: MemoryDatabase) -> None:
        record = DocumentRecord(
            document_sha256="E" * 64,
            content_sha256="F" * 64,
            deployment_id=uuid4(),
            producing_node_id=None,
            document_type="ReplayAttachmentDocument",
            name="broken-attachment.txt",
            attachment_names=("details.txt",),
            attachment_descriptions=("broken attachment",),
            attachment_sha256s=("G" * 64,),
            attachment_mime_types=("text/plain",),
            attachment_sizes=(12,),
        )
        await memory_database.save_document(record)
        await memory_database.save_blob(BlobRecord(content_sha256=record.content_sha256, content=b"main", size_bytes=4))
        ref = DocumentRef.model_validate({
            "$doc_ref": record.document_sha256,
            "class_name": record.document_type,
            "name": record.name,
        })

        with pytest.raises(FileNotFoundError, match="Attachment blob"):
            await resolve_document_ref(ref, memory_database)

    @pytest.mark.asyncio
    async def test_resolve_missing_class_in_registry_raises(self, memory_database: MemoryDatabase) -> None:
        record = DocumentRecord(
            document_sha256="C" * 64,
            content_sha256="D" * 64,
            deployment_id=uuid4(),
            producing_node_id=None,
            document_type="NonExistentDocumentXYZ",
            name="fake.txt",
        )
        await memory_database.save_document(record)
        await memory_database.save_blob(BlobRecord(content_sha256=record.content_sha256, content=b"fake content", size_bytes=12))
        ref = DocumentRef.model_validate({
            "$doc_ref": record.document_sha256,
            "class_name": record.document_type,
            "name": record.name,
        })

        with pytest.raises(ValueError, match="No Document subclass found"):
            await resolve_document_ref(ref, memory_database)

    @pytest.mark.asyncio
    async def test_resolve_test_document_not_in_registry(
        self,
        memory_database: MemoryDatabase,
        stored_text_document: ReplayTextDocument,
    ) -> None:
        assert "ReplayTextDocument" not in _class_name_registry

        ref = DocumentRef.model_validate(doc_ref_dict(stored_text_document))
        resolved = await resolve_document_ref(ref, memory_database)
        assert isinstance(resolved, ReplayTextDocument)
        assert resolved.content == stored_text_document.content
