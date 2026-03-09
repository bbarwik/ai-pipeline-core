"""Document reference resolution from DatabaseReader backends."""

from ai_pipeline_core.database._protocol import DatabaseReader
from ai_pipeline_core.database._types import BlobRecord, DocumentRecord
from ai_pipeline_core.documents import Attachment
from ai_pipeline_core.documents._context import DocumentSha256, _suppress_document_registration
from ai_pipeline_core.documents.document import Document, _class_name_registry
from ai_pipeline_core.logging import get_pipeline_logger

from .types import DocumentRef

__all__ = ["resolve_document_ref"]

logger = get_pipeline_logger(__name__)


def _find_document_class(class_name: str) -> type[Document]:
    """Find a Document subclass by name, checking the registry first."""
    if class_name in _class_name_registry:
        return _class_name_registry[class_name]

    queue: list[type[Document]] = list(Document.__subclasses__())
    while queue:
        cls = queue.pop()
        if cls.__name__ == class_name:
            return cls
        queue.extend(cls.__subclasses__())

    raise ValueError(f"No Document subclass found for class_name '{class_name}'. Import the module that defines this Document subclass before replaying.")


def _build_attachments_from_record(
    record: DocumentRecord,
    attachment_blobs: dict[str, BlobRecord],
) -> tuple[Attachment, ...] | None:
    """Reconstruct attachments from DocumentRecord metadata and fetched blobs."""
    attachments: list[Attachment] = []
    for index, attachment_name in enumerate(record.attachment_names):
        blob_sha = record.attachment_sha256s[index]
        blob = attachment_blobs.get(blob_sha)
        if blob is None:
            raise FileNotFoundError(
                f"Attachment blob {blob_sha[:12]}... is missing for document {record.document_sha256[:12]}... "
                f"attachment '{attachment_name}'. Replay requires every attachment blob referenced by the stored document metadata."
            )
        description = record.attachment_descriptions[index] if index < len(record.attachment_descriptions) else ""
        attachments.append(
            Attachment(
                name=attachment_name,
                content=blob.content,
                description=description or None,
            )
        )
    return tuple(attachments) if attachments else None


async def resolve_document_ref(ref: DocumentRef, database: DatabaseReader) -> Document:
    """Resolve a replay document reference from the database."""
    doc_record = await database.get_document(ref.doc_ref)
    if doc_record is None:
        raise FileNotFoundError(
            f"Document {ref.doc_ref[:12]}... not found in database. Replay payload requested class_name={ref.class_name!r}, name={ref.name!r}."
        )

    main_blob = await database.get_blob(doc_record.content_sha256)
    if main_blob is None:
        raise FileNotFoundError(f"Blob {doc_record.content_sha256[:12]}... missing for document {doc_record.document_sha256[:12]}...")

    attachment_blobs = await database.get_blobs_batch(list(doc_record.attachment_sha256s))
    attachments = _build_attachments_from_record(doc_record, attachment_blobs)
    doc_cls = _find_document_class(doc_record.document_type)

    with _suppress_document_registration():
        return doc_cls(
            name=doc_record.name,
            content=main_blob.content,
            description=doc_record.description or None,
            derived_from=doc_record.derived_from,
            triggered_by=tuple(DocumentSha256(sha) for sha in doc_record.triggered_by),
            attachments=attachments,
        )
