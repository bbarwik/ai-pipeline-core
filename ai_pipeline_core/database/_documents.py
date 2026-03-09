"""Document reconstruction from database records.

Converts DocumentRecord + BlobRecord pairs back into typed Document instances.
Used for flow resume (loading cached flow outputs) and build_result().
"""

from ai_pipeline_core.database._protocol import DatabaseReader
from ai_pipeline_core.database._types import DocumentRecord
from ai_pipeline_core.documents._context import DocumentSha256, _suppress_document_registration
from ai_pipeline_core.documents.document import Document, _class_name_registry
from ai_pipeline_core.logging import get_pipeline_logger

__all__ = [
    "load_documents_from_database",
]

logger = get_pipeline_logger(__name__)


def _find_document_class(class_name: str) -> type[Document] | None:
    """Find a Document subclass by name from the registry."""
    if class_name in _class_name_registry:
        return _class_name_registry[class_name]

    # Search all loaded Document subclasses (covers test-defined classes not in registry)
    queue: list[type[Document]] = list(Document.__subclasses__())
    while queue:
        cls = queue.pop()
        if cls.__name__ == class_name:
            return cls
        queue.extend(cls.__subclasses__())

    return None


def _reconstruct_document(
    record: DocumentRecord,
    content: bytes,
    attachment_contents: dict[str, bytes],
) -> Document | None:
    """Reconstruct a typed Document from a DocumentRecord + content bytes."""
    doc_cls = _find_document_class(record.document_type)
    if doc_cls is None:
        logger.warning(
            "Cannot reconstruct document '%s': Document subclass '%s' not found. Import the module that defines this Document subclass.",
            record.name,
            record.document_type,
        )
        return None

    # Build attachments
    from ai_pipeline_core.documents.attachment import Attachment

    attachments: list[Attachment] = []
    for i, att_name in enumerate(record.attachment_names):
        att_sha = record.attachment_sha256s[i] if i < len(record.attachment_sha256s) else ""
        att_content = attachment_contents.get(att_sha)
        if att_content is None:
            logger.warning("Attachment '%s' content not found (sha256=%s...)", att_name, att_sha[:12])
            continue
        att_desc = record.attachment_descriptions[i] if i < len(record.attachment_descriptions) else ""
        attachments.append(
            Attachment(
                name=att_name,
                content=att_content,
                description=att_desc or None,
            )
        )

    with _suppress_document_registration():
        return doc_cls(
            name=record.name,
            content=content,
            description=record.description or None,
            derived_from=record.derived_from,
            triggered_by=tuple(DocumentSha256(t) for t in record.triggered_by),
            attachments=tuple(attachments) if attachments else None,
        )


async def load_documents_from_database(
    reader: DatabaseReader,
    sha256s: set[str],
    *,
    filter_types: list[type[Document]] | None = None,
) -> list[Document]:
    """Load and reconstruct typed Document instances from the database.

    Args:
        reader: Database reader for fetching records and blobs.
        sha256s: Set of document SHA256 hashes to load.
        filter_types: If provided, only return documents matching these types.

    Returns:
        List of reconstructed Document instances.
    """
    if not sha256s:
        return []

    sha256_list = [DocumentSha256(s) for s in sha256s]
    records = await reader.get_documents_batch(sha256_list)
    if not records:
        return []

    filter_type_names: set[str] | None = None
    if filter_types is not None:
        filter_type_names = {t.__name__ for t in filter_types}

    # Collect content SHA256s we need
    content_sha256s: set[str] = set()
    att_sha256s: set[str] = set()
    filtered_records: list[DocumentRecord] = []

    for record in records.values():
        if filter_type_names is not None and record.document_type not in filter_type_names:
            continue
        filtered_records.append(record)
        content_sha256s.add(record.content_sha256)
        att_sha256s.update(record.attachment_sha256s)

    if not filtered_records:
        return []

    # Batch-fetch content blobs
    all_blob_sha256s = sorted(content_sha256s | att_sha256s)
    blobs = await reader.get_blobs_batch(all_blob_sha256s)

    # Reconstruct documents
    result: list[Document] = []
    for record in filtered_records:
        blob = blobs.get(record.content_sha256)
        if blob is None:
            logger.warning(
                "Content blob not found for document '%s' (content_sha256=%s...)",
                record.name,
                record.content_sha256[:12],
            )
            continue

        att_contents: dict[str, bytes] = {}
        for att_sha in record.attachment_sha256s:
            att_blob = blobs.get(att_sha)
            if att_blob is not None:
                att_contents[att_sha] = att_blob.content

        doc = _reconstruct_document(record, blob.content, att_contents)
        if doc is not None:
            result.append(doc)

    return result
