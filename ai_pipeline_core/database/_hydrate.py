"""Shared document hydration helpers for stored document records."""

from ai_pipeline_core.database._types import HydratedDocument
from ai_pipeline_core.documents._context import DocumentSha256
from ai_pipeline_core.documents.attachment import Attachment
from ai_pipeline_core.documents.document import Document

__all__ = [
    "hydrate_document",
]


def hydrate_document(document_cls: type[Document], hydrated: HydratedDocument) -> Document:
    """Build a concrete Document instance from hydrated metadata and blobs."""
    return document_cls(
        name=hydrated.record.name,
        content=hydrated.content,
        description=hydrated.record.description or None,
        summary=hydrated.record.summary,
        derived_from=hydrated.record.derived_from,
        triggered_by=tuple(DocumentSha256(sha) for sha in hydrated.record.triggered_by),
        attachments=_hydrate_attachments(hydrated),
    )


def _hydrate_attachments(hydrated: HydratedDocument) -> tuple[Attachment, ...] | None:
    built_attachments: list[Attachment] = []
    for name, description, content_sha256 in zip(
        hydrated.record.attachment_names,
        hydrated.record.attachment_descriptions,
        hydrated.record.attachment_content_sha256s,
        strict=True,
    ):
        content = hydrated.attachment_contents.get(content_sha256)
        if content is None:
            raise ValueError(
                f"Document {hydrated.record.document_sha256[:12]}... is missing attachment blob {content_sha256[:12]}... for {name!r}. "
                "Persist every attachment blob before hydrating this stored document."
            )
        built_attachments.append(
            Attachment(
                name=name,
                content=content,
                description=description or None,
            )
        )
    return tuple(built_attachments) if built_attachments else None
