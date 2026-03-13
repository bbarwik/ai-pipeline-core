"""ClickHouse-specific row conversion helpers.

Shared helpers (SPAN_COLUMNS, LOG_COLUMNS, row_to_span, row_to_log, decode_text, etc.)
live in database/_serialization.py. This module handles document/blob rows and
INSERT row builders that are only needed by the ClickHouse backend.
"""

from typing import Any

from ai_pipeline_core.database._serialization import (
    decode_text,
    string_tuple,
    to_datetime,
)
from ai_pipeline_core.database._types import BlobRecord, DocumentRecord, SpanRecord
from ai_pipeline_core.logger._types import LogRecord

__all__ = [
    "BLOB_COLUMNS",
    "DOCUMENT_COLUMNS",
    "blob_to_row",
    "document_to_row",
    "log_to_row",
    "row_to_blob",
    "row_to_document",
    "span_to_row",
]

DOCUMENT_COLUMNS = (
    "document_sha256",
    "content_sha256",
    "document_type",
    "name",
    "description",
    "mime_type",
    "size_bytes",
    "summary",
    "derived_from",
    "triggered_by",
    "attachments.name",
    "attachments.description",
    "attachments.content_sha256",
    "attachments.mime_type",
    "attachments.size_bytes",
    "created_at",
)

BLOB_COLUMNS = (
    "content_sha256",
    "content",
    "created_at",
)


def span_to_row(span: SpanRecord) -> list[Any]:
    return [
        span.span_id,
        span.parent_span_id,
        span.deployment_id,
        span.root_deployment_id,
        span.run_id,
        span.deployment_name,
        span.kind,
        span.name,
        span.description,
        span.status,
        span.sequence_no,
        span.started_at,
        span.ended_at,
        span.version,
        span.cache_key,
        span.previous_conversation_id,
        span.cost_usd,
        span.error_type,
        span.error_message,
        list(span.input_document_shas),
        list(span.output_document_shas),
        span.target,
        span.receiver_json,
        span.input_json,
        span.output_json,
        span.error_json,
        span.meta_json,
        span.metrics_json,
        list(span.input_blob_shas),
        list(span.output_blob_shas),
    ]


def document_to_row(record: DocumentRecord) -> list[Any]:
    return [
        record.document_sha256,
        record.content_sha256,
        record.document_type,
        record.name,
        record.description,
        record.mime_type,
        record.size_bytes,
        record.summary,
        list(record.derived_from),
        list(record.triggered_by),
        list(record.attachment_names),
        list(record.attachment_descriptions),
        list(record.attachment_content_sha256s),
        list(record.attachment_mime_types),
        list(record.attachment_size_bytes),
        record.created_at,
    ]


def row_to_document(row: tuple[Any, ...]) -> DocumentRecord:
    fields: dict[str, Any] = dict(zip(DOCUMENT_COLUMNS, row, strict=True))
    fields["document_sha256"] = decode_text(fields["document_sha256"])
    fields["content_sha256"] = decode_text(fields["content_sha256"])
    fields["document_type"] = decode_text(fields["document_type"])
    fields["name"] = decode_text(fields["name"])
    fields["description"] = decode_text(fields["description"])
    fields["mime_type"] = decode_text(fields["mime_type"])
    fields["size_bytes"] = int(fields["size_bytes"])
    fields["summary"] = decode_text(fields["summary"])
    fields["derived_from"] = string_tuple(fields["derived_from"])
    fields["triggered_by"] = string_tuple(fields["triggered_by"])
    fields["attachment_names"] = string_tuple(fields["attachments.name"])
    fields["attachment_descriptions"] = string_tuple(fields["attachments.description"])
    fields["attachment_content_sha256s"] = string_tuple(fields["attachments.content_sha256"])
    fields["attachment_mime_types"] = string_tuple(fields["attachments.mime_type"])
    raw_attachment_sizes = fields["attachments.size_bytes"]
    if isinstance(raw_attachment_sizes, (list, tuple)):
        fields["attachment_size_bytes"] = tuple(int(value) for value in raw_attachment_sizes)
    elif raw_attachment_sizes is None:
        fields["attachment_size_bytes"] = ()
    else:
        fields["attachment_size_bytes"] = (int(raw_attachment_sizes),)
    del fields["attachments.name"]
    del fields["attachments.description"]
    del fields["attachments.content_sha256"]
    del fields["attachments.mime_type"]
    del fields["attachments.size_bytes"]
    fields["created_at"] = to_datetime(fields["created_at"])
    return DocumentRecord(**fields)


def blob_to_row(blob: BlobRecord) -> list[Any]:
    return [blob.content_sha256, blob.content, blob.created_at]


def row_to_blob(row: tuple[Any, ...]) -> BlobRecord:
    fields = dict(zip(BLOB_COLUMNS, row, strict=True))
    content = fields["content"]
    if isinstance(content, bytes):
        fields["content"] = content
    elif isinstance(content, str):
        fields["content"] = content.encode("utf-8")
    else:
        fields["content"] = bytes(content)
    fields["content_sha256"] = decode_text(fields["content_sha256"])
    fields["created_at"] = to_datetime(fields["created_at"])
    return BlobRecord(**fields)


def log_to_row(log: LogRecord) -> list[Any]:
    return [
        log.deployment_id,
        log.span_id,
        log.timestamp,
        log.sequence_no,
        log.level,
        log.category,
        log.event_type,
        log.logger_name,
        log.message,
        log.fields_json,
        log.exception_text,
    ]
