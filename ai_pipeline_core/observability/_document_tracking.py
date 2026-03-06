"""Document tracking helpers for pipeline instrumentation.

Sets OTel span attributes for document lineage (SHA256 arrays).
Document lineage is persisted via input_doc_sha256s/output_doc_sha256s
columns on pipeline_spans.

All functions are no-ops when no spans are active.
"""

from collections.abc import Sequence
from typing import Any, cast

from opentelemetry import trace as otel_trace
from pydantic import BaseModel

from ai_pipeline_core.documents import Document, DocumentSha256
from ai_pipeline_core.llm.conversation import Conversation
from ai_pipeline_core.observability._span_data import ATTR_INPUT_DOC_SHA256S, ATTR_OUTPUT_DOC_SHA256S


def _collect_sha256s(obj: object, sha256s: list[DocumentSha256]) -> None:
    """Collect document SHA256s from single Documents, lists/tuples, or nested containers."""
    if isinstance(obj, Document):
        sha256s.append(obj.sha256)
    elif isinstance(obj, Conversation):
        for document in obj.context:
            _collect_sha256s(document, sha256s)
        for message in obj.messages:
            _collect_sha256s(message, sha256s)
    elif isinstance(obj, BaseModel):
        for field_name in type(obj).model_fields:
            _collect_sha256s(getattr(obj, field_name), sha256s)
    elif isinstance(obj, dict):
        for value in cast(dict[str, Any], obj).values():
            _collect_sha256s(value, sha256s)
    elif isinstance(obj, (list, tuple)) and obj:
        if all(isinstance(x, Document) for x in cast(list[object], obj)):
            sha256s.extend(doc.sha256 for doc in cast(list[Document], obj))
        else:
            for item in cast(list[object], obj):
                _collect_sha256s(item, sha256s)


def _track_io(inputs: Sequence[object], output: object) -> None:
    """Set OTel span attributes for input/output document SHA256 arrays."""
    span = otel_trace.get_current_span()
    ctx = span.get_span_context()
    if not ctx or not ctx.span_id:
        return

    input_sha256s: list[DocumentSha256] = []
    output_sha256s: list[DocumentSha256] = []

    for arg in inputs:
        _collect_sha256s(arg, input_sha256s)
    _collect_sha256s(output, output_sha256s)

    if input_sha256s:
        span.set_attribute(ATTR_INPUT_DOC_SHA256S, input_sha256s)
    if output_sha256s:
        span.set_attribute(ATTR_OUTPUT_DOC_SHA256S, output_sha256s)


def track_task_io(args: tuple[object, ...], kwargs: dict[str, object], result: object) -> None:
    """Track input/output documents for a pipeline task."""
    _track_io(list(args) + list(kwargs.values()), result)


def track_flow_io(input_documents: list[Document], output_documents: list[Document]) -> None:
    """Track input/output documents for a pipeline flow."""
    _track_io(input_documents, output_documents)
