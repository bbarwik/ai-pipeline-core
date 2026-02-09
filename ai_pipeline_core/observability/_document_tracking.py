"""Document tracking helpers for pipeline instrumentation.

Emits document lifecycle events and sets OTel span attributes for
document lineage. All functions are no-ops when tracking is not initialized.
"""

from typing import cast

from opentelemetry import trace as otel_trace

from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents._types import DocumentSha256
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._initialization import get_tracking_service
from ai_pipeline_core.observability._tracking._models import ATTR_INPUT_DOCUMENT_SHA256S, ATTR_OUTPUT_DOCUMENT_SHA256S, DocumentEventType
from ai_pipeline_core.observability._tracking._service import TrackingService

logger = get_pipeline_logger(__name__)


def get_current_span_id() -> str:
    """Return the current OTel span ID as hex, or empty string."""
    span = otel_trace.get_current_span()
    ctx = span.get_span_context()
    if ctx and ctx.span_id:
        return format(ctx.span_id, "016x")
    return ""


def _collect_and_track(
    obj: object,
    sha256s: list[DocumentSha256],
    service: TrackingService,
    span_id: str,
    event_type: DocumentEventType,
) -> None:
    """Collect document SHA256s and emit tracking events.

    Handles single Documents, homogeneous lists/tuples of Documents,
    and nested containers (e.g., tuple[list[DocA], list[DocB]]).
    """
    if isinstance(obj, Document):
        sha256s.append(obj.sha256)
        service.track_document_event(document_sha256=obj.sha256, span_id=span_id, event_type=event_type)
    elif isinstance(obj, (list, tuple)) and obj:
        if all(isinstance(x, Document) for x in obj):
            for doc in cast(list[Document], obj):
                sha256s.append(doc.sha256)
                service.track_document_event(document_sha256=doc.sha256, span_id=span_id, event_type=event_type)
        else:
            for item in obj:
                _collect_and_track(item, sha256s, service, span_id, event_type)


def track_task_io(args: tuple[object, ...], kwargs: dict[str, object], result: object) -> None:
    """Track input/output documents for a pipeline task."""
    service = get_tracking_service()
    if service is None:
        return

    span_id = get_current_span_id()
    input_sha256s: list[DocumentSha256] = []
    output_sha256s: list[DocumentSha256] = []

    for arg in (*args, *kwargs.values()):
        _collect_and_track(arg, input_sha256s, service, span_id, DocumentEventType.TASK_INPUT)

    _collect_and_track(result, output_sha256s, service, span_id, DocumentEventType.TASK_OUTPUT)

    if input_sha256s or output_sha256s:
        span = otel_trace.get_current_span()
        if input_sha256s:
            span.set_attribute(ATTR_INPUT_DOCUMENT_SHA256S, input_sha256s)
        if output_sha256s:
            span.set_attribute(ATTR_OUTPUT_DOCUMENT_SHA256S, output_sha256s)


def track_flow_io(input_documents: list[Document], output_documents: list[Document]) -> None:
    """Track input/output documents for a pipeline flow."""
    service = get_tracking_service()
    if service is None:
        return

    span_id = get_current_span_id()
    input_sha256s: list[DocumentSha256] = []
    output_sha256s: list[DocumentSha256] = []

    _collect_and_track(input_documents, input_sha256s, service, span_id, DocumentEventType.FLOW_INPUT)
    _collect_and_track(output_documents, output_sha256s, service, span_id, DocumentEventType.FLOW_OUTPUT)

    if input_sha256s or output_sha256s:
        span = otel_trace.get_current_span()
        if input_sha256s:
            span.set_attribute(ATTR_INPUT_DOCUMENT_SHA256S, input_sha256s)
        if output_sha256s:
            span.set_attribute(ATTR_OUTPUT_DOCUMENT_SHA256S, output_sha256s)


def track_llm_documents(context: object | None, messages: object | None) -> None:
    """Track documents used in LLM calls (context and messages)."""
    service = get_tracking_service()
    if service is None:
        return

    span_id = get_current_span_id()

    if context is not None:
        _track_docs_from_messages(service, context, span_id, DocumentEventType.LLM_CONTEXT)

    if messages is not None:
        _track_docs_from_messages(service, messages, span_id, DocumentEventType.LLM_MESSAGE)


def _track_docs_from_messages(service: TrackingService, messages: object, span_id: str, event_type: DocumentEventType) -> None:
    """Extract and track documents from AIMessages or similar containers."""
    if not isinstance(messages, list):
        return
    for item in cast(list[object], messages):
        if isinstance(item, Document):
            service.track_document_event(
                document_sha256=item.sha256,
                span_id=span_id,
                event_type=event_type,
            )
