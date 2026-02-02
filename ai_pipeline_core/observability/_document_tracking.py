"""Document tracking helpers for pipeline instrumentation.

Emits document lifecycle events and sets OTel span attributes for
document lineage. All functions are no-ops when tracking is not initialized.
"""

from typing import cast

from opentelemetry import trace as otel_trace

from ai_pipeline_core.documents import Document
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._initialization import TrackingServiceProtocol, get_tracking_service
from ai_pipeline_core.observability._tracking._models import ATTR_INPUT_DOCUMENT_SHA256S, ATTR_OUTPUT_DOCUMENT_SHA256S, DocumentEventType

logger = get_pipeline_logger(__name__)


def get_current_span_id() -> str:
    """Return the current OTel span ID as hex, or empty string."""
    span = otel_trace.get_current_span()
    ctx = span.get_span_context()
    if ctx and ctx.span_id:
        return format(ctx.span_id, "016x")
    return ""


def _get_tracking_service() -> TrackingServiceProtocol | None:
    """Return the global tracking service, or None if not initialized."""
    return get_tracking_service()


def track_task_io(task_name: str, args: tuple[object, ...], kwargs: dict[str, object], result: object) -> None:  # noqa: ARG001
    """Track input/output documents for a pipeline task."""
    service = _get_tracking_service()
    if service is None:
        return

    span_id = get_current_span_id()
    input_sha256s: list[str] = []
    output_sha256s: list[str] = []

    # Track input documents
    for arg in (*args, *kwargs.values()):
        if isinstance(arg, Document):
            input_sha256s.append(arg.sha256)
            service.track_document_event(
                document_sha256=arg.sha256,
                span_id=span_id,
                event_type=DocumentEventType.TASK_INPUT,
            )
        elif isinstance(arg, list) and arg and isinstance(arg[0], Document):
            for doc in cast(list[Document], arg):
                input_sha256s.append(doc.sha256)
                service.track_document_event(
                    document_sha256=doc.sha256,
                    span_id=span_id,
                    event_type=DocumentEventType.TASK_INPUT,
                )

    # Track output documents
    if isinstance(result, Document):
        output_sha256s.append(result.sha256)
        service.track_document_event(
            document_sha256=result.sha256,
            span_id=span_id,
            event_type=DocumentEventType.TASK_OUTPUT,
        )
    elif isinstance(result, list) and result and isinstance(result[0], Document):
        for doc in cast(list[Document], result):
            output_sha256s.append(doc.sha256)
            service.track_document_event(
                document_sha256=doc.sha256,
                span_id=span_id,
                event_type=DocumentEventType.TASK_OUTPUT,
            )

    # Set span attributes for TrackingSpanProcessor to populate tracked_spans columns
    if input_sha256s or output_sha256s:
        span = otel_trace.get_current_span()
        if input_sha256s:
            span.set_attribute(ATTR_INPUT_DOCUMENT_SHA256S, input_sha256s)
        if output_sha256s:
            span.set_attribute(ATTR_OUTPUT_DOCUMENT_SHA256S, output_sha256s)


def track_flow_io(flow_name: str, input_documents: list[Document], output_documents: list[Document]) -> None:  # noqa: ARG001
    """Track input/output documents for a pipeline flow."""
    service = _get_tracking_service()
    if service is None:
        return

    span_id = get_current_span_id()
    input_sha256s: list[str] = []
    output_sha256s: list[str] = []

    for doc in input_documents:
        input_sha256s.append(doc.sha256)
        service.track_document_event(
            document_sha256=doc.sha256,
            span_id=span_id,
            event_type=DocumentEventType.FLOW_INPUT,
        )

    for doc in output_documents:
        output_sha256s.append(doc.sha256)
        service.track_document_event(
            document_sha256=doc.sha256,
            span_id=span_id,
            event_type=DocumentEventType.FLOW_OUTPUT,
        )

    if input_sha256s or output_sha256s:
        span = otel_trace.get_current_span()
        if input_sha256s:
            span.set_attribute(ATTR_INPUT_DOCUMENT_SHA256S, input_sha256s)
        if output_sha256s:
            span.set_attribute(ATTR_OUTPUT_DOCUMENT_SHA256S, output_sha256s)


def track_llm_documents(context: object | None, messages: object | None) -> None:
    """Track documents used in LLM calls (context and messages)."""
    service = _get_tracking_service()
    if service is None:
        return

    span_id = get_current_span_id()

    if context is not None:
        _track_docs_from_messages(service, context, span_id, DocumentEventType.LLM_CONTEXT)

    if messages is not None:
        _track_docs_from_messages(service, messages, span_id, DocumentEventType.LLM_MESSAGE)


def _track_docs_from_messages(service: TrackingServiceProtocol, messages: object, span_id: str, event_type: DocumentEventType) -> None:
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
