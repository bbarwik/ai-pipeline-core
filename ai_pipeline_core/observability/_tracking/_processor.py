"""OpenTelemetry SpanProcessor that feeds the tracking system."""

from datetime import UTC, datetime
from typing import Any

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace import StatusCode

from ai_pipeline_core.logging import get_pipeline_logger

from ._internal import is_internal_tracking
from ._models import ATTR_INPUT_DOCUMENT_SHA256S, ATTR_OUTPUT_DOCUMENT_SHA256S, SpanType
from ._service import TrackingService

logger = get_pipeline_logger(__name__)


def _hex_span_id(span_id: int) -> str:
    """Convert integer span ID to hex string."""
    return format(span_id, "016x")


def _hex_trace_id(trace_id: int) -> str:
    """Convert integer trace ID to hex string."""
    return format(trace_id, "032x")


def _ns_to_datetime(ns: int) -> datetime:
    """Convert nanosecond timestamp to datetime."""
    return datetime.fromtimestamp(ns / 1e9, tz=UTC)


def _classify_span(attrs: dict[str, Any]) -> SpanType:
    """Determine span type from attributes."""
    span_type_str = str(attrs.get("lmnr.span.type", ""))
    if span_type_str == "LLM":
        return SpanType.LLM
    if attrs.get("prefect.flow.name"):
        return SpanType.FLOW
    if attrs.get("prefect.task.name"):
        return SpanType.TASK
    return SpanType.TRACE


class TrackingSpanProcessor(SpanProcessor):
    """Forwards completed spans to TrackingService.

    Skips internal tracking spans (summary LLM calls) to prevent recursion.
    """

    def __init__(self, service: TrackingService) -> None:
        """Initialize with tracking service."""
        self._service = service

    @staticmethod
    def _parent_span_id(span: Span | ReadableSpan) -> str | None:
        """Extract parent span ID as hex string, or None."""
        parent = span.parent
        if parent is None:
            return None
        return _hex_span_id(parent.span_id)

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        """Record span start."""
        if is_internal_tracking():
            return
        try:
            ctx = span.get_span_context()
            if ctx is None:
                return
            attrs: dict[str, Any] = dict(span.attributes or {})
            self._service.track_span_start(
                span_id=_hex_span_id(ctx.span_id),
                trace_id=_hex_trace_id(ctx.trace_id),
                parent_span_id=self._parent_span_id(span),
                name=span.name,
                span_type=_classify_span(attrs),
            )
        except Exception as e:
            logger.debug(f"TrackingSpanProcessor.on_start failed: {e}")

    def on_end(self, span: ReadableSpan) -> None:  # noqa: PLR0914
        """Record span completion with full details."""
        if is_internal_tracking():
            return
        try:
            ctx = span.get_span_context()
            if ctx is None:
                return
            attrs: dict[str, Any] = dict(span.attributes or {})

            start_ns = span.start_time or 0
            end_ns = span.end_time or 0
            start_time = _ns_to_datetime(start_ns)
            end_time = _ns_to_datetime(end_ns)
            duration_ms = max(0, (end_ns - start_ns) // 1_000_000)

            status = "failed" if span.status.status_code == StatusCode.ERROR else "completed"

            # Extract LLM-specific attributes
            cost = float(attrs.get("gen_ai.usage.cost", 0.0))
            tokens_input = int(attrs.get("gen_ai.usage.input_tokens", 0))
            tokens_output = int(attrs.get("gen_ai.usage.output_tokens", 0))
            llm_model = str(attrs.get("gen_ai.request.model", "")) or None

            # Extract document SHA256 arrays set by track_task_io
            raw_input_sha256s = attrs.get(ATTR_INPUT_DOCUMENT_SHA256S)
            input_doc_sha256s = list(raw_input_sha256s) if raw_input_sha256s else None
            raw_output_sha256s = attrs.get(ATTR_OUTPUT_DOCUMENT_SHA256S)
            output_doc_sha256s = list(raw_output_sha256s) if raw_output_sha256s else None

            span_id = _hex_span_id(ctx.span_id)
            self._service.track_span_end(
                span_id=span_id,
                trace_id=_hex_trace_id(ctx.trace_id),
                parent_span_id=self._parent_span_id(span),
                name=span.name,
                span_type=_classify_span(attrs),
                status=status,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                cost=cost,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                llm_model=llm_model,
                input_document_sha256s=input_doc_sha256s,
                output_document_sha256s=output_doc_sha256s,
            )

            # Forward span events
            if span.events:
                events: list[tuple[str, datetime, dict[str, str], str | None]] = []
                for event in span.events:
                    event_attrs = dict(event.attributes) if event.attributes else {}
                    level = str(event_attrs.pop("log.level", "")) or None
                    events.append((
                        event.name,
                        _ns_to_datetime(event.timestamp),
                        {k: str(v) for k, v in event_attrs.items()},
                        level,
                    ))
                self._service.track_span_events(
                    span_id=span_id,
                    events=events,
                )
        except Exception as e:
            logger.debug(f"TrackingSpanProcessor.on_end failed: {e}")

    def shutdown(self) -> None:
        """Shutdown the tracking service."""
        self._service.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # noqa: PLR6301
        """Force flush is a no-op — the writer flushes on its own schedule."""
        _ = timeout_millis
        return True
