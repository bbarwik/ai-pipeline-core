"""OpenTelemetry SpanProcessor for local trace debugging."""

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace import StatusCode

from .writer import LocalTraceWriter, WriteJob


class LocalDebugSpanProcessor(SpanProcessor):
    """OpenTelemetry SpanProcessor that writes spans to local filesystem.

    Integrates with the OpenTelemetry SDK to capture all spans and write them
    to a structured directory hierarchy for debugging.

    Usage:
        writer = LocalTraceWriter(config)
        processor = LocalDebugSpanProcessor(writer)
        tracer_provider.add_span_processor(processor)
    """

    def __init__(self, writer: LocalTraceWriter):
        """Initialize span processor with writer."""
        self._writer = writer

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        """Handle span start - create directories.

        Creates the span directory early so we can see "running" spans.
        Input/output data is not available yet - will be captured in on_end().
        """
        try:
            if span.context is None:
                return
            trace_id = format(span.context.trace_id, "032x")
            span_id = format(span.context.span_id, "016x")
            parent_id = self._get_parent_span_id(span)

            self._writer.on_span_start(trace_id, span_id, parent_id, span.name)
        except Exception:
            # Never fail the actual span - debug tracing should be transparent
            pass

    def on_end(self, span: ReadableSpan) -> None:
        """Handle span end - queue full span data for background write.

        All data (input, output, attributes, events) is captured here because
        Laminar sets these attributes after span start.
        """
        try:
            if span.context is None or span.start_time is None or span.end_time is None:
                return
            job = WriteJob(
                trace_id=format(span.context.trace_id, "032x"),
                span_id=format(span.context.span_id, "016x"),
                name=span.name,
                parent_id=self._get_parent_span_id_from_readable(span),
                attributes=dict(span.attributes) if span.attributes else {},
                events=list(span.events) if span.events else [],
                status_code=self._get_status_code(span),
                status_description=span.status.description,
                start_time_ns=span.start_time,
                end_time_ns=span.end_time,
            )
            self._writer.on_span_end(job)
        except Exception:
            # Never fail the actual span
            pass

    def shutdown(self) -> None:
        """Shutdown the processor and writer."""
        self._writer.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush is not needed for this processor."""
        return True

    def _get_parent_span_id(self, span: Span) -> str | None:
        """Extract parent span ID from a writable Span."""
        if hasattr(span, "parent") and span.parent:
            parent_ctx = span.parent
            if hasattr(parent_ctx, "span_id") and parent_ctx.span_id:
                return format(parent_ctx.span_id, "016x")
        return None

    def _get_parent_span_id_from_readable(self, span: ReadableSpan) -> str | None:
        """Extract parent span ID from a ReadableSpan."""
        if span.parent:
            if hasattr(span.parent, "span_id") and span.parent.span_id:
                return format(span.parent.span_id, "016x")
        return None

    def _get_status_code(self, span: ReadableSpan) -> str:
        """Get status code as string."""
        if span.status.status_code == StatusCode.OK:
            return "OK"
        elif span.status.status_code == StatusCode.ERROR:
            return "ERROR"
        return "UNSET"
