"""OpenTelemetry SpanProcessor for local trace debugging."""

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace import StatusCode

from ai_pipeline_core.logging import get_pipeline_logger

from ._writer import LocalTraceWriter, WriteJob

logger = get_pipeline_logger(__name__)

# Span processors must never raise — exceptions here would break tracing for the
# entire application.  We catch the broad set of exceptions that can occur from
# filesystem I/O, serialization, and formatting operations in the trace writer.
_PROCESSOR_ERRORS = (OSError, ValueError, TypeError, RuntimeError, AttributeError)


class LocalDebugSpanProcessor(SpanProcessor):
    """OpenTelemetry SpanProcessor that writes spans to local filesystem.

    Integrates with the OpenTelemetry SDK to capture all spans and write them
    to a structured directory hierarchy for debugging.

    When verbose=False (default), LLM-type spans are filtered out to avoid
    duplicate directories (every Conversation call creates both a DEFAULT and
    an inner LLM span). Filtered span metrics are still captured for totals.
    """

    def __init__(self, writer: LocalTraceWriter, *, verbose: bool = False):
        self._writer = writer
        self._verbose = verbose
        self._filtered_span_ids: set[str] = set()

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        """Handle span start - create directories.

        Creates the span directory early so we can see "running" spans.
        Input/output data is not available yet - will be captured in on_end().

        LLM-type spans are filtered unless verbose mode is enabled.
        """
        try:
            if span.context is None:
                return
            trace_id = format(span.context.trace_id, "032x")
            span_id = format(span.context.span_id, "016x")
            parent_id = self._get_parent_span_id(span)

            # Filter LLM-type spans unless verbose mode
            if not self._verbose:
                attrs = span.attributes
                if attrs and attrs.get("lmnr.span.type") == "LLM":
                    self._filtered_span_ids.add(span_id)
                    return

            self._writer.on_span_start(trace_id, span_id, parent_id, span.name)
        except _PROCESSOR_ERRORS as e:
            logger.debug("Failed to process span start for '%s': %s", getattr(span, "name", "?"), e)

    def on_end(self, span: ReadableSpan) -> None:
        """Handle span end - queue full span data for background write.

        All data (input, output, attributes, events) is captured here because
        Laminar sets these attributes after span start.

        Filtered spans are not written to disk but their metrics are passed
        to the writer for trace-level cost/token totals.
        """
        try:
            if span.context is None or span.start_time is None or span.end_time is None:
                return

            span_id = format(span.context.span_id, "016x")

            if span_id in self._filtered_span_ids:
                self._filtered_span_ids.discard(span_id)
                # Pass metrics to writer for document_summary cost tracking
                trace_id = format(span.context.trace_id, "032x")
                attributes = dict(span.attributes) if span.attributes else {}
                self._writer.record_filtered_llm_metrics(trace_id, attributes)
                return

            job = WriteJob(
                trace_id=format(span.context.trace_id, "032x"),
                span_id=span_id,
                name=span.name,
                parent_id=self._get_parent_span_id(span),
                attributes=dict(span.attributes) if span.attributes else {},
                events=list(span.events) if span.events else [],
                status_code=self._get_status_code(span),
                status_description=span.status.description,
                start_time_ns=span.start_time,
                end_time_ns=span.end_time,
            )
            self._writer.on_span_end(job)
        except _PROCESSOR_ERRORS as e:
            logger.debug("Failed to process span end for '%s': %s", getattr(span, "name", "?"), e)

    def shutdown(self) -> None:
        """Shutdown the processor and writer."""
        self._writer.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Flush all queued spans to disk, blocking until complete or timeout."""
        return self._writer.flush(timeout=timeout_millis / 1000)

    @staticmethod
    def _get_parent_span_id(span: Span | ReadableSpan) -> str | None:
        """Extract parent span ID as hex string, or None."""
        parent = getattr(span, "parent", None)
        if parent is not None and hasattr(parent, "span_id") and parent.span_id:
            return format(parent.span_id, "016x")
        return None

    @staticmethod
    def _get_status_code(span: ReadableSpan) -> str:
        """Get status code as string."""
        if span.status.status_code == StatusCode.OK:
            return "OK"
        if span.status.status_code == StatusCode.ERROR:
            return "ERROR"
        return "UNSET"
