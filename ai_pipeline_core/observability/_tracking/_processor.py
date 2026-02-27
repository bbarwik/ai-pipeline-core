"""Unified OTel SpanProcessor with pluggable backends."""

import dataclasses
import threading
from datetime import UTC, datetime
from typing import override
from uuid import UUID

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._span_data import SpanData
from ai_pipeline_core.observability._tracking._types import SpanBackend

logger = get_pipeline_logger(__name__)

_PROCESSOR_ERRORS = (OSError, ValueError, TypeError, RuntimeError, AttributeError)


class PipelineSpanProcessor(SpanProcessor):
    """Single OTel SpanProcessor that dispatches to multiple backends.

    Extracts SpanData once on span end, sends to all registered backends.
    Manages the span_order counter, LLM span filtering, and run context
    propagation (execution_id, run_id, flow_name, run_scope).

    When verbose=False (default), LLM-type spans skip directory creation
    but are still sent to backends for metrics accumulation.
    """

    def __init__(
        self,
        backends: tuple[SpanBackend, ...],
        *,
        verbose: bool = False,
    ) -> None:
        self._backends = backends
        self._verbose = verbose
        self._lock = threading.Lock()
        self._span_order: int = 0
        self._span_order_map: dict[str, int] = {}
        self._filtered_span_ids: set[str] = set()

        # Run context — set by PipelineDeployment.run(), injected into SpanData.
        # WARNING: instance-level mutable state — assumes single flow per process.
        # If concurrent flows run in one process, spans get tagged with the wrong
        # execution_id. Fix (when needed): switch to contextvars.ContextVar.
        self._execution_id: UUID | None = None
        self._run_id: str = ""
        self._flow_name: str = ""
        self._run_scope: str = ""

    def set_run_context(self, *, execution_id: UUID, run_id: str, flow_name: str, run_scope: str) -> None:
        """Store pipeline run context for injection into child spans."""
        with self._lock:
            self._execution_id = execution_id
            self._run_id = run_id
            self._flow_name = flow_name
            self._run_scope = run_scope

    def clear_run_context(self) -> None:
        """Clear pipeline run context after run completes."""
        with self._lock:
            self._execution_id = None
            self._run_id = ""
            self._flow_name = ""
            self._run_scope = ""

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        """Handle span start -- assign order and notify backends that need early directory creation."""
        try:
            if span.context is None:
                return
            span_id = format(span.context.span_id, "016x")

            with self._lock:
                # Filter LLM spans unless verbose
                if not self._verbose:
                    attrs = span.attributes
                    if attrs and attrs.get("lmnr.span.type") == "LLM":
                        self._filtered_span_ids.add(span_id)
                        return

                self._span_order += 1
                order = self._span_order
                self._span_order_map[span_id] = order

            # Build minimal SpanData for on_start (FilesystemBackend creates dirs)
            trace_id = format(span.context.trace_id, "032x")
            parent = getattr(span, "parent", None)
            parent_span_id = format(parent.span_id, "016x") if parent and hasattr(parent, "span_id") and parent.span_id else None

            start_data = _build_start_span_data(
                span_id=span_id,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                name=span.name,
                span_order=order,
            )

            for backend in self._backends:
                try:
                    backend.on_span_start(start_data)
                except _PROCESSOR_ERRORS as e:
                    logger.debug("Backend %s.on_span_start failed: %s", type(backend).__name__, e)
        except _PROCESSOR_ERRORS as e:
            logger.debug("PipelineSpanProcessor.on_start failed: %s", e)

    def on_end(self, span: ReadableSpan) -> None:
        """Handle span end -- extract SpanData and dispatch to all backends."""
        try:
            ctx = span.get_span_context()
            if ctx is None or span.start_time is None or span.end_time is None:
                return

            span_id = format(ctx.span_id, "016x")

            with self._lock:
                is_filtered = span_id in self._filtered_span_ids
                if is_filtered:
                    self._filtered_span_ids.discard(span_id)
                    order = 0
                else:
                    order = self._span_order_map.pop(span_id, 0)

                # Capture run context under lock
                run_ctx_execution_id = self._execution_id
                run_ctx_run_id = self._run_id
                run_ctx_flow_name = self._flow_name
                run_ctx_run_scope = self._run_scope

            span_data = SpanData.from_otel_span(span, span_order=order)

            # Inject run context if SpanData didn't get it from span/resource attributes
            if span_data.execution_id is None and run_ctx_execution_id is not None:
                span_data = dataclasses.replace(
                    span_data,
                    execution_id=run_ctx_execution_id,
                    run_id=span_data.run_id or run_ctx_run_id,
                    flow_name=span_data.flow_name or run_ctx_flow_name,
                    run_scope=span_data.run_scope or run_ctx_run_scope,
                )

            for backend in self._backends:
                try:
                    backend.on_span_end(span_data)
                except _PROCESSOR_ERRORS as e:
                    logger.debug("Backend %s.on_span_end failed: %s", type(backend).__name__, e)
        except _PROCESSOR_ERRORS as e:
            logger.debug("PipelineSpanProcessor.on_end failed: %s", e)

    def shutdown(self) -> None:
        """Shutdown all backends."""
        for backend in self._backends:
            try:
                backend.shutdown()
            except _PROCESSOR_ERRORS as e:
                logger.debug("Backend %s.shutdown failed: %s", type(backend).__name__, e)

    @override
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """No-op — backends flush independently via their own threads."""
        del timeout_millis
        return True


_EPOCH = datetime.fromtimestamp(0, tz=UTC)


def _build_start_span_data(
    *,
    span_id: str,
    trace_id: str,
    parent_span_id: str | None,
    name: str,
    span_order: int,
) -> SpanData:
    """Build a minimal SpanData for on_span_start notifications.

    Only identity and ordering fields are populated; the full data
    comes in on_end via SpanData.from_otel_span().
    """
    return SpanData(
        execution_id=None,
        span_id=span_id,
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        name=name,
        run_id="",
        flow_name="",
        run_scope="",
        span_type="trace",
        status="running",
        start_time=_EPOCH,
        end_time=_EPOCH,
        duration_ms=0,
        span_order=span_order,
        cost=0.0,
        tokens_input=0,
        tokens_output=0,
        tokens_cached=0,
        llm_model=None,
        error_message="",
        input_json="",
        output_json="",
        replay_payload="",
        attributes={},
        events=(),
        input_doc_sha256s=(),
        output_doc_sha256s=(),
    )
