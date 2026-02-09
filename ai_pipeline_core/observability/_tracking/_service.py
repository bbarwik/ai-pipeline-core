"""TrackingService — central coordinator for pipeline observability.

Manages run context, version counters, row caching for summary updates,
and coordinates the ClickHouse writer thread.
"""

import json
import time
from datetime import UTC, datetime
from threading import Lock
from uuid import UUID, uuid4

from lmnr.opentelemetry_lib.tracing import context as laminar_context
from opentelemetry import context as otel_context

from ai_pipeline_core.documents._types import DocumentSha256
from ai_pipeline_core.logging import get_pipeline_logger

from ._client import ClickHouseClient
from ._models import (
    TABLE_DOCUMENT_EVENTS,
    TABLE_PIPELINE_RUNS,
    TABLE_SPAN_EVENTS,
    TABLE_TRACKED_SPANS,
    DocumentEventRow,
    DocumentEventType,
    PipelineRunRow,
    RunStatus,
    SpanEventRow,
    SpanType,
    TrackedSpanRow,
)
from ._writer import ClickHouseWriter, SpanSummaryFn, SummaryJob

logger = get_pipeline_logger(__name__)


class TrackingService:
    """Central tracking coordinator.

    Thread-safe — all mutable state is protected by ``_lock``.
    """

    def __init__(
        self,
        client: ClickHouseClient,
        *,
        summary_model: str = "gemini-3-flash",
        span_summary_fn: SpanSummaryFn | None = None,
    ) -> None:
        """Initialize tracking service and start writer thread."""
        self._client = client
        self._summary_model = summary_model

        self._writer = ClickHouseWriter(
            client,
            summary_row_builder=self.build_span_summary_update,
            span_summary_fn=span_summary_fn,
        )
        self._writer.start()

        # Run context
        self._run_id: UUID | None = None
        self._project_name: str = ""
        self._flow_name: str = ""
        self._run_scope: str = ""
        self._run_start_time: datetime | None = None

        # Monotonic version counter
        self._last_version: int = 0
        self._lock = Lock()

        # Row caches for summary updates
        self._span_cache: dict[str, TrackedSpanRow] = {}

    # --- Run context ---

    def set_run_context(self, *, run_id: UUID, project_name: str, flow_name: str, run_scope: str = "") -> None:
        """Set the current run context. Called at pipeline start."""
        with self._lock:
            self._run_id = run_id
            self._project_name = project_name
            self._flow_name = flow_name
            self._run_scope = run_scope

    def clear_run_context(self) -> None:
        """Clear run context and caches. Called by flush() and shutdown()."""
        with self._lock:
            self._run_id = None
            self._project_name = ""
            self._flow_name = ""
            self._run_scope = ""
            self._run_start_time = None
            self._span_cache.clear()

    # --- Version management ---

    def _next_version(self) -> int:
        """Generate a monotonically increasing version using nanosecond timestamps."""
        now = time.time_ns()
        if now <= self._last_version:
            now = self._last_version + 1
        self._last_version = now
        return now

    # --- Run tracking ---

    def track_run_start(self, *, run_id: UUID, project_name: str, flow_name: str, run_scope: str = "") -> None:
        """Record pipeline run start."""
        now = datetime.now(UTC)
        with self._lock:
            self._run_start_time = now
            version = self._next_version()
        row = PipelineRunRow(
            run_id=run_id,
            project_name=project_name,
            flow_name=flow_name,
            run_scope=run_scope,
            status=RunStatus.RUNNING,
            start_time=now,
            version=version,
        )
        self._writer.write(TABLE_PIPELINE_RUNS, [row])

    def track_run_end(
        self,
        *,
        run_id: UUID,
        status: RunStatus,
        total_cost: float = 0.0,
        total_tokens: int = 0,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Record pipeline run completion or failure."""
        now = datetime.now(UTC)
        with self._lock:
            version = self._next_version()
            start_time = self._run_start_time or now
        row = PipelineRunRow(
            run_id=run_id,
            project_name=self._project_name,
            flow_name=self._flow_name,
            run_scope=self._run_scope,
            status=status,
            start_time=start_time,
            end_time=now,
            total_cost=total_cost,
            total_tokens=total_tokens,
            metadata=json.dumps(metadata) if metadata else "{}",
            version=version,
        )
        self._writer.write(TABLE_PIPELINE_RUNS, [row])

    # --- Span tracking ---

    def track_span_start(self, *, span_id: str, trace_id: str, parent_span_id: str | None, name: str, span_type: SpanType) -> None:
        """Record span start."""
        if self._run_id is None:
            return
        now = datetime.now(UTC)
        with self._lock:
            version = self._next_version()
        row = TrackedSpanRow(
            span_id=span_id,
            trace_id=trace_id,
            run_id=self._run_id,
            parent_span_id=parent_span_id,
            name=name,
            span_type=span_type,
            status="running",
            start_time=now,
            version=version,
        )
        self._writer.write(TABLE_TRACKED_SPANS, [row])

    def track_span_end(
        self,
        *,
        span_id: str,
        trace_id: str,
        parent_span_id: str | None,
        name: str,
        span_type: SpanType,
        status: str,
        start_time: datetime,
        end_time: datetime,
        duration_ms: int,
        cost: float = 0.0,
        tokens_input: int = 0,
        tokens_output: int = 0,
        llm_model: str | None = None,
        user_summary: str | None = None,
        user_visible: bool = False,
        user_label: str | None = None,
        input_document_sha256s: list[DocumentSha256] | None = None,
        output_document_sha256s: list[DocumentSha256] | None = None,
    ) -> None:
        """Record span completion with full details."""
        if self._run_id is None:
            return
        with self._lock:
            version = self._next_version()
        row = TrackedSpanRow(
            span_id=span_id,
            trace_id=trace_id,
            run_id=self._run_id,
            parent_span_id=parent_span_id,
            name=name,
            span_type=span_type,
            status=status,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            cost=cost,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            llm_model=llm_model,
            user_summary=user_summary,
            user_visible=user_visible,
            user_label=user_label,
            input_document_sha256s=tuple(input_document_sha256s) if input_document_sha256s else (),
            output_document_sha256s=tuple(output_document_sha256s) if output_document_sha256s else (),
            version=version,
        )
        self._writer.write(TABLE_TRACKED_SPANS, [row])
        with self._lock:
            self._span_cache[span_id] = row

    def track_span_events(self, *, span_id: str, events: list[tuple[str, datetime, dict[str, str], str | None]]) -> None:
        """Record span events (including bridged log events)."""
        if self._run_id is None or not events:
            return
        rows = [
            SpanEventRow(
                event_id=uuid4(),
                run_id=self._run_id,
                span_id=span_id,
                name=name,
                timestamp=ts,
                attributes=json.dumps(attrs) if attrs else "{}",
                level=level,
            )
            for name, ts, attrs, level in events
        ]
        self._writer.write(TABLE_SPAN_EVENTS, list(rows))

    # --- Document tracking ---

    def track_document_event(
        self,
        *,
        document_sha256: DocumentSha256,
        span_id: str,
        event_type: DocumentEventType,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Record a document lifecycle event."""
        if self._run_id is None:
            return
        row = DocumentEventRow(
            event_id=uuid4(),
            run_id=self._run_id,
            document_sha256=document_sha256,
            span_id=span_id,
            event_type=event_type,
            timestamp=datetime.now(UTC),
            metadata=json.dumps(metadata) if metadata else "{}",
        )
        self._writer.write(TABLE_DOCUMENT_EVENTS, [row])

    # --- Summary scheduling ---

    def schedule_summary(self, span_id: str, label: str, output_hint: str) -> None:
        """Schedule LLM summary generation for a span."""
        self._writer.write_job(
            SummaryJob(
                span_id=span_id,
                label=label,
                output_hint=output_hint,
                summary_model=self._summary_model,
                parent_otel_context=otel_context.get_current(),
                parent_laminar_context=laminar_context.get_current_context(),
            )
        )

    # --- Summary row builders ---

    def build_span_summary_update(self, span_id: str, summary: str) -> TrackedSpanRow | None:
        """Build a replacement row with summary filled. Called from writer thread."""
        with self._lock:
            cached = self._span_cache.get(span_id)
            if cached is None:
                return None
            version = self._next_version()
        return cached.model_copy(update={"user_summary": summary, "version": version})

    # --- Lifecycle ---

    def flush(self, timeout: float = 30.0) -> None:
        """Wait for all pending items (including summary LLM jobs) to complete, then clear run context.

        Use between runs in long-lived processes to prevent unbounded cache growth.
        """
        self._writer.flush(timeout=timeout)
        self.clear_run_context()

    def shutdown(self, timeout: float = 30.0) -> None:
        """Shutdown the writer thread and clear run context.

        Writer drains all pending items (including summary LLM jobs)
        before caches are cleared, ensuring summaries can look up span data.
        """
        self._writer.shutdown(timeout=timeout)
        self.clear_run_context()
