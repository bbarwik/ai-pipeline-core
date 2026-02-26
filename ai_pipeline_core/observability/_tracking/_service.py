"""TrackingService — central coordinator for pipeline observability.

Manages run context, version counters, and coordinates the ClickHouse writer thread.
"""

import json
import time
from datetime import UTC, datetime
from threading import Lock
from uuid import UUID, uuid4

from ai_pipeline_core.documents.types import DocumentSha256
from ai_pipeline_core.logging import get_pipeline_logger

from ._client import ClickHouseClient
from ._models import (
    TABLE_DOCUMENT_EVENTS,
    TABLE_PIPELINE_RUNS,
    TABLE_SPAN_EVENTS,
    TABLE_TRACE_SPAN_CONTENT,
    TABLE_TRACKED_SPANS,
    DocumentEventRow,
    DocumentEventType,
    PipelineRunRow,
    RunStatus,
    SpanEventRow,
    SpanType,
    TraceSpanContentRow,
    TrackedSpanRow,
)
from ._writer import ClickHouseWriter

logger = get_pipeline_logger(__name__)


class TrackingService:
    """Central tracking coordinator.

    Thread-safe — all mutable state is protected by ``_lock``.
    """

    def __init__(
        self,
        client: ClickHouseClient,
    ) -> None:
        """Initialize tracking service and start writer thread."""
        self._client = client

        self._writer = ClickHouseWriter(client)
        self._writer.start()

        # Run context
        self._execution_id: UUID | None = None
        self._run_id: str = ""
        self._flow_name: str = ""
        self._run_scope: str = ""
        self._run_start_time: datetime | None = None

        # Monotonic version counter
        self._last_version: int = 0
        self._span_order_counter: int = 0
        self._lock = Lock()

    # --- Run context ---

    def set_run_context(self, *, execution_id: UUID, run_id: str, flow_name: str, run_scope: str = "") -> None:
        """Set the current run context. Called at pipeline start."""
        with self._lock:
            self._execution_id = execution_id
            self._run_id = run_id
            self._flow_name = flow_name
            self._run_scope = run_scope
            self._span_order_counter = 0

    def clear_run_context(self) -> None:
        """Clear run context. Called by flush() and shutdown()."""
        with self._lock:
            self._execution_id = None
            self._run_id = ""
            self._flow_name = ""
            self._run_scope = ""
            self._run_start_time = None
            self._span_order_counter = 0

    # --- Version management ---

    def _next_version(self) -> int:
        """Generate a monotonically increasing version using nanosecond timestamps."""
        now = time.time_ns()
        if now <= self._last_version:
            now = self._last_version + 1
        self._last_version = now
        return now

    # --- Run tracking ---

    def track_run_start(self, *, execution_id: UUID, run_id: str, flow_name: str, run_scope: str = "") -> None:
        """Record pipeline run start."""
        now = datetime.now(UTC)
        with self._lock:
            self._run_start_time = now
            version = self._next_version()
        row = PipelineRunRow(
            execution_id=execution_id,
            run_id=run_id,
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
        execution_id: UUID,
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
            execution_id=execution_id,
            run_id=self._run_id,
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
        if self._execution_id is None:
            return
        now = datetime.now(UTC)
        with self._lock:
            version = self._next_version()
        row = TrackedSpanRow(
            span_id=span_id,
            trace_id=trace_id,
            execution_id=self._execution_id,
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
        input_document_sha256s: list[DocumentSha256] | None = None,
        output_document_sha256s: list[DocumentSha256] | None = None,
    ) -> None:
        """Record span completion with full details."""
        if self._execution_id is None:
            return
        with self._lock:
            version = self._next_version()
        row = TrackedSpanRow(
            span_id=span_id,
            trace_id=trace_id,
            execution_id=self._execution_id,
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
            input_document_sha256s=tuple(input_document_sha256s) if input_document_sha256s else (),
            output_document_sha256s=tuple(output_document_sha256s) if output_document_sha256s else (),
            version=version,
        )
        self._writer.write(TABLE_TRACKED_SPANS, [row])

    def track_span_events(self, *, span_id: str, events: list[tuple[str, datetime, dict[str, str], str | None]]) -> None:
        """Record span events (including bridged log events)."""
        if self._execution_id is None or not events:
            return
        rows = [
            SpanEventRow(
                event_id=uuid4(),
                execution_id=self._execution_id,
                span_id=span_id,
                name=name,
                timestamp=ts,
                attributes=json.dumps(attrs) if attrs else "{}",
                level=level,
            )
            for name, ts, attrs, level in events
        ]
        self._writer.write(TABLE_SPAN_EVENTS, list(rows))

    # --- Span ordering & content ---

    def assign_span_order(self) -> int:
        """Assign monotonically increasing order to a span. Thread-safe."""
        with self._lock:
            self._span_order_counter += 1
            return self._span_order_counter

    def track_span_content(
        self,
        *,
        span_id: str,
        trace_id: str,
        span_order: int,
        input_json: str,
        output_json: str,
        replay_payload: str,
        attributes_json: str,
        events_json: str,
    ) -> None:
        """Record span content for remote trace reconstruction."""
        if self._execution_id is None:
            return
        row = TraceSpanContentRow(
            span_id=span_id,
            trace_id=trace_id,
            execution_id=self._execution_id,
            span_order=span_order,
            input_json=input_json,
            output_json=output_json,
            replay_payload=replay_payload,
            attributes_json=attributes_json,
            events_json=events_json,
            stored_at=datetime.now(UTC),
        )
        self._writer.write(TABLE_TRACE_SPAN_CONTENT, [row])

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
        if self._execution_id is None:
            return
        row = DocumentEventRow(
            event_id=uuid4(),
            execution_id=self._execution_id,
            document_sha256=document_sha256,
            span_id=span_id,
            event_type=event_type,
            timestamp=datetime.now(UTC),
            metadata=json.dumps(metadata) if metadata else "{}",
        )
        self._writer.write(TABLE_DOCUMENT_EVENTS, [row])

    # --- Lifecycle ---

    def flush(self, timeout: float = 30.0) -> None:
        """Wait for all pending items to complete, then clear run context."""
        self._writer.flush(timeout=timeout)
        self.clear_run_context()

    def shutdown(self, timeout: float = 30.0) -> None:
        """Shutdown the writer thread and clear run context."""
        self._writer.shutdown(timeout=timeout)
        self.clear_run_context()
