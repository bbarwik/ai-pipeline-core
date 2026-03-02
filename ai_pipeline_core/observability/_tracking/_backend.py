"""ClickHouse backend for the unified span processor."""

import json
import time
from datetime import UTC, datetime
from threading import Lock
from uuid import UUID

from ai_pipeline_core.observability._span_data import SpanData

from ._models import TABLE_PIPELINE_RUNS, TABLE_PIPELINE_SPANS, PipelineRunRow, PipelineSpanRow, RunStatus
from ._writer import ClickHouseWriter


class ClickHouseBackend:
    """Writes span data and run lifecycle to ClickHouse.

    Implements the SpanBackend protocol: on_span_end builds a PipelineSpanRow
    and queues it to ClickHouseWriter. Run start/end are separate methods
    called from PipelineDeployment.run(), not from OTel span callbacks.
    """

    def __init__(self, writer: ClickHouseWriter) -> None:
        self._writer = writer
        self._last_version: int = 0
        self._lock = Lock()
        self._run_execution_id: UUID | None = None

    def _next_version(self) -> int:
        """Generate a monotonically increasing version (nanosecond timestamp)."""
        now = time.time_ns()
        with self._lock:
            if now <= self._last_version:
                now = self._last_version + 1
            self._last_version = now
        return now

    # --- SpanBackend protocol ---

    def on_span_start(self, span_data: SpanData) -> None:
        """No-op for ClickHouse -- spans are written on end only."""

    def on_span_end(self, span_data: SpanData) -> None:
        """Build a PipelineSpanRow and queue it for insertion."""
        execution_id = span_data.execution_id or self._run_execution_id
        if execution_id is None:
            return
        row = PipelineSpanRow(
            execution_id=execution_id,
            span_id=span_data.span_id,
            trace_id=span_data.trace_id,
            parent_span_id=span_data.parent_span_id,
            run_id=span_data.run_id,
            flow_name=span_data.flow_name,
            run_scope=span_data.run_scope,
            name=span_data.name,
            span_type=span_data.span_type,
            status=span_data.status,
            start_time=span_data.start_time,
            end_time=span_data.end_time,
            duration_ms=span_data.duration_ms,
            span_order=span_data.span_order,
            cost=span_data.cost,
            tokens_input=span_data.tokens_input,
            tokens_output=span_data.tokens_output,
            tokens_cached=span_data.tokens_cached,
            llm_model=span_data.llm_model,
            error_message=span_data.error_message,
            input_json=span_data.input_json,
            output_json=span_data.output_json,
            replay_payload=span_data.replay_payload,
            attributes_json=json.dumps(span_data.attributes, default=str),
            events_json=json.dumps(list(span_data.events), default=str),
            input_doc_sha256s=tuple(span_data.input_doc_sha256s),
            output_doc_sha256s=tuple(span_data.output_doc_sha256s),
        )
        self._writer.write(TABLE_PIPELINE_SPANS, [row])

    def shutdown(self) -> None:
        """Flush pending batches and stop the writer thread. Idempotent."""
        self._writer.shutdown()

    # --- Run lifecycle (separate from SpanBackend) ---

    def track_run_start(
        self,
        *,
        execution_id: UUID,
        run_id: str,
        flow_name: str,
        run_scope: str = "",
    ) -> datetime:
        """Write a running-status row to pipeline_runs. Returns the start_time for use in track_run_end."""
        self._run_execution_id = execution_id
        start_time = datetime.now(UTC)
        row = PipelineRunRow(
            execution_id=execution_id,
            run_id=run_id,
            flow_name=flow_name,
            run_scope=run_scope,
            status=RunStatus.RUNNING,
            start_time=start_time,
            version=self._next_version(),
        )
        self._writer.write(TABLE_PIPELINE_RUNS, [row])
        return start_time

    def track_run_end(
        self,
        *,
        execution_id: UUID,
        run_id: str,
        flow_name: str,
        run_scope: str = "",
        status: RunStatus,
        start_time: datetime,
        total_cost: float = 0.0,
        total_tokens: int = 0,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Write a completed/failed-status row to pipeline_runs."""
        self._run_execution_id = None
        row = PipelineRunRow(
            execution_id=execution_id,
            run_id=run_id,
            flow_name=flow_name,
            run_scope=run_scope,
            status=status,
            start_time=start_time,
            end_time=datetime.now(UTC),
            total_cost=total_cost,
            total_tokens=total_tokens,
            metadata_json=json.dumps(metadata) if metadata else "{}",
            version=self._next_version(),
        )
        self._writer.write(TABLE_PIPELINE_RUNS, [row])

    def flush(self, timeout: float = 30.0) -> None:
        """Flush the underlying writer."""
        self._writer.flush(timeout=timeout)
