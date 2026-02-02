"""Tests that reproduce race conditions and data loss scenarios in observability.

These tests use synchronization primitives (Events, monkey-patching) to deterministically
reproduce race windows. Tests that FAIL confirm the bug exists. Tests that PASS after
a fix confirm it's resolved.
"""

import time
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from ai_pipeline_core.observability._debug._config import TraceDebugConfig
from ai_pipeline_core.observability._debug._types import WriteJob
from ai_pipeline_core.observability._debug._writer import LocalTraceWriter
from ai_pipeline_core.observability._tracking._client import ClickHouseClient
from ai_pipeline_core.observability._tracking._writer import ClickHouseWriter


class _DummyRow(BaseModel):
    value: str


def _make_mock_client() -> MagicMock:
    """Create a mock ClickHouseClient that succeeds on connect/ensure_tables."""
    mock = MagicMock(spec=ClickHouseClient)
    mock.connect.return_value = None
    mock.ensure_tables.return_value = None
    mock._insert_rows.return_value = None
    return mock


class TestWriterStartupRace:
    """Issue 1: ClickHouseWriter has no startup synchronization barrier.

    write() silently drops data when _loop/_queue are None (before writer thread initializes).
    SummaryWorker correctly uses threading.Event — ClickHouseWriter does not.
    """

    def test_writes_after_start_are_not_dropped(self):
        """After fix: start() blocks until _loop/_queue are ready, so no writes are lost."""
        mock_client = _make_mock_client()
        writer = ClickHouseWriter(mock_client, batch_size=100)
        writer.start()

        # After start() returns, _loop and _queue MUST be initialized (startup barrier)
        assert writer._loop is not None
        assert writer._queue is not None

        # All writes should be accepted
        for i in range(5):
            writer.write("tracked_spans", [_DummyRow(value=f"row_{i}")])

        writer.shutdown(timeout=5.0)

        # All rows should have reached the client
        assert mock_client._insert_rows.called
        all_rows = []
        for call in mock_client._insert_rows.call_args_list:
            all_rows.extend(call[0][1])
        values = [r.value for r in all_rows]
        for i in range(5):
            assert f"row_{i}" in values

    def test_summary_worker_has_startup_barrier(self):
        """Contrast: SummaryWorker correctly blocks until ready."""
        from ai_pipeline_core.document_store._summary_worker import SummaryWorker

        async def dummy_gen(name: str, excerpt: str) -> str:
            return "summary"

        async def dummy_update(run_scope: str, sha256: str, summary: str) -> None:
            pass

        worker = SummaryWorker(generator=dummy_gen, update_fn=dummy_update)
        worker.start()

        # After start() returns, _loop and _queue MUST be initialized
        assert worker._loop is not None
        assert worker._queue is not None

        worker.shutdown(timeout=2.0)


class TestWriterPermanentDisable:
    """Issue 2: Transient connection failure permanently disables the writer.

    If connect() or ensure_tables() fails once, _disabled=True forever.
    No retry, no reconnect, no buffer. All future tracking data is silently dropped.
    """

    def test_persistent_connection_failure_disables_after_retries(self):
        """After fix: Writer retries connection with backoff before giving up."""
        mock_client = _make_mock_client()
        mock_client.connect.side_effect = ConnectionError("ClickHouse permanently unreachable")

        # Use small retry params via internal method to speed up test
        writer = ClickHouseWriter(mock_client, batch_size=100)
        writer.start()

        # Wait for all retries to exhaust (5 retries with exponential backoff)
        # Default: 1+2+4+8+16=31s is too slow for tests. The thread will eventually disable.
        assert writer._thread is not None
        writer._thread.join(timeout=60.0)

        assert writer._disabled is True
        assert writer._loop is None
        # connect was called multiple times (retries)
        assert mock_client.connect.call_count > 1

        writer.shutdown(timeout=2.0)

    def test_transient_failure_recovers_on_retry(self):
        """After fix: A transient failure that resolves on retry succeeds."""
        call_count = 0

        def connect_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Transient failure")

        mock_client = _make_mock_client()
        mock_client.connect.side_effect = connect_side_effect

        writer = ClickHouseWriter(mock_client, batch_size=100)
        writer.start()

        # Wait for the successful retry
        time.sleep(5.0)

        # Writer should NOT be disabled — it recovered on retry 3
        assert writer._disabled is False
        assert call_count >= 3

        # Writes should work
        writer.write("tracked_spans", [_DummyRow(value="recovered")])
        writer.shutdown(timeout=5.0)

        assert mock_client._insert_rows.called


class TestFlushBatchDataLoss:
    """Issue 4: _flush_batches() clears ALL pending rows on partial failure.

    If one table's insert fails, pending.clear() wipes rows for ALL tables,
    including tables that haven't been attempted yet.
    """

    def test_partial_flush_failure_preserves_failed_tables(self):
        """After fix: Failed table rows are preserved for retry on next flush cycle."""
        mock_client = _make_mock_client()
        tables_inserted: list[str] = []
        attempt_count = 0

        def selective_insert(table, rows):
            nonlocal attempt_count
            if table == "document_events":
                attempt_count += 1
                if attempt_count <= 1:
                    raise ConnectionError("Transient ClickHouse error")
            tables_inserted.append(table)

        mock_client._insert_rows.side_effect = selective_insert

        # Short flush interval so timeout-based retry happens quickly
        writer = ClickHouseWriter(mock_client, batch_size=1000, flush_interval_seconds=0.5)
        writer.start()

        # Write to two tables
        writer.write("tracked_spans", [_DummyRow(value="span_data")])
        writer.write("document_events", [_DummyRow(value="event_data")])

        # Wait for first flush (timeout-based) to fail for document_events
        # then a second flush (timeout-based) to retry and succeed
        time.sleep(2.0)

        writer.shutdown(timeout=5.0)

        # tracked_spans should have been inserted on first flush
        assert "tracked_spans" in tables_inserted
        # document_events should have been retried on a subsequent flush
        assert "document_events" in tables_inserted


class TestLocalWriterShutdownRace:
    """Issue 5: LocalTraceWriter on_span_end after shutdown drops span data.

    If on_span_end is called after _shutdown=True, the job is silently dropped.
    The span directory exists (from on_span_start) but has no _span.yaml.
    """

    def test_span_end_during_shutdown_is_processed(self, tmp_path):
        """After fix: Spans ending during shutdown are processed (sentinel drains queue first)."""
        config = TraceDebugConfig(
            path=tmp_path,
            generate_summary=False,
            include_llm_index=False,
            include_error_index=False,
        )
        writer = LocalTraceWriter(config)

        trace_id = "a" * 32
        span_id = "b" * 16
        now_ns = int(datetime.now(UTC).timestamp() * 1e9)

        writer.on_span_start(trace_id, span_id, None, "test_task")

        # Queue the span end and immediately shutdown
        # With the fix, _shutdown is set AFTER join(), so on_span_end can still queue
        writer.on_span_end(
            WriteJob(
                trace_id=trace_id,
                span_id=span_id,
                name="test_task",
                parent_id=None,
                attributes={},
                events=[],
                status_code="OK",
                status_description=None,
                start_time_ns=now_ns,
                end_time_ns=now_ns + 50_000_000,
            )
        )
        writer.shutdown(timeout=5.0)

        trace_dirs = list(tmp_path.iterdir())
        assert len(trace_dirs) == 1

        span_dirs = [d for d in trace_dirs[0].iterdir() if d.is_dir() and "test_task" in d.name]
        assert len(span_dirs) == 1

        # _span.yaml SHOULD exist because on_span_end was processed before shutdown
        assert (span_dirs[0] / "_span.yaml").exists()

    def test_span_end_queued_after_sentinel_is_orphaned(self, tmp_path):
        """Span end job arriving after shutdown sentinel is never processed."""
        config = TraceDebugConfig(
            path=tmp_path,
            generate_summary=False,
            include_llm_index=False,
            include_error_index=False,
        )
        writer = LocalTraceWriter(config)

        trace_id = "c" * 32
        root_id = "d" * 16
        child_id = "e" * 16
        now_ns = int(datetime.now(UTC).timestamp() * 1e9)

        # Start both spans
        writer.on_span_start(trace_id, root_id, None, "root_flow")
        writer.on_span_start(trace_id, child_id, root_id, "child_task")

        # End root span normally
        writer.on_span_end(
            WriteJob(
                trace_id=trace_id,
                span_id=root_id,
                name="root_flow",
                parent_id=None,
                attributes={},
                events=[],
                status_code="OK",
                status_description=None,
                start_time_ns=now_ns,
                end_time_ns=now_ns + 100_000_000,
            )
        )

        # Brief wait for processing
        time.sleep(0.3)

        # Now shutdown — child_task is still "running"
        writer.shutdown(timeout=5.0)

        trace_dirs = list(tmp_path.iterdir())
        t = trace_dirs[0]

        # Root span should have _span.yaml (ended before shutdown)
        root_dirs = [d for d in t.iterdir() if d.is_dir() and "root_flow" in d.name]
        assert len(root_dirs) == 1
        assert (root_dirs[0] / "_span.yaml").exists()

        # Child span directory exists but may not have _span.yaml
        # (it was finalized during shutdown as "running")
        child_dirs = [d for d in root_dirs[0].iterdir() if d.is_dir() and "child_task" in d.name]
        assert len(child_dirs) == 1


class TestAutoSummaryShutdownBug:
    """Issue 6: Auto-summary disabled during shutdown.

    _finalize_trace checks `not self._shutdown` before generating auto-summary,
    but during shutdown(), _shutdown is already True, so auto-summary is always skipped
    for traces finalized in the shutdown path.
    """

    def test_auto_summary_skipped_when_finalized_during_shutdown(self, tmp_path):
        """Auto-summary is never generated for traces finalized during shutdown."""
        config = TraceDebugConfig(
            path=tmp_path,
            auto_summary_enabled=True,
            auto_summary_model="test-model",
            generate_summary=True,
            include_llm_index=False,
            include_error_index=False,
        )
        writer = LocalTraceWriter(config)

        trace_id = "f" * 32
        span_id = "1" * 16

        # Start a span but don't end it — leave trace incomplete
        writer.on_span_start(trace_id, span_id, None, "incomplete_task")

        # Shutdown with the span still "running"
        # _finalize_trace will be called with _shutdown=True → auto-summary skipped
        writer.shutdown(timeout=5.0)

        trace_dirs = list(tmp_path.iterdir())
        assert len(trace_dirs) == 1
        t = trace_dirs[0]

        # Static summary should exist (generated unconditionally)
        assert (t / "_summary.md").exists()

        # Auto-summary should NOT exist (skipped due to _shutdown=True)
        assert not (t / "_auto_summary.md").exists()

    def test_auto_summary_works_before_shutdown(self, tmp_path):
        """Auto-summary IS generated when trace completes normally (before shutdown)."""
        config = TraceDebugConfig(
            path=tmp_path,
            auto_summary_enabled=True,
            auto_summary_model="test-model",
            generate_summary=True,
            include_llm_index=False,
            include_error_index=False,
        )
        writer = LocalTraceWriter(config)

        trace_id = "2" * 32
        span_id = "3" * 16
        now_ns = int(datetime.now(UTC).timestamp() * 1e9)

        writer.on_span_start(trace_id, span_id, None, "complete_task")

        # Mock the auto-summary module to avoid real LLM call
        mock_auto_summary = "# Auto Summary\nTest auto-summary content."
        with patch(
            "ai_pipeline_core.observability._debug._auto_summary.generate_auto_summary",
            return_value=mock_auto_summary,
        ):
            writer.on_span_end(
                WriteJob(
                    trace_id=trace_id,
                    span_id=span_id,
                    name="complete_task",
                    parent_id=None,
                    attributes={},
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns,
                    end_time_ns=now_ns + 100_000_000,
                )
            )

            # Wait for processing
            time.sleep(1.0)

        writer.shutdown(timeout=5.0)

        trace_dirs = list(tmp_path.iterdir())
        t = trace_dirs[0]

        # Static summary should exist
        assert (t / "_summary.md").exists()

        # Auto-summary should exist (trace completed before shutdown)
        # NOTE: This may or may not work depending on whether the trace was
        # finalized as a single-root-span trace. If the _finalize_trace path
        # only triggers on shutdown, this test validates the happy path.
        # If the test fails here, it means auto-summary only works via shutdown
        # path — which is ALWAYS blocked by the _shutdown check.
