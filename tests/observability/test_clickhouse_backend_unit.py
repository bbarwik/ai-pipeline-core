"""Unit tests for ClickHouseBackend — span handling, run tracking, version monotonicity."""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

clickhouse_connect = pytest.importorskip("clickhouse_connect")

from ai_pipeline_core.observability._span_data import SpanData
from ai_pipeline_core.observability._tracking._backend import ClickHouseBackend
from ai_pipeline_core.observability._tracking._models import (
    TABLE_PIPELINE_RUNS,
    TABLE_PIPELINE_SPANS,
    PipelineRunRow,
    PipelineSpanRow,
    RunStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPOCH = datetime(2024, 1, 1, tzinfo=UTC)
_TEST_UUID = UUID("12345678-1234-5678-1234-567812345678")


def _make_span_data(
    *,
    execution_id: UUID | None = _TEST_UUID,
    span_id: str = "span1",
    trace_id: str = "trace1",
    cost: float = 0.05,
    tokens_input: int = 100,
    span_order: int = 1,
    attributes: dict[str, object] | None = None,
) -> SpanData:
    return SpanData(
        execution_id=execution_id,
        span_id=span_id,
        trace_id=trace_id,
        parent_span_id=None,
        name="test",
        run_id="run-1",
        flow_name="flow-1",
        run_scope="scope/run-1",
        span_type="task",
        status="completed",
        start_time=_EPOCH,
        end_time=_EPOCH,
        duration_ms=500,
        span_order=span_order,
        cost=cost,
        tokens_input=tokens_input,
        tokens_output=50,
        tokens_cached=0,
        llm_model="gpt-5",
        error_message="",
        input_json='{"x": 1}',
        output_json='{"y": 2}',
        replay_payload="",
        attributes=attributes or {"custom": "val"},
        events=({"name": "evt"},),
        input_doc_sha256s=("sha_in",),
        output_doc_sha256s=("sha_out",),
    )


def _make_backend() -> tuple[ClickHouseBackend, MagicMock]:
    mock_writer = MagicMock()
    backend = ClickHouseBackend(mock_writer)
    return backend, mock_writer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOnSpanEnd:
    def test_no_execution_id_skips(self):
        backend, writer = _make_backend()
        span_data = _make_span_data(execution_id=None)
        backend.on_span_end(span_data)
        writer.write.assert_not_called()

    def test_builds_correct_row(self):
        backend, writer = _make_backend()
        span_data = _make_span_data()
        backend.on_span_end(span_data)
        writer.write.assert_called_once()
        table, rows = writer.write.call_args[0]
        assert table == TABLE_PIPELINE_SPANS
        assert len(rows) == 1
        row = rows[0]
        assert isinstance(row, PipelineSpanRow)
        assert row.execution_id == _TEST_UUID
        assert row.span_id == "span1"
        assert row.trace_id == "trace1"
        assert row.cost == 0.05
        assert row.tokens_input == 100
        assert row.span_type == "task"
        assert row.llm_model == "gpt-5"
        assert row.input_doc_sha256s == ("sha_in",)
        assert row.output_doc_sha256s == ("sha_out",)

    def test_attributes_json_serialized(self):
        backend, writer = _make_backend()
        span_data = _make_span_data(attributes={"key": "value", "nested": {"a": 1}})
        backend.on_span_end(span_data)
        row = writer.write.call_args[0][1][0]
        parsed = json.loads(row.attributes_json)
        assert parsed["key"] == "value"
        assert parsed["nested"]["a"] == 1


class TestOnSpanStart:
    def test_is_noop(self):
        backend, writer = _make_backend()
        span_data = _make_span_data()
        backend.on_span_start(span_data)
        writer.write.assert_not_called()


class TestTrackRunStart:
    def test_writes_running_row(self):
        backend, writer = _make_backend()
        uid = uuid4()
        start_time = backend.track_run_start(
            execution_id=uid,
            run_id="run-1",
            flow_name="my_flow",
            run_scope="scope/run-1",
        )
        writer.write.assert_called_once()
        table, rows = writer.write.call_args[0]
        assert table == TABLE_PIPELINE_RUNS
        row = rows[0]
        assert isinstance(row, PipelineRunRow)
        assert row.execution_id == uid
        assert row.status == RunStatus.RUNNING
        assert row.start_time == start_time


class TestTrackRunEnd:
    def test_writes_completed_row(self):
        backend, writer = _make_backend()
        uid = uuid4()
        start = datetime.now(UTC)
        backend.track_run_end(
            execution_id=uid,
            run_id="run-1",
            flow_name="my_flow",
            status=RunStatus.COMPLETED,
            start_time=start,
            total_cost=1.5,
            total_tokens=5000,
        )
        writer.write.assert_called_once()
        row = writer.write.call_args[0][1][0]
        assert row.status == RunStatus.COMPLETED
        assert row.total_cost == 1.5
        assert row.total_tokens == 5000
        assert row.end_time is not None


class TestVersionMonotonic:
    def test_each_version_greater(self):
        backend, _ = _make_backend()
        versions = [backend._next_version() for _ in range(100)]
        for i in range(1, len(versions)):
            assert versions[i] > versions[i - 1]


class TestFlushAndShutdown:
    def test_flush_delegates(self):
        backend, writer = _make_backend()
        backend.flush(timeout=5.0)
        writer.flush.assert_called_once_with(timeout=5.0)

    def test_shutdown_calls_writer_shutdown(self):
        backend, writer = _make_backend()
        backend.shutdown()
        writer.shutdown.assert_called_once()
