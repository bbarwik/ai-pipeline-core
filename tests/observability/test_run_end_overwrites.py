# pyright: reportPrivateUsage=false
"""Regression tests: track_run_end must preserve metadata, parent linkage, and costs.

Previously, track_run_end wrote a higher-version row to ReplacingMergeTree
WITHOUT metadata_json, parent_execution_id, or parent_span_id from the start row.
ClickHouse FINAL returned only the end row, losing flow_plan and parent linkage.
Cost/tokens were also lost because base.py didn't aggregate them.

Fix: ClickHouseBackend now stores start-row fields and accumulates span costs,
then automatically includes them in the end row.
"""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

_EPOCH = datetime(2024, 1, 1, tzinfo=UTC)

clickhouse_connect = pytest.importorskip("clickhouse_connect")

from ai_pipeline_core.observability._span_data import SpanData
from ai_pipeline_core.observability._tracking._backend import ClickHouseBackend
from ai_pipeline_core.observability._tracking._models import PipelineRunRow, RunStatus


def _make_backend() -> tuple[ClickHouseBackend, MagicMock]:
    mock_writer = MagicMock()
    return ClickHouseBackend(mock_writer), mock_writer


def _make_span(*, cost: float = 0.0, tokens_input: int = 0, tokens_output: int = 0) -> SpanData:
    return SpanData(
        execution_id=None,
        span_id="s1",
        trace_id="t1",
        parent_span_id=None,
        name="test",
        run_id="run-1",
        flow_name="flow",
        run_scope="scope",
        span_type="llm",
        status="completed",
        start_time=_EPOCH,
        end_time=_EPOCH,
        duration_ms=100,
        span_order=1,
        cost=cost,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        tokens_cached=0,
        llm_model="test-model",
        error_message="",
        input_json="",
        output_json="",
        replay_payload="",
        attributes={},
        events=(),
        input_doc_sha256s=(),
        output_doc_sha256s=(),
    )


SAMPLE_FLOW_PLAN = [
    {"name": "AnalyzeFlow", "class": "AnalyzeFlow", "step": 1, "estimated_minutes": 3.0, "params": {}, "expected_tasks": ["AnalyzeTask"]},
    {"name": "SynthesisFlow", "class": "SynthesisFlow", "step": 2, "estimated_minutes": 2.0, "params": {}, "expected_tasks": ["SynthesizeTask"]},
]


def test_run_end_preserves_metadata_from_start() -> None:
    """After track_run_start + track_run_end, the end row must contain the same metadata (flow_plan)."""
    backend, writer = _make_backend()
    uid = uuid4()

    start_time = backend.track_run_start(
        execution_id=uid,
        run_id="run-1",
        flow_name="TestPipeline",
        run_scope="run-1:abc123",
        metadata={"flow_plan": SAMPLE_FLOW_PLAN},
    )

    backend.track_run_end(
        execution_id=uid,
        run_id="run-1",
        flow_name="TestPipeline",
        run_scope="run-1:abc123",
        status=RunStatus.COMPLETED,
        start_time=start_time,
    )

    assert writer.write.call_count == 2
    start_row: PipelineRunRow = writer.write.call_args_list[0][0][1][0]
    end_row: PipelineRunRow = writer.write.call_args_list[1][0][1][0]

    assert end_row.version > start_row.version

    end_metadata = json.loads(end_row.metadata_json)
    assert "flow_plan" in end_metadata, f"End row metadata_json lost flow_plan: {end_row.metadata_json}"
    assert end_metadata["flow_plan"] == SAMPLE_FLOW_PLAN


def test_run_end_preserves_parent_linkage() -> None:
    """After track_run_start + track_run_end, the end row must contain parent_execution_id and parent_span_id."""
    backend, writer = _make_backend()
    uid = uuid4()
    parent_uid = uuid4()

    start_time = backend.track_run_start(
        execution_id=uid,
        run_id="run-child",
        flow_name="ChildPipeline",
        parent_execution_id=parent_uid,
        parent_span_id="parent-span-xyz",
    )

    backend.track_run_end(
        execution_id=uid,
        run_id="run-child",
        flow_name="ChildPipeline",
        status=RunStatus.COMPLETED,
        start_time=start_time,
    )

    end_row: PipelineRunRow = writer.write.call_args_list[1][0][1][0]

    assert end_row.parent_execution_id == parent_uid
    assert end_row.parent_span_id == "parent-span-xyz"


def test_run_end_aggregates_span_costs() -> None:
    """track_run_end auto-aggregates cost and tokens from spans seen during the run."""
    backend, writer = _make_backend()
    uid = uuid4()

    start_time = backend.track_run_start(
        execution_id=uid,
        run_id="run-1",
        flow_name="Pipeline",
    )

    # Simulate two LLM spans completing during the run
    backend.on_span_end(_make_span(cost=0.50, tokens_input=1000, tokens_output=200))
    backend.on_span_end(_make_span(cost=0.75, tokens_input=2000, tokens_output=300))

    # base.py calls track_run_end without cost/tokens — backend fills them in
    backend.track_run_end(
        execution_id=uid,
        run_id="run-1",
        flow_name="Pipeline",
        status=RunStatus.COMPLETED,
        start_time=start_time,
    )

    # start_row + 2 span rows + end_row = 4 writes
    end_row: PipelineRunRow = writer.write.call_args_list[-1][0][1][0]

    assert end_row.total_cost == pytest.approx(1.25)
    assert end_row.total_tokens == 3500  # (1000+200) + (2000+300)


def test_run_end_metadata_merge() -> None:
    """track_run_end metadata merges with (not replaces) start metadata."""
    backend, writer = _make_backend()
    uid = uuid4()

    start_time = backend.track_run_start(
        execution_id=uid,
        run_id="run-1",
        flow_name="Pipeline",
        metadata={"flow_plan": SAMPLE_FLOW_PLAN},
    )

    backend.track_run_end(
        execution_id=uid,
        run_id="run-1",
        flow_name="Pipeline",
        status=RunStatus.COMPLETED,
        start_time=start_time,
        metadata={"extra_key": "extra_value"},
    )

    end_row: PipelineRunRow = writer.write.call_args_list[-1][0][1][0]
    end_metadata = json.loads(end_row.metadata_json)

    assert end_metadata["flow_plan"] == SAMPLE_FLOW_PLAN
    assert end_metadata["extra_key"] == "extra_value"


def test_explicit_cost_overrides_accumulated() -> None:
    """Caller-provided cost takes precedence over auto-accumulated value."""
    backend, writer = _make_backend()
    uid = uuid4()

    start_time = backend.track_run_start(execution_id=uid, run_id="run-1", flow_name="Pipeline")
    backend.on_span_end(_make_span(cost=0.50, tokens_input=100, tokens_output=50))

    backend.track_run_end(
        execution_id=uid,
        run_id="run-1",
        flow_name="Pipeline",
        status=RunStatus.COMPLETED,
        start_time=start_time,
        total_cost=99.0,
        total_tokens=9999,
    )

    end_row: PipelineRunRow = writer.write.call_args_list[-1][0][1][0]
    assert end_row.total_cost == 99.0
    assert end_row.total_tokens == 9999
