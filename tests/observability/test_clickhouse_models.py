"""Tests for ClickHouse tracking models."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from ai_pipeline_core.observability._tracking._models import (
    PipelineRunRow,
    PipelineSpanRow,
    RunStatus,
)


class TestEnums:
    """Test enum definitions."""

    def test_run_status_values(self):
        assert RunStatus.RUNNING == "running"
        assert RunStatus.COMPLETED == "completed"
        assert RunStatus.FAILED == "failed"


class TestRowModels:
    """Test Pydantic row models."""

    def test_pipeline_run_row_defaults(self):
        row = PipelineRunRow(
            execution_id=uuid4(),
            run_id="test",
            flow_name="my_flow",
            status=RunStatus.RUNNING,
            start_time=datetime.now(UTC),
        )
        assert row.total_cost == pytest.approx(0.0)
        assert row.total_tokens == 0
        assert row.metadata_json == "{}"
        assert row.version == 1

    def test_pipeline_span_row_defaults(self):
        row = PipelineSpanRow(
            execution_id=uuid4(),
            span_id="abcdef1234567890",
            trace_id="0" * 32,
            name="my_task",
            span_type="task",
            status="completed",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
        )
        assert row.input_doc_sha256s == ()
        assert row.output_doc_sha256s == ()
        assert row.cost == 0.0
        assert row.tokens_cached == 0

    def test_pipeline_span_row_tuple_defaults_safe(self):
        """Verify that tuple defaults don't share state between instances."""
        row1 = PipelineSpanRow(
            execution_id=uuid4(),
            span_id="a" * 16,
            trace_id="0" * 32,
            name="t1",
            span_type="task",
            status="completed",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
        )
        row2 = PipelineSpanRow(
            execution_id=uuid4(),
            span_id="b" * 16,
            trace_id="0" * 32,
            name="t2",
            span_type="task",
            status="completed",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
        )
        assert row1.input_doc_sha256s == row2.input_doc_sha256s == ()

    def test_row_models_are_frozen(self):
        row = PipelineRunRow(
            execution_id=uuid4(),
            run_id="test",
            flow_name="flow",
            status=RunStatus.RUNNING,
            start_time=datetime.now(UTC),
        )
        with pytest.raises(ValidationError):
            row.run_id = "changed"  # type: ignore[misc]
