"""Tests for ClickHouse tracking models."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from ai_pipeline_core.observability._tracking._models import (
    DocumentEventRow,
    DocumentEventType,
    PipelineRunRow,
    RunStatus,
    SpanEventRow,
    SpanType,
    TrackedSpanRow,
)


class TestEnums:
    """Test enum definitions."""

    def test_run_status_values(self):
        assert RunStatus.RUNNING == "running"
        assert RunStatus.COMPLETED == "completed"
        assert RunStatus.FAILED == "failed"

    def test_span_type_values(self):
        assert SpanType.TASK == "task"
        assert SpanType.FLOW == "flow"
        assert SpanType.LLM == "llm"
        assert SpanType.TRACE == "trace"

    def test_document_event_type_has_llm_message(self):
        assert DocumentEventType.LLM_MESSAGE == "llm_message"
        assert DocumentEventType.TASK_INPUT == "task_input"
        assert DocumentEventType.TASK_OUTPUT == "task_output"
        assert DocumentEventType.LLM_CONTEXT == "llm_context"
        assert DocumentEventType.STORE_SAVED == "store_saved"
        assert DocumentEventType.STORE_SAVE_FAILED == "store_save_failed"


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
        assert row.metadata == "{}"
        assert row.version == 1

    def test_tracked_span_row_defaults(self):
        row = TrackedSpanRow(
            span_id="abcdef1234567890",
            trace_id="0" * 32,
            execution_id=uuid4(),
            name="my_task",
            span_type=SpanType.TASK,
            status="running",
            start_time=datetime.now(UTC),
        )
        assert row.input_document_sha256s == ()
        assert row.output_document_sha256s == ()

    def test_tracked_span_row_mutable_defaults_safe(self):
        """Verify that list defaults don't share state between instances."""
        row1 = TrackedSpanRow(
            span_id="a" * 16,
            trace_id="0" * 32,
            execution_id=uuid4(),
            name="t1",
            span_type=SpanType.TASK,
            status="running",
            start_time=datetime.now(UTC),
        )
        row2 = TrackedSpanRow(
            span_id="b" * 16,
            trace_id="0" * 32,
            execution_id=uuid4(),
            name="t2",
            span_type=SpanType.TASK,
            status="running",
            start_time=datetime.now(UTC),
        )
        assert row1.input_document_sha256s == row2.input_document_sha256s == ()

    def test_document_event_row(self):
        row = DocumentEventRow(
            event_id=uuid4(),
            execution_id=uuid4(),
            document_sha256="a" * 64,
            span_id="b" * 16,
            event_type=DocumentEventType.TASK_INPUT,
            timestamp=datetime.now(UTC),
        )
        assert row.metadata == "{}"

    def test_span_event_row(self):
        row = SpanEventRow(
            event_id=uuid4(),
            execution_id=uuid4(),
            span_id="c" * 16,
            name="log",
            timestamp=datetime.now(UTC),
        )
        assert row.level is None
        assert row.attributes == "{}"

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

    def test_model_copy_with_update(self):
        """Test ReplacingMergeTree update pattern via model_copy."""
        row = TrackedSpanRow(
            span_id="a" * 16,
            trace_id="0" * 32,
            execution_id=uuid4(),
            name="task",
            span_type=SpanType.TASK,
            status="completed",
            start_time=datetime.now(UTC),
            version=1,
        )
        updated = row.model_copy(update={"status": "failed", "version": 2})
        assert updated.status == "failed"
        assert updated.version == 2
        assert row.status == "completed"  # original unchanged
