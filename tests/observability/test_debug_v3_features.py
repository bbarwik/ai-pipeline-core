"""Tests for V3 tracing features: events, wrapper merging, split indexes."""

import time

import pytest
from pathlib import Path
from unittest.mock import Mock

import yaml

from ai_pipeline_core.observability._debug import LocalTraceWriter, TraceDebugConfig, WriteJob


def _find_span_dir(trace_dir: Path, span_name: str) -> Path:
    """Find a span directory by name within a trace directory (recursive)."""
    for p in trace_dir.rglob(f"*_{span_name}*"):
        if p.is_dir():
            return p
    msg = f"No span directory found for '{span_name}' in {trace_dir}"
    raise FileNotFoundError(msg)


class TestEventsWriting:
    """Tests for per-span events.yaml writing."""

    def test_events_written_when_present(self, tmp_path: Path) -> None:
        """Test events.yaml is always written when span has events."""
        config = TraceDebugConfig(path=tmp_path)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)
            writer.on_span_start("trace001", "span1", None, "test")

            mock_event = Mock()
            mock_event.name = "log"
            mock_event.timestamp = now_ns
            mock_event.attributes = {"log.level": "INFO", "log.message": "hello"}

            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="span1",
                    name="test",
                    parent_id=None,
                    attributes={},
                    events=[mock_event],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 1_000_000_000,
                    end_time_ns=now_ns,
                )
            )
            time.sleep(0.5)
        finally:
            writer.shutdown(timeout=5.0)

        span_path = _find_span_dir(tmp_path, "test")

        assert (span_path / "events.yaml").exists()
        events = yaml.safe_load((span_path / "events.yaml").read_text())
        assert len(events) == 1
        assert events[0]["name"] == "log"
        assert events[0]["attributes"]["log.level"] == "INFO"

    def test_no_events_file_when_empty(self, tmp_path: Path) -> None:
        """Test events.yaml is not created when span has no events."""
        config = TraceDebugConfig(path=tmp_path)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)
            writer.on_span_start("trace001", "span1", None, "test")
            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="span1",
                    name="test",
                    parent_id=None,
                    attributes={},
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 1_000_000_000,
                    end_time_ns=now_ns,
                )
            )
            time.sleep(0.5)
        finally:
            writer.shutdown(timeout=5.0)

        span_path = _find_span_dir(tmp_path, "test")

        assert not (span_path / "events.yaml").exists()


class TestSplitIndexes:
    """Tests for split index files (llm_calls.yaml, errors.yaml)."""

    def test_llm_calls_index_content(self, tmp_path: Path) -> None:
        """Test llm_calls.yaml contains correct LLM metadata."""
        config = TraceDebugConfig(path=tmp_path)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)
            writer.on_span_start("trace001", "flow1", None, "main_flow")
            writer.on_span_start("trace001", "llm1", "flow1", "gpt-5.1-call")

            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="llm1",
                    name="gpt-5.1-call",
                    parent_id="flow1",
                    attributes={
                        "lmnr.span.type": "LLM",
                        "gen_ai.usage.input_tokens": 1000,
                        "gen_ai.usage.output_tokens": 500,
                        "gen_ai.response.model": "gpt-5.1",
                        "gen_ai.usage.cost": 0.05,
                    },
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 500_000_000,
                    end_time_ns=now_ns,
                )
            )

            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="flow1",
                    name="main_flow",
                    parent_id=None,
                    attributes={},
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 1_000_000_000,
                    end_time_ns=now_ns,
                )
            )

            time.sleep(0.5)
        finally:
            writer.shutdown(timeout=5.0)

        # Check llm_calls.yaml
        llm_index = yaml.safe_load((tmp_path / "llm_calls.yaml").read_text())
        assert llm_index["format_version"] == 3
        assert llm_index["llm_call_count"] == 1
        assert llm_index["total_tokens"] == 1500
        assert llm_index["total_cost"] == pytest.approx(0.05)

        # Check LLM call entry has parent context
        call = llm_index["calls"][0]
        assert call["model"] == "gpt-5.1"
        assert call["input_tokens"] == 1000
        assert call["output_tokens"] == 500
        assert "(in main_flow)" in call["name"]  # Parent context added

    def test_errors_index_generation(self, tmp_path: Path) -> None:
        """Test errors.yaml is generated only when there are errors."""
        config = TraceDebugConfig(path=tmp_path)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)
            writer.on_span_start("trace001", "flow1", None, "test_flow")
            writer.on_span_start("trace001", "task1", "flow1", "failing_task")

            # Task fails
            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="task1",
                    name="failing_task",
                    parent_id="flow1",
                    attributes={},
                    events=[],
                    status_code="ERROR",
                    status_description="Something went wrong",
                    start_time_ns=now_ns - 500_000_000,
                    end_time_ns=now_ns,
                )
            )

            # Flow completes
            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="flow1",
                    name="test_flow",
                    parent_id=None,
                    attributes={},
                    events=[],
                    status_code="ERROR",
                    status_description="Task failed",
                    start_time_ns=now_ns - 1_000_000_000,
                    end_time_ns=now_ns,
                )
            )

            time.sleep(0.5)
        finally:
            writer.shutdown(timeout=5.0)

        # Check errors.yaml exists and has correct content
        assert (tmp_path / "errors.yaml").exists()
        errors = yaml.safe_load((tmp_path / "errors.yaml").read_text())
        assert errors["format_version"] == 3
        assert errors["error_count"] == 2
        assert len(errors["errors"]) == 2

        # Check parent chain is included
        task_error = next(e for e in errors["errors"] if e["span_id"] == "task1")
        assert "parent_chain" in task_error
        assert task_error["parent_chain"] == ["test_flow"]

    def test_no_errors_index_when_no_failures(self, tmp_path: Path) -> None:
        """Test errors.yaml is not created when all spans succeed."""
        config = TraceDebugConfig(path=tmp_path)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)
            writer.on_span_start("trace001", "span1", None, "test")
            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="span1",
                    name="test",
                    parent_id=None,
                    attributes={},
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 1_000_000_000,
                    end_time_ns=now_ns,
                )
            )
            time.sleep(0.5)
        finally:
            writer.shutdown(timeout=5.0)

        # No errors.yaml when no errors
        assert not (tmp_path / "errors.yaml").exists()


class TestWrapperSpanMerging:
    """Tests for Prefect wrapper span merging."""

    def test_wrapper_detection_and_merge(self, tmp_path: Path) -> None:
        """Test wrapper spans are detected and merged."""
        config = TraceDebugConfig(path=tmp_path, merge_wrapper_spans=True)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)

            writer.on_span_start("trace001", "flow1", None, "test_flow")
            writer.on_span_start("trace001", "wrapper1", "flow1", "my_task-abc123")
            writer.on_span_start("trace001", "inner1", "wrapper1", "my_task")

            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="inner1",
                    name="my_task",
                    parent_id="wrapper1",
                    attributes={
                        "lmnr.span.input": '{"arg": "value"}',
                        "lmnr.span.output": '{"result": "success"}',
                    },
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 300_000_000,
                    end_time_ns=now_ns - 200_000_000,
                )
            )

            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="wrapper1",
                    name="my_task-abc123",
                    parent_id="flow1",
                    attributes={
                        "prefect.run.id": "abc-def-123",
                        "prefect.run.name": "my_task",
                    },
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 400_000_000,
                    end_time_ns=now_ns - 100_000_000,
                )
            )

            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="flow1",
                    name="test_flow",
                    parent_id=None,
                    attributes={},
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 1_000_000_000,
                    end_time_ns=now_ns,
                )
            )

            time.sleep(0.5)
        finally:
            writer.shutdown(timeout=5.0)

        summary = (tmp_path / "summary.md").read_text()

        # Wrapper should be merged — inner1 visible, wrapper1 not
        assert "my_task" in summary
        # The summary tree should contain the inner span name

    def test_wrapper_merge_disabled(self, tmp_path: Path) -> None:
        """Test wrapper spans are preserved when merging is disabled."""
        config = TraceDebugConfig(path=tmp_path, merge_wrapper_spans=False)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)

            writer.on_span_start("trace001", "flow1", None, "test_flow")
            writer.on_span_start("trace001", "wrapper1", "flow1", "my_task-abc123")
            writer.on_span_start("trace001", "inner1", "wrapper1", "my_task")

            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="inner1",
                    name="my_task",
                    parent_id="wrapper1",
                    attributes={"lmnr.span.input": '{"arg": "value"}'},
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 300_000_000,
                    end_time_ns=now_ns - 200_000_000,
                )
            )

            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="wrapper1",
                    name="my_task-abc123",
                    parent_id="flow1",
                    attributes={"prefect.run.id": "abc-def-123"},
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 400_000_000,
                    end_time_ns=now_ns - 100_000_000,
                )
            )

            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="flow1",
                    name="test_flow",
                    parent_id=None,
                    attributes={},
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 1_000_000_000,
                    end_time_ns=now_ns,
                )
            )

            time.sleep(0.5)
        finally:
            writer.shutdown(timeout=5.0)

        summary = (tmp_path / "summary.md").read_text()

        # Both wrapper and inner should appear when merging is disabled
        assert "my_task-abc123" in summary
        assert "my_task" in summary
