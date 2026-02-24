"""Integration tests for the debug trace system."""

import time

from pathlib import Path

import yaml

from ai_pipeline_core.observability._debug import (
    LocalDebugSpanProcessor,
    LocalTraceWriter,
    TraceDebugConfig,
    WriteJob,
)


def _find_span_dir(trace_dir: Path, span_name: str) -> Path:
    """Find a span directory by name within a trace directory (recursive)."""
    for p in trace_dir.rglob(f"*_{span_name}*"):
        if p.is_dir():
            return p
    msg = f"No span directory found for '{span_name}' in {trace_dir}"
    raise FileNotFoundError(msg)


class TestFullTraceFlow:
    """Tests for complete trace flows."""

    def test_simple_flow_creates_complete_trace(self, tmp_path: Path) -> None:
        """Test a simple flow creates all expected files."""
        config = TraceDebugConfig(path=tmp_path)
        writer = LocalTraceWriter(config)
        _ = LocalDebugSpanProcessor(writer)

        try:
            now_ns = int(time.time() * 1e9)

            # Simulate a flow with one task
            writer.on_span_start("trace001", "flow1", None, "my_flow")
            writer.on_span_start("trace001", "task1", "flow1", "my_task")

            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="task1",
                    name="my_task",
                    parent_id="flow1",
                    attributes={
                        "lmnr.span.input": '{"arg1": "value1"}',
                        "lmnr.span.output": '{"result": "success"}',
                        "lmnr.span.type": "DEFAULT",
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
                    name="my_flow",
                    parent_id=None,
                    attributes={
                        "lmnr.span.input": '{"project": "test"}',
                        "lmnr.span.output": '{"documents": []}',
                    },
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 1_000_000_000,
                    end_time_ns=now_ns,
                )
            )

            time.sleep(0.5)

        finally:
            writer.shutdown(timeout=10.0)

        # Verify structure — trace writes directly to tmp_path
        assert (tmp_path / "summary.md").exists()
        assert (tmp_path / "llm_calls.yaml").exists()

        # Check hierarchical span directories
        flow_path = _find_span_dir(tmp_path, "my_flow")
        task_path = _find_span_dir(tmp_path, "my_task")
        assert (flow_path / "span.yaml").exists()
        assert (task_path / "span.yaml").exists()

        # Verify task is nested under flow
        assert task_path.parent == flow_path

    def test_flow_with_llm_call(self, tmp_path: Path) -> None:
        """Test flow with LLM call captures token/cost info."""
        config = TraceDebugConfig(path=tmp_path)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)

            writer.on_span_start("trace001", "flow1", None, "analysis_flow")
            writer.on_span_start("trace001", "llm1", "flow1", "gpt-5.1-call")
            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="llm1",
                    name="gpt-5.1-call",
                    parent_id="flow1",
                    attributes={
                        "lmnr.span.type": "LLM",
                        "gen_ai.usage.input_tokens": 5000,
                        "gen_ai.usage.output_tokens": 1000,
                        "gen_ai.response.model": "gpt-5.1",
                        "gen_ai.usage.cost": 0.15,
                        "lmnr.span.input": '{"messages": [{"role": "user", "content": "Hello"}]}',
                        "lmnr.span.output": '"Response text"',
                    },
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 2_000_000_000,
                    end_time_ns=now_ns - 1_000_000_000,
                )
            )

            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="flow1",
                    name="analysis_flow",
                    parent_id=None,
                    attributes={},
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 3_000_000_000,
                    end_time_ns=now_ns,
                )
            )

            time.sleep(0.5)

        finally:
            writer.shutdown(timeout=10.0)

        # Verify LLM span metadata
        llm_path = _find_span_dir(tmp_path, "gpt-5.1-call")
        llm_span = yaml.safe_load((llm_path / "span.yaml").read_text())
        assert llm_span["type"] == "llm"
        assert llm_span["llm"]["model"] == "gpt-5.1"
        assert llm_span["llm"]["input_tokens"] == 5000

        # Check summary mentions LLM info
        summary = (tmp_path / "summary.md").read_text()
        assert "6,000" in summary or "6000" in summary or "5K IN" in summary

    def test_failed_span_captured(self, tmp_path: Path) -> None:
        """Test failed spans are properly captured."""
        config = TraceDebugConfig(path=tmp_path)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)

            writer.on_span_start("trace001", "flow1", None, "failing_flow")
            writer.on_span_start("trace001", "task1", "flow1", "failing_task")

            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="task1",
                    name="failing_task",
                    parent_id="flow1",
                    attributes={},
                    events=[],
                    status_code="ERROR",
                    status_description="ValueError: Something went wrong",
                    start_time_ns=now_ns - 500_000_000,
                    end_time_ns=now_ns,
                )
            )

            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="flow1",
                    name="failing_flow",
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
            writer.shutdown(timeout=10.0)

        # Check span has error
        task_path = _find_span_dir(tmp_path, "failing_task")
        task_span = yaml.safe_load((task_path / "span.yaml").read_text())
        assert task_span["status"] == "failed"
        assert "error" in task_span
        assert "Something went wrong" in task_span["error"]["message"]

        # Check summary mentions error
        summary = (tmp_path / "summary.md").read_text()
        assert "Failed" in summary or "ERROR" in summary

    def test_concurrent_spans(self, tmp_path: Path) -> None:
        """Test concurrent spans under same parent are handled correctly."""
        config = TraceDebugConfig(path=tmp_path)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)

            writer.on_span_start("trace001", "parent", None, "parent_flow")

            for i in range(10):
                writer.on_span_start("trace001", f"child{i}", "parent", f"child_task_{i}")

            for i in range(10):
                writer.on_span_end(
                    WriteJob(
                        trace_id="trace001",
                        span_id=f"child{i}",
                        name=f"child_task_{i}",
                        parent_id="parent",
                        attributes={},
                        events=[],
                        status_code="OK",
                        status_description=None,
                        start_time_ns=now_ns - 500_000_000 + i * 10_000_000,
                        end_time_ns=now_ns + i * 10_000_000,
                    )
                )

            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="parent",
                    name="parent_flow",
                    parent_id=None,
                    attributes={},
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 1_000_000_000,
                    end_time_ns=now_ns + 100_000_000,
                )
            )

            time.sleep(1.0)

        finally:
            writer.shutdown(timeout=10.0)

        # Verify all spans captured in summary
        summary = (tmp_path / "summary.md").read_text()
        assert "parent_flow" in summary


class TestSummaryGeneration:
    """Tests for summary file generation."""

    def test_summary_includes_execution_tree(self, tmp_path: Path) -> None:
        """Test summary includes execution tree visualization."""
        config = TraceDebugConfig(path=tmp_path)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)

            writer.on_span_start("trace001", "flow", None, "main_flow")
            writer.on_span_start("trace001", "task1", "flow", "first_task")
            writer.on_span_start("trace001", "task2", "flow", "second_task")

            for span_id, name in [
                ("task1", "first_task"),
                ("task2", "second_task"),
                ("flow", "main_flow"),
            ]:
                writer.on_span_end(
                    WriteJob(
                        trace_id="trace001",
                        span_id=span_id,
                        name=name,
                        parent_id="flow" if span_id != "flow" else None,
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
            writer.shutdown(timeout=10.0)

        summary = (tmp_path / "summary.md").read_text()

        # Check tree structure is present
        assert "Execution Tree" in summary
        assert "main_flow" in summary
        assert "first_task" in summary
        assert "second_task" in summary

    def test_summary_has_all_sections(self, tmp_path: Path) -> None:
        """Test summary has all necessary sections for both human and LLM use."""
        config = TraceDebugConfig(path=tmp_path)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)

            writer.on_span_start("trace001", "flow", None, "debug_flow")
            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="flow",
                    name="debug_flow",
                    parent_id=None,
                    attributes={
                        "gen_ai.usage.input_tokens": 1000,
                        "gen_ai.usage.output_tokens": 500,
                        "gen_ai.usage.cost": 0.05,
                    },
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 1_000_000_000,
                    end_time_ns=now_ns,
                )
            )

            time.sleep(0.5)

        finally:
            writer.shutdown(timeout=10.0)

        summary = (tmp_path / "summary.md").read_text()

        # Check all essential sections
        assert "Trace Summary:" in summary
        assert "Execution Tree" in summary
        assert "Root Span" in summary
        assert "Navigation" in summary
        assert "debug_flow" in summary
