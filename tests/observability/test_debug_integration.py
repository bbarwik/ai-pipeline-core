"""Integration tests for the debug trace system."""

import time
from pathlib import Path

import yaml

from ai_pipeline_core.observability import (
    LocalDebugSpanProcessor,
    LocalTraceWriter,
    TraceDebugConfig,
    WriteJob,
)


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
            # 1. Flow starts
            writer.on_span_start("trace001", "flow1", None, "my_flow")

            # 2. Task starts
            writer.on_span_start("trace001", "task1", "flow1", "my_task")

            # 3. Task ends
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

            # 4. Flow ends
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

            # Wait for processing
            time.sleep(0.5)

        finally:
            writer.shutdown(timeout=10.0)

        # Verify structure
        trace_dirs = list(tmp_path.iterdir())
        assert len(trace_dirs) == 1
        trace_dir = trace_dirs[0]

        # Check all expected files exist
        assert (trace_dir / "_trace.yaml").exists()
        assert (trace_dir / "_tree.yaml").exists()  # V3 split index
        assert (trace_dir / "_llm_calls.yaml").exists()  # V3 LLM index
        assert (trace_dir / "_summary.md").exists()

        # Verify _trace.yaml content
        trace_meta = yaml.safe_load((trace_dir / "_trace.yaml").read_text())
        assert trace_meta["name"] == "my_flow"
        assert trace_meta["status"] == "completed"
        assert trace_meta["stats"]["total_spans"] == 2

        # Verify _tree.yaml content (V3)
        tree = yaml.safe_load((trace_dir / "_tree.yaml").read_text())
        assert tree["format_version"] == 3
        assert tree["span_count"] == 2
        assert tree["root_span_id"] == "flow1"

        # Verify hierarchy
        flow_entry = next(s for s in tree["tree"] if s["span_id"] == "flow1")
        assert "task1" in flow_entry["children"]

        # Check hierarchical span directories using span_paths
        flow_path = trace_dir / tree["span_paths"]["flow1"].rstrip("/")
        task_path = trace_dir / tree["span_paths"]["task1"].rstrip("/")
        assert (flow_path / "_span.yaml").exists()
        assert (task_path / "_span.yaml").exists()

        # Verify task is nested under flow (hierarchical structure)
        assert task_path.parent == flow_path

    def test_flow_with_llm_call(self, tmp_path: Path) -> None:
        """Test flow with LLM call captures token/cost info."""
        config = TraceDebugConfig(path=tmp_path)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)

            # Flow
            writer.on_span_start("trace001", "flow1", None, "analysis_flow")

            # LLM call
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

            # End flow
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

        # Verify LLM info in trace
        trace_dir = list(tmp_path.iterdir())[0]
        trace_meta = yaml.safe_load((trace_dir / "_trace.yaml").read_text())

        assert trace_meta["stats"]["llm_calls"] == 1
        assert trace_meta["stats"]["total_tokens"] == 6000
        assert trace_meta["stats"]["total_cost"] == 0.15

        # Verify LLM span metadata using index
        index = yaml.safe_load((trace_dir / "_tree.yaml").read_text())
        llm_path = trace_dir / index["span_paths"]["llm1"].rstrip("/")
        llm_span = yaml.safe_load((llm_path / "_span.yaml").read_text())
        assert llm_span["type"] == "llm"
        assert llm_span["llm"]["model"] == "gpt-5.1"
        assert llm_span["llm"]["input_tokens"] == 5000

        # Check summary mentions LLM
        summary = (trace_dir / "_summary.md").read_text()
        assert "gpt-5.1" in summary
        assert "6,000" in summary or "6000" in summary  # Token count

    def test_failed_span_captured(self, tmp_path: Path) -> None:
        """Test failed spans are properly captured."""
        config = TraceDebugConfig(path=tmp_path)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)

            writer.on_span_start("trace001", "flow1", None, "failing_flow")
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
                    status_description="ValueError: Something went wrong",
                    start_time_ns=now_ns - 500_000_000,
                    end_time_ns=now_ns,
                )
            )

            # Flow ends with error
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

        # Verify error captured
        trace_dir = list(tmp_path.iterdir())[0]
        trace_meta = yaml.safe_load((trace_dir / "_trace.yaml").read_text())
        assert trace_meta["status"] == "failed"

        # Check span has error using index
        index = yaml.safe_load((trace_dir / "_tree.yaml").read_text())
        task_path = trace_dir / index["span_paths"]["task1"].rstrip("/")
        task_span = yaml.safe_load((task_path / "_span.yaml").read_text())
        assert task_span["status"] == "failed"
        assert "error" in task_span
        assert "Something went wrong" in task_span["error"]["message"]

        # Check summary mentions error
        summary = (trace_dir / "_summary.md").read_text()
        assert "❌" in summary or "Failed" in summary

    def test_concurrent_spans(self, tmp_path: Path) -> None:
        """Test concurrent spans under same parent are handled correctly."""
        config = TraceDebugConfig(path=tmp_path)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)

            # Start parent
            writer.on_span_start("trace001", "parent", None, "parent_flow")

            # Start multiple children concurrently
            for i in range(10):
                writer.on_span_start("trace001", f"child{i}", "parent", f"child_task_{i}")

            # End children
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

            # End parent
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

            time.sleep(1.0)  # More time for concurrent processing

        finally:
            writer.shutdown(timeout=10.0)

        # Verify all spans captured
        trace_dir = list(tmp_path.iterdir())[0]
        index = yaml.safe_load((trace_dir / "_tree.yaml").read_text())

        assert index["span_count"] == 11  # parent + 10 children

        # Check all children in parent's children list
        parent_entry = next(s for s in index["tree"] if s["span_id"] == "parent")
        assert len(parent_entry["children"]) == 10


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

        trace_dir = list(tmp_path.iterdir())[0]
        summary = (trace_dir / "_summary.md").read_text()

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

        trace_dir = list(tmp_path.iterdir())[0]
        summary = (trace_dir / "_summary.md").read_text()

        # Check all essential sections
        assert "Trace Summary:" in summary
        assert "Execution Tree" in summary
        assert "Root Span" in summary
        assert "Navigation" in summary
        assert "debug_flow" in summary
