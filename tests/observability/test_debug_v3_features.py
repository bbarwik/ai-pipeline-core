"""Tests for V3 tracing features: events, wrapper merging, split indexes."""

import time
from pathlib import Path
from unittest.mock import Mock

import yaml

from ai_pipeline_core.observability import LocalTraceWriter, TraceDebugConfig, WriteJob


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

        trace_dir = list(tmp_path.iterdir())[0]
        tree = yaml.safe_load((trace_dir / "_tree.yaml").read_text())
        span_path = trace_dir / tree["span_paths"]["span1"].rstrip("/")

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

        trace_dir = list(tmp_path.iterdir())[0]
        tree = yaml.safe_load((trace_dir / "_tree.yaml").read_text())
        span_path = trace_dir / tree["span_paths"]["span1"].rstrip("/")

        assert not (span_path / "events.yaml").exists()


class TestSplitIndexes:
    """Tests for split index files (_tree.yaml, _llm_calls.yaml, _errors.yaml)."""

    def test_llm_calls_index_content(self, tmp_path: Path) -> None:
        """Test _llm_calls.yaml contains correct LLM metadata."""
        config = TraceDebugConfig(path=tmp_path, include_llm_index=True)
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

        trace_dirs = list(tmp_path.iterdir())
        trace_dir = trace_dirs[0]

        # Check _llm_calls.yaml
        llm_index = yaml.safe_load((trace_dir / "_llm_calls.yaml").read_text())
        assert llm_index["format_version"] == 3
        assert llm_index["llm_call_count"] == 1
        assert llm_index["total_tokens"] == 1500
        assert llm_index["total_cost"] == 0.05

        # Check LLM call entry has parent context
        call = llm_index["calls"][0]
        assert call["model"] == "gpt-5.1"
        assert call["input_tokens"] == 1000
        assert call["output_tokens"] == 500
        assert "(in main_flow)" in call["name"]  # Parent context added

    def test_errors_index_generation(self, tmp_path: Path) -> None:
        """Test _errors.yaml is generated only when there are errors."""
        config = TraceDebugConfig(path=tmp_path, include_error_index=True)
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

        trace_dirs = list(tmp_path.iterdir())
        trace_dir = trace_dirs[0]

        # Check _errors.yaml exists and has correct content
        assert (trace_dir / "_errors.yaml").exists()
        errors = yaml.safe_load((trace_dir / "_errors.yaml").read_text())
        assert errors["format_version"] == 3
        assert errors["error_count"] == 2
        assert len(errors["errors"]) == 2

        # Check parent chain is included
        task_error = next(e for e in errors["errors"] if e["span_id"] == "task1")
        assert "parent_chain" in task_error
        assert task_error["parent_chain"] == ["test_flow"]

    def test_no_errors_index_when_no_failures(self, tmp_path: Path) -> None:
        """Test _errors.yaml is not created when all spans succeed."""
        config = TraceDebugConfig(path=tmp_path, include_error_index=True)
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

        trace_dirs = list(tmp_path.iterdir())
        trace_dir = trace_dirs[0]

        # No _errors.yaml when no errors
        assert not (trace_dir / "_errors.yaml").exists()


class TestTreeConsistency:
    """Tests for tree structural invariants."""

    def test_span_count_matches_tree_entries(self, tmp_path: Path) -> None:
        """Test span_count always equals actual tree entries after merge."""
        config = TraceDebugConfig(path=tmp_path, merge_wrapper_spans=True)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)

            # Create structure with wrapper that will be merged
            writer.on_span_start("trace001", "flow1", None, "main_flow")
            writer.on_span_start("trace001", "wrapper1", "flow1", "task-abc123")
            writer.on_span_start("trace001", "inner1", "wrapper1", "task")

            # End spans
            for span_id, name, parent in [
                ("inner1", "task", "wrapper1"),
                ("wrapper1", "task-abc123", "flow1"),
                ("flow1", "main_flow", None),
            ]:
                writer.on_span_end(
                    WriteJob(
                        trace_id="trace001",
                        span_id=span_id,
                        name=name,
                        parent_id=parent,
                        attributes={"prefect.run.id": "test"} if span_id == "wrapper1" else {},
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

        trace_dir = list(tmp_path.iterdir())[0]
        tree = yaml.safe_load((trace_dir / "_tree.yaml").read_text())

        # Critical invariants: span_count must match actual entries
        assert tree["span_count"] == len(tree["tree"])
        assert tree["span_count"] == len(tree["span_paths"])
        assert tree["span_count"] == 2  # flow1 + inner1 (wrapper merged)

        # Verify all span_ids in tree exist in span_paths
        tree_ids = {entry["span_id"] for entry in tree["tree"]}
        assert tree_ids == set(tree["span_paths"].keys())

        # Verify wrapper was merged out
        assert "wrapper1" not in tree_ids
        assert "inner1" in tree_ids
        assert "flow1" in tree_ids

    def test_span_paths_consistency(self, tmp_path: Path) -> None:
        """Test span_paths only includes visible spans."""
        config = TraceDebugConfig(path=tmp_path, merge_wrapper_spans=True)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)

            writer.on_span_start("trace001", "root", None, "root")
            writer.on_span_start("trace001", "w1", "root", "task-abc")
            writer.on_span_start("trace001", "i1", "w1", "task")
            writer.on_span_start("trace001", "w2", "root", "other-def")
            writer.on_span_start("trace001", "i2", "w2", "other")

            for span_id, name, parent, has_prefect in [
                ("i1", "task", "w1", False),
                ("w1", "task-abc", "root", True),
                ("i2", "other", "w2", False),
                ("w2", "other-def", "root", True),
                ("root", "root", None, False),
            ]:
                writer.on_span_end(
                    WriteJob(
                        trace_id="trace001",
                        span_id=span_id,
                        name=name,
                        parent_id=parent,
                        attributes={"prefect.run.id": "x"} if has_prefect else {},
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

        trace_dir = list(tmp_path.iterdir())[0]
        tree = yaml.safe_load((trace_dir / "_tree.yaml").read_text())

        # Both wrappers should be merged
        assert tree["span_count"] == 3  # root + i1 + i2
        assert len(tree["span_paths"]) == 3
        assert set(tree["span_paths"].keys()) == {"root", "i1", "i2"}


class TestWrapperSpanMerging:
    """Tests for Prefect wrapper span merging."""

    def test_childless_prefect_leaf_preserved(self, tmp_path: Path) -> None:
        """Test legitimate Prefect leaf task is NOT incorrectly merged."""
        config = TraceDebugConfig(path=tmp_path, merge_wrapper_spans=True)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)

            # Create flow with a wrapper+inner pair AND a legitimate leaf
            writer.on_span_start("trace001", "flow1", None, "main_flow")

            # This will be merged (wrapper pattern)
            writer.on_span_start("trace001", "wrapper1", "flow1", "task-abc123")
            writer.on_span_start("trace001", "inner1", "wrapper1", "task")

            # This is a legitimate Prefect leaf - should NOT be merged
            writer.on_span_start("trace001", "leaf_task", "flow1", "cleanup")

            # End spans
            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="inner1",
                    name="task",
                    parent_id="wrapper1",
                    attributes={"lmnr.span.input": '{"data": "test"}'},
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 2_000_000_000,
                    end_time_ns=now_ns - 1_500_000_000,
                )
            )

            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="wrapper1",
                    name="task-abc123",
                    parent_id="flow1",
                    attributes={"prefect.run.id": "run-123"},
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 2_500_000_000,
                    end_time_ns=now_ns - 1_000_000_000,
                )
            )

            # Leaf task: has prefect info, no children, but HAS I/O
            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="leaf_task",
                    name="cleanup",
                    parent_id="flow1",
                    attributes={
                        "prefect.run.id": "run-456",
                        "lmnr.span.output": '{"cleaned": true}',
                    },
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 1_000_000_000,
                    end_time_ns=now_ns - 500_000_000,
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
                    start_time_ns=now_ns - 3_000_000_000,
                    end_time_ns=now_ns,
                )
            )

            time.sleep(0.5)
        finally:
            writer.shutdown(timeout=5.0)

        trace_dir = list(tmp_path.iterdir())[0]
        tree = yaml.safe_load((trace_dir / "_tree.yaml").read_text())

        # Verify: wrapper merged, but leaf_task preserved
        span_ids = {entry["span_id"] for entry in tree["tree"]}
        assert "wrapper1" not in span_ids  # Wrapper merged
        assert "inner1" in span_ids  # Inner preserved
        assert "leaf_task" in span_ids  # Leaf NOT dropped (this is the bug fix)
        assert "flow1" in span_ids

        # Verify counts
        assert tree["span_count"] == 3  # flow1 + inner1 + leaf_task

    def test_root_wrapper_merge(self, tmp_path: Path) -> None:
        """Test merging when wrapper is the root span."""
        config = TraceDebugConfig(path=tmp_path, merge_wrapper_spans=True)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)

            # Root is a wrapper wrapping the actual task
            writer.on_span_start("trace001", "wrapper_root", None, "task-abc123")
            writer.on_span_start("trace001", "inner", "wrapper_root", "task")

            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="inner",
                    name="task",
                    parent_id="wrapper_root",
                    attributes={"lmnr.span.input": '{"x": 1}'},
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 1_000_000_000,
                    end_time_ns=now_ns - 500_000_000,
                )
            )

            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="wrapper_root",
                    name="task-abc123",
                    parent_id=None,
                    attributes={"prefect.run.id": "run-789"},
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 1_500_000_000,
                    end_time_ns=now_ns,
                )
            )

            time.sleep(0.5)
        finally:
            writer.shutdown(timeout=5.0)

        trace_dir = list(tmp_path.iterdir())[0]
        tree = yaml.safe_load((trace_dir / "_tree.yaml").read_text())

        # After merge: inner becomes new root
        assert tree["root_span_id"] == "inner"
        assert tree["span_count"] == 1

        span_ids = {entry["span_id"] for entry in tree["tree"]}
        assert "wrapper_root" not in span_ids
        assert "inner" in span_ids

        # Inner should have no parent (it's now root)
        inner_entry = tree["tree"][0]
        assert "parent_id" not in inner_entry

    def test_nested_wrapper_chain(self, tmp_path: Path) -> None:
        """Test merging nested wrappers (wrapper -> wrapper -> inner)."""
        config = TraceDebugConfig(path=tmp_path, merge_wrapper_spans=True)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)

            # Create nested wrapper chain
            writer.on_span_start("trace001", "flow_wrapper", None, "my_flow-abc123")
            writer.on_span_start("trace001", "flow_inner", "flow_wrapper", "my_flow")
            writer.on_span_start("trace001", "task_wrapper", "flow_inner", "my_task-def456")
            writer.on_span_start("trace001", "task_inner", "task_wrapper", "my_task")

            # End from innermost to outermost
            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="task_inner",
                    name="my_task",
                    parent_id="task_wrapper",
                    attributes={"lmnr.span.input": '{"data": 1}'},
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 1_000_000_000,
                    end_time_ns=now_ns - 750_000_000,
                )
            )

            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="task_wrapper",
                    name="my_task-def456",
                    parent_id="flow_inner",
                    attributes={"prefect.run.id": "task-run"},
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 1_500_000_000,
                    end_time_ns=now_ns - 500_000_000,
                )
            )

            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="flow_inner",
                    name="my_flow",
                    parent_id="flow_wrapper",
                    attributes={"lmnr.span.input": '{"config": "x"}'},
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 2_000_000_000,
                    end_time_ns=now_ns - 250_000_000,
                )
            )

            writer.on_span_end(
                WriteJob(
                    trace_id="trace001",
                    span_id="flow_wrapper",
                    name="my_flow-abc123",
                    parent_id=None,
                    attributes={"prefect.run.id": "flow-run"},
                    events=[],
                    status_code="OK",
                    status_description=None,
                    start_time_ns=now_ns - 2_500_000_000,
                    end_time_ns=now_ns,
                )
            )

            time.sleep(0.5)
        finally:
            writer.shutdown(timeout=5.0)

        trace_dir = list(tmp_path.iterdir())[0]
        tree = yaml.safe_load((trace_dir / "_tree.yaml").read_text())

        # Both wrappers should be merged
        span_ids = {entry["span_id"] for entry in tree["tree"]}
        assert "flow_wrapper" not in span_ids
        assert "task_wrapper" not in span_ids
        assert "flow_inner" in span_ids
        assert "task_inner" in span_ids

        # Verify hierarchy after double merge
        assert tree["root_span_id"] == "flow_inner"
        assert tree["span_count"] == 2

        # task_inner should be child of flow_inner
        task_entry = next(e for e in tree["tree"] if e["span_id"] == "task_inner")
        assert task_entry["parent_id"] == "flow_inner"

        # flow_inner should have task_inner as child
        flow_entry = next(e for e in tree["tree"] if e["span_id"] == "flow_inner")
        assert "task_inner" in flow_entry["children"]

    def test_wrapper_detection_and_merge(self, tmp_path: Path) -> None:
        """Test wrapper spans are detected and merged in tree index."""
        config = TraceDebugConfig(path=tmp_path, merge_wrapper_spans=True)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)

            # Flow span
            writer.on_span_start("trace001", "flow1", None, "test_flow")

            # Wrapper span (Prefect-generated)
            writer.on_span_start("trace001", "wrapper1", "flow1", "my_task-abc123")

            # Inner span (actual traced function)
            writer.on_span_start("trace001", "inner1", "wrapper1", "my_task")

            # End inner
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

            # End wrapper (no I/O)
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

            # End flow
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

        trace_dirs = list(tmp_path.iterdir())
        trace_dir = trace_dirs[0]

        # Check tree index
        tree = yaml.safe_load((trace_dir / "_tree.yaml").read_text())

        # Wrapper should be skipped in tree
        span_ids = [entry["span_id"] for entry in tree["tree"]]
        assert "wrapper1" not in span_ids
        assert "inner1" in span_ids
        assert "flow1" in span_ids

        # Inner span should be reparented to flow
        inner_entry = next(e for e in tree["tree"] if e["span_id"] == "inner1")
        assert inner_entry["parent_id"] == "flow1"

        # Flow should have inner as direct child
        flow_entry = next(e for e in tree["tree"] if e["span_id"] == "flow1")
        assert "inner1" in flow_entry["children"]
        assert "wrapper1" not in flow_entry["children"]

    def test_wrapper_merge_disabled(self, tmp_path: Path) -> None:
        """Test wrapper spans are preserved when merging is disabled."""
        config = TraceDebugConfig(path=tmp_path, merge_wrapper_spans=False)
        writer = LocalTraceWriter(config)

        try:
            now_ns = int(time.time() * 1e9)

            # Create same structure as above
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

        trace_dirs = list(tmp_path.iterdir())
        trace_dir = trace_dirs[0]

        # Check tree index
        tree = yaml.safe_load((trace_dir / "_tree.yaml").read_text())

        # Wrapper should be present when merging disabled
        span_ids = [entry["span_id"] for entry in tree["tree"]]
        assert "wrapper1" in span_ids
        assert "inner1" in span_ids

        # Original hierarchy preserved
        inner_entry = next(e for e in tree["tree"] if e["span_id"] == "inner1")
        assert inner_entry["parent_id"] == "wrapper1"

        wrapper_entry = next(e for e in tree["tree"] if e["span_id"] == "wrapper1")
        assert wrapper_entry["parent_id"] == "flow1"
