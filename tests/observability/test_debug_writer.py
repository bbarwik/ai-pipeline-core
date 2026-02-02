"""Tests for LocalTraceWriter."""

import time
from collections.abc import Generator
from pathlib import Path

import pytest
import yaml

from ai_pipeline_core.observability import LocalTraceWriter, TraceDebugConfig, WriteJob


@pytest.fixture
def config(tmp_path: Path) -> TraceDebugConfig:
    """Create test configuration."""
    return TraceDebugConfig(path=tmp_path)


@pytest.fixture
def writer(config: TraceDebugConfig) -> Generator[LocalTraceWriter, None, None]:
    """Create test LocalTraceWriter."""
    w = LocalTraceWriter(config)
    yield w
    w.shutdown(timeout=5.0)


class TestLocalTraceWriter:
    """Tests for LocalTraceWriter."""

    def test_creates_base_directory(self, config: TraceDebugConfig) -> None:
        """Test base directory is created on init."""
        writer = LocalTraceWriter(config)
        try:
            assert config.path.exists()
        finally:
            writer.shutdown(timeout=2.0)

    def test_span_start_creates_directories(self, writer: LocalTraceWriter, config: TraceDebugConfig) -> None:
        """Test span start creates trace and span directories."""
        writer.on_span_start(
            trace_id="abc123",
            span_id="span001",
            parent_id=None,
            name="test_flow",
        )

        # Find the trace directory
        trace_dirs = list(config.path.iterdir())
        assert len(trace_dirs) == 1

        trace_dir = trace_dirs[0]
        # Hierarchical: root span is directly under trace dir with counter prefix (4-digit)
        span_dirs = [d for d in trace_dir.iterdir() if d.is_dir() and d.name.startswith("0001_")]
        assert len(span_dirs) == 1

    def test_span_end_writes_metadata(self, writer: LocalTraceWriter, config: TraceDebugConfig) -> None:
        """Test span end writes span metadata."""
        # Start span
        writer.on_span_start(
            trace_id="abc123",
            span_id="span001",
            parent_id=None,
            name="test_task",
        )

        # End span
        now_ns = int(time.time() * 1e9)
        job = WriteJob(
            trace_id="abc123",
            span_id="span001",
            name="test_task",
            parent_id=None,
            attributes={
                "lmnr.span.input": '{"arg": "value"}',
                "lmnr.span.output": '{"result": "success"}',
            },
            events=[],
            status_code="OK",
            status_description=None,
            start_time_ns=now_ns - 1_000_000_000,  # 1 second ago
            end_time_ns=now_ns,
        )
        writer.on_span_end(job)

        # Wait for background processing
        time.sleep(0.5)

        # Check span metadata was written using index
        trace_dirs = list(config.path.iterdir())
        trace_dir = trace_dirs[0]
        index = yaml.safe_load((trace_dir / "_tree.yaml").read_text())
        span_dir = trace_dir / index["span_paths"]["span001"].rstrip("/")

        assert (span_dir / "_span.yaml").exists()

        span_meta = yaml.safe_load((span_dir / "_span.yaml").read_text())
        assert span_meta["name"] == "test_task"
        assert span_meta["status"] == "completed"

    def test_multiple_spans_same_trace(self, writer: LocalTraceWriter, config: TraceDebugConfig) -> None:
        """Test multiple spans in the same trace."""
        # Start parent span
        writer.on_span_start(
            trace_id="trace001",
            span_id="parent",
            parent_id=None,
            name="parent_flow",
        )

        # Start child span
        writer.on_span_start(
            trace_id="trace001",
            span_id="child1",
            parent_id="parent",
            name="child_task",
        )

        # End child span
        now_ns = int(time.time() * 1e9)
        writer.on_span_end(
            WriteJob(
                trace_id="trace001",
                span_id="child1",
                name="child_task",
                parent_id="parent",
                attributes={},
                events=[],
                status_code="OK",
                status_description=None,
                start_time_ns=now_ns - 500_000_000,
                end_time_ns=now_ns,
            )
        )

        # Wait for processing
        time.sleep(0.5)

        # Check both spans exist using index
        trace_dirs = list(config.path.iterdir())
        trace_dir = trace_dirs[0]
        index = yaml.safe_load((trace_dir / "_tree.yaml").read_text())

        parent_path = trace_dir / index["span_paths"]["parent"].rstrip("/")
        child_path = trace_dir / index["span_paths"]["child1"].rstrip("/")
        assert parent_path.exists()
        assert child_path.exists()
        # Verify hierarchical: child is nested under parent
        assert child_path.parent == parent_path

    def test_index_tracks_hierarchy(self, writer: LocalTraceWriter, config: TraceDebugConfig) -> None:
        """Test _tree.yaml tracks span hierarchy."""
        now_ns = int(time.time() * 1e9)

        # Create parent and child
        writer.on_span_start("trace001", "parent", None, "parent_flow")
        writer.on_span_start("trace001", "child1", "parent", "child_task")

        writer.on_span_end(
            WriteJob(
                trace_id="trace001",
                span_id="child1",
                name="child_task",
                parent_id="parent",
                attributes={},
                events=[],
                status_code="OK",
                status_description=None,
                start_time_ns=now_ns - 100_000_000,
                end_time_ns=now_ns,
            )
        )

        time.sleep(0.5)

        # Check index
        trace_dirs = list(config.path.iterdir())
        trace_dir = trace_dirs[0]
        index = yaml.safe_load((trace_dir / "_tree.yaml").read_text())

        assert "tree" in index
        parent_entry = next((s for s in index["tree"] if s["span_id"] == "parent"), None)
        assert parent_entry is not None
        assert "child1" in parent_entry["children"]

    def test_shutdown_finalizes_traces(self, config: TraceDebugConfig) -> None:
        """Test shutdown generates summary files."""
        writer = LocalTraceWriter(config)

        now_ns = int(time.time() * 1e9)
        writer.on_span_start("trace001", "span1", None, "test_flow")
        writer.on_span_end(
            WriteJob(
                trace_id="trace001",
                span_id="span1",
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

        time.sleep(0.3)
        writer.shutdown(timeout=5.0)

        # Check summary file was generated
        trace_dirs = list(config.path.iterdir())
        trace_dir = trace_dirs[0]

        assert (trace_dir / "_summary.md").exists()

        # Check content
        summary = (trace_dir / "_summary.md").read_text()
        assert "test_flow" in summary

    def test_llm_info_extraction(self, writer: LocalTraceWriter, config: TraceDebugConfig) -> None:
        """Test LLM metadata is extracted from attributes."""
        now_ns = int(time.time() * 1e9)

        writer.on_span_start("trace001", "llm_span", None, "llm_call")
        writer.on_span_end(
            WriteJob(
                trace_id="trace001",
                span_id="llm_span",
                name="llm_call",
                parent_id=None,
                attributes={
                    "gen_ai.usage.input_tokens": 1000,
                    "gen_ai.usage.output_tokens": 500,
                    "gen_ai.response.model": "gpt-5.1",
                    "gen_ai.usage.cost": 0.05,
                    "lmnr.span.type": "LLM",
                },
                events=[],
                status_code="OK",
                status_description=None,
                start_time_ns=now_ns - 2_000_000_000,
                end_time_ns=now_ns,
            )
        )

        time.sleep(0.5)

        trace_dirs = list(config.path.iterdir())
        trace_dir = trace_dirs[0]
        index = yaml.safe_load((trace_dir / "_tree.yaml").read_text())
        span_path = trace_dir / index["span_paths"]["llm_span"].rstrip("/")
        span_meta = yaml.safe_load((span_path / "_span.yaml").read_text())

        assert span_meta["type"] == "llm"
        assert "llm" in span_meta
        assert span_meta["llm"]["model"] == "gpt-5.1"
        assert span_meta["llm"]["input_tokens"] == 1000
        assert span_meta["llm"]["cost"] == 0.05

    def test_unset_status_treated_as_completed(self, writer: LocalTraceWriter, config: TraceDebugConfig) -> None:
        """Test UNSET status code (from Laminar @observe) is treated as completed."""
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
                status_code="UNSET",  # Laminar's @observe uses UNSET, not OK
                status_description=None,
                start_time_ns=now_ns - 1_000_000_000,
                end_time_ns=now_ns,
            )
        )

        time.sleep(0.5)

        trace_dirs = list(config.path.iterdir())
        trace_dir = trace_dirs[0]
        index = yaml.safe_load((trace_dir / "_tree.yaml").read_text())
        span_path = trace_dir / index["span_paths"]["span1"].rstrip("/")
        span_meta = yaml.safe_load((span_path / "_span.yaml").read_text())

        # UNSET should be treated as completed, not failed
        assert span_meta["status"] == "completed"


class TestTraceCleanup:
    """Tests for trace cleanup functionality."""

    def test_old_traces_cleaned_up(self, tmp_path: Path) -> None:
        """Test old traces are deleted when max_traces exceeded."""
        config = TraceDebugConfig(path=tmp_path, max_traces=2)

        # Create 3 traces manually
        for i in range(3):
            trace_dir = tmp_path / f"trace_{i}"
            trace_dir.mkdir()
            (trace_dir / "_trace.yaml").write_text("trace_id: test")
            time.sleep(0.1)  # Ensure different mtime

        # Create writer - should clean up oldest
        writer = LocalTraceWriter(config)
        writer.shutdown(timeout=2.0)

        # Should only have 2 traces left
        trace_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(trace_dirs) == 2
