"""Tests for LocalTraceWriter."""

import time
from collections.abc import Generator
from pathlib import Path

import pytest
import yaml

from ai_pipeline_core.observability._debug import LocalTraceWriter, TraceDebugConfig, WriteJob


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


def _find_span_dir(trace_dir: Path, span_name: str) -> Path:
    """Find a span directory by name within a trace directory (recursive)."""
    for p in trace_dir.rglob(f"*_{span_name}*"):
        if p.is_dir():
            return p
    msg = f"No span directory found for '{span_name}' in {trace_dir}"
    raise FileNotFoundError(msg)


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

        # Span dir is directly under config.path (trace writes directly there)
        span_dirs = [d for d in config.path.iterdir() if d.is_dir() and d.name.startswith("001_")]
        assert len(span_dirs) == 1

    def test_span_end_writes_metadata(self, writer: LocalTraceWriter, config: TraceDebugConfig) -> None:
        """Test span end writes span metadata."""
        writer.on_span_start(
            trace_id="abc123",
            span_id="span001",
            parent_id=None,
            name="test_task",
        )

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
            start_time_ns=now_ns - 1_000_000_000,
            end_time_ns=now_ns,
        )
        writer.on_span_end(job)

        time.sleep(0.5)

        span_dir = _find_span_dir(config.path, "test_task")

        assert (span_dir / "span.yaml").exists()

        span_meta = yaml.safe_load((span_dir / "span.yaml").read_text())
        assert span_meta["name"] == "test_task"
        assert span_meta["status"] == "completed"

    def test_multiple_spans_same_trace(self, writer: LocalTraceWriter, config: TraceDebugConfig) -> None:
        """Test multiple spans in the same trace."""
        writer.on_span_start(
            trace_id="trace001",
            span_id="parent",
            parent_id=None,
            name="parent_flow",
        )

        writer.on_span_start(
            trace_id="trace001",
            span_id="child1",
            parent_id="parent",
            name="child_task",
        )

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

        time.sleep(0.5)

        parent_path = _find_span_dir(config.path, "parent_flow")
        child_path = _find_span_dir(config.path, "child_task")
        assert parent_path.exists()
        assert child_path.exists()
        # Verify hierarchical: child is nested under parent
        assert child_path.parent == parent_path

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

        assert (config.path / "summary.md").exists()

        summary = (config.path / "summary.md").read_text()
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
                    "gen_ai.usage.cache_read_input_tokens": 800,
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

        span_path = _find_span_dir(config.path, "llm_call")
        span_meta = yaml.safe_load((span_path / "span.yaml").read_text())

        assert span_meta["type"] == "llm"
        assert "llm" in span_meta
        assert span_meta["llm"]["model"] == "gpt-5.1"
        assert span_meta["llm"]["input_tokens"] == 1000
        assert span_meta["llm"]["cached_tokens"] == 800
        assert span_meta["llm"]["cost"] == pytest.approx(0.05)

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

        span_path = _find_span_dir(config.path, "test")
        span_meta = yaml.safe_load((span_path / "span.yaml").read_text())

        # UNSET should be treated as completed, not failed
        assert span_meta["status"] == "completed"
