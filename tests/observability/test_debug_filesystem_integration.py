"""Integration tests for local debug trace writing to real filesystem.

Tests LocalTraceWriter, ContentWriter, and summary generation
with real filesystem I/O. No mocking.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Generator

import pytest
import yaml

from ai_pipeline_core.observability._debug._config import TraceDebugConfig
from ai_pipeline_core.observability._debug._content import ContentWriter
from ai_pipeline_core.observability._debug._summary import generate_summary
from ai_pipeline_core.observability._debug._config import SpanInfo, TraceState, WriteJob
from ai_pipeline_core.observability._debug._writer import LocalTraceWriter


@pytest.fixture
def trace_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for trace output."""
    return tmp_path / "traces"


@pytest.fixture
def config(trace_dir: Path) -> TraceDebugConfig:
    """Create a TraceDebugConfig for testing."""
    return TraceDebugConfig(
        path=trace_dir,
        max_file_bytes=50_000,
        generate_summary=True,
        merge_wrapper_spans=False,
    )


@pytest.fixture
def writer(config: TraceDebugConfig) -> Generator[LocalTraceWriter, None, None]:
    """Create and return a LocalTraceWriter."""
    w = LocalTraceWriter(config)
    yield w
    w.shutdown(timeout=5.0)


class TestLocalTraceWriterLifecycle:
    """Test full trace lifecycle on filesystem."""

    def test_single_span_creates_trace_directory(self, writer, trace_dir):
        trace_id = "a" * 32
        span_id = "b" * 16

        writer.on_span_start(trace_id, span_id, None, "my_function")
        job = WriteJob(
            trace_id=trace_id,
            span_id=span_id,
            name="my_function",
            parent_id=None,
            attributes={"lmnr.span.input": json.dumps({"x": 1}), "lmnr.span.output": json.dumps({"result": "ok"})},
            events=[],
            status_code="OK",
            status_description=None,
            start_time_ns=int(datetime.now(UTC).timestamp() * 1e9),
            end_time_ns=int(datetime.now(UTC).timestamp() * 1e9) + 100_000_000,
        )
        writer.on_span_end(job)
        writer.shutdown(timeout=5.0)

        assert (trace_dir / "summary.md").exists()
        assert (trace_dir / "llm_calls.yaml").exists()

    def test_span_directory_contains_expected_files(self, writer, trace_dir):
        trace_id = "c" * 32
        span_id = "d" * 16

        writer.on_span_start(trace_id, span_id, None, "process")
        job = WriteJob(
            trace_id=trace_id,
            span_id=span_id,
            name="process",
            parent_id=None,
            attributes={"lmnr.span.input": json.dumps("hello"), "lmnr.span.output": json.dumps("world")},
            events=[],
            status_code="OK",
            status_description=None,
            start_time_ns=int(datetime.now(UTC).timestamp() * 1e9),
            end_time_ns=int(datetime.now(UTC).timestamp() * 1e9) + 50_000_000,
        )
        writer.on_span_end(job)
        writer.shutdown(timeout=5.0)

        span_dirs = [d for d in trace_dir.iterdir() if d.is_dir() and d.name.startswith("001_")]
        assert len(span_dirs) == 1

        span_dir = span_dirs[0]
        assert (span_dir / "span.yaml").exists()
        assert (span_dir / "input.yaml").exists()
        assert (span_dir / "output.yaml").exists()

        span_meta = yaml.safe_load((span_dir / "span.yaml").read_text())
        assert span_meta["name"] == "process"
        assert span_meta["status"] == "completed"
        assert span_meta["span_id"] == span_id

    def test_nested_spans_create_hierarchical_dirs(self, writer, trace_dir):
        trace_id = "e" * 32
        parent_id = "f" * 16
        child_id = "1" * 16

        now_ns = int(datetime.now(UTC).timestamp() * 1e9)

        writer.on_span_start(trace_id, parent_id, None, "parent_flow")
        writer.on_span_start(trace_id, child_id, parent_id, "child_task")

        writer.on_span_end(
            WriteJob(
                trace_id=trace_id,
                span_id=child_id,
                name="child_task",
                parent_id=parent_id,
                attributes={},
                events=[],
                status_code="OK",
                status_description=None,
                start_time_ns=now_ns,
                end_time_ns=now_ns + 30_000_000,
            )
        )
        writer.on_span_end(
            WriteJob(
                trace_id=trace_id,
                span_id=parent_id,
                name="parent_flow",
                parent_id=None,
                attributes={},
                events=[],
                status_code="OK",
                status_description=None,
                start_time_ns=now_ns,
                end_time_ns=now_ns + 60_000_000,
            )
        )
        writer.shutdown(timeout=5.0)

        parent_dirs = [d for d in trace_dir.iterdir() if d.is_dir() and "parent_flow" in d.name]
        assert len(parent_dirs) == 1

        child_dirs = [d for d in parent_dirs[0].iterdir() if d.is_dir() and "child_task" in d.name]
        assert len(child_dirs) == 1

    def test_failed_span_recorded_in_errors_index(self, writer, trace_dir):
        trace_id = "g" * 32
        span_id = "h" * 16
        now_ns = int(datetime.now(UTC).timestamp() * 1e9)

        writer.on_span_start(trace_id, span_id, None, "failing_task")
        writer.on_span_end(
            WriteJob(
                trace_id=trace_id,
                span_id=span_id,
                name="failing_task",
                parent_id=None,
                attributes={},
                events=[],
                status_code="ERROR",
                status_description="ValueError: bad input",
                start_time_ns=now_ns,
                end_time_ns=now_ns + 10_000_000,
            )
        )
        writer.shutdown(timeout=5.0)

        errors_path = trace_dir / "errors.yaml"
        assert errors_path.exists()
        errors = yaml.safe_load(errors_path.read_text())
        assert errors["error_count"] == 1
        assert errors["errors"][0]["name"] == "failing_task"

    def test_llm_span_recorded_in_llm_index(self, writer, trace_dir):
        trace_id = "i" * 32
        span_id = "j" * 16
        now_ns = int(datetime.now(UTC).timestamp() * 1e9)

        writer.on_span_start(trace_id, span_id, None, "llm_call")
        writer.on_span_end(
            WriteJob(
                trace_id=trace_id,
                span_id=span_id,
                name="llm_call",
                parent_id=None,
                attributes={
                    "lmnr.span.type": "LLM",
                    "gen_ai.usage.input_tokens": 500,
                    "gen_ai.usage.output_tokens": 100,
                    "gen_ai.request.model": "gpt-5.1",
                    "gen_ai.system": "openai",
                    "gen_ai.usage.cost": 0.003,
                    "purpose": "summarize",
                },
                events=[],
                status_code="OK",
                status_description=None,
                start_time_ns=now_ns,
                end_time_ns=now_ns + 200_000_000,
            )
        )
        writer.shutdown(timeout=5.0)

        llm_path = trace_dir / "llm_calls.yaml"
        assert llm_path.exists()
        llm_data = yaml.safe_load(llm_path.read_text())
        assert llm_data["llm_call_count"] == 1
        assert llm_data["total_tokens"] == 600
        assert llm_data["calls"][0]["model"] == "gpt-5.1"
        assert llm_data["calls"][0]["purpose"] == "summarize"


class TestRedaction:
    """Test secret redaction in content writing."""

    def test_api_key_redacted(self, tmp_path, config):
        cw = ContentWriter(config)

        span_dir = tmp_path / "span"
        span_dir.mkdir()

        text_with_secret = "Using key sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ12345678 for auth"
        cw.write(text_with_secret, span_dir, "input")

        content = (span_dir / "input.yaml").read_text()
        assert "sk-proj-" not in content
        assert "[REDACTED]" in content


class TestSummaryGeneration:
    """Test static summary generation for traces."""

    def test_summary_contains_tree(self, tmp_path):
        now = datetime.now(UTC)
        trace = TraceState(trace_id="aaa", name="test_flow", path=tmp_path, start_time=now)
        span = SpanInfo(
            span_id="s1",
            parent_id=None,
            name="root_task",
            span_type="task",
            status="completed",
            start_time=now,
            path=tmp_path / "001_root_task",
            end_time=now,
            duration_ms=150,
            order=1,
        )
        (tmp_path / "001_root_task").mkdir()
        trace.spans["s1"] = span
        trace.root_span_id = "s1"

        summary = generate_summary(trace)
        assert "# Trace Summary: test_flow" in summary
        assert "root_task" in summary
        assert "Execution Tree" in summary

    def test_summary_shows_llm_info_in_tree(self, tmp_path):
        now = datetime.now(UTC)
        trace = TraceState(trace_id="bbb", name="llm_flow", path=tmp_path, start_time=now)

        root = SpanInfo(
            span_id="r1",
            parent_id=None,
            name="flow",
            span_type="flow",
            status="completed",
            start_time=now,
            path=tmp_path / "001_flow",
            end_time=now,
            duration_ms=500,
            order=1,
        )
        llm = SpanInfo(
            span_id="l1",
            parent_id="r1",
            name="generate",
            span_type="llm",
            status="completed",
            start_time=now,
            path=tmp_path / "001_flow" / "002_generate",
            end_time=now,
            duration_ms=300,
            order=2,
            llm_info={
                "model": "gpt-5.1",
                "input_tokens": 1000,
                "output_tokens": 200,
                "cached_tokens": 0,
                "total_tokens": 1200,
                "cost": 0.01,
                "purpose": "research",
            },
        )
        (tmp_path / "001_flow").mkdir()
        (tmp_path / "001_flow" / "002_generate").mkdir()

        trace.spans["r1"] = root
        trace.spans["l1"] = llm
        trace.root_span_id = "r1"
        root.children = ["l1"]
        trace.llm_call_count = 1
        trace.total_tokens = 1200
        trace.total_cost = 0.01

        summary = generate_summary(trace)
        # LLM info appears in compact tree format: token counts and cost
        assert "1K IN" in summary
        assert "200 OUT" in summary
        assert "$0.010" in summary

    def test_summary_shows_error_status(self, tmp_path):
        now = datetime.now(UTC)
        trace = TraceState(trace_id="ccc", name="error_flow", path=tmp_path, start_time=now)
        span = SpanInfo(
            span_id="e1",
            parent_id=None,
            name="failing_task",
            span_type="task",
            status="failed",
            start_time=now,
            path=tmp_path / "001_failing",
            end_time=now,
            duration_ms=50,
            order=1,
        )
        (tmp_path / "001_failing").mkdir()
        trace.spans["e1"] = span
        trace.root_span_id = "e1"

        summary = generate_summary(trace)
        assert "Failed" in summary
        assert "ERROR" in summary
        assert "failing_task" in summary
