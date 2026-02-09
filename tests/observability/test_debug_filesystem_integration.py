"""Integration tests for local debug trace writing to real filesystem.

Tests LocalTraceWriter, ArtifactStore, ContentWriter, and summary generation
with real filesystem I/O. No mocking.
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Generator

import pytest
import yaml

from ai_pipeline_core.observability._debug._config import TraceDebugConfig
from ai_pipeline_core.observability._debug._content import ArtifactStore, ContentWriter

from tests.observability.test_helpers import reconstruct_span_content
from ai_pipeline_core.observability._debug._summary import generate_summary
from ai_pipeline_core.observability._debug._types import SpanInfo, TraceState, WriteJob
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
        max_element_bytes=500,
        element_excerpt_bytes=100,
        max_file_bytes=50_000,
        generate_summary=True,
        include_llm_index=True,
        include_error_index=True,
        merge_wrapper_spans=False,
        auto_summary_enabled=False,
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

        # Find the trace directory
        trace_dirs = list(trace_dir.iterdir())
        assert len(trace_dirs) == 1

        t = trace_dirs[0]
        assert (t / "_trace.yaml").exists()
        assert (t / "_tree.yaml").exists()
        assert (t / "_summary.md").exists()

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

        trace_dirs = list(trace_dir.iterdir())
        t = trace_dirs[0]

        # Find the span directory (format: 0001_process)
        span_dirs = [d for d in t.iterdir() if d.is_dir() and d.name.startswith("0001_")]
        assert len(span_dirs) == 1

        span_dir = span_dirs[0]
        assert (span_dir / "_span.yaml").exists()
        assert (span_dir / "input.yaml").exists()
        assert (span_dir / "output.yaml").exists()

        # Verify _span.yaml content
        span_meta = yaml.safe_load((span_dir / "_span.yaml").read_text())
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

        # End child first, then parent
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

        trace_dirs = list(trace_dir.iterdir())
        t = trace_dirs[0]

        # Find parent dir
        parent_dirs = [d for d in t.iterdir() if d.is_dir() and "parent_flow" in d.name]
        assert len(parent_dirs) == 1

        # Child should be nested inside parent
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

        trace_dirs = list(trace_dir.iterdir())
        t = trace_dirs[0]

        # _errors.yaml should exist
        errors_path = t / "_errors.yaml"
        assert errors_path.exists()
        errors = yaml.safe_load(errors_path.read_text())
        assert errors["error_count"] == 1
        assert errors["errors"][0]["name"] == "failing_task"

        # _trace.yaml should show failed status
        trace_meta = yaml.safe_load((t / "_trace.yaml").read_text())
        assert trace_meta["status"] == "failed"

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

        trace_dirs = list(trace_dir.iterdir())
        t = trace_dirs[0]

        llm_path = t / "_llm_calls.yaml"
        assert llm_path.exists()
        llm_data = yaml.safe_load(llm_path.read_text())
        assert llm_data["llm_call_count"] == 1
        assert llm_data["total_tokens"] == 600
        assert llm_data["calls"][0]["model"] == "gpt-5.1"
        assert llm_data["calls"][0]["purpose"] == "summarize"


class TestArtifactStoreIntegration:
    """Test ArtifactStore deduplication with real filesystem."""

    def test_text_deduplication(self, tmp_path):
        store = ArtifactStore(tmp_path)

        ref1 = store.store_text("Hello, world!")
        ref2 = store.store_text("Hello, world!")
        ref3 = store.store_text("Different content")

        assert ref1.hash == ref2.hash
        assert ref1.path == ref2.path
        assert ref1.hash != ref3.hash

        stats = store.get_stats()
        assert stats["unique_artifacts"] == 2
        # total_references counts unique hashes stored, not total store calls
        assert stats["total_references"] >= 2

    def test_binary_deduplication(self, tmp_path):
        store = ArtifactStore(tmp_path)

        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        ref1 = store.store_binary(data, "image/png")
        ref2 = store.store_binary(data, "image/png")

        assert ref1.hash == ref2.hash
        assert ref1.path.endswith(".png")

        # Verify file exists and content matches
        artifact_path = tmp_path / ref1.path
        assert artifact_path.exists()
        assert artifact_path.read_bytes() == data

    def test_flat_directory_structure(self, tmp_path):
        store = ArtifactStore(tmp_path)
        ref = store.store_text("test content for sharding")

        # Path should have format: artifacts/sha256/<hash>.txt
        parts = Path(ref.path).parts
        assert parts[0] == "artifacts"
        assert parts[1] == "sha256"
        assert len(parts) == 3
        assert parts[2].endswith(".txt")


class TestContentWriterExternalization:
    """Test ContentWriter with artifact externalization."""

    def test_large_text_externalized_via_llm_messages(self, tmp_path, config):
        store = ArtifactStore(tmp_path)
        cw = ContentWriter(config, store)

        span_dir = tmp_path / "span"
        span_dir.mkdir()

        # ContentWriter externalizes text inside LLM messages or document lists
        large_text = "x" * 1000  # Exceeds max_element_bytes=500
        messages = [{"role": "user", "content": large_text}]
        ref = cw.write(messages, span_dir, "input")

        assert ref["type"] == "file"
        assert (span_dir / "input.yaml").exists()

        content = yaml.safe_load((span_dir / "input.yaml").read_text())
        # The text part inside the message should have a content_ref
        parts = content["messages"][0]["parts"]
        assert any("content_ref" in p for p in parts)

    def test_small_text_stays_inline(self, tmp_path, config):
        store = ArtifactStore(tmp_path)
        cw = ContentWriter(config, store)

        span_dir = tmp_path / "span"
        span_dir.mkdir()

        small_text = "short"
        cw.write(small_text, span_dir, "input")

        content = yaml.safe_load((span_dir / "input.yaml").read_text())
        assert content.get("content") == "short"

    def test_reconstruct_span_content_resolves_refs(self, tmp_path, config):
        store = ArtifactStore(tmp_path)
        cw = ContentWriter(config, store)

        span_dir = tmp_path / "span"
        span_dir.mkdir()

        large_text = "y" * 1000
        messages = [{"role": "assistant", "content": large_text}]
        cw.write(messages, span_dir, "output")

        reconstructed = reconstruct_span_content(tmp_path, span_dir, "output")
        # The text part should have its content resolved from artifact
        parts = reconstructed["messages"][0]["parts"]
        text_part = parts[0]
        assert "content" in text_part
        assert len(text_part["content"]) == 1000


class TestRedaction:
    """Test secret redaction in content writing."""

    def test_api_key_redacted(self, tmp_path, config):
        store = ArtifactStore(tmp_path)
        cw = ContentWriter(config, store)

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
            path=tmp_path / "0001_root_task",
            end_time=now,
            duration_ms=150,
        )
        (tmp_path / "0001_root_task").mkdir()
        trace.spans["s1"] = span
        trace.root_span_id = "s1"

        summary = generate_summary(trace)
        assert "# Trace Summary: test_flow" in summary
        assert "root_task" in summary
        assert "Execution Tree" in summary

    def test_summary_includes_llm_calls_table(self, tmp_path):
        now = datetime.now(UTC)
        trace = TraceState(trace_id="bbb", name="llm_flow", path=tmp_path, start_time=now)

        root = SpanInfo(
            span_id="r1",
            parent_id=None,
            name="flow",
            span_type="flow",
            status="completed",
            start_time=now,
            path=tmp_path / "0001_flow",
            end_time=now,
            duration_ms=500,
        )
        llm = SpanInfo(
            span_id="l1",
            parent_id="r1",
            name="generate",
            span_type="llm",
            status="completed",
            start_time=now,
            path=tmp_path / "0001_flow" / "0002_generate",
            end_time=now,
            duration_ms=300,
            llm_info={"model": "gpt-5.1", "input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200, "cost": 0.01, "purpose": "research"},
        )
        (tmp_path / "0001_flow").mkdir()
        (tmp_path / "0001_flow" / "0002_generate").mkdir()

        trace.spans["r1"] = root
        trace.spans["l1"] = llm
        trace.root_span_id = "r1"
        root.children = ["l1"]
        trace.llm_call_count = 1
        trace.total_tokens = 1200
        trace.total_cost = 0.01

        summary = generate_summary(trace)
        assert "LLM Calls" in summary
        assert "gpt-5.1" in summary
        assert "research" in summary

    def test_summary_includes_error_section(self, tmp_path):
        now = datetime.now(UTC)
        trace = TraceState(trace_id="ccc", name="error_flow", path=tmp_path, start_time=now)
        span = SpanInfo(
            span_id="e1",
            parent_id=None,
            name="failing_task",
            span_type="task",
            status="failed",
            start_time=now,
            path=tmp_path / "0001_failing",
            end_time=now,
            duration_ms=50,
        )
        (tmp_path / "0001_failing").mkdir()
        trace.spans["e1"] = span
        trace.root_span_id = "e1"

        summary = generate_summary(trace)
        assert "Errors" in summary
        assert "failing_task" in summary


class TestMaxTracesCleanup:
    """Test old trace cleanup when max_traces is set."""

    def test_old_traces_cleaned_up(self, trace_dir):
        # Create 3 existing trace dirs
        trace_dir.mkdir(parents=True)
        for i in range(3):
            d = trace_dir / f"20250101_00000{i}_aaaa_trace{i}"
            d.mkdir()
            (d / "_trace.yaml").write_text(f"trace_id: trace{i}")
            # Stagger modification times
            time.sleep(0.05)

        config = TraceDebugConfig(
            path=trace_dir,
            max_traces=2,
            generate_summary=False,
            include_llm_index=False,
            include_error_index=False,
        )
        writer = LocalTraceWriter(config)
        writer.shutdown(timeout=2.0)

        # Only max_traces (2) should remain
        remaining = [d for d in trace_dir.iterdir() if d.is_dir() and (d / "_trace.yaml").exists()]
        assert len(remaining) <= 2
