"""Tests for replay payload extraction in the trace writer.

Verifies that the LocalTraceWriter recognizes `replay.payload` in span
attributes, parses it as JSON, and writes typed YAML files
(conversation.yaml, task.yaml, flow.yaml) into the span directory —
while excluding `replay.payload` from the persisted span.yaml attributes.
"""

import json
import time
from pathlib import Path
from typing import Any

import yaml

from ai_pipeline_core.observability._debug import LocalTraceWriter, TraceDebugConfig, WriteJob


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_span_dir(trace_dir: Path, span_name: str) -> Path:
    """Find the first span directory whose name contains *span_name*."""
    for path in trace_dir.rglob(f"*{span_name}*"):
        if path.is_dir():
            return path
    raise FileNotFoundError(f"No span directory for {span_name}")


def _make_job(
    *,
    span_id: str = "span001",
    name: str = "my_span",
    attributes: dict[str, Any] | None = None,
) -> WriteJob:
    """Build a minimal WriteJob with sensible defaults."""
    now_ns = int(time.time() * 1e9)
    return WriteJob(
        trace_id="trace001",
        span_id=span_id,
        name=name,
        parent_id=None,
        attributes=attributes or {},
        events=[],
        status_code="OK",
        status_description=None,
        start_time_ns=now_ns - 500_000_000,
        end_time_ns=now_ns,
    )


def _run_span(writer: LocalTraceWriter, job: WriteJob) -> None:
    """Register a span start, enqueue end, wait for background processing."""
    writer.on_span_start(
        trace_id=job.trace_id,
        span_id=job.span_id,
        parent_id=job.parent_id,
        name=job.name,
    )
    writer.on_span_end(job)
    time.sleep(0.3)
    writer.shutdown(timeout=5.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWriterReplayHook:
    """Replay payload extraction from span attributes."""

    def test_writer_writes_conversation_yaml(self, tmp_path: Path) -> None:
        """Span with replay.payload (payload_type=conversation) produces conversation.yaml."""
        payload = {
            "payload_type": "conversation",
            "model": "gpt-4o",
            "prompt": "Summarize this document.",
            "context": [],
            "history": [],
        }
        writer = LocalTraceWriter(TraceDebugConfig(path=tmp_path))
        job = _make_job(
            name="llm_conversation",
            attributes={"replay.payload": json.dumps(payload)},
        )
        _run_span(writer, job)

        span_dir = _find_span_dir(tmp_path, "llm_conversation")
        replay_path = span_dir / "conversation.yaml"
        assert replay_path.exists(), f"Expected conversation.yaml in {span_dir}"

        data = yaml.safe_load(replay_path.read_text())
        assert data["payload_type"] == "conversation"
        assert data["model"] == "gpt-4o"
        assert data["prompt"] == "Summarize this document."

    def test_writer_writes_task_yaml(self, tmp_path: Path) -> None:
        """Span with replay.payload (payload_type=pipeline_task) produces task.yaml."""
        payload = {
            "payload_type": "pipeline_task",
            "function_path": "my_package.tasks.extract",
            "arguments": {"label": "demo"},
        }
        writer = LocalTraceWriter(TraceDebugConfig(path=tmp_path))
        job = _make_job(
            name="extract_task",
            attributes={"replay.payload": json.dumps(payload)},
        )
        _run_span(writer, job)

        span_dir = _find_span_dir(tmp_path, "extract_task")
        replay_path = span_dir / "task.yaml"
        assert replay_path.exists(), f"Expected task.yaml in {span_dir}"

        data = yaml.safe_load(replay_path.read_text())
        assert data["payload_type"] == "pipeline_task"
        assert data["function_path"] == "my_package.tasks.extract"
        assert data["arguments"]["label"] == "demo"

    def test_writer_writes_flow_yaml(self, tmp_path: Path) -> None:
        """Span with replay.payload (payload_type=pipeline_flow) produces flow.yaml."""
        payload = {
            "payload_type": "pipeline_flow",
            "function_path": "my_package.flows.main_flow",
            "run_id": "run-42",
            "documents": [],
            "flow_options": {"replay_label": "baseline"},
        }
        writer = LocalTraceWriter(TraceDebugConfig(path=tmp_path))
        job = _make_job(
            name="main_flow",
            attributes={"replay.payload": json.dumps(payload)},
        )
        _run_span(writer, job)

        span_dir = _find_span_dir(tmp_path, "main_flow")
        replay_path = span_dir / "flow.yaml"
        assert replay_path.exists(), f"Expected flow.yaml in {span_dir}"

        data = yaml.safe_load(replay_path.read_text())
        assert data["payload_type"] == "pipeline_flow"
        assert data["run_id"] == "run-42"

    def test_writer_excludes_replay_from_span_yaml(self, tmp_path: Path) -> None:
        """replay.payload must NOT appear in span.yaml attributes."""
        payload = {
            "payload_type": "conversation",
            "model": "gpt-4o",
            "prompt": "Hello",
            "context": [],
            "history": [],
        }
        writer = LocalTraceWriter(TraceDebugConfig(path=tmp_path))
        job = _make_job(
            name="check_exclusion",
            attributes={
                "replay.payload": json.dumps(payload),
                "other_attr": "keep_me",
            },
        )
        _run_span(writer, job)

        span_dir = _find_span_dir(tmp_path, "check_exclusion")
        span_yaml = yaml.safe_load((span_dir / "span.yaml").read_text())

        attrs = span_yaml.get("attributes", {})
        assert "replay.payload" not in attrs, "replay.payload should be excluded from span.yaml"
        assert attrs.get("other_attr") == "keep_me"

    def test_writer_no_replay_file_without_payload(self, tmp_path: Path) -> None:
        """Without replay.payload, no conversation/task/flow YAML is written."""
        writer = LocalTraceWriter(TraceDebugConfig(path=tmp_path))
        job = _make_job(
            name="plain_span",
            attributes={"some_attr": "value"},
        )
        _run_span(writer, job)

        span_dir = _find_span_dir(tmp_path, "plain_span")
        assert not (span_dir / "conversation.yaml").exists()
        assert not (span_dir / "task.yaml").exists()
        assert not (span_dir / "flow.yaml").exists()
