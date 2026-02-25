"""Tests for replay section generation in trace summary."""

from datetime import UTC, datetime
from pathlib import Path

import yaml

from ai_pipeline_core.observability._debug._config import SpanInfo, TraceState
from ai_pipeline_core.observability._debug._summary import (
    _build_replay_section,
    _replay_entry_detail,
    generate_summary,
)


def _make_trace(tmp_path: Path, name: str = "test-trace") -> TraceState:
    """Create a minimal TraceState."""
    trace_path = tmp_path / ".trace"
    trace_path.mkdir(parents=True, exist_ok=True)
    return TraceState(
        trace_id="trace-001",
        name=name,
        path=trace_path,
        start_time=datetime.now(UTC),
    )


def _add_span(trace: TraceState, name: str, parent_id: str | None = None) -> SpanInfo:
    """Add a completed span with a directory to the trace."""
    trace.span_counter += 1
    span_id = f"span-{trace.span_counter:03d}"
    parent_path = trace.path
    if parent_id and parent_id in trace.spans:
        parent_path = trace.spans[parent_id].path

    span_dir = parent_path / f"{trace.span_counter:03d}_{name}"
    span_dir.mkdir(parents=True, exist_ok=True)

    span = SpanInfo(
        span_id=span_id,
        parent_id=parent_id,
        name=name,
        span_type="default",
        status="completed",
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC),
        path=span_dir,
        depth=0,
        order=trace.span_counter,
        duration_ms=100,
    )
    trace.spans[span_id] = span
    if trace.root_span_id is None:
        trace.root_span_id = span_id
    return span


class TestBuildReplaySection:
    """Tests for _build_replay_section."""

    def test_empty_when_no_replay_files(self, tmp_path: Path) -> None:
        """Returns empty list when spans have no replay YAML files."""
        trace = _make_trace(tmp_path)
        _add_span(trace, "my_span")
        assert _build_replay_section(trace) == []

    def test_conversation_yaml_listed(self, tmp_path: Path) -> None:
        """Conversation replay files appear under Conversations heading."""
        trace = _make_trace(tmp_path)
        span = _add_span(trace, "llm_call")
        (span.path / "conversation.yaml").write_text(
            yaml.dump({
                "payload_type": "conversation",
                "model": "gemini-3-flash",
            })
        )

        lines = _build_replay_section(trace)
        text = "\n".join(lines)
        assert "## Replay" in text
        assert "**Conversations** (1):" in text
        assert "conversation.yaml" in text
        assert "gemini-3-flash" in text

    def test_task_yaml_listed(self, tmp_path: Path) -> None:
        """Task replay files appear under Tasks heading."""
        trace = _make_trace(tmp_path)
        span = _add_span(trace, "extract")
        (span.path / "task.yaml").write_text(
            yaml.dump({
                "payload_type": "pipeline_task",
                "function_path": "app:extract_insights",
            })
        )

        lines = _build_replay_section(trace)
        text = "\n".join(lines)
        assert "**Tasks** (1):" in text
        assert "extract_insights" in text

    def test_flow_yaml_listed(self, tmp_path: Path) -> None:
        """Flow replay files appear under Flows heading with doc count."""
        trace = _make_trace(tmp_path)
        span = _add_span(trace, "analysis")
        (span.path / "flow.yaml").write_text(
            yaml.dump({
                "payload_type": "pipeline_flow",
                "function_path": "app:analysis_flow",
                "documents": [{"$doc_ref": "ABC123"}, {"$doc_ref": "DEF456"}],
            })
        )

        lines = _build_replay_section(trace)
        text = "\n".join(lines)
        assert "**Flows** (1):" in text
        assert "analysis_flow" in text
        assert "2 docs" in text

    def test_all_three_types(self, tmp_path: Path) -> None:
        """All three payload types grouped correctly."""
        trace = _make_trace(tmp_path)

        s1 = _add_span(trace, "conv_span")
        (s1.path / "conversation.yaml").write_text(yaml.dump({"payload_type": "conversation", "model": "gpt-4o"}))

        s2 = _add_span(trace, "task_span")
        (s2.path / "task.yaml").write_text(yaml.dump({"payload_type": "pipeline_task", "function_path": "m:fn"}))

        s3 = _add_span(trace, "flow_span")
        (s3.path / "flow.yaml").write_text(yaml.dump({"payload_type": "pipeline_flow", "function_path": "m:fl", "documents": []}))

        lines = _build_replay_section(trace)
        text = "\n".join(lines)
        assert "**Conversations** (1):" in text
        assert "**Tasks** (1):" in text
        assert "**Flows** (1):" in text

    def test_multiple_conversations(self, tmp_path: Path) -> None:
        """Multiple replay files of the same type are counted correctly."""
        trace = _make_trace(tmp_path)
        for i in range(3):
            span = _add_span(trace, f"llm_{i}")
            (span.path / "conversation.yaml").write_text(
                yaml.dump({
                    "payload_type": "conversation",
                    "model": f"model-{i}",
                })
            )

        lines = _build_replay_section(trace)
        text = "\n".join(lines)
        assert "**Conversations** (3):" in text

    def test_example_commands_use_first_path(self, tmp_path: Path) -> None:
        """Example CLI commands reference the first found replay file."""
        trace = _make_trace(tmp_path)
        span = _add_span(trace, "my_task")
        (span.path / "task.yaml").write_text(yaml.dump({"payload_type": "pipeline_task", "function_path": "m:f"}))

        lines = _build_replay_section(trace)
        text = "\n".join(lines)
        assert "ai-replay show" in text
        assert "ai-replay run" in text
        assert "task.yaml" in text

    def test_ordering_by_span_order(self, tmp_path: Path) -> None:
        """Replay files are listed in span execution order."""
        trace = _make_trace(tmp_path)
        s1 = _add_span(trace, "first")
        (s1.path / "conversation.yaml").write_text(yaml.dump({"payload_type": "conversation", "model": "model-A"}))
        s2 = _add_span(trace, "second")
        (s2.path / "conversation.yaml").write_text(yaml.dump({"payload_type": "conversation", "model": "model-B"}))

        lines = _build_replay_section(trace)
        text = "\n".join(lines)
        idx_a = text.index("model-A")
        idx_b = text.index("model-B")
        assert idx_a < idx_b

    def test_conversation_with_original_cost(self, tmp_path: Path) -> None:
        """Conversation entries show original cost when available."""
        trace = _make_trace(tmp_path)
        span = _add_span(trace, "llm")
        (span.path / "conversation.yaml").write_text(
            yaml.dump({
                "payload_type": "conversation",
                "model": "gemini-3-pro",
                "original": {"cost": 0.0123, "tokens": {"input": 100, "output": 200}},
            })
        )

        lines = _build_replay_section(trace)
        text = "\n".join(lines)
        assert "$0.0123" in text

    def test_integrated_in_generate_summary(self, tmp_path: Path) -> None:
        """Replay section appears in the full summary between Navigation and Root Span."""
        trace = _make_trace(tmp_path)
        span = _add_span(trace, "my_llm")
        (span.path / "conversation.yaml").write_text(yaml.dump({"payload_type": "conversation", "model": "gpt-4o"}))

        summary = generate_summary(trace)
        assert "## Navigation" in summary
        assert "## Replay" in summary
        # Replay should come after Navigation
        assert summary.index("## Navigation") < summary.index("## Replay")

    def test_no_replay_section_in_summary_when_empty(self, tmp_path: Path) -> None:
        """Summary does not contain Replay section when no replay files exist."""
        trace = _make_trace(tmp_path)
        _add_span(trace, "plain_span")

        summary = generate_summary(trace)
        assert "## Replay" not in summary

    def test_malformed_yaml_skipped_gracefully(self, tmp_path: Path) -> None:
        """Malformed YAML files are skipped without error."""
        trace = _make_trace(tmp_path)
        span = _add_span(trace, "bad")
        (span.path / "conversation.yaml").write_text("{{invalid yaml content")

        # Should still produce a section (file exists but data is empty dict)
        lines = _build_replay_section(trace)
        assert len(lines) > 0
        text = "\n".join(lines)
        assert "**Conversations** (1):" in text


class TestReplayEntryDetail:
    """Tests for _replay_entry_detail formatting."""

    def test_conversation_model_and_cost(self) -> None:
        data = {"model": "gemini-3-flash", "original": {"cost": 0.005}}
        assert _replay_entry_detail("conversation", data) == " — gemini-3-flash ($0.0050)"

    def test_conversation_model_only(self) -> None:
        data = {"model": "gpt-4o"}
        assert _replay_entry_detail("conversation", data) == " — gpt-4o"

    def test_conversation_empty(self) -> None:
        assert _replay_entry_detail("conversation", {}) == ""

    def test_task_function_path(self) -> None:
        data = {"function_path": "my_app.tasks:extract_insights"}
        assert _replay_entry_detail("task", data) == " — extract_insights"

    def test_flow_with_docs(self) -> None:
        data = {"function_path": "app:my_flow", "documents": [1, 2, 3]}
        assert _replay_entry_detail("flow", data) == " — my_flow (3 docs)"

    def test_flow_no_docs(self) -> None:
        data = {"function_path": "app:my_flow", "documents": []}
        assert _replay_entry_detail("flow", data) == " — my_flow"

    def test_unknown_label(self) -> None:
        assert _replay_entry_detail("unknown", {}) == ""
