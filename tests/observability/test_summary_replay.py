"""Tests for replay section generation in trace summary."""

from datetime import UTC, datetime
from pathlib import Path

import yaml

from ai_pipeline_core.observability._debug._config import SpanInfo, TraceState
from ai_pipeline_core.observability._debug._summary import (
    _build_replay_section,
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

    def test_conversation_yaml_counted(self, tmp_path: Path) -> None:
        """Conversation replay files appear as a count."""
        trace = _make_trace(tmp_path)
        span = _add_span(trace, "llm_call")
        (span.path / "conversation.yaml").write_text(yaml.dump({"payload_type": "conversation", "model": "gemini-3-flash"}))

        lines = _build_replay_section(trace)
        text = "\n".join(lines)
        assert "## Replay" in text
        assert "1 conversation" in text

    def test_task_yaml_counted(self, tmp_path: Path) -> None:
        """Task replay files appear as a count."""
        trace = _make_trace(tmp_path)
        span = _add_span(trace, "extract")
        (span.path / "task.yaml").write_text(yaml.dump({"payload_type": "pipeline_task", "function_path": "app:extract_insights"}))

        lines = _build_replay_section(trace)
        text = "\n".join(lines)
        assert "1 task" in text

    def test_multiple_conversations_shows_plural(self, tmp_path: Path) -> None:
        """Multiple replay files show plural count."""
        trace = _make_trace(tmp_path)
        for i in range(10):
            span = _add_span(trace, f"llm_{i}")
            (span.path / "conversation.yaml").write_text(yaml.dump({"payload_type": "conversation", "model": f"model-{i}"}))

        lines = _build_replay_section(trace)
        text = "\n".join(lines)
        assert "10 conversations" in text

    def test_correct_pluralization_singular(self, tmp_path: Path) -> None:
        """Single item uses singular form."""
        trace = _make_trace(tmp_path)
        span = _add_span(trace, "conv_span")
        (span.path / "conversation.yaml").write_text(yaml.dump({"payload_type": "conversation"}))

        lines = _build_replay_section(trace)
        text = "\n".join(lines)
        assert "1 conversation" in text
        assert "conversations" not in text

    def test_example_commands_included(self, tmp_path: Path) -> None:
        """Example CLI commands reference the first found replay file."""
        trace = _make_trace(tmp_path)
        span = _add_span(trace, "my_task")
        (span.path / "task.yaml").write_text(yaml.dump({"payload_type": "pipeline_task"}))

        lines = _build_replay_section(trace)
        text = "\n".join(lines)
        assert "ai-replay show" in text
        assert "ai-replay run" in text
        assert "task.yaml" in text

    def test_all_three_types(self, tmp_path: Path) -> None:
        """All three payload types counted correctly."""
        trace = _make_trace(tmp_path)

        s1 = _add_span(trace, "conv_span")
        (s1.path / "conversation.yaml").write_text(yaml.dump({"payload_type": "conversation"}))

        s2 = _add_span(trace, "task_span")
        (s2.path / "task.yaml").write_text(yaml.dump({"payload_type": "pipeline_task"}))

        s3 = _add_span(trace, "flow_span")
        (s3.path / "flow.yaml").write_text(yaml.dump({"payload_type": "pipeline_flow"}))

        lines = _build_replay_section(trace)
        text = "\n".join(lines)
        assert "1 conversation" in text
        assert "1 task" in text
        assert "1 flow" in text

    def test_integrated_in_generate_summary(self, tmp_path: Path) -> None:
        """Replay section appears in the full summary."""
        trace = _make_trace(tmp_path)
        span = _add_span(trace, "my_llm")
        (span.path / "conversation.yaml").write_text(yaml.dump({"payload_type": "conversation"}))

        summary = generate_summary(trace)
        assert "## Replay" in summary

    def test_no_replay_section_in_summary_when_empty(self, tmp_path: Path) -> None:
        """Summary does not contain Replay section when no replay files exist."""
        trace = _make_trace(tmp_path)
        _add_span(trace, "plain_span")

        summary = generate_summary(trace)
        assert "## Replay" not in summary

    def test_malformed_yaml_still_counted(self, tmp_path: Path) -> None:
        """Malformed YAML files are still counted (file exists)."""
        trace = _make_trace(tmp_path)
        span = _add_span(trace, "bad")
        (span.path / "conversation.yaml").write_text("{{invalid yaml content")

        lines = _build_replay_section(trace)
        assert len(lines) > 0
        text = "\n".join(lines)
        assert "1 conversation" in text


# ---------------------------------------------------------------------------
# Bug 7 (summary): execution tree indentation
# ---------------------------------------------------------------------------


class TestExecutionTreeIndentation:
    def test_execution_tree_has_indentation(self, tmp_path: Path) -> None:
        """Execution tree shows increasing indentation for nested spans."""
        trace = _make_trace(tmp_path)
        root = _add_span(trace, "root_span")
        child = _add_span(trace, "child_span", parent_id=root.span_id)
        child.depth = 1
        root.children.append(child.span_id)
        grandchild = _add_span(trace, "gc_span", parent_id=child.span_id)
        grandchild.depth = 2
        child.children.append(grandchild.span_id)

        summary = generate_summary(trace)
        lines = summary.split("\n")
        root_line = next(ln for ln in lines if "root_span" in ln)
        child_line = next(ln for ln in lines if "child_span" in ln)
        gc_line = next(ln for ln in lines if "gc_span" in ln)
        root_indent = len(root_line) - len(root_line.lstrip())
        child_indent = len(child_line) - len(child_line.lstrip())
        gc_indent = len(gc_line) - len(gc_line.lstrip())
        assert child_indent > root_indent
        assert gc_indent > child_indent
