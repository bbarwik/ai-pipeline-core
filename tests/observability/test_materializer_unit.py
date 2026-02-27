"""Unit tests for TraceMaterializer — directory creation, span writing, metrics, finalize, wrapper merging."""

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from ai_pipeline_core.observability._debug._config import TraceDebugConfig
from ai_pipeline_core.observability._debug._materializer import TraceMaterializer, _sanitize_name
from ai_pipeline_core.observability._span_data import SpanData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPOCH = datetime(2024, 1, 1, tzinfo=UTC)
_LATER = datetime(2024, 1, 1, 0, 0, 1, tzinfo=UTC)


def _make_span_data(
    *,
    span_id: str = "span1",
    trace_id: str = "trace1",
    parent_span_id: str | None = None,
    name: str = "test_span",
    span_order: int = 1,
    cost: float = 0.0,
    tokens_input: int = 0,
    tokens_output: int = 0,
    status: str = "completed",
    input_json: str = "",
    output_json: str = "",
    replay_payload: str = "",
    events: tuple[dict[str, object], ...] = (),
    attributes: dict[str, object] | None = None,
) -> SpanData:
    return SpanData(
        execution_id=None,
        span_id=span_id,
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        name=name,
        run_id="",
        flow_name="",
        run_scope="",
        span_type="trace",
        status=status,
        start_time=_EPOCH,
        end_time=_LATER,
        duration_ms=1000,
        span_order=span_order,
        cost=cost,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        tokens_cached=0,
        llm_model=None,
        error_message="",
        input_json=input_json,
        output_json=output_json,
        replay_payload=replay_payload,
        attributes=attributes or {},
        events=events,
        input_doc_sha256s=(),
        output_doc_sha256s=(),
    )


def _make_materializer(tmp_path: Path, *, generate_summary: bool = True) -> TraceMaterializer:
    config = TraceDebugConfig(path=tmp_path, generate_summary=generate_summary)
    return TraceMaterializer(config)


# ---------------------------------------------------------------------------
# on_span_start — directory creation
# ---------------------------------------------------------------------------


class TestOnSpanStart:
    def test_creates_dir_for_root_span(self, tmp_path):
        mat = _make_materializer(tmp_path)
        span = _make_span_data(span_id="root1", name="my_root")
        mat.on_span_start(span)
        dirs = list(tmp_path.iterdir())
        assert len(dirs) == 1
        assert "my_root" in dirs[0].name

    def test_nested_child_under_parent(self, tmp_path):
        mat = _make_materializer(tmp_path)
        parent = _make_span_data(span_id="parent1", name="parent")
        mat.on_span_start(parent)
        child = _make_span_data(span_id="child1", parent_span_id="parent1", name="child")
        mat.on_span_start(child)
        # Child dir should be nested inside parent dir
        parent_dir = next(tmp_path.iterdir())
        child_dirs = [d for d in parent_dir.iterdir() if d.is_dir()]
        assert len(child_dirs) == 1
        assert "child" in child_dirs[0].name


# ---------------------------------------------------------------------------
# add_span — file writing
# ---------------------------------------------------------------------------


def _find_span_dir(base: Path, name_fragment: str) -> Path:
    """Find the first span directory containing name_fragment."""
    for d in base.iterdir():
        if d.is_dir() and name_fragment in d.name:
            return d
    raise FileNotFoundError(f"No span dir matching '{name_fragment}' in {base}")


class TestAddSpan:
    def test_writes_span_yaml(self, tmp_path):
        mat = _make_materializer(tmp_path)
        span = _make_span_data(
            span_id="s1",
            name="my_span",
            input_json='{"key": "val"}',
            output_json='"result"',
        )
        mat.on_span_start(span)
        mat.add_span(span)
        span_dir = _find_span_dir(tmp_path, "my_span")
        span_yaml = span_dir / "span.yaml"
        assert span_yaml.exists()
        content = yaml.safe_load(span_yaml.read_text())
        assert content["span_id"] == "s1"
        assert content["name"] == "my_span"
        assert content["status"] == "completed"

    def test_writes_replay_conversation(self, tmp_path):
        mat = _make_materializer(tmp_path)
        replay = json.dumps({"payload_type": "conversation", "model": "gpt-5"})
        span = _make_span_data(span_id="s1", replay_payload=replay)
        mat.on_span_start(span)
        mat.add_span(span)
        span_dir = _find_span_dir(tmp_path, "test")
        assert (span_dir / "conversation.yaml").exists()

    def test_writes_replay_task(self, tmp_path):
        mat = _make_materializer(tmp_path)
        replay = json.dumps({"payload_type": "pipeline_task", "function_path": "mod:fn"})
        span = _make_span_data(span_id="s1", replay_payload=replay)
        mat.on_span_start(span)
        mat.add_span(span)
        span_dir = _find_span_dir(tmp_path, "test")
        assert (span_dir / "task.yaml").exists()

    def test_writes_events_yaml(self, tmp_path):
        mat = _make_materializer(tmp_path)
        span = _make_span_data(
            span_id="s1",
            events=({"name": "evt1", "timestamp": 1_000_000_000, "attributes": {"k": "v"}},),
        )
        mat.on_span_start(span)
        mat.add_span(span)
        span_dir = _find_span_dir(tmp_path, "test")
        assert (span_dir / "events.yaml").exists()

    def test_without_prior_on_start(self, tmp_path):
        """Download path: add_span without on_span_start creates dir on demand."""
        mat = _make_materializer(tmp_path)
        span = _make_span_data(span_id="s1", name="downloaded")
        mat.add_span(span)
        span_dir = _find_span_dir(tmp_path, "downloaded")
        assert (span_dir / "span.yaml").exists()


# ---------------------------------------------------------------------------
# record_filtered_llm_metrics
# ---------------------------------------------------------------------------


class TestRecordFilteredLlmMetrics:
    def test_accumulates_costs(self, tmp_path):
        mat = _make_materializer(tmp_path)
        # Create a trace first
        root = _make_span_data(span_id="root", trace_id="t1")
        mat.on_span_start(root)
        # Record filtered metrics
        filtered1 = _make_span_data(trace_id="t1", cost=0.01, tokens_input=100, tokens_output=50)
        filtered2 = _make_span_data(trace_id="t1", cost=0.02, tokens_input=200, tokens_output=100)
        mat.record_filtered_llm_metrics(filtered1)
        mat.record_filtered_llm_metrics(filtered2)
        trace = mat._traces["t1"]
        assert trace.llm_call_count == 2
        assert trace.total_cost == 0.03
        assert trace.total_tokens == 450  # 100+50 + 200+100

    def test_zero_cost_skipped(self, tmp_path):
        mat = _make_materializer(tmp_path)
        root = _make_span_data(span_id="root", trace_id="t1")
        mat.on_span_start(root)
        filtered = _make_span_data(trace_id="t1", cost=0.0, tokens_input=0, tokens_output=0)
        mat.record_filtered_llm_metrics(filtered)
        trace = mat._traces["t1"]
        assert trace.llm_call_count == 0

    def test_unknown_trace_noop(self, tmp_path):
        mat = _make_materializer(tmp_path)
        filtered = _make_span_data(trace_id="nonexistent", cost=0.1)
        # Should not raise
        mat.record_filtered_llm_metrics(filtered)


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------


class TestFinalize:
    def test_generates_summary_and_costs(self, tmp_path):
        mat = _make_materializer(tmp_path, generate_summary=True)
        span = _make_span_data(
            span_id="root",
            trace_id="t1",
            attributes={"gen_ai.usage.input_tokens": 100, "gen_ai.usage.output_tokens": 50, "gen_ai.usage.cost": 0.01},
        )
        mat.on_span_start(span)
        mat.add_span(span)
        # After add_span, trace should be auto-finalized
        assert (tmp_path / "summary.md").exists()

    def test_finalize_all(self, tmp_path):
        mat = _make_materializer(tmp_path, generate_summary=False)
        # Create two traces that won't auto-finalize (spans still "running")
        for tid in ("t1", "t2"):
            root = _make_span_data(span_id=f"root_{tid}", trace_id=tid, name=f"trace_{tid}")
            mat.on_span_start(root)
        assert len(mat._traces) == 2
        mat.finalize_all()
        assert len(mat._traces) == 0


# ---------------------------------------------------------------------------
# Initial clean
# ---------------------------------------------------------------------------


class TestInitialClean:
    def test_directory_cleaned_only_once(self, tmp_path):
        mat = _make_materializer(tmp_path)
        # Create some pre-existing data
        old_dir = tmp_path / "old_span"
        old_dir.mkdir()
        (old_dir / "data.txt").write_text("old data")
        # First trace creation cleans
        span1 = _make_span_data(span_id="s1", trace_id="t1", name="span1")
        mat.on_span_start(span1)
        assert not old_dir.exists()
        assert mat._initial_clean_done is True
        # Add second trace data
        span2 = _make_span_data(span_id="s2", trace_id="t2", name="span2")
        mat.on_span_start(span2)
        # First trace dir should still exist (not cleaned again)
        dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(dirs) == 2


# ---------------------------------------------------------------------------
# Wrapper span detection and merging
# ---------------------------------------------------------------------------


class TestWrapperSpanDetection:
    def test_wrapper_identified(self, tmp_path):
        mat = _make_materializer(tmp_path)
        # Create parent (wrapper) and child with matching names
        parent = _make_span_data(span_id="wrapper1", trace_id="t1", name="my_task-abc123")
        mat.on_span_start(parent)
        mat.add_span(parent)
        # Write span.yaml with input type "none" for wrapper detection
        trace = mat._traces.get("t1")
        if trace is None:
            # trace was auto-finalized, re-create
            mat._traces["t1"] = mat._get_or_create_trace("t1", "test")
            trace = mat._traces["t1"]
            mat.on_span_start(parent)
            mat.add_span(parent)
            trace = mat._traces.get("t1")
        # Re-do with span that keeps trace alive
        mat2 = _make_materializer(tmp_path / "wrapper_test")
        wrapper = _make_span_data(span_id="w1", trace_id="t1", name="my_task-abc")
        mat2.on_span_start(wrapper)
        child = _make_span_data(span_id="c1", trace_id="t1", parent_span_id="w1", name="my_task-abc")
        mat2.on_span_start(child)
        # Write wrapper span.yaml with input type "none"
        trace = mat2._traces["t1"]
        wrapper_info = trace.spans["w1"]
        wrapper_info.prefect_info = {"run_id": "r1"}
        span_meta = {"input": {"type": "none"}, "output": {"type": "file"}}
        (wrapper_info.path / "span.yaml").write_text(yaml.dump(span_meta))
        from ai_pipeline_core.observability._debug._materializer import _detect_wrapper_spans

        wrappers = _detect_wrapper_spans(trace)
        assert "w1" in wrappers


class TestWrapperMerge:
    def test_reparents_child(self, tmp_path):
        mat = _make_materializer(tmp_path / "merge_test")
        grandparent = _make_span_data(span_id="gp", trace_id="t1", name="grandparent")
        mat.on_span_start(grandparent)
        wrapper = _make_span_data(span_id="w1", trace_id="t1", parent_span_id="gp", name="task-abc")
        mat.on_span_start(wrapper)
        child = _make_span_data(span_id="c1", trace_id="t1", parent_span_id="w1", name="task-abc")
        mat.on_span_start(child)
        trace = mat._traces["t1"]
        # Set up wrapper conditions
        wrapper_info = trace.spans["w1"]
        wrapper_info.prefect_info = {"run_id": "r1"}
        span_meta = {"input": {"type": "none"}}
        (wrapper_info.path / "span.yaml").write_text(yaml.dump(span_meta))
        from ai_pipeline_core.observability._debug._materializer import _merge_wrapper_spans

        _merge_wrapper_spans(trace)
        # Child should now point to grandparent
        assert trace.spans["c1"].parent_id == "gp"
        # Wrapper's children should be cleared
        assert trace.spans["w1"].children == []


# ---------------------------------------------------------------------------
# _sanitize_name
# ---------------------------------------------------------------------------


class TestSanitizeName:
    def test_special_chars_replaced(self):
        result = _sanitize_name('file<>:"/\\|?*name')
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result

    def test_truncation(self):
        long_name = "a" * 100
        result = _sanitize_name(long_name)
        assert len(result) <= 20

    def test_empty_fallback(self):
        assert _sanitize_name("...") == "span"
