"""Tests for trace reconstruction from ClickHouse data."""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
import yaml

clickhouse_connect = pytest.importorskip("clickhouse_connect")

from ai_pipeline_core.observability._debug._reconstruction import (
    TraceDownloader,
    _parse_json_safe,
)


@pytest.fixture
def mock_client():
    """Create a mock ClickHouse client."""
    client = MagicMock()
    client.query.return_value = MagicMock(result_rows=[])
    return client


@pytest.fixture
def downloader(mock_client):
    return TraceDownloader(client=mock_client)


def _make_span_row(
    *,
    span_id: str = "span01",
    trace_id: str = "trace01",
    parent_span_id: str | None = None,
    name: str = "test_span",
    span_type: str = "default",
    status: str = "completed",
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    duration_ms: int = 100,
    cost: float = 0.0,
    tokens_input: int = 0,
    tokens_output: int = 0,
    llm_model: str | None = None,
    span_order: int = 1,
    input_json: str = "",
    output_json: str = "",
    replay_payload: str = "",
    attributes_json: str = "{}",
    events_json: str = "[]",
) -> tuple[str, ...]:
    """Build a row tuple matching the joined query columns."""
    now = start_time or datetime.now(UTC)
    end = end_time or now
    return (
        span_id,
        trace_id,
        parent_span_id,
        name,
        span_type,
        status,
        now,
        end,
        duration_ms,
        cost,
        tokens_input,
        tokens_output,
        llm_model,
        span_order,
        input_json,
        output_json,
        replay_payload,
        attributes_json,
        events_json,
    )


class TestParseJsonSafe:
    def test_empty_string(self):
        assert _parse_json_safe("") is None

    def test_valid_json(self):
        assert _parse_json_safe('{"key": "value"}') == {"key": "value"}

    def test_invalid_json_returns_raw(self):
        assert _parse_json_safe("not json") == "not json"

    def test_json_list(self):
        assert _parse_json_safe("[1, 2, 3]") == [1, 2, 3]


class TestTraceReconstruction:
    def test_builds_hierarchy_from_parent_child_spans(self, downloader, mock_client, tmp_path):
        execution_id = uuid4()
        root = _make_span_row(span_id="root", name="root_flow", span_order=1)
        child1 = _make_span_row(span_id="child1", parent_span_id="root", name="task_a", span_order=2)
        child2 = _make_span_row(span_id="child2", parent_span_id="root", name="task_b", span_order=3)

        mock_client.query.side_effect = [
            MagicMock(result_rows=[root, child1, child2]),  # span query
            MagicMock(result_rows=[]),  # run metadata query
        ]

        result = downloader.download_trace(execution_id, tmp_path / "trace")
        assert result.exists()

        # Root should have a directory
        root_dirs = [d for d in result.iterdir() if d.is_dir() and d.name.startswith("001_")]
        assert len(root_dirs) == 1

        # Children should be nested inside root
        root_dir = root_dirs[0]
        child_dirs = sorted(d.name for d in root_dir.iterdir() if d.is_dir())
        assert len(child_dirs) == 2
        assert any("task_a" in d for d in child_dirs)
        assert any("task_b" in d for d in child_dirs)

    def test_writes_span_yaml_input_output(self, downloader, mock_client, tmp_path):
        execution_id = uuid4()
        row = _make_span_row(
            span_id="span01",
            name="my_task",
            span_order=1,
            input_json='{"prompt": "hello"}',
            output_json='{"result": "world"}',
        )
        mock_client.query.side_effect = [
            MagicMock(result_rows=[row]),
            MagicMock(result_rows=[]),
        ]

        result = downloader.download_trace(execution_id, tmp_path / "trace")
        span_dirs = [d for d in result.iterdir() if d.is_dir()]
        assert len(span_dirs) == 1

        span_dir = span_dirs[0]
        assert (span_dir / "span.yaml").exists()
        assert (span_dir / "input.yaml").exists()
        assert (span_dir / "output.yaml").exists()

    def test_writes_replay_files_by_type(self, downloader, mock_client, tmp_path):
        execution_id = uuid4()
        conv_row = _make_span_row(
            span_id="s1",
            name="conv_span",
            span_order=1,
            replay_payload=json.dumps({"payload_type": "conversation", "model": "gpt-4"}),
        )
        task_row = _make_span_row(
            span_id="s2",
            name="task_span",
            span_order=2,
            replay_payload=json.dumps({"payload_type": "pipeline_task", "function_path": "mod:func"}),
        )
        flow_row = _make_span_row(
            span_id="s3",
            name="flow_span",
            span_order=3,
            replay_payload=json.dumps({"payload_type": "pipeline_flow", "function_path": "mod:flow"}),
        )
        mock_client.query.side_effect = [
            MagicMock(result_rows=[conv_row, task_row, flow_row]),
            MagicMock(result_rows=[]),
        ]

        result = downloader.download_trace(execution_id, tmp_path / "trace")
        span_dirs = sorted(d for d in result.iterdir() if d.is_dir())
        assert (span_dirs[0] / "conversation.yaml").exists()
        assert (span_dirs[1] / "task.yaml").exists()
        assert (span_dirs[2] / "flow.yaml").exists()

    def test_generates_summary_and_indexes(self, downloader, mock_client, tmp_path):
        execution_id = uuid4()
        row = _make_span_row(span_id="s1", name="root_task", span_order=1)
        mock_client.query.side_effect = [
            MagicMock(result_rows=[row]),
            MagicMock(result_rows=[]),
        ]

        result = downloader.download_trace(execution_id, tmp_path / "trace")
        assert (result / "summary.md").exists()
        assert (result / "llm_calls.yaml").exists()

    def test_writes_task_replay_with_doc_refs(self, downloader, mock_client, tmp_path):
        execution_id = uuid4()
        replay = json.dumps({
            "payload_type": "pipeline_task",
            "function_path": "mod:func",
            "kwargs": {
                "doc": {
                    "$doc_ref": "ABCDEF123456789012345678901234567890123456789012",
                    "class_name": "MyDoc",
                    "name": "test.md",
                },
            },
        })
        row = _make_span_row(span_id="s1", name="task_span", span_order=1, replay_payload=replay)
        mock_client.query.side_effect = [
            MagicMock(result_rows=[row]),
            MagicMock(result_rows=[]),
        ]

        result = downloader.download_trace(execution_id, tmp_path / "trace")
        span_dirs = [d for d in result.iterdir() if d.is_dir()]
        assert (span_dirs[0] / "task.yaml").exists()

    def test_writes_events_yaml(self, downloader, mock_client, tmp_path):
        execution_id = uuid4()
        events = json.dumps([{"name": "log_event", "timestamp": 1500000000, "attributes": {"key": "val"}}])
        row = _make_span_row(span_id="s1", name="my_span", span_order=1, events_json=events)
        mock_client.query.side_effect = [
            MagicMock(result_rows=[row]),
            MagicMock(result_rows=[]),
        ]

        result = downloader.download_trace(execution_id, tmp_path / "trace")
        span_dirs = [d for d in result.iterdir() if d.is_dir()]
        assert (span_dirs[0] / "events.yaml").exists()

    def test_empty_trace_creates_directory(self, downloader, mock_client, tmp_path):
        execution_id = uuid4()
        mock_client.query.side_effect = [
            MagicMock(result_rows=[]),
            MagicMock(result_rows=[]),
        ]

        result = downloader.download_trace(execution_id, tmp_path / "trace")
        assert result.exists()

    def test_span_with_llm_info_included_in_llm_index(self, downloader, mock_client, tmp_path):
        execution_id = uuid4()
        attrs = json.dumps({
            "lmnr.span.type": "LLM",
            "gen_ai.usage.input_tokens": 100,
            "gen_ai.usage.output_tokens": 50,
            "gen_ai.usage.cost": 0.005,
            "gen_ai.request.model": "gpt-4",
        })
        row = _make_span_row(
            span_id="s1",
            name="llm_call",
            span_type="llm",
            span_order=1,
            cost=0.005,
            tokens_input=100,
            tokens_output=50,
            llm_model="gpt-4",
            attributes_json=attrs,
        )
        mock_client.query.side_effect = [
            MagicMock(result_rows=[row]),
            MagicMock(result_rows=[]),
        ]

        result = downloader.download_trace(execution_id, tmp_path / "trace")
        llm_calls_path = result / "llm_calls.yaml"
        assert llm_calls_path.exists()
        llm_data = yaml.safe_load(llm_calls_path.read_text())
        assert llm_data["llm_call_count"] >= 1


class TestCrossPipelineDownload:
    def test_follow_children_discovers_and_downloads_child_traces(self, mock_client, tmp_path):
        downloader = TraceDownloader(client=mock_client)
        parent_id = uuid4()
        child_exec_id = uuid4()

        parent_span = _make_span_row(span_id="p1", name="parent_flow", span_order=1)
        child_span = _make_span_row(span_id="c1", name="child_flow", span_order=1)

        mock_client.query.side_effect = [
            MagicMock(result_rows=[parent_span]),  # parent span query
            MagicMock(result_rows=[("parent-run", "parent_flow", "scope", "completed", None, None, 0, 0, "{}")]),  # parent run meta
            MagicMock(result_rows=[(str(child_exec_id), "parent-run-abc123")]),  # child runs query
            MagicMock(result_rows=[child_span]),  # child span query
            MagicMock(result_rows=[]),  # child run meta
        ]

        result = downloader.download_trace(parent_id, tmp_path / "trace", follow_children=True)
        assert result.exists()
        child_dir = tmp_path / "trace" / "child_parent-run-abc123"
        assert child_dir.exists()

    def test_follow_children_noop_without_run_metadata(self, mock_client, tmp_path):
        downloader = TraceDownloader(client=mock_client)
        execution_id = uuid4()
        row = _make_span_row(span_id="s1", name="task", span_order=1)

        mock_client.query.side_effect = [
            MagicMock(result_rows=[row]),
            MagicMock(result_rows=[]),  # no run metadata
        ]

        result = downloader.download_trace(execution_id, tmp_path / "trace", follow_children=True)
        assert result.exists()
        # Only the trace dir, no child dirs
        child_dirs = [d for d in result.iterdir() if d.is_dir() and d.name.startswith("child_")]
        assert len(child_dirs) == 0


class TestDocumentDownload:
    def test_include_documents_downloads_referenced_docs(self, mock_client, tmp_path):
        downloader = TraceDownloader(client=mock_client)
        execution_id = uuid4()

        doc_sha = "ABCDEF" + "0" * 46
        replay = json.dumps({
            "payload_type": "pipeline_task",
            "kwargs": {"doc": {"$doc_ref": doc_sha, "class_name": "MyDoc", "name": "report.md"}},
        })
        row = _make_span_row(span_id="s1", name="task", span_order=1, replay_payload=replay)

        content_sha = "CONTENT" + "0" * 45
        mock_client.query.side_effect = [
            MagicMock(result_rows=[row]),  # span query
            MagicMock(result_rows=[]),  # run meta
            # document_index query
            MagicMock(result_rows=[(doc_sha, content_sha, "MyDoc", "report.md", "", "text/markdown", [], [], [], [], [])]),
            # document_content query
            MagicMock(result_rows=[(content_sha, b"# Hello World", 13)]),
        ]

        result = downloader.download_trace(execution_id, tmp_path / "trace", include_documents=True)
        doc_dir = result / "MyDoc"
        assert doc_dir.exists()
        doc_files = list(doc_dir.glob("report_*"))
        assert len(doc_files) >= 1
        meta_files = list(doc_dir.glob("*.meta.json"))
        assert len(meta_files) >= 1

    def test_include_documents_noop_without_doc_refs(self, mock_client, tmp_path):
        downloader = TraceDownloader(client=mock_client)
        execution_id = uuid4()
        row = _make_span_row(span_id="s1", name="task", span_order=1)

        mock_client.query.side_effect = [
            MagicMock(result_rows=[row]),
            MagicMock(result_rows=[]),
        ]

        result = downloader.download_trace(execution_id, tmp_path / "trace", include_documents=True)
        assert result.exists()
        # No document class directories created
        assert not any(d.is_dir() and not d.name.startswith("0") for d in result.iterdir())
