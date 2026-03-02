"""Tests for document download from ClickHouse for replay support."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

clickhouse_connect = pytest.importorskip("clickhouse_connect")

from ai_pipeline_core.observability._download_docs import (
    _collect_doc_refs,
    _extract_doc_refs,
    fetch_trace_documents,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_replay_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")


def _make_ch_doc_row(
    *,
    doc_sha256: str = "ABCDEF123456",
    content_sha256: str = "content_sha_1",
    class_name: str = "MyDocument",
    name: str = "report.md",
    content: bytes = b"# Report content",
) -> tuple[Any, ...]:
    """Build a tuple matching the document query result columns."""
    return (
        doc_sha256,
        content_sha256,
        class_name,
        name,
        "",  # description
        "text/markdown",  # mime_type
        [],  # derived_from
        [],  # triggered_by
        [],  # attachment_names
        [],  # attachment_descriptions
        [],  # attachment_sha256s
        content,
        len(content),
    )


# ---------------------------------------------------------------------------
# _extract_doc_refs
# ---------------------------------------------------------------------------


class TestExtractDocRefs:
    def test_extracts_from_flat_dict(self):
        data = {"$doc_ref": "SHA256HASH", "class_name": "MyDoc", "name": "doc.md"}
        refs: dict[str, tuple[str, str]] = {}
        _extract_doc_refs(data, refs)
        assert refs == {"SHA256HASH": ("MyDoc", "doc.md")}

    def test_extracts_from_nested_task_arguments(self):
        data = {
            "payload_type": "pipeline_task",
            "arguments": {
                "input_doc": {"$doc_ref": "SHA1", "class_name": "InputDoc", "name": "input.json"},
                "config": {"$doc_ref": "SHA2", "class_name": "Config", "name": "config.yaml"},
                "plain_arg": "not a doc ref",
            },
        }
        refs: dict[str, tuple[str, str]] = {}
        _extract_doc_refs(data, refs)
        assert len(refs) == 2
        assert refs["SHA1"] == ("InputDoc", "input.json")
        assert refs["SHA2"] == ("Config", "config.yaml")

    def test_extracts_from_list(self):
        data = {
            "context": [
                {"$doc_ref": "SHA_A", "class_name": "DocA", "name": "a.md"},
                {"$doc_ref": "SHA_B", "class_name": "DocB", "name": "b.md"},
            ]
        }
        refs: dict[str, tuple[str, str]] = {}
        _extract_doc_refs(data, refs)
        assert len(refs) == 2

    def test_empty_dict(self):
        refs: dict[str, tuple[str, str]] = {}
        _extract_doc_refs({}, refs)
        assert refs == {}

    def test_defaults_for_missing_class_name_and_name(self):
        data = {"$doc_ref": "SHA_ONLY"}
        refs: dict[str, tuple[str, str]] = {}
        _extract_doc_refs(data, refs)
        assert refs["SHA_ONLY"] == ("Document", "unknown")


# ---------------------------------------------------------------------------
# _collect_doc_refs
# ---------------------------------------------------------------------------


class TestCollectDocRefs:
    def test_scans_conversation_yaml(self, tmp_path: Path):
        trace = tmp_path / ".trace"
        span_dir = trace / "001_my_span"
        _write_replay_yaml(
            span_dir / "conversation.yaml",
            {
                "payload_type": "conversation",
                "context": [{"$doc_ref": "CONV_SHA", "class_name": "SpecDoc", "name": "spec.json"}],
            },
        )
        refs = _collect_doc_refs(trace)
        assert "CONV_SHA" in refs
        assert refs["CONV_SHA"] == ("SpecDoc", "spec.json")

    def test_scans_task_yaml(self, tmp_path: Path):
        trace = tmp_path / ".trace"
        span_dir = trace / "002_task"
        _write_replay_yaml(
            span_dir / "task.yaml",
            {
                "payload_type": "pipeline_task",
                "arguments": {"doc": {"$doc_ref": "TASK_SHA", "class_name": "TaskDoc", "name": "t.md"}},
            },
        )
        refs = _collect_doc_refs(trace)
        assert "TASK_SHA" in refs

    def test_scans_flow_yaml(self, tmp_path: Path):
        trace = tmp_path / ".trace"
        span_dir = trace / "003_flow"
        _write_replay_yaml(
            span_dir / "flow.yaml",
            {
                "payload_type": "pipeline_flow",
                "documents": [{"$doc_ref": "FLOW_SHA", "class_name": "FlowDoc", "name": "f.md"}],
            },
        )
        refs = _collect_doc_refs(trace)
        assert "FLOW_SHA" in refs

    def test_deduplicates_across_files(self, tmp_path: Path):
        trace = tmp_path / ".trace"
        for i, filename in enumerate(["conversation.yaml", "task.yaml"]):
            span_dir = trace / f"00{i}_span"
            _write_replay_yaml(
                span_dir / filename,
                {
                    "context": [{"$doc_ref": "SAME_SHA", "class_name": "Doc", "name": "d.md"}],
                },
            )
        refs = _collect_doc_refs(trace)
        assert len(refs) == 1

    def test_ignores_non_replay_yaml(self, tmp_path: Path):
        trace = tmp_path / ".trace"
        span_dir = trace / "001_span"
        _write_replay_yaml(span_dir / "span.yaml", {"$doc_ref": "SHOULD_IGNORE", "class_name": "X", "name": "x"})
        refs = _collect_doc_refs(trace)
        assert refs == {}

    def test_empty_trace_dir(self, tmp_path: Path):
        trace = tmp_path / ".trace"
        trace.mkdir()
        refs = _collect_doc_refs(trace)
        assert refs == {}


# ---------------------------------------------------------------------------
# fetch_trace_documents
# ---------------------------------------------------------------------------


class TestFetchTraceDocuments:
    def test_writes_document_in_local_store_format(self, tmp_path: Path):
        # Set up trace with a replay file referencing a document
        trace = tmp_path / ".trace"
        span_dir = trace / "001_span"
        _write_replay_yaml(
            span_dir / "conversation.yaml",
            {
                "context": [{"$doc_ref": "ABCDEF123456", "class_name": "MyDocument", "name": "report.md"}],
            },
        )

        # Mock CH client returning the document
        client = MagicMock()
        client.query.return_value = MagicMock(result_rows=[_make_ch_doc_row()])

        found, total = fetch_trace_documents(client, tmp_path)

        assert total == 1
        assert found == 1

        # Verify file layout matches LocalDocumentStore
        doc_dir = tmp_path / "MyDocument"
        assert doc_dir.is_dir()
        content_file = doc_dir / "report_ABCDEF.md"
        assert content_file.exists()
        assert content_file.read_bytes() == b"# Report content"
        meta_file = doc_dir / "report_ABCDEF.md.meta.json"
        assert meta_file.exists()
        meta = json.loads(meta_file.read_text())
        assert meta["document_sha256"] == "ABCDEF123456"
        assert meta["class_name"] == "MyDocument"
        assert meta["name"] == "report.md"
        assert meta["mime_type"] == "text/markdown"

    def test_writes_multiple_documents(self, tmp_path: Path):
        trace = tmp_path / ".trace"
        span_dir = trace / "001_span"
        _write_replay_yaml(
            span_dir / "task.yaml",
            {
                "arguments": {
                    "doc_a": {"$doc_ref": "SHA_A_123456", "class_name": "DocA", "name": "a.json"},
                    "doc_b": {"$doc_ref": "SHA_B_789012", "class_name": "DocB", "name": "b.txt"},
                },
            },
        )

        client = MagicMock()
        client.query.return_value = MagicMock(
            result_rows=[
                _make_ch_doc_row(doc_sha256="SHA_A_123456", class_name="DocA", name="a.json", content=b'{"key": 1}'),
                _make_ch_doc_row(doc_sha256="SHA_B_789012", class_name="DocB", name="b.txt", content=b"text content"),
            ]
        )

        found, total = fetch_trace_documents(client, tmp_path)

        assert found == 2
        assert total == 2
        assert (tmp_path / "DocA" / "a_SHA_A_.json").exists()
        assert (tmp_path / "DocB" / "b_SHA_B_.txt").exists()

    def test_handles_documents_with_attachments(self, tmp_path: Path):
        trace = tmp_path / ".trace"
        span_dir = trace / "001_span"
        _write_replay_yaml(
            span_dir / "conversation.yaml",
            {
                "context": [{"$doc_ref": "DOC_SHA_12345", "class_name": "Report", "name": "report.md"}],
            },
        )

        # Document with one attachment
        doc_row = (
            "DOC_SHA_12345",
            "content_sha_1",
            "Report",
            "report.md",
            "A report",
            "text/markdown",
            [],  # derived_from
            [],  # triggered_by
            ["screenshot.png"],  # attachment_names
            ["Screenshot"],  # attachment_descriptions
            ["att_sha_abc"],  # attachment_sha256s
            b"# Report",
            8,
        )

        call_count = 0

        def mock_query(query, parameters=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Main doc query
                return MagicMock(result_rows=[doc_row])
            # Attachment query
            return MagicMock(result_rows=[("att_sha_abc", b"\x89PNG", 4)])

        client = MagicMock()
        client.query.side_effect = mock_query

        found, total = fetch_trace_documents(client, tmp_path)

        assert found == 1
        att_dir = tmp_path / "Report" / "report_DOC_SH.md.att"
        assert att_dir.is_dir()
        assert (att_dir / "screenshot.png").exists()
        assert (att_dir / "screenshot.png").read_bytes() == b"\x89PNG"

    def test_missing_sha256_warns_and_continues(self, tmp_path: Path):
        trace = tmp_path / ".trace"
        span_dir = trace / "001_span"
        _write_replay_yaml(
            span_dir / "conversation.yaml",
            {
                "context": [
                    {"$doc_ref": "FOUND_SHA_1234", "class_name": "Doc", "name": "found.md"},
                    {"$doc_ref": "MISSING_SHA_12", "class_name": "Doc", "name": "missing.md"},
                ],
            },
        )

        # CH only returns one document
        client = MagicMock()
        client.query.return_value = MagicMock(
            result_rows=[
                _make_ch_doc_row(doc_sha256="FOUND_SHA_1234", name="found.md"),
            ]
        )

        found, total = fetch_trace_documents(client, tmp_path)

        assert total == 2
        assert found == 1
        assert (tmp_path / "MyDocument" / "found_FOUND_.md").exists()

    def test_no_trace_dir_returns_zero(self, tmp_path: Path):
        found, total = fetch_trace_documents(MagicMock(), tmp_path)
        assert found == 0
        assert total == 0

    def test_no_replay_files_returns_zero(self, tmp_path: Path):
        trace = tmp_path / ".trace"
        span_dir = trace / "001_span"
        span_dir.mkdir(parents=True)
        (span_dir / "span.yaml").write_text("name: test")

        found, total = fetch_trace_documents(MagicMock(), tmp_path)
        assert found == 0
        assert total == 0
