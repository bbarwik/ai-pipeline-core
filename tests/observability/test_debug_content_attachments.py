"""Tests for attachment handling in ContentWriter._structure_documents()."""

import base64
from pathlib import Path

import pytest
import yaml

from ai_pipeline_core.observability._debug import ContentWriter, TraceDebugConfig


@pytest.fixture
def config(tmp_path: Path) -> TraceDebugConfig:
    """Create config for testing."""
    return TraceDebugConfig(path=tmp_path)


@pytest.fixture
def writer(config: TraceDebugConfig) -> ContentWriter:
    """Create a ContentWriter."""
    return ContentWriter(config)


class TestStructureDocumentsWithAttachments:
    """Tests for attachment handling in _structure_documents()."""

    def test_binary_attachment_placeholder(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Binary attachment is replaced with placeholder."""
        binary_data = b"\x89PNG" + b"\x00" * 200
        b64_content = base64.b64encode(binary_data).decode("ascii")

        docs = [
            {
                "class_name": "SampleTaskDocument",
                "name": "doc.txt",
                "content": "text content",
                "attachments": [
                    {
                        "name": "screenshot.png",
                        "description": "A screenshot",
                        "content": f"data:image/png;base64,{b64_content}",
                        "mime_type": "image/png",
                    },
                ],
            },
        ]

        tmp_path.mkdir(parents=True, exist_ok=True)
        writer.write(docs, tmp_path, "output")

        content = yaml.safe_load((tmp_path / "output.yaml").read_text())
        doc_entry = content["documents"][0]
        assert doc_entry["attachment_count"] == 1
        att = doc_entry["attachments"][0]
        assert att["name"] == "screenshot.png"
        assert att["description"] == "A screenshot"
        # Binary content should be replaced with placeholder
        assert att["content"] == "[binary content removed]"

    def test_text_attachment_inline(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Small text attachment kept inline."""
        docs = [
            {
                "class_name": "SampleTaskDocument",
                "name": "doc.txt",
                "content": "text content",
                "attachments": [
                    {
                        "name": "notes.txt",
                        "content": "Short notes here",
                    },
                ],
            },
        ]

        tmp_path.mkdir(parents=True, exist_ok=True)
        writer.write(docs, tmp_path, "output")

        content = yaml.safe_load((tmp_path / "output.yaml").read_text())
        doc_entry = content["documents"][0]
        assert doc_entry["attachment_count"] == 1
        att = doc_entry["attachments"][0]
        assert att["name"] == "notes.txt"
        assert att["content"] == "Short notes here"
        assert att["size_bytes"] == len(b"Short notes here")

    def test_no_attachments_backwards_compatible(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Document without attachments has no attachments key in output."""
        docs = [
            {
                "class_name": "SampleFlowDocument",
                "name": "report.md",
                "content": "Report text",
            },
        ]

        tmp_path.mkdir(parents=True, exist_ok=True)
        writer.write(docs, tmp_path, "output")

        content = yaml.safe_load((tmp_path / "output.yaml").read_text())
        doc_entry = content["documents"][0]
        assert "attachments" not in doc_entry
        assert "attachment_count" not in doc_entry

    def test_attachment_count_in_output(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Verify attachment_count field appears when attachments present."""
        docs = [
            {
                "class_name": "SampleTaskDocument",
                "name": "doc.txt",
                "content": "text",
                "attachments": [
                    {"name": "a.txt", "content": "aaa"},
                    {"name": "b.txt", "content": "bbb"},
                    {"name": "c.txt", "content": "ccc"},
                ],
            },
        ]

        tmp_path.mkdir(parents=True, exist_ok=True)
        writer.write(docs, tmp_path, "output")

        content = yaml.safe_load((tmp_path / "output.yaml").read_text())
        doc_entry = content["documents"][0]
        assert doc_entry["attachment_count"] == 3
        assert len(doc_entry["attachments"]) == 3
