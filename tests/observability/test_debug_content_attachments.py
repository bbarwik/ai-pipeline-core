"""Tests for attachment handling in ContentWriter._structure_documents()."""

import base64
from pathlib import Path

import pytest
import yaml

from ai_pipeline_core.observability import ArtifactStore, ContentWriter, TraceDebugConfig, reconstruct_span_content


@pytest.fixture
def config_with_store(tmp_path: Path) -> tuple[TraceDebugConfig, ArtifactStore]:
    """Create config and artifact store for testing."""
    config = TraceDebugConfig(path=tmp_path, max_element_bytes=100)
    store = ArtifactStore(tmp_path)
    return config, store


@pytest.fixture
def writer_with_store(config_with_store: tuple[TraceDebugConfig, ArtifactStore]) -> ContentWriter:
    """Create a ContentWriter with an artifact store."""
    config, store = config_with_store
    return ContentWriter(config, artifact_store=store)


@pytest.fixture
def writer_no_store(tmp_path: Path) -> ContentWriter:
    """Create a ContentWriter without an artifact store."""
    config = TraceDebugConfig(path=tmp_path, max_element_bytes=100)
    return ContentWriter(config)


class TestStructureDocumentsWithAttachments:
    """Tests for attachment handling in _structure_documents()."""

    def test_binary_attachment_externalized(self, writer_with_store: ContentWriter, tmp_path: Path) -> None:
        """Binary attachment exceeding max_element_bytes is externalized."""
        # Create binary data > 100 bytes (max_element_bytes)
        binary_data = b"\x89PNG" + b"\x00" * 200
        b64_content = base64.b64encode(binary_data).decode("ascii")

        docs = [
            {
                "class_name": "SampleTaskDocument",
                "name": "doc.txt",
                "content": "text content",
                "content_encoding": "utf-8",
                "attachments": [
                    {
                        "name": "screenshot.png",
                        "description": "A screenshot",
                        "content": b64_content,
                        "content_encoding": "base64",
                        "mime_type": "image/png",
                    },
                ],
            },
        ]

        tmp_path.mkdir(parents=True, exist_ok=True)
        writer_with_store.write(docs, tmp_path, "output")

        content = yaml.safe_load((tmp_path / "output.yaml").read_text())
        doc_entry = content["documents"][0]
        assert doc_entry["attachment_count"] == 1
        att = doc_entry["attachments"][0]
        assert att["name"] == "screenshot.png"
        assert att["description"] == "A screenshot"
        assert att["encoding"] == "base64"
        assert att["size_bytes"] == len(binary_data)
        assert "content_ref" in att
        assert "content" not in att
        assert att["content_ref"]["mime_type"] == "image/png"
        assert att["preview"] == f"[Binary attachment, {len(binary_data)} bytes]"

    def test_text_attachment_inline(self, writer_with_store: ContentWriter, tmp_path: Path) -> None:
        """Small text attachment kept inline."""
        docs = [
            {
                "class_name": "SampleTaskDocument",
                "name": "doc.txt",
                "content": "text content",
                "content_encoding": "utf-8",
                "attachments": [
                    {
                        "name": "notes.txt",
                        "content": "Short notes here",
                        "content_encoding": "utf-8",
                    },
                ],
            },
        ]

        tmp_path.mkdir(parents=True, exist_ok=True)
        writer_with_store.write(docs, tmp_path, "output")

        content = yaml.safe_load((tmp_path / "output.yaml").read_text())
        doc_entry = content["documents"][0]
        assert doc_entry["attachment_count"] == 1
        att = doc_entry["attachments"][0]
        assert att["name"] == "notes.txt"
        assert att["content"] == "Short notes here"
        assert att["size_bytes"] == len(b"Short notes here")

    def test_small_binary_attachment_inline(self, writer_with_store: ContentWriter, tmp_path: Path) -> None:
        """Binary attachment below max_element_bytes kept inline as base64."""
        small_binary = b"\x01\x02\x03"  # 3 bytes, well under 100
        b64_content = base64.b64encode(small_binary).decode("ascii")

        docs = [
            {
                "class_name": "SampleTaskDocument",
                "name": "doc.txt",
                "content": "text",
                "content_encoding": "utf-8",
                "attachments": [
                    {
                        "name": "tiny.bin",
                        "content": b64_content,
                        "content_encoding": "base64",
                    },
                ],
            },
        ]

        tmp_path.mkdir(parents=True, exist_ok=True)
        writer_with_store.write(docs, tmp_path, "output")

        content = yaml.safe_load((tmp_path / "output.yaml").read_text())
        att = content["documents"][0]["attachments"][0]
        assert att["content"] == b64_content  # inline
        assert "content_ref" not in att
        assert att["size_bytes"] == len(small_binary)

    def test_no_attachments_backwards_compatible(self, writer_with_store: ContentWriter, tmp_path: Path) -> None:
        """Document without attachments has no attachments key in output."""
        docs = [
            {
                "class_name": "SampleFlowDocument",
                "name": "report.md",
                "content": "Report text",
                "content_encoding": "utf-8",
            },
        ]

        tmp_path.mkdir(parents=True, exist_ok=True)
        writer_with_store.write(docs, tmp_path, "output")

        content = yaml.safe_load((tmp_path / "output.yaml").read_text())
        doc_entry = content["documents"][0]
        assert "attachments" not in doc_entry
        assert "attachment_count" not in doc_entry

    def test_attachment_count_in_output(self, writer_with_store: ContentWriter, tmp_path: Path) -> None:
        """Verify attachment_count field appears when attachments present."""
        docs = [
            {
                "class_name": "SampleTaskDocument",
                "name": "doc.txt",
                "content": "text",
                "content_encoding": "utf-8",
                "attachments": [
                    {"name": "a.txt", "content": "aaa", "content_encoding": "utf-8"},
                    {"name": "b.txt", "content": "bbb", "content_encoding": "utf-8"},
                    {"name": "c.txt", "content": "ccc", "content_encoding": "utf-8"},
                ],
            },
        ]

        tmp_path.mkdir(parents=True, exist_ok=True)
        writer_with_store.write(docs, tmp_path, "output")

        content = yaml.safe_load((tmp_path / "output.yaml").read_text())
        doc_entry = content["documents"][0]
        assert doc_entry["attachment_count"] == 3
        assert len(doc_entry["attachments"]) == 3

    def test_rehydrate_attachment_content_ref(self, writer_with_store: ContentWriter, tmp_path: Path) -> None:
        """Verify _rehydrate() resolves content_ref inside attachment entries."""
        # Create a large binary attachment that will be externalized
        binary_data = b"\xff" * 200  # > 100 bytes (max_element_bytes)
        b64_content = base64.b64encode(binary_data).decode("ascii")

        docs = [
            {
                "class_name": "SampleTaskDocument",
                "name": "doc.txt",
                "content": "text",
                "content_encoding": "utf-8",
                "attachments": [
                    {
                        "name": "large.bin",
                        "content": b64_content,
                        "content_encoding": "base64",
                        "mime_type": "application/octet-stream",
                    },
                ],
            },
        ]

        # Write to create the externalized artifacts
        span_dir = tmp_path / "span"
        span_dir.mkdir(parents=True, exist_ok=True)
        writer_with_store.write(docs, span_dir, "output")

        # Verify the YAML has a content_ref and no inline content
        raw_content = yaml.safe_load((span_dir / "output.yaml").read_text())
        att = raw_content["documents"][0]["attachments"][0]
        assert "content_ref" in att
        assert "content" not in att

        # Reconstruct and verify the content_ref is resolved
        reconstructed = reconstruct_span_content(tmp_path, span_dir, "output")
        rehydrated_att = reconstructed["documents"][0]["attachments"][0]
        assert "content_ref" not in rehydrated_att
        assert rehydrated_att["content"] == binary_data

    def test_large_text_attachment_externalized(self, writer_with_store: ContentWriter, tmp_path: Path) -> None:
        """Large text attachment is externalized with excerpt."""
        long_text = "a" * 500  # > 100 bytes

        docs = [
            {
                "class_name": "SampleTaskDocument",
                "name": "doc.txt",
                "content": "text",
                "content_encoding": "utf-8",
                "attachments": [
                    {
                        "name": "essay.txt",
                        "content": long_text,
                        "content_encoding": "utf-8",
                    },
                ],
            },
        ]

        tmp_path.mkdir(parents=True, exist_ok=True)
        writer_with_store.write(docs, tmp_path, "output")

        content = yaml.safe_load((tmp_path / "output.yaml").read_text())
        att = content["documents"][0]["attachments"][0]
        assert "content_ref" in att
        assert "content" not in att
        assert "excerpt" in att
        assert att["excerpt"].startswith("a" * 100)  # excerpt from beginning
        assert "[TRUNCATED" in att["excerpt"]
