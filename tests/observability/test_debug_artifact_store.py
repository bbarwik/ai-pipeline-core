"""Tests for ArtifactStore and content externalization (V3 features)."""

from pathlib import Path

import yaml

from ai_pipeline_core.observability import (
    ArtifactStore,
    ContentWriter,
    TraceDebugConfig,
)

from tests.observability.test_helpers import reconstruct_span_content


class TestArtifactStore:
    """Tests for artifact store functionality."""

    def test_store_text_creates_artifact(self, tmp_path: Path) -> None:
        """Test storing text creates artifact file."""
        store = ArtifactStore(tmp_path)
        ref = store.store_text("Hello world")

        assert ref.hash.startswith("sha256:")
        assert ref.path.startswith("artifacts/sha256/")
        assert ref.size_bytes == len(b"Hello world")
        assert ref.mime_type == "text/plain"
        assert ref.encoding == "utf-8"

        # Verify artifact file exists
        artifact_path = tmp_path / ref.path
        assert artifact_path.exists()
        assert artifact_path.read_text(encoding="utf-8") == "Hello world"

    def test_store_binary_creates_artifact(self, tmp_path: Path) -> None:
        """Test storing binary data creates artifact file."""
        store = ArtifactStore(tmp_path)
        binary_data = b"\x89PNG\r\n\x1a\n"
        ref = store.store_binary(binary_data, "image/png")

        assert ref.hash.startswith("sha256:")
        assert ref.path.endswith(".png")
        assert ref.size_bytes == len(binary_data)
        assert ref.mime_type == "image/png"
        assert ref.encoding == "binary"

        # Verify artifact file exists
        artifact_path = tmp_path / ref.path
        assert artifact_path.exists()
        assert artifact_path.read_bytes() == binary_data

    def test_deduplication(self, tmp_path: Path) -> None:
        """Test identical content is deduplicated."""
        store = ArtifactStore(tmp_path)

        ref1 = store.store_text("Duplicate content")
        ref2 = store.store_text("Duplicate content")

        # Same hash and path
        assert ref1.hash == ref2.hash
        assert ref1.path == ref2.path

        # Only one physical file
        artifact_files = list((tmp_path / "artifacts" / "sha256").glob("*.txt"))
        assert len(artifact_files) == 1

    def test_different_content_different_files(self, tmp_path: Path) -> None:
        """Test different content creates different files."""
        store = ArtifactStore(tmp_path)

        ref1 = store.store_text("Content A")
        ref2 = store.store_text("Content B")

        # Different hashes and paths
        assert ref1.hash != ref2.hash
        assert ref1.path != ref2.path

        # Two physical files
        artifact_files = list((tmp_path / "artifacts" / "sha256").glob("*.txt"))
        assert len(artifact_files) == 2

    def test_flat_directory_structure(self, tmp_path: Path) -> None:
        """Test artifacts are stored in flat directory."""
        store = ArtifactStore(tmp_path)
        ref = store.store_text("Test content")

        # Path should follow pattern: artifacts/sha256/<hash>.txt
        parts = Path(ref.path).parts
        assert parts[0] == "artifacts"
        assert parts[1] == "sha256"
        assert len(parts) == 3
        assert parts[2].endswith(".txt")

    def test_get_stats(self, tmp_path: Path) -> None:
        """Test getting deduplication statistics."""
        store = ArtifactStore(tmp_path)

        store.store_text("Content 1")
        store.store_text("Content 2")
        store.store_text("Content 1")  # Duplicate - returns same ref

        stats = store.get_stats()
        assert stats["unique_artifacts"] == 2  # Only 2 unique files
        assert stats["total_references"] == 2  # 2 unique hashes tracked
        assert stats["dedup_ratio"] == 1.0  # 2/2


class TestContentExternalization:
    """Tests for content externalization to artifacts."""

    def test_large_text_element_externalized(self, tmp_path: Path) -> None:
        """Test large text element is externalized to artifact."""
        config = TraceDebugConfig(path=tmp_path, max_element_bytes=100)
        store = ArtifactStore(tmp_path)
        writer = ContentWriter(config, store)

        messages = [
            {"role": "system", "content": "x" * 500},  # Large content
        ]

        tmp_path.mkdir(parents=True, exist_ok=True)
        writer.write(messages, tmp_path, "input")

        # Check content structure
        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        text_part = content["messages"][0]["parts"][0]

        # Should have content_ref and excerpt
        assert "content_ref" in text_part
        assert "excerpt" in text_part
        assert "content" not in text_part or len(text_part.get("content", "")) < 500

        # Verify artifact exists
        ref_path = text_part["content_ref"]["path"]
        artifact_path = tmp_path / ref_path
        assert artifact_path.exists()
        assert artifact_path.read_text(encoding="utf-8") == "x" * 500

    def test_small_text_element_inline(self, tmp_path: Path) -> None:
        """Test small text element stays inline."""
        config = TraceDebugConfig(path=tmp_path, max_element_bytes=1000)
        store = ArtifactStore(tmp_path)
        writer = ContentWriter(config, store)

        messages = [
            {"role": "user", "content": "Hello!"},  # Small content
        ]

        tmp_path.mkdir(parents=True, exist_ok=True)
        writer.write(messages, tmp_path, "input")

        # Check content structure
        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        text_part = content["messages"][0]["parts"][0]

        # Should be inline
        assert "content" in text_part
        assert "content_ref" not in text_part
        assert text_part["content"] == "Hello!"

    def test_mixed_inline_and_external(self, tmp_path: Path) -> None:
        """Test mix of inline and externalized content."""
        config = TraceDebugConfig(path=tmp_path, max_element_bytes=100)
        store = ArtifactStore(tmp_path)
        writer = ContentWriter(config, store)

        messages = [
            {"role": "system", "content": "x" * 500},  # External
            {"role": "user", "content": "Hello!"},  # Inline
            {"role": "assistant", "content": "y" * 300},  # External
        ]

        tmp_path.mkdir(parents=True, exist_ok=True)
        writer.write(messages, tmp_path, "input")

        content = yaml.safe_load((tmp_path / "input.yaml").read_text())

        # First message - externalized
        assert "content_ref" in content["messages"][0]["parts"][0]

        # Second message - inline
        assert content["messages"][1]["parts"][0]["content"] == "Hello!"

        # Third message - externalized
        assert "content_ref" in content["messages"][2]["parts"][0]

    def test_document_externalization(self, tmp_path: Path) -> None:
        """Test large document content is externalized."""
        config = TraceDebugConfig(path=tmp_path, max_element_bytes=100)
        store = ArtifactStore(tmp_path)
        writer = ContentWriter(config, store)

        docs = [
            {
                "class_name": "SampleFlowDocument",
                "name": "large_doc.txt",
                "content": "z" * 500,
                "content_encoding": "utf-8",
            },
        ]

        tmp_path.mkdir(parents=True, exist_ok=True)
        writer.write(docs, tmp_path, "output")

        content = yaml.safe_load((tmp_path / "output.yaml").read_text())
        doc = content["documents"][0]

        # Should be externalized
        assert "content_ref" in doc
        assert "excerpt" in doc

        # Verify artifact
        artifact_path = tmp_path / doc["content_ref"]["path"]
        assert artifact_path.read_text(encoding="utf-8") == "z" * 500


class TestContentReconstruction:
    """Tests for lossless content reconstruction."""

    def test_reconstruct_externalized_text(self, tmp_path: Path) -> None:
        """Test reconstructing externalized text content."""
        config = TraceDebugConfig(path=tmp_path, max_element_bytes=100)
        store = ArtifactStore(tmp_path)
        writer = ContentWriter(config, store)

        original_text = "a" * 500
        messages = [{"role": "system", "content": original_text}]

        span_dir = tmp_path / "span"
        span_dir.mkdir(parents=True, exist_ok=True)
        writer.write(messages, span_dir, "input")

        # Reconstruct
        reconstructed = reconstruct_span_content(tmp_path, span_dir, "input")

        # Should have full content
        assert reconstructed["messages"][0]["parts"][0]["content"] == original_text
        assert "content_ref" not in reconstructed["messages"][0]["parts"][0]

    def test_reconstruct_mixed_content(self, tmp_path: Path) -> None:
        """Test reconstructing mix of inline and externalized content."""
        config = TraceDebugConfig(path=tmp_path, max_element_bytes=100)
        store = ArtifactStore(tmp_path)
        writer = ContentWriter(config, store)

        messages = [
            {"role": "system", "content": "x" * 500},  # External
            {"role": "user", "content": "Hello!"},  # Inline
        ]

        span_dir = tmp_path / "span"
        span_dir.mkdir(parents=True, exist_ok=True)
        writer.write(messages, span_dir, "input")

        # Reconstruct
        reconstructed = reconstruct_span_content(tmp_path, span_dir, "input")

        # Both should have full content
        assert reconstructed["messages"][0]["parts"][0]["content"] == "x" * 500
        assert reconstructed["messages"][1]["parts"][0]["content"] == "Hello!"

    def test_reconstruct_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        """Test reconstructing nonexistent file returns empty dict."""
        span_dir = tmp_path / "span"
        span_dir.mkdir(parents=True, exist_ok=True)

        reconstructed = reconstruct_span_content(tmp_path, span_dir, "input")
        assert reconstructed == {}

    def test_roundtrip_preserves_structure(self, tmp_path: Path) -> None:
        """Test roundtrip write->reconstruct preserves all data."""
        config = TraceDebugConfig(path=tmp_path, max_element_bytes=100)
        store = ArtifactStore(tmp_path)
        writer = ContentWriter(config, store)

        original = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "b" * 300},
                    {"type": "text", "text": "Small"},
                ],
            },
            {"role": "assistant", "content": "c" * 400},
        ]

        span_dir = tmp_path / "span"
        span_dir.mkdir(parents=True, exist_ok=True)
        writer.write(original, span_dir, "input")

        # Reconstruct
        reconstructed = reconstruct_span_content(tmp_path, span_dir, "input")

        # Verify structure preserved
        assert reconstructed["type"] == "llm_messages"
        assert reconstructed["message_count"] == 2
        assert len(reconstructed["messages"][0]["parts"]) == 2
        assert reconstructed["messages"][0]["parts"][0]["content"] == "b" * 300
        assert reconstructed["messages"][0]["parts"][1]["content"] == "Small"
        assert reconstructed["messages"][1]["parts"][0]["content"] == "c" * 400
