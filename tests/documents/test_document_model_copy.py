"""Tests for Document.model_copy method (Pydantic's built-in method)."""

from enum import StrEnum

from ai_pipeline_core.documents import FlowDocument, TemporaryDocument


class SampleFlowDoc(FlowDocument):
    """Sample flow document for testing."""

    pass


class RestrictedFlowDoc(FlowDocument):
    """Flow document with file name restrictions."""

    class FILES(StrEnum):
        CONFIG = "config.json"
        DATA = "data.yaml"


class TestModelCopy:
    """Test suite for Document.model_copy method (Pydantic built-in)."""

    def test_basic_copy(self):
        """Test basic document copying."""
        doc = SampleFlowDoc.create(
            name="test.json", content={"data": "value"}, description="Original"
        )

        copied = doc.model_copy()

        assert isinstance(copied, SampleFlowDoc)
        assert copied.name == doc.name
        assert copied.content == doc.content
        assert copied.description == doc.description
        assert copied.sources == doc.sources
        assert copied.sha256 == doc.sha256

    def test_copy_with_update(self):
        """Test copying with attribute updates."""
        doc = SampleFlowDoc.create(
            name="test.json", content={"data": "value"}, description="Original"
        )

        copied = doc.model_copy(update={"description": "Updated"})

        assert copied.name == doc.name
        assert copied.content == doc.content
        assert copied.description == "Updated"
        assert copied.sources == doc.sources

    def test_copy_with_name_update(self):
        """Test updating name during copy."""
        doc = SampleFlowDoc.create(name="original.json", content={"data": "value"})

        copied = doc.model_copy(update={"name": "copied.json"})

        assert copied.name == "copied.json"
        assert copied.content == doc.content

    def test_copy_with_content_update(self):
        """Test updating content during copy."""
        doc = SampleFlowDoc.create(name="test.json", content={"old": "data"})

        # model_copy doesn't validate - content must be bytes already
        new_content = b'{"new": "data"}'
        copied = doc.model_copy(update={"content": new_content})

        # Content is raw bytes
        assert copied.content == new_content
        assert copied.sha256 != doc.sha256  # Different content = different hash

    def test_copy_with_sources_update(self):
        """Test updating sources during copy."""
        doc = SampleFlowDoc.create(name="test.json", content="data", sources=["source1"])

        copied = doc.model_copy(update={"sources": ["source2", "source3"]})

        assert copied.sources == ["source2", "source3"]
        assert copied.sources != doc.sources

    def test_copy_preserves_type(self):
        """Test that model_copy preserves the document type."""
        # Test with FlowDocument
        flow_doc = SampleFlowDoc.create(name="flow.json", content={})
        flow_copy = flow_doc.model_copy()
        assert isinstance(flow_copy, SampleFlowDoc)
        assert flow_copy.is_flow

        # Test with TemporaryDocument
        temp_doc = TemporaryDocument.create(name="temp.json", content={})
        temp_copy = temp_doc.model_copy()
        assert isinstance(temp_copy, TemporaryDocument)
        assert temp_copy.is_temporary

    def test_copy_with_invalid_name_for_restricted(self):
        """Test that model_copy doesn't validate name for restricted documents."""
        doc = RestrictedFlowDoc.create(name="config.json", content={})

        # Valid name update should work
        copied = doc.model_copy(update={"name": "data.yaml"})
        assert copied.name == "data.yaml"

        # Invalid name is allowed - model_copy doesn't validate
        # (validation only happens on create/init)
        copied2 = doc.model_copy(update={"name": "invalid.txt"})
        assert copied2.name == "invalid.txt"

    def test_deep_copy(self):
        """Test deep copy of document."""
        sources = ["source1", "source2"]
        doc = SampleFlowDoc.create(name="test.json", content={"data": "value"}, sources=sources)

        # Shallow copy (default)
        shallow = doc.model_copy(deep=False)
        # Deep copy
        deep = doc.model_copy(deep=True)

        # Both should have same values
        assert shallow.sources == sources
        assert deep.sources == sources

        # Shallow copy shares the same sources list
        assert shallow.sources is doc.sources
        # Deep copy creates a new list
        assert deep.sources is not doc.sources

    def test_copy_with_null_description(self):
        """Test copying with None description."""
        doc = SampleFlowDoc.create(name="test.json", content={}, description="Original")

        copied = doc.model_copy(update={"description": None})

        assert copied.description is None

    def test_copy_preserves_sha256_when_content_unchanged(self):
        """Test that SHA256 is preserved when content is unchanged."""
        doc = SampleFlowDoc.create(name="test.json", content={"data": "value"})

        copied = doc.model_copy(update={"description": "New description"})

        # Same content = same SHA256
        assert copied.sha256 == doc.sha256
        assert copied.id == doc.id

    def test_copy_changes_sha256_when_content_changed(self):
        """Test that SHA256 changes when content is changed."""
        doc = SampleFlowDoc.create(name="test.json", content={"data": "value"})

        # model_copy requires bytes for content
        new_content = b'{"other": "value"}'
        copied = doc.model_copy(update={"content": new_content})

        # Different content = different SHA256
        assert copied.sha256 != doc.sha256
        assert copied.id != doc.id

    def test_copy_binary_content(self):
        """Test copying document with binary content."""
        binary_data = b"\x00\x01\x02\x03\xff"
        doc = SampleFlowDoc(name="binary.bin", content=binary_data)

        copied = doc.model_copy()

        assert copied.content == binary_data
        assert copied.sha256 == doc.sha256

    def test_copy_with_multiple_updates(self):
        """Test copying with multiple attribute updates."""
        doc = SampleFlowDoc.create(
            name="original.json",
            content={"original": "data"},
            description="Original description",
            sources=["source1"],
        )

        # model_copy requires bytes for content
        copied = doc.model_copy(
            update={
                "name": "updated.json",
                "content": b'{"updated": "data"}',
                "description": "Updated description",
                "sources": ["source2", "source3"],
            }
        )

        assert copied.name == "updated.json"
        assert copied.content == b'{"updated": "data"}'
        assert copied.description == "Updated description"
        assert copied.sources == ["source2", "source3"]

    def test_copy_immutability(self):
        """Test that original document remains unchanged after copy."""
        original_content = {"original": "data"}
        original_sources = ["source1"]

        doc = SampleFlowDoc.create(
            name="test.json",
            content=original_content.copy(),
            description="Original",
            sources=original_sources.copy(),
        )

        # Make copy with updates
        copied = doc.model_copy(
            update={
                "name": "copied.json",
                "content": {"new": "data"},
                "description": "Copied",
                "sources": ["source2"],
            }
        )

        # Original should be unchanged
        assert doc.name == "test.json"
        assert doc.parse(dict) == original_content
        assert doc.description == "Original"
        assert doc.sources == original_sources

        # Copy should have new values
        assert copied.name == "copied.json"
        assert copied.content == {"new": "data"}  # model_copy doesn't validate
        assert copied.description == "Copied"
        assert copied.sources == ["source2"]
