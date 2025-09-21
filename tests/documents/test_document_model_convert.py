"""Tests for Document.model_convert method."""

from enum import StrEnum

import pytest

from ai_pipeline_core.documents import Document, FlowDocument, TaskDocument, TemporaryDocument
from ai_pipeline_core.exceptions import DocumentNameError


class SampleFlowDoc(FlowDocument):
    """Sample flow document for testing."""

    pass


class AnotherFlowDoc(FlowDocument):
    """Another flow document for testing."""

    pass


class SampleTaskDoc(TaskDocument):
    """Sample task document for testing."""

    pass


class AnotherTaskDoc(TaskDocument):
    """Another task document for testing."""

    pass


class RestrictedFlowDoc(FlowDocument):
    """Flow document with file name restrictions."""

    class FILES(StrEnum):
        CONFIG = "config.json"
        DATA = "data.yaml"


class TestModelConvert:
    """Test suite for Document.model_convert method."""

    def test_convert_task_to_flow(self):
        """Test converting TaskDocument to FlowDocument."""
        task_doc = SampleTaskDoc.create(
            name="temp.json", content={"data": "value"}, description="Task data"
        )

        flow_doc = task_doc.model_convert(SampleFlowDoc)

        assert isinstance(flow_doc, SampleFlowDoc)
        assert flow_doc.is_flow
        assert not flow_doc.is_task
        assert flow_doc.name == "temp.json"
        assert flow_doc.content == task_doc.content
        assert flow_doc.description == "Task data"
        assert flow_doc.sources == []

    def test_convert_flow_to_task(self):
        """Test converting FlowDocument to TaskDocument."""
        flow_doc = SampleFlowDoc.create(
            name="permanent.yaml",
            content={"key": "value"},
            description="Flow data",
            sources=["source1", "source2"],
        )

        task_doc = flow_doc.model_convert(SampleTaskDoc)

        assert isinstance(task_doc, SampleTaskDoc)
        assert task_doc.is_task
        assert not task_doc.is_flow
        assert task_doc.name == "permanent.yaml"
        assert task_doc.content == flow_doc.content
        assert task_doc.description == "Flow data"
        assert task_doc.sources == ["source1", "source2"]

    def test_convert_flow_to_different_flow(self):
        """Test converting between different FlowDocument types."""
        doc1 = SampleFlowDoc.create(name="data.json", content=[1, 2, 3], description="Numbers")

        doc2 = doc1.model_convert(AnotherFlowDoc)

        assert isinstance(doc2, AnotherFlowDoc)
        assert type(doc2) is not type(doc1)
        assert doc2.is_flow
        assert doc2.content == doc1.content

    def test_convert_with_updates(self):
        """Test converting with attribute updates."""
        original = SampleTaskDoc.create(
            name="original.json", content={"old": "data"}, description="Original"
        )

        converted = original.model_convert(
            SampleFlowDoc, update={"name": "updated.json", "description": "Updated description"}
        )

        assert converted.name == "updated.json"
        assert converted.description == "Updated description"
        assert converted.content == original.content  # Content unchanged

    def test_convert_with_content_update(self):
        """Test converting with content update."""
        original = SampleTaskDoc.create(name="data.json", content={"old": "value"})

        new_content = {"new": "value", "extra": "field"}
        converted = original.model_convert(SampleFlowDoc, update={"content": new_content.copy()})

        # Content should be updated via validator
        assert converted.parse(dict) == new_content
        assert converted.content != original.content

    def test_convert_with_sources_update(self):
        """Test updating sources during conversion."""
        source_doc = SampleFlowDoc.create(name="source.txt", content="Source data")

        derived = SampleTaskDoc.create(name="derived.txt", content="Processed data")

        final = derived.model_convert(
            AnotherFlowDoc, update={"sources": [source_doc.sha256, "https://api.example.com"]}
        )

        assert source_doc.sha256 in final.sources
        assert "https://api.example.com" in final.sources
        assert final.has_source(source_doc)

    def test_convert_preserves_sha256(self):
        """Test that converting preserves SHA256 when content unchanged."""
        doc = SampleTaskDoc.create(name="data.json", content={"test": "data"})

        converted = doc.model_convert(SampleFlowDoc)

        # Same content should have same SHA256
        assert converted.sha256 == doc.sha256
        assert converted.id == doc.id

    def test_convert_to_temporary(self):
        """Test converting to TemporaryDocument."""
        flow_doc = SampleFlowDoc.create(name="flow.json", content={"data": "value"})

        temp_doc = flow_doc.model_convert(TemporaryDocument)

        assert isinstance(temp_doc, TemporaryDocument)
        assert temp_doc.is_temporary
        assert not temp_doc.is_flow
        assert not temp_doc.is_task

    def test_convert_from_temporary(self):
        """Test converting from TemporaryDocument."""
        temp_doc = TemporaryDocument.create(name="temp.json", content={"temp": "data"})

        flow_doc = temp_doc.model_convert(SampleFlowDoc)

        assert isinstance(flow_doc, SampleFlowDoc)
        assert flow_doc.is_flow
        assert not flow_doc.is_temporary

    def test_deep_copy_sources(self):
        """Test deep copy of sources list."""
        original_sources = ["source1", "source2"]
        doc = SampleTaskDoc.create(name="data.json", content="data", sources=original_sources)

        # Without deep copy
        shallow = doc.model_convert(SampleFlowDoc, deep=False)
        # With deep copy
        deep = doc.model_convert(SampleFlowDoc, deep=True)

        # Both should have the same sources initially
        assert shallow.sources == original_sources
        assert deep.sources == original_sources

        # Both should be independent since Documents are frozen
        assert deep.sources is not doc.sources
        assert shallow.sources is not doc.sources
        # But they should have the same content
        assert deep.sources == doc.sources
        assert shallow.sources == doc.sources

    def test_convert_with_files_restriction(self):
        """Test converting to document with FILES restriction."""
        doc = SampleTaskDoc.create(name="config.json", content={"setting": "value"})

        # Should succeed with valid name
        converted = doc.model_convert(RestrictedFlowDoc)
        assert converted.name == "config.json"

        # Should fail with invalid name
        doc2 = SampleTaskDoc.create(name="invalid.json", content={"data": "value"})

        with pytest.raises(DocumentNameError):
            doc2.model_convert(RestrictedFlowDoc)

    def test_convert_with_name_update_to_restricted(self):
        """Test updating name when converting to restricted document."""
        doc = SampleTaskDoc.create(name="whatever.json", content={"data": "value"})

        # Update to valid name should work
        converted = doc.model_convert(RestrictedFlowDoc, update={"name": "data.yaml"})
        assert converted.name == "data.yaml"

        # Update to invalid name should fail
        with pytest.raises(DocumentNameError):
            doc.model_convert(RestrictedFlowDoc, update={"name": "invalid.txt"})

    def test_cannot_convert_to_abstract_document(self):
        """Test that converting to abstract Document class raises error."""
        doc = SampleTaskDoc.create(name="test.json", content={})

        with pytest.raises(TypeError, match="Cannot instantiate abstract Document"):
            doc.model_convert(Document)

    def test_cannot_convert_to_abstract_flow_document(self):
        """Test that converting to abstract FlowDocument class raises error."""
        doc = SampleTaskDoc.create(name="test.json", content={})

        with pytest.raises(TypeError, match="Cannot instantiate abstract FlowDocument"):
            doc.model_convert(FlowDocument)

    def test_cannot_convert_to_abstract_task_document(self):
        """Test that converting to abstract TaskDocument class raises error."""
        doc = SampleFlowDoc.create(name="test.json", content={})

        with pytest.raises(TypeError, match="Cannot instantiate abstract TaskDocument"):
            doc.model_convert(TaskDocument)

    def test_cannot_convert_to_non_document(self):
        """Test that converting to non-Document class raises error."""
        doc = SampleTaskDoc.create(name="test.json", content={})

        with pytest.raises(TypeError, match="must be a subclass of Document"):
            doc.model_convert(dict)  # type: ignore

        with pytest.raises(TypeError, match="must be a subclass of Document"):
            doc.model_convert(str)  # type: ignore

        with pytest.raises(TypeError, match="must be a subclass of Document"):
            doc.model_convert("not a class")  # type: ignore

    def test_model_copy_still_works(self):
        """Test that standard model_copy still works for same-type copying."""
        doc = SampleFlowDoc.create(name="test.json", content={"data": "value"}, description="Test")

        # Standard Pydantic model_copy should still work
        copied = doc.model_copy(update={"description": "Copied"})

        assert isinstance(copied, SampleFlowDoc)
        assert copied.name == doc.name
        assert copied.content == doc.content
        assert copied.description == "Copied"

    def test_convert_preserves_binary_content(self):
        """Test that binary content is preserved during conversion."""
        binary_data = b"\x00\x01\x02\x03\xff"
        doc = SampleTaskDoc(name="binary.bin", content=binary_data)

        converted = doc.model_convert(SampleFlowDoc)

        assert converted.content == binary_data
        assert converted.sha256 == doc.sha256

    def test_convert_with_null_description(self):
        """Test converting with None description."""
        doc = SampleTaskDoc.create(name="test.json", content={}, description="Original")

        # Clear description
        converted = doc.model_convert(SampleFlowDoc, update={"description": None})

        assert converted.description is None

    def test_convert_chain(self):
        """Test chaining multiple conversions."""
        # Start with task doc
        task = SampleTaskDoc.create(name="data.json", content={"step": 1})

        # Convert to flow
        flow = task.model_convert(SampleFlowDoc)
        assert flow.is_flow

        # Convert to another task
        task2 = flow.model_convert(AnotherTaskDoc)
        assert task2.is_task

        # Convert to temporary
        temp = task2.model_convert(TemporaryDocument)
        assert temp.is_temporary

        # All should have same content
        assert temp.content == task.content
        assert temp.sha256 == task.sha256
