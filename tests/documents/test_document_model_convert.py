"""Tests for Document.retype method."""

from enum import StrEnum

import pytest

from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents._context import _suppress_document_registration
from ai_pipeline_core.exceptions import DocumentNameError


class SampleFlowDoc(Document):
    pass


class AnotherFlowDoc(Document):
    pass


class SampleTaskDoc(Document):
    pass


class AnotherTaskDoc(Document):
    pass


class RestrictedFlowDoc(Document):
    class FILES(StrEnum):
        CONFIG = "config.json"
        DATA = "data.yaml"


@pytest.fixture(autouse=True)
def _suppress_registration():
    with _suppress_document_registration():
        yield


class TestModelConvert:
    """Test suite for Document.retype method."""

    def test_convert_between_types(self):
        """Test converting between Document subclasses."""
        task_doc = SampleTaskDoc.create_root(name="temp.json", content={"data": "value"}, description="Task data", reason="test input")

        flow_doc = task_doc.retype(SampleFlowDoc, preserve_provenance=True)

        assert isinstance(flow_doc, SampleFlowDoc)
        assert flow_doc.name == "temp.json"
        assert flow_doc.content == task_doc.content
        assert flow_doc.description == "Task data"
        assert flow_doc.derived_from == ()

    def test_convert_preserves_derived_from(self):
        """Test conversion preserves derived_from."""
        flow_doc = SampleFlowDoc.create(
            name="permanent.yaml",
            content={"key": "value"},
            description="Flow data",
            derived_from=("https://example.com/source1", "https://example.com/source2"),
        )

        task_doc = flow_doc.retype(SampleTaskDoc, preserve_provenance=True)

        assert isinstance(task_doc, SampleTaskDoc)
        assert task_doc.name == "permanent.yaml"
        assert task_doc.content == flow_doc.content
        assert task_doc.description == "Flow data"
        assert task_doc.derived_from == ("https://example.com/source1", "https://example.com/source2")

    def test_convert_to_different_type(self):
        """Test converting between different Document types."""
        doc1 = SampleFlowDoc.create_root(name="data.json", content=[1, 2, 3], description="Numbers", reason="test input")

        doc2 = doc1.retype(AnotherFlowDoc, preserve_provenance=True)

        assert isinstance(doc2, AnotherFlowDoc)
        assert type(doc2) is not type(doc1)
        assert doc2.content == doc1.content

    def test_convert_with_updates(self):
        """Test converting with attribute updates."""
        original = SampleTaskDoc.create_root(name="original.json", content={"old": "data"}, description="Original", reason="test input")

        converted = original.retype(SampleFlowDoc, preserve_provenance=True, update={"name": "updated.json", "description": "Updated description"})

        assert converted.name == "updated.json"
        assert converted.description == "Updated description"
        assert converted.content == original.content

    def test_convert_with_content_update(self):
        """Test converting with content update."""
        original = SampleTaskDoc.create_root(name="data.json", content={"old": "value"}, reason="test input")

        new_content = {"new": "value", "extra": "field"}
        converted = original.retype(SampleFlowDoc, preserve_provenance=True, update={"content": new_content.copy()})

        assert converted.parse(dict) == new_content
        assert converted.content != original.content

    def test_convert_with_derived_from_update(self):
        """Test updating derived_from during conversion."""
        source_doc = SampleFlowDoc.create_root(name="source.txt", content="Source data", reason="test input")

        derived = SampleTaskDoc.create_root(name="derived.txt", content="Processed data", reason="test input")

        final = derived.retype(AnotherFlowDoc, preserve_provenance=True, update={"derived_from": (source_doc.sha256, "https://api.example.com")})

        assert source_doc.sha256 in final.derived_from
        assert "https://api.example.com" in final.derived_from
        assert final.has_derived_from(source_doc)

    def test_convert_preserves_sha256(self):
        """Test that converting preserves SHA256 when content and name unchanged."""
        doc = SampleTaskDoc.create_root(name="data.json", content={"test": "data"}, reason="test input")

        converted = doc.retype(SampleFlowDoc, preserve_provenance=True)

        assert converted.sha256 == doc.sha256
        assert converted.id == doc.id

    def test_derived_from_tuple_preserved(self):
        """Test that derived_from tuple is preserved through conversion."""
        original_derived_from = ("https://example.com/source1", "https://example.com/source2")
        doc = SampleTaskDoc.create(name="data.json", content="data", derived_from=original_derived_from)
        converted = doc.retype(SampleFlowDoc, preserve_provenance=True)
        assert converted.derived_from == original_derived_from

    def test_convert_with_files_restriction(self):
        """Test converting to document with FILES restriction."""
        doc = SampleTaskDoc.create_root(name="config.json", content={"setting": "value"}, reason="test input")

        converted = doc.retype(RestrictedFlowDoc, preserve_provenance=True)
        assert converted.name == "config.json"

        doc2 = SampleTaskDoc.create_root(name="invalid.json", content={"data": "value"}, reason="test input")

        with pytest.raises(DocumentNameError):
            doc2.retype(RestrictedFlowDoc, preserve_provenance=True)

    def test_convert_with_name_update_to_restricted(self):
        """Test updating name when converting to restricted document."""
        doc = SampleTaskDoc.create_root(name="whatever.json", content={"data": "value"}, reason="test input")

        converted = doc.retype(RestrictedFlowDoc, preserve_provenance=True, update={"name": "data.yaml"})
        assert converted.name == "data.yaml"

        with pytest.raises(DocumentNameError):
            doc.retype(RestrictedFlowDoc, preserve_provenance=True, update={"name": "invalid.txt"})

    def test_cannot_convert_to_document(self):
        """Test that converting to base Document class raises error."""
        doc = SampleTaskDoc.create_root(name="test.json", content={}, reason="test input")

        with pytest.raises(TypeError):
            doc.retype(Document, preserve_provenance=True)

    def test_cannot_convert_to_non_document(self):
        """Test that converting to non-Document class raises error."""
        doc = SampleTaskDoc.create_root(name="test.json", content={}, reason="test input")

        with pytest.raises(TypeError, match="must be a subclass of Document"):
            doc.retype(dict, preserve_provenance=True)  # type: ignore

        with pytest.raises(TypeError, match="must be a subclass of Document"):
            doc.retype(str, preserve_provenance=True)  # type: ignore

    def test_convert_preserves_binary_content(self):
        """Test that binary content is preserved during conversion."""
        binary_data = b"\x00\x01\x02\x03\xff"
        doc = SampleTaskDoc(name="binary.bin", content=binary_data)

        converted = doc.retype(SampleFlowDoc, preserve_provenance=True)

        assert converted.content == binary_data
        assert converted.sha256 == doc.sha256

    def test_convert_with_null_description(self):
        """Test converting with None description."""
        doc = SampleTaskDoc.create_root(name="test.json", content={}, description="Original", reason="test input")

        converted = doc.retype(SampleFlowDoc, preserve_provenance=True, update={"description": None})

        assert converted.description is None

    def test_convert_chain(self):
        """Test chaining multiple conversions."""
        task = SampleTaskDoc.create_root(name="data.json", content={"step": 1}, reason="test input")

        flow = task.retype(SampleFlowDoc, preserve_provenance=True)
        assert isinstance(flow, SampleFlowDoc)

        task2 = flow.retype(AnotherTaskDoc, preserve_provenance=True)
        assert isinstance(task2, AnotherTaskDoc)

        # All should have same content and hash
        assert task2.content == task.content
        assert task2.sha256 == task.sha256
