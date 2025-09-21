"""Test document serialization debug fields."""

import pytest

from ai_pipeline_core.documents import FlowDocument, TaskDocument, TemporaryDocument


class DebugSampleDocument(FlowDocument):
    """Sample document for debug field testing."""

    pass


class VeryLongNamedDebugSampleFlowDocument(FlowDocument):
    """Sample document with long name for canonical key testing."""

    pass


@pytest.mark.asyncio
async def test_serialize_includes_debug_fields():
    """Test that serialize_model includes canonical_name and class_name."""
    doc = DebugSampleDocument(name="test.txt", content=b"test content")
    serialized = doc.serialize_model()

    # Check debug fields are present
    assert "canonical_name" in serialized
    assert "class_name" in serialized

    # Check values are correct
    assert serialized["canonical_name"] == "debug_sample"  # Strips Document and Flow suffixes
    assert serialized["class_name"] == "DebugSampleDocument"


@pytest.mark.asyncio
async def test_canonical_name_strips_suffixes():
    """Test that canonical_name properly strips parent class suffixes."""
    doc = VeryLongNamedDebugSampleFlowDocument(name="test.txt", content=b"test")
    serialized = doc.serialize_model()

    # Should strip FlowDocument and Document suffixes
    assert serialized["canonical_name"] == "very_long_named_debug_sample"
    assert serialized["class_name"] == "VeryLongNamedDebugSampleFlowDocument"


@pytest.mark.asyncio
async def test_from_dict_ignores_debug_fields():
    """Test that from_dict ignores canonical_name and class_name."""
    # Create document and serialize
    doc = DebugSampleDocument(name="test.txt", content=b"test content")
    serialized = doc.serialize_model()

    # Verify debug fields are present
    assert "canonical_name" in serialized
    assert "class_name" in serialized

    # Deserialize should work even with these fields
    restored = DebugSampleDocument.from_dict(serialized)
    assert restored.name == doc.name
    assert restored.content == doc.content

    # The canonical_name is a classmethod, not an instance attribute
    # Verify debug fields from serialized dict are not stored as instance attrs
    # (canonical_name exists as a classmethod, but the value from dict shouldn't override it)
    assert DebugSampleDocument.canonical_name() == "debug_sample"
    # class_name should not be an instance attribute at all
    assert not hasattr(restored, "class_name")


@pytest.mark.asyncio
async def test_different_document_types_have_correct_debug_fields():
    """Test debug fields for different document types."""

    class MyTaskDoc(TaskDocument):
        pass

    # Test TaskDocument
    task_doc = MyTaskDoc(name="task.txt", content=b"task")
    task_serialized = task_doc.serialize_model()
    assert (
        task_serialized["canonical_name"] == "my_task_doc"
    )  # Doesn't strip since it doesn't end with parent names
    assert task_serialized["class_name"] == "MyTaskDoc"

    # Test TemporaryDocument (cannot be subclassed, so use directly)
    temp_doc = TemporaryDocument(name="temp.txt", content=b"temp")
    temp_serialized = temp_doc.serialize_model()
    assert temp_serialized["canonical_name"] == "temporary"  # No suffixes to strip
    assert temp_serialized["class_name"] == "TemporaryDocument"


@pytest.mark.asyncio
async def test_roundtrip_preserves_content_ignores_debug():
    """Test that serialize/deserialize roundtrip preserves content but ignores debug fields."""
    doc = DebugSampleDocument(
        name="test.txt",
        content=b"test content",
        description="Test description",
        sources=["source1", "source2"],
    )

    # Serialize
    serialized = doc.serialize_model()

    # Modify debug fields to prove they're ignored
    serialized["canonical_name"] = "wrong_name"
    serialized["class_name"] = "WrongClass"

    # Deserialize
    restored = DebugSampleDocument.from_dict(serialized)

    # Content should be preserved
    assert restored.name == doc.name
    assert restored.content == doc.content
    assert restored.description == doc.description
    assert restored.sources == doc.sources

    # Re-serialize and check debug fields are correct (not the modified values)
    re_serialized = restored.serialize_model()
    assert re_serialized["canonical_name"] == "debug_sample"
    assert re_serialized["class_name"] == "DebugSampleDocument"
