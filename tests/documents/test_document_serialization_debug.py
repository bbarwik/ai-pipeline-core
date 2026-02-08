"""Test document serialization debug fields."""

import pytest

from ai_pipeline_core.documents import Attachment, Document


class DebugSampleDocument(Document):
    """Sample document for debug field testing."""


class VeryLongNamedDebugSampleDocument(Document):
    """Sample document with long name for testing."""


@pytest.mark.asyncio
async def test_serialize_includes_class_name():
    """Test that serialize_model includes class_name."""
    doc = DebugSampleDocument(name="test.txt", content=b"test content")
    serialized = doc.serialize_model()

    assert "class_name" in serialized
    assert serialized["class_name"] == "DebugSampleDocument"
    assert "canonical_name" not in serialized


@pytest.mark.asyncio
async def test_long_named_document_class_name():
    """Test class_name for longer document names."""
    doc = VeryLongNamedDebugSampleDocument(name="test.txt", content=b"test")
    serialized = doc.serialize_model()

    assert serialized["class_name"] == "VeryLongNamedDebugSampleDocument"


@pytest.mark.asyncio
async def test_from_dict_ignores_class_name():
    """Test that from_dict ignores class_name."""
    doc = DebugSampleDocument(name="test.txt", content=b"test content")
    serialized = doc.serialize_model()

    assert "class_name" in serialized

    restored = DebugSampleDocument.from_dict(serialized)
    assert restored.name == doc.name
    assert restored.content == doc.content
    assert not hasattr(restored, "class_name")


@pytest.mark.asyncio
async def test_different_document_types_have_correct_class_name():
    """Test class_name for different document types."""

    class MyTaskDoc(Document):
        pass

    task_doc = MyTaskDoc(name="task.txt", content=b"task")
    task_serialized = task_doc.serialize_model()
    assert task_serialized["class_name"] == "MyTaskDoc"


@pytest.mark.asyncio
async def test_roundtrip_preserves_content():
    """Test that serialize/deserialize roundtrip preserves content."""
    doc = DebugSampleDocument(
        name="test.txt",
        content=b"test content",
        description="Test description",
        sources=("https://example.com/source1", "https://example.com/source2"),
    )

    serialized = doc.serialize_model()

    # Modify class_name to prove it's ignored
    serialized["class_name"] = "WrongClass"

    restored = DebugSampleDocument.from_dict(serialized)

    assert restored.name == doc.name
    assert restored.content == doc.content
    assert restored.description == doc.description
    assert restored.sources == doc.sources

    re_serialized = restored.serialize_model()
    assert re_serialized["class_name"] == "DebugSampleDocument"


# --- Attachment serialization metadata tests ---


JPEG_HEADER = b"\xff\xd8\xff\xe0" + b"\x00" * 100
# PDF_HEADER needs invalid UTF-8 bytes to trigger base64 encoding
PDF_HEADER = b"%PDF-1.4\xff" + b"\x00" * 100  # \xff is invalid UTF-8 when standalone


@pytest.mark.asyncio
async def test_serialize_attachment_includes_mime_type_and_size():
    """Test that serialized attachments include mime_type and size keys."""
    att = Attachment(name="screenshot.jpg", content=JPEG_HEADER)
    doc = DebugSampleDocument(name="report.txt", content=b"text", attachments=(att,))
    serialized = doc.serialize_model()

    assert len(serialized["attachments"]) == 1
    att_dict = serialized["attachments"][0]
    assert "mime_type" in att_dict
    assert "size" in att_dict
    assert att_dict["mime_type"] == att.mime_type
    assert att_dict["size"] == att.size


@pytest.mark.asyncio
async def test_serialize_text_attachment_metadata():
    """Test mime_type and size for a text attachment."""
    content = b"hello world"
    att = Attachment(name="notes.txt", content=content)
    doc = DebugSampleDocument(name="report.txt", content=b"body", attachments=(att,))
    serialized = doc.serialize_model()

    att_dict = serialized["attachments"][0]
    # New format: content is a plain string for text
    assert isinstance(att_dict["content"], str)
    assert att_dict["content"] == "hello world"
    assert att_dict["mime_type"] == "text/plain"
    assert att_dict["size"] == len(content)


@pytest.mark.asyncio
async def test_serialize_pdf_attachment_metadata():
    """Test mime_type and size for a PDF attachment."""
    att = Attachment(name="doc.pdf", content=PDF_HEADER)
    doc = DebugSampleDocument(name="report.txt", content=b"body", attachments=(att,))
    serialized = doc.serialize_model()

    att_dict = serialized["attachments"][0]
    # New format: content is a data URI for binary
    assert isinstance(att_dict["content"], str)
    assert att_dict["content"].startswith("data:")
    assert att_dict["mime_type"] == "application/pdf"
    assert att_dict["size"] == len(PDF_HEADER)


@pytest.mark.asyncio
async def test_serialize_empty_attachments():
    """Test that empty attachments serialize to an empty list."""
    doc = DebugSampleDocument(name="report.txt", content=b"body")
    serialized = doc.serialize_model()
    assert serialized["attachments"] == []


@pytest.mark.asyncio
async def test_roundtrip_with_attachments_ignores_extra_fields():
    """Test that from_dict ignores mime_type and size in attachment dicts."""
    att = Attachment(name="screenshot.jpg", content=JPEG_HEADER)
    doc = DebugSampleDocument(name="report.txt", content=b"body", attachments=(att,))

    serialized = doc.serialize_model()
    assert "mime_type" in serialized["attachments"][0]
    assert "size" in serialized["attachments"][0]

    restored = DebugSampleDocument.from_dict(serialized)
    assert restored.name == doc.name
    assert restored.content == doc.content
    assert len(restored.attachments) == 1
    assert restored.attachments[0].name == att.name
    assert restored.attachments[0].content == att.content
