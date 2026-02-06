"""Tests proving the Pydantic serialization bug for binary content.

These tests demonstrate that binary content is CORRUPTED when using
Pydantic's model_dump() / model_validate() path, while serialize_model() /
from_dict() works correctly.

TDD approach: These tests should FAIL before the fix, PASS after.
"""

import base64


from ai_pipeline_core.documents import Attachment

from tests.support.helpers import ConcreteDocument


# Minimal valid PNG (1x1 transparent pixel)
MINIMAL_PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")

# Minimal valid JPEG
MINIMAL_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof"
    "Hh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwh"
    "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAAR"
    "CAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAA"
    "AAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMB"
    "AAIRAxEAPwCwAB//2Q=="
)

# Some arbitrary binary data (not valid image)
BINARY_DATA = bytes(range(256))


class TestDocumentPydanticSerializationBug:
    """Tests proving Document binary content is corrupted via Pydantic path."""

    def test_text_content_pydantic_roundtrip_works(self):
        """Text content should roundtrip correctly via Pydantic (sanity check)."""
        original = ConcreteDocument.create(
            name="test.txt",
            content="Hello, World! 你好世界 🎉",
        )
        original_sha256 = original.sha256

        # Pydantic serialization path
        dumped = original.model_dump(mode="json")
        restored = ConcreteDocument.model_validate(dumped)

        assert restored.sha256 == original_sha256, "Text content should roundtrip correctly"
        assert restored.content == original.content

    def test_binary_content_pydantic_roundtrip_corrupted(self):
        """PROVES BUG: Binary content is CORRUPTED via Pydantic path.

        This test should FAIL before the fix (content gets corrupted).
        After fix, this test should PASS.
        """
        original = ConcreteDocument(
            name="image.png",
            content=MINIMAL_PNG,
        )
        original_sha256 = original.sha256
        original_content = original.content

        # Pydantic serialization path
        dumped = original.model_dump(mode="json")
        restored = ConcreteDocument.model_validate(dumped)

        # These assertions prove the bug exists
        assert restored.sha256 == original_sha256, (
            f"Binary content SHA256 mismatch!\n"
            f"Original: {original_sha256}\n"
            f"Restored: {restored.sha256}\n"
            f"Original content length: {len(original_content)}\n"
            f"Restored content length: {len(restored.content)}\n"
            f"This proves the Pydantic serialization bug exists."
        )
        assert restored.content == original_content, "Binary content bytes should match"

    def test_jpeg_content_pydantic_roundtrip_corrupted(self):
        """PROVES BUG: JPEG content is CORRUPTED via Pydantic path."""
        original = ConcreteDocument(
            name="photo.jpg",
            content=MINIMAL_JPEG,
        )
        original_sha256 = original.sha256

        dumped = original.model_dump(mode="json")
        restored = ConcreteDocument.model_validate(dumped)

        assert restored.sha256 == original_sha256, "JPEG content should roundtrip correctly"

    def test_arbitrary_binary_pydantic_roundtrip_corrupted(self):
        """PROVES BUG: Arbitrary binary data is CORRUPTED via Pydantic path."""
        original = ConcreteDocument(
            name="data.bin",
            content=BINARY_DATA,
        )
        original_sha256 = original.sha256

        dumped = original.model_dump(mode="json")
        restored = ConcreteDocument.model_validate(dumped)

        assert restored.sha256 == original_sha256, "Binary data should roundtrip correctly"

    def test_serialize_model_roundtrip_works(self):
        """serialize_model/from_dict path works correctly (control test)."""
        original = ConcreteDocument(
            name="image.png",
            content=MINIMAL_PNG,
        )
        original_sha256 = original.sha256

        # Correct serialization path
        serialized = original.serialize_model()
        restored = ConcreteDocument.from_dict(serialized)

        assert restored.sha256 == original_sha256, "serialize_model path should work"
        assert restored.content == original.content


class TestAttachmentPydanticSerializationBug:
    """Tests proving Attachment binary content is corrupted via Pydantic path."""

    def test_text_attachment_pydantic_roundtrip_works(self):
        """Text attachment should roundtrip correctly via Pydantic (sanity check)."""
        original = Attachment(
            name="notes.txt",
            content=b"Hello, World!",
        )
        original_content = original.content

        dumped = original.model_dump(mode="json")
        restored = Attachment.model_validate(dumped)

        assert restored.content == original_content, "Text attachment should roundtrip"

    def test_binary_attachment_pydantic_roundtrip_corrupted(self):
        """PROVES BUG: Binary attachment is CORRUPTED via Pydantic path.

        This test should FAIL before the fix.
        After fix, this test should PASS.
        """
        original = Attachment(
            name="screenshot.png",
            content=MINIMAL_PNG,
        )
        original_content = original.content

        dumped = original.model_dump(mode="json")
        restored = Attachment.model_validate(dumped)

        assert restored.content == original_content, (
            f"Binary attachment content mismatch!\n"
            f"Original length: {len(original_content)}\n"
            f"Restored length: {len(restored.content)}\n"
            f"This proves the Pydantic serialization bug exists."
        )

    def test_jpeg_attachment_pydantic_roundtrip_corrupted(self):
        """PROVES BUG: JPEG attachment is CORRUPTED via Pydantic path."""
        original = Attachment(
            name="photo.jpg",
            content=MINIMAL_JPEG,
        )
        original_content = original.content

        dumped = original.model_dump(mode="json")
        restored = Attachment.model_validate(dumped)

        assert restored.content == original_content, "JPEG attachment should roundtrip"


class TestDocumentWithBinaryAttachmentsBug:
    """Tests proving Document with binary attachments is corrupted via Pydantic path."""

    def test_document_with_binary_attachment_pydantic_roundtrip_corrupted(self):
        """PROVES BUG: Document with binary attachment is CORRUPTED via Pydantic path."""
        original = ConcreteDocument.create(
            name="report.md",
            content="# Report\n\nSee attached screenshot.",
            attachments=(Attachment(name="screenshot.png", content=MINIMAL_PNG),),
        )
        original_sha256 = original.sha256
        original_attachment_content = original.attachments[0].content

        dumped = original.model_dump(mode="json")
        restored = ConcreteDocument.model_validate(dumped)

        # Document SHA256 includes attachments, so this will fail if attachment is corrupted
        assert restored.sha256 == original_sha256, f"Document with attachment SHA256 mismatch!\nOriginal: {original_sha256}\nRestored: {restored.sha256}"
        assert restored.attachments[0].content == original_attachment_content, "Attachment content should match"

    def test_document_with_multiple_binary_attachments_corrupted(self):
        """PROVES BUG: Multiple binary attachments all get corrupted."""
        original = ConcreteDocument.create(
            name="gallery.md",
            content="# Gallery",
            attachments=(
                Attachment(name="img1.png", content=MINIMAL_PNG),
                Attachment(name="img2.jpg", content=MINIMAL_JPEG),
                Attachment(name="data.bin", content=BINARY_DATA),
            ),
        )
        original_sha256 = original.sha256

        dumped = original.model_dump(mode="json")
        restored = ConcreteDocument.model_validate(dumped)

        assert restored.sha256 == original_sha256, "All attachments should roundtrip correctly"

        for i, (orig_att, rest_att) in enumerate(zip(original.attachments, restored.attachments, strict=True)):
            assert rest_att.content == orig_att.content, f"Attachment {i} content mismatch"


class TestPydanticVsSerializeModelComparison:
    """Direct comparison showing the two paths now both work correctly."""

    def test_serialization_output_format(self):
        """Verify both Pydantic and serialize_model use the same {v,e} format.

        After Option E, serialize_model delegates to model_dump, so both use
        the same {v: str, e: encoding} format for content.
        """
        doc = ConcreteDocument(
            name="image.png",
            content=MINIMAL_PNG,
        )

        # Pydantic path
        pydantic_dump = doc.model_dump(mode="json")
        pydantic_content = pydantic_dump["content"]

        # New format: dict with v and e keys
        assert isinstance(pydantic_content, dict), "Pydantic content should be dict"
        assert "v" in pydantic_content, "Should have 'v' key for value"
        assert "e" in pydantic_content, "Should have 'e' key for encoding"
        assert pydantic_content["e"] == "base64", "Binary should be marked as base64"

        # serialize_model path - now uses same format (delegates to model_dump)
        serialize_dump = doc.serialize_model()
        serialize_content = serialize_dump["content"]

        # Both should use the same {v, e} format
        assert isinstance(serialize_content, dict), "serialize_model content should be dict"
        assert serialize_content["e"] == "base64", "serialize_model should mark as base64"
        assert serialize_content["v"] == pydantic_content["v"], "Both should have same base64 value"

        # serialize_model adds metadata
        assert "sha256" in serialize_dump, "serialize_model should add metadata"
        assert "class_name" in serialize_dump, "serialize_model should add class_name"

    def test_both_paths_restore_binary_correctly(self):
        """Verify both Pydantic and from_dict now restore binary correctly.

        After the fix, both paths produce identical results.
        """
        doc = ConcreteDocument(
            name="image.png",
            content=MINIMAL_PNG,
        )

        # Pydantic deserialization
        pydantic_dump = doc.model_dump(mode="json")
        pydantic_restored = ConcreteDocument.model_validate(pydantic_dump)

        # serialize_model deserialization
        serialize_dump = doc.serialize_model()
        from_dict_restored = ConcreteDocument.from_dict(serialize_dump)

        # Both should restore correctly
        assert from_dict_restored.content == MINIMAL_PNG, "from_dict restores correctly"
        assert pydantic_restored.content == MINIMAL_PNG, "Pydantic restores correctly"

        # Both should produce identical documents
        assert pydantic_restored.sha256 == from_dict_restored.sha256, "Both paths produce same document"
