"""Tests for ai_pipeline_core.llm.validation module."""

from io import BytesIO

import pytest
from PIL import Image
from pypdf import PdfWriter

from ai_pipeline_core.documents.attachment import Attachment
from ai_pipeline_core.llm.ai_messages import AIMessages
from ai_pipeline_core.llm.validation import (
    _validate_attachment,
    _validate_document,
    _validate_image_content,
    _validate_pdf_content,
    _validate_text_content,
    validate_messages,
)
from tests.support.helpers import ConcreteDocument


# ============================================================================
# Fixtures for generating valid test content
# ============================================================================


@pytest.fixture
def valid_png_1x1() -> bytes:
    """Minimal valid 1x1 white PNG."""
    buf = BytesIO()
    img = Image.new("RGB", (1, 1), color="white")
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def valid_jpeg_1x1() -> bytes:
    """Minimal valid 1x1 JPEG."""
    buf = BytesIO()
    img = Image.new("RGB", (1, 1), color="white")
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def valid_pdf_1page() -> bytes:
    """Minimal valid PDF with 1 blank page."""
    buf = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def valid_pdf_multipage() -> bytes:
    """Valid PDF with 3 pages."""
    buf = BytesIO()
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=72, height=72)
    writer.write(buf)
    return buf.getvalue()


# ============================================================================
# _validate_image_content tests
# ============================================================================


class TestValidateImageContent:
    """Tests for _validate_image_content function."""

    def test_empty_content_returns_error(self):
        """Empty bytes should return error."""
        result = _validate_image_content(b"", "test.png")
        assert result is not None
        assert "empty image content" in result

    def test_valid_png(self, valid_png_1x1: bytes):
        """Valid PNG should return None."""
        result = _validate_image_content(valid_png_1x1, "test.png")
        assert result is None

    def test_valid_jpeg(self, valid_jpeg_1x1: bytes):
        """Valid JPEG should return None."""
        result = _validate_image_content(valid_jpeg_1x1, "test.jpg")
        assert result is None

    def test_truncated_png_header_only(self, valid_png_1x1: bytes):
        """PNG with only header (truncated) should return error."""
        truncated = valid_png_1x1[:16]
        result = _validate_image_content(truncated, "truncated.png")
        assert result is not None
        assert "invalid image" in result

    def test_random_binary_garbage(self):
        """Random binary data should return error."""
        garbage = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"
        result = _validate_image_content(garbage, "garbage.png")
        assert result is not None
        assert "invalid image" in result

    def test_text_content_as_image(self):
        """Text content with image extension should return error."""
        text_content = b"This is not an image"
        result = _validate_image_content(text_content, "fake.png")
        assert result is not None
        assert "invalid image" in result

    def test_png_magic_bytes_only(self):
        """PNG signature without valid chunks should fail."""
        png_signature = b"\x89PNG\r\n\x1a\n"
        result = _validate_image_content(png_signature, "partial.png")
        assert result is not None
        assert "invalid image" in result

    def test_error_message_includes_name(self):
        """Error message should include the provided name."""
        result = _validate_image_content(b"invalid", "my_special_image.png")
        assert result is not None
        assert "my_special_image.png" in result

    def test_gif_image_format(self):
        """GIF image format validation."""
        buf = BytesIO()
        img = Image.new("P", (1, 1), color=0)
        img.save(buf, format="GIF")
        gif_content = buf.getvalue()
        result = _validate_image_content(gif_content, "test.gif")
        assert result is None

    def test_webp_image_format(self):
        """WebP image format validation."""
        buf = BytesIO()
        img = Image.new("RGB", (1, 1), color="white")
        img.save(buf, format="WEBP")
        webp_content = buf.getvalue()
        result = _validate_image_content(webp_content, "test.webp")
        assert result is None


# ============================================================================
# _validate_pdf_content tests
# ============================================================================


class TestValidatePdfContent:
    """Tests for _validate_pdf_content function."""

    def test_empty_content_returns_error(self):
        """Empty bytes should return error."""
        result = _validate_pdf_content(b"", "test.pdf")
        assert result is not None
        assert "empty PDF content" in result

    def test_valid_single_page_pdf(self, valid_pdf_1page: bytes):
        """Valid single-page PDF should return None."""
        result = _validate_pdf_content(valid_pdf_1page, "doc.pdf")
        assert result is None

    def test_valid_multipage_pdf(self, valid_pdf_multipage: bytes):
        """Valid multi-page PDF should return None."""
        result = _validate_pdf_content(valid_pdf_multipage, "doc.pdf")
        assert result is None

    def test_missing_pdf_header(self):
        """Content without %PDF- header should return error."""
        fake_pdf = b"This is not a PDF file"
        result = _validate_pdf_content(fake_pdf, "fake.pdf")
        assert result is not None
        assert "invalid PDF header" in result

    def test_pdf_header_with_leading_whitespace(self, valid_pdf_1page: bytes):
        """PDF with leading whitespace before %PDF- should be valid."""
        pdf_with_whitespace = b"   \n\t" + valid_pdf_1page
        result = _validate_pdf_content(pdf_with_whitespace, "doc.pdf")
        assert result is None

    def test_pdf_header_only_no_content(self):
        """Just %PDF- header without valid structure should fail."""
        fake_pdf = b"%PDF-1.4\n"
        result = _validate_pdf_content(fake_pdf, "header_only.pdf")
        assert result is not None
        assert "corrupted PDF" in result or "PDF has no pages" in result

    def test_pdf_header_with_garbage(self):
        """PDF header followed by garbage should fail."""
        fake_pdf = b"%PDF-1.7\x00\x01\x02\x03garbage"
        result = _validate_pdf_content(fake_pdf, "garbage.pdf")
        assert result is not None
        assert "corrupted PDF" in result or "PDF has no pages" in result

    def test_truncated_valid_pdf(self, valid_pdf_1page: bytes):
        """Truncated PDF (partial content) should fail."""
        truncated = valid_pdf_1page[: len(valid_pdf_1page) // 2]
        result = _validate_pdf_content(truncated, "truncated.pdf")
        assert result is not None
        assert "corrupted PDF" in result or "PDF has no pages" in result

    def test_image_content_as_pdf(self, valid_png_1x1: bytes):
        """Image content with PDF extension should fail header check."""
        result = _validate_pdf_content(valid_png_1x1, "fake.pdf")
        assert result is not None
        assert "invalid PDF header" in result

    def test_error_message_includes_name(self):
        """Error message should include the provided name."""
        result = _validate_pdf_content(b"not a pdf", "important_document.pdf")
        assert result is not None
        assert "important_document.pdf" in result

    def test_pdf_zero_pages(self):
        """PDF with zero pages should return error."""
        buf = BytesIO()
        writer = PdfWriter()
        # Don't add any pages
        writer.write(buf)
        zero_page_pdf = buf.getvalue()
        result = _validate_pdf_content(zero_page_pdf, "empty.pdf")
        assert result is not None
        assert "PDF has no pages" in result


# ============================================================================
# _validate_text_content tests
# ============================================================================


class TestValidateTextContent:
    """Tests for _validate_text_content function."""

    def test_empty_content_returns_error(self):
        """Empty bytes should return error."""
        result = _validate_text_content(b"", "test.txt")
        assert result is not None
        assert "empty text content" in result

    def test_valid_ascii_text(self):
        """Valid ASCII text should return None."""
        result = _validate_text_content(b"Hello, World!", "hello.txt")
        assert result is None

    def test_valid_utf8_text(self):
        """Valid UTF-8 with non-ASCII characters should return None."""
        result = _validate_text_content("Héllo Wörld! 日本語".encode(), "unicode.txt")
        assert result is None

    def test_valid_utf8_with_bom(self):
        """UTF-8 with BOM should return None."""
        content = b"\xef\xbb\xbfHello with BOM"
        result = _validate_text_content(content, "bom.txt")
        assert result is None

    def test_null_bytes_detected(self):
        """Content with null bytes should return error."""
        binary_content = b"Hello\x00World"
        result = _validate_text_content(binary_content, "binary.txt")
        assert result is not None
        assert "binary content" in result
        assert "null bytes" in result

    def test_invalid_utf8_sequence(self):
        """Invalid UTF-8 byte sequence should return error."""
        invalid_utf8 = b"\x80\x81\x82"
        result = _validate_text_content(invalid_utf8, "invalid.txt")
        assert result is not None
        assert "invalid UTF-8 encoding" in result

    def test_partial_utf8_sequence(self):
        """Incomplete UTF-8 multi-byte sequence should return error."""
        incomplete = b"Hello \xe2\x82"
        result = _validate_text_content(incomplete, "incomplete.txt")
        assert result is not None
        assert "invalid UTF-8 encoding" in result

    def test_latin1_not_utf8(self):
        """Latin-1 encoded text (not valid UTF-8) should return error."""
        latin1_content = "Héllo".encode("latin-1")
        result = _validate_text_content(latin1_content, "latin1.txt")
        assert result is not None
        assert "invalid UTF-8 encoding" in result

    def test_whitespace_only_is_valid(self):
        """Whitespace-only content should be valid."""
        result = _validate_text_content(b"   \n\t\r\n   ", "whitespace.txt")
        assert result is None

    def test_single_character(self):
        """Single character should be valid."""
        result = _validate_text_content(b"x", "single.txt")
        assert result is None

    def test_error_message_includes_name(self):
        """Error message should include the provided name."""
        result = _validate_text_content(b"\x80\x81", "my_file.txt")
        assert result is not None
        assert "my_file.txt" in result


# ============================================================================
# _validate_attachment tests
# ============================================================================


class TestValidateAttachment:
    """Tests for _validate_attachment function."""

    def test_valid_text_attachment(self):
        """Valid text attachment should return None."""
        att = Attachment(name="notes.txt", content=b"Some notes")
        result = _validate_attachment(att, "parent_doc")
        assert result is None

    def test_valid_image_attachment(self, valid_png_1x1: bytes):
        """Valid image attachment should return None."""
        att = Attachment(name="screenshot.png", content=valid_png_1x1)
        result = _validate_attachment(att, "parent_doc")
        assert result is None

    def test_valid_pdf_attachment(self, valid_pdf_1page: bytes):
        """Valid PDF attachment should return None."""
        att = Attachment(name="report.pdf", content=valid_pdf_1page)
        result = _validate_attachment(att, "parent_doc")
        assert result is None

    def test_invalid_image_attachment(self):
        """Invalid image attachment should return error."""
        att = Attachment(name="broken.png", content=b"not an image")
        result = _validate_attachment(att, "parent_doc")
        assert result is not None
        assert "invalid image" in result
        assert "attachment 'broken.png' of 'parent_doc'" in result

    def test_invalid_pdf_attachment(self):
        """Invalid PDF attachment should return error."""
        att = Attachment(name="broken.pdf", content=b"not a pdf")
        result = _validate_attachment(att, "parent_doc")
        assert result is not None
        assert "invalid PDF header" in result

    def test_invalid_text_attachment(self):
        """Text attachment with null bytes should return error."""
        att = Attachment(name="binary.txt", content=b"data\x00more")
        result = _validate_attachment(att, "parent_doc")
        assert result is not None
        assert "binary content" in result

    def test_unknown_type_passes_through(self):
        """Unknown MIME type attachment should pass (return None)."""
        att = Attachment(name="data.dat", content=b"\x00\x01\x02\x03")
        result = _validate_attachment(att, "parent_doc")
        assert result is None

    def test_error_includes_parent_name(self):
        """Error should include both attachment and parent names."""
        att = Attachment(name="bad_image.jpg", content=b"garbage")
        result = _validate_attachment(att, "important_document.md")
        assert result is not None
        assert "bad_image.jpg" in result
        assert "important_document.md" in result


# ============================================================================
# _validate_document tests
# ============================================================================


class TestValidateDocument:
    """Tests for _validate_document function."""

    def test_valid_text_document_no_attachments(self):
        """Valid text document without attachments."""
        doc = ConcreteDocument(name="readme.md", content=b"# Hello")
        result_doc, errors = _validate_document(doc)
        assert result_doc is doc
        assert errors == []

    def test_valid_image_document(self, valid_png_1x1: bytes):
        """Valid image document."""
        doc = ConcreteDocument(name="image.png", content=valid_png_1x1)
        result_doc, errors = _validate_document(doc)
        assert result_doc is doc
        assert errors == []

    def test_valid_pdf_document(self, valid_pdf_1page: bytes):
        """Valid PDF document."""
        doc = ConcreteDocument(name="report.pdf", content=valid_pdf_1page)
        result_doc, errors = _validate_document(doc)
        assert result_doc is doc
        assert errors == []

    def test_invalid_main_content_returns_none(self):
        """Invalid main content should return None document."""
        doc = ConcreteDocument(name="broken.png", content=b"not an image")
        result_doc, errors = _validate_document(doc)
        assert result_doc is None
        assert len(errors) == 1
        assert "invalid image" in errors[0]

    def test_valid_document_with_valid_attachments(self, valid_png_1x1: bytes):
        """Document with all valid attachments."""
        att1 = Attachment(name="notes.txt", content=b"Notes")
        att2 = Attachment(name="img.png", content=valid_png_1x1)
        doc = ConcreteDocument(name="main.md", content=b"# Main", attachments=(att1, att2))

        result_doc, errors = _validate_document(doc)
        assert result_doc is doc
        assert errors == []

    def test_valid_document_with_one_invalid_attachment(self, valid_png_1x1: bytes):
        """Document with mix of valid and invalid attachments."""
        valid_att = Attachment(name="good.png", content=valid_png_1x1)
        invalid_att = Attachment(name="bad.png", content=b"garbage")
        doc = ConcreteDocument(name="main.md", content=b"# Main", attachments=(valid_att, invalid_att))

        result_doc, errors = _validate_document(doc)
        assert result_doc is not None
        assert result_doc is not doc
        assert len(result_doc.attachments) == 1
        assert result_doc.attachments[0].name == "good.png"
        assert len(errors) == 1
        assert "bad.png" in errors[0]

    def test_document_with_all_invalid_attachments(self):
        """Document with all attachments invalid should keep document, remove all attachments."""
        bad_att1 = Attachment(name="bad1.png", content=b"garbage1")
        bad_att2 = Attachment(name="bad2.png", content=b"garbage2")
        doc = ConcreteDocument(name="main.md", content=b"# Main", attachments=(bad_att1, bad_att2))

        result_doc, errors = _validate_document(doc)
        assert result_doc is not None
        assert len(result_doc.attachments) == 0
        assert len(errors) == 2

    def test_invalid_document_attachments_not_validated(self):
        """If main content is invalid, attachments are not validated (early return)."""
        bad_att = Attachment(name="bad.txt", content=b"\x00null")
        doc = ConcreteDocument(name="broken.png", content=b"not image", attachments=(bad_att,))

        result_doc, errors = _validate_document(doc)
        assert result_doc is None
        assert len(errors) == 1
        assert "invalid image" in errors[0]

    def test_document_with_empty_attachments_tuple(self):
        """Document with empty attachments tuple."""
        doc = ConcreteDocument(name="test.md", content=b"content", attachments=())
        result_doc, errors = _validate_document(doc)
        assert result_doc is doc
        assert errors == []

    def test_unknown_document_type_passes_through(self):
        """Document with unknown MIME type passes through without validation."""
        doc = ConcreteDocument(name="data.dat", content=b"\x00\x01\x02\x03")
        result_doc, errors = _validate_document(doc)
        assert result_doc is doc
        assert errors == []


# ============================================================================
# validate_messages tests
# ============================================================================


class TestValidateMessages:
    """Tests for validate_messages function."""

    def test_empty_aimessages(self):
        """Empty AIMessages returns same object."""
        messages = AIMessages([])
        result, warnings = validate_messages(messages)
        assert result is messages
        assert warnings == []

    def test_aimessages_with_only_strings(self):
        """AIMessages with only strings returns same object (no validation needed)."""
        messages = AIMessages(["Hello", "World"])
        result, warnings = validate_messages(messages)
        assert result is messages
        assert warnings == []

    def test_aimessages_with_valid_document(self):
        """AIMessages with valid document returns same object."""
        doc = ConcreteDocument(name="test.md", content=b"# Test")
        messages = AIMessages([doc])
        result, warnings = validate_messages(messages)
        assert result is messages
        assert warnings == []

    def test_aimessages_with_invalid_document_filtered(self):
        """Invalid document is filtered from AIMessages."""
        invalid_doc = ConcreteDocument(name="broken.png", content=b"not image")
        messages = AIMessages([invalid_doc])

        result, warnings = validate_messages(messages)
        assert len(result) == 0
        assert len(warnings) == 1
        assert "LLM input validation: filtering" in warnings[0]

    def test_aimessages_mixed_strings_and_documents(self):
        """Mix of strings and documents, only invalid documents filtered."""
        doc = ConcreteDocument(name="test.md", content=b"content")
        invalid_doc = ConcreteDocument(name="bad.png", content=b"garbage")
        messages = AIMessages(["Hello", doc, "World", invalid_doc])

        result, warnings = validate_messages(messages)
        assert len(result) == 3
        assert result[0] == "Hello"
        assert result[1] == doc
        assert result[2] == "World"
        assert len(warnings) == 1

    def test_aimessages_all_valid_documents(self, valid_png_1x1: bytes, valid_pdf_1page: bytes):
        """All valid documents returns same object."""
        doc1 = ConcreteDocument(name="text.md", content=b"# Text")
        doc2 = ConcreteDocument(name="image.png", content=valid_png_1x1)
        doc3 = ConcreteDocument(name="report.pdf", content=valid_pdf_1page)
        messages = AIMessages([doc1, doc2, doc3])

        result, warnings = validate_messages(messages)
        assert result is messages
        assert warnings == []

    def test_aimessages_all_invalid_documents(self):
        """All invalid documents results in empty AIMessages."""
        doc1 = ConcreteDocument(name="bad1.png", content=b"garbage1")
        doc2 = ConcreteDocument(name="bad2.pdf", content=b"not pdf")
        messages = AIMessages([doc1, doc2])

        result, warnings = validate_messages(messages)
        assert len(result) == 0
        assert len(warnings) == 2

    def test_aimessages_document_with_filtered_attachments(self, valid_png_1x1: bytes):
        """Document attachment filtering propagates to result."""
        valid_att = Attachment(name="good.txt", content=b"good")
        invalid_att = Attachment(name="bad.png", content=b"bad")
        doc = ConcreteDocument(name="main.md", content=b"# Main", attachments=(valid_att, invalid_att))
        messages = AIMessages([doc])

        result, warnings = validate_messages(messages)
        assert len(result) == 1
        assert len(result[0].attachments) == 1
        assert result[0].attachments[0].name == "good.txt"
        assert len(warnings) == 1

    def test_aimessages_preserves_order(self, valid_png_1x1: bytes):
        """Order of valid messages is preserved."""
        invalid = ConcreteDocument(name="bad.png", content=b"garbage")
        doc1 = ConcreteDocument(name="first.md", content=b"first")
        doc2 = ConcreteDocument(name="second.md", content=b"second")
        messages = AIMessages([doc1, invalid, "middle", doc2])

        result, warnings = validate_messages(messages)
        assert len(result) == 3
        assert result[0].name == "first.md"
        assert result[1] == "middle"
        assert result[2].name == "second.md"

    def test_aimessages_returns_new_instance_when_changed(self):
        """When filtering occurs, a new AIMessages instance is returned."""
        valid = ConcreteDocument(name="good.md", content=b"good")
        invalid = ConcreteDocument(name="bad.png", content=b"garbage")
        messages = AIMessages([valid, invalid])

        result, warnings = validate_messages(messages)
        assert result is not messages
        assert isinstance(result, AIMessages)

    def test_aimessages_returns_same_instance_when_unchanged(self):
        """When no filtering needed, same AIMessages instance returned."""
        doc = ConcreteDocument(name="good.md", content=b"content")
        messages = AIMessages(["Hello", doc])

        result, warnings = validate_messages(messages)
        assert result is messages

    def test_warnings_include_full_context(self):
        """Warnings include full descriptive message."""
        invalid = ConcreteDocument(name="corrupted.pdf", content=b"not a pdf")
        messages = AIMessages([invalid])

        result, warnings = validate_messages(messages)
        assert len(warnings) == 1
        assert "LLM input validation: filtering" in warnings[0]
        assert "corrupted.pdf" in warnings[0]


# ============================================================================
# Edge case tests
# ============================================================================


class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_text_with_only_newlines(self):
        """Text with only newlines is valid."""
        result = _validate_text_content(b"\n\n\n\n", "newlines.txt")
        assert result is None

    def test_pdf_with_many_whitespace_before_header(self, valid_pdf_1page: bytes):
        """PDF with excessive whitespace before header."""
        padded_pdf = b" " * 100 + valid_pdf_1page
        result = _validate_pdf_content(padded_pdf, "padded.pdf")
        assert result is None

    def test_document_name_with_special_characters(self):
        """Document names with special characters in error messages."""
        doc = ConcreteDocument(name="file with spaces & symbols!.png", content=b"garbage")
        result_doc, errors = _validate_document(doc)
        assert result_doc is None
        assert "file with spaces & symbols!.png" in errors[0]

    def test_deeply_nested_attachment_errors(self):
        """Attachment error message includes full context chain."""
        att = Attachment(name="deep.png", content=b"bad")
        doc = ConcreteDocument(name="parent.md", content=b"content", attachments=(att,))
        result_doc, errors = _validate_document(doc)
        assert "attachment 'deep.png' of 'parent.md'" in errors[0]

    def test_multiple_null_bytes_in_text(self):
        """Multiple null bytes detected."""
        content = b"\x00\x00\x00\x00"
        result = _validate_text_content(content, "nulls.txt")
        assert result is not None
        assert "null bytes" in result

    def test_attachment_empty_description(self, valid_png_1x1: bytes):
        """Attachment with None description works."""
        att = Attachment(name="img.png", content=valid_png_1x1, description=None)
        result = _validate_attachment(att, "parent")
        assert result is None

    def test_attachment_with_description(self, valid_png_1x1: bytes):
        """Attachment with description works."""
        att = Attachment(name="img.png", content=valid_png_1x1, description="A screenshot")
        result = _validate_attachment(att, "parent")
        assert result is None

    def test_empty_pdf_filtered(self):
        """Empty PDF document is filtered (falls back to text validation on empty)."""
        doc = ConcreteDocument(name="empty.pdf", content=b"")
        messages = AIMessages([doc])
        result, warnings = validate_messages(messages)
        assert len(result) == 0
        # Empty content can't be detected as PDF, falls back to text validation
        assert any("empty" in w for w in warnings)

    def test_empty_image_filtered(self):
        """Empty image document is filtered (falls back to text validation on empty)."""
        doc = ConcreteDocument(name="empty.png", content=b"")
        messages = AIMessages([doc])
        result, warnings = validate_messages(messages)
        assert len(result) == 0
        # Empty content can't be detected as image, falls back to text validation
        assert any("empty" in w for w in warnings)

    def test_empty_text_filtered(self):
        """Empty text document is filtered."""
        doc = ConcreteDocument(name="empty.txt", content=b"")
        messages = AIMessages([doc])
        result, warnings = validate_messages(messages)
        assert len(result) == 0
        assert any("empty text content" in w for w in warnings)
