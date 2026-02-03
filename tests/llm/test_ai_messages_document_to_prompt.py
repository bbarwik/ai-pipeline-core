"""Tests for AIMessages.document_to_prompt conversion."""

import base64
import io

from PIL import Image

from ai_pipeline_core.llm import AIMessages
from tests.support.helpers import ConcreteDocument


class TestDocumentToPrompt:
    """Test document to prompt conversion."""

    def test_text_document_conversion(self):
        """Test converting text document to prompt format."""
        doc = ConcreteDocument(name="test.txt", content=b"This is the document content.", description="A test document")

        parts = AIMessages.document_to_prompt(doc)

        # Should return a single text part for text documents
        assert len(parts) == 1
        assert parts[0]["type"] == "text"

        text = parts[0]["text"]  # type: ignore[index]
        # Check for required XML-like tags
        assert "<document>" in text
        assert "</document>" in text
        assert f"<id>{doc.id}</id>" in text
        assert "<name>test.txt</name>" in text
        assert "<description>A test document</description>" in text
        assert "<content>" in text
        assert "This is the document content." in text
        assert "</content>" in text

    def test_text_document_without_description(self):
        """Test text document without description."""
        doc = ConcreteDocument(name="test.txt", content=b"Content only")

        parts = AIMessages.document_to_prompt(doc)

        assert len(parts) == 1
        text = parts[0]["text"]  # type: ignore[index]
        # Description tag should not be present
        assert "<description>" not in text
        assert "Content only" in text

    def test_image_document_conversion(self):
        """Test converting image document to prompt format."""
        # Minimal PNG header
        png_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        doc = ConcreteDocument(name="image.png", content=png_content, description="Test image")

        parts = AIMessages.document_to_prompt(doc)

        # Should return 3 parts: header text, image, closing text
        assert len(parts) == 3

        # First part: header
        assert parts[0]["type"] == "text"
        header = parts[0]["text"]
        assert "<document>" in header
        assert f"<id>{doc.id}</id>" in header
        assert "<name>image.png</name>" in header
        assert "<content>" in header

        # Second part: image
        assert parts[1]["type"] == "image_url"
        assert "image_url" in parts[1]
        image_data = parts[1]["image_url"]
        assert "url" in image_data
        assert "detail" in image_data
        assert image_data["detail"] == "high"

        # Check base64 encoding
        expected_b64 = base64.b64encode(png_content).decode("utf-8")
        assert image_data["url"].startswith("data:image/png;base64,")
        assert expected_b64 in image_data["url"]

        # Third part: closing
        assert parts[2]["type"] == "text"
        assert parts[2]["text"] == "</content>\n</document>\n"

    def test_pdf_document_conversion(self):
        """Test converting PDF document to prompt format."""
        pdf_content = b"%PDF-1.4\n%\xd3\xeb\xe9\xe1\n1 0 obj\n<</Type/Catalog>>\nendobj"
        doc = ConcreteDocument(name="document.pdf", content=pdf_content)

        parts = AIMessages.document_to_prompt(doc)

        # Should return 3 parts: header text, file, closing text
        assert len(parts) == 3

        # First part: header
        assert parts[0]["type"] == "text"
        header = parts[0]["text"]
        assert "<document>" in header

        # Second part: file (not image_url)
        assert parts[1]["type"] == "file"
        assert "file" in parts[1]
        file_data = parts[1]["file"]
        assert "file_data" in file_data

        # Check base64 encoding
        expected_b64 = base64.b64encode(pdf_content).decode("utf-8")
        assert file_data["file_data"].startswith("data:application/pdf;base64,")
        assert expected_b64 in file_data["file_data"]

        # Third part: closing
        assert parts[2]["type"] == "text"
        assert parts[2]["text"] == "</content>\n</document>\n"

    def test_unsupported_mime_type(self):
        """Test handling of unsupported MIME types - skip since we can't mock frozen models."""
        # Skip this test as Document is frozen and we can't patch its properties
        # The actual behavior is tested in integration tests

    def test_markdown_document(self):
        """Test markdown document conversion."""
        doc = ConcreteDocument(name="readme.md", content=b"# Header\n\nThis is **markdown** content.")

        parts = AIMessages.document_to_prompt(doc)

        # Markdown should be treated as text
        assert len(parts) == 1
        assert parts[0]["type"] == "text"
        text = parts[0]["text"]  # type: ignore[index]
        assert "# Header" in text
        assert "This is **markdown** content." in text

    def test_json_document(self):
        """Test JSON document conversion."""
        import json

        json_data = {"key": "value", "number": 42}
        doc = ConcreteDocument(name="data.json", content=json.dumps(json_data).encode())

        parts = AIMessages.document_to_prompt(doc)

        # JSON should be treated as text
        assert len(parts) == 1
        assert parts[0]["type"] == "text"
        assert '{"key": "value", "number": 42}' in parts[0]["text"]

    def test_large_text_document(self):
        """Test handling of large text documents."""
        # Create a large text document
        large_content = "x" * 10000
        doc = ConcreteDocument(name="large.txt", content=large_content.encode())

        parts = AIMessages.document_to_prompt(doc)

        assert len(parts) == 1
        assert parts[0]["type"] == "text"
        assert large_content in parts[0]["text"]

    def test_unicode_text_document(self):
        """Test handling of Unicode text."""
        unicode_content = "Hello 世界! 🌍 Здравствуй мир!"
        doc = ConcreteDocument(name="unicode.txt", content=unicode_content.encode("utf-8"))

        parts = AIMessages.document_to_prompt(doc)

        assert len(parts) == 1
        text = parts[0]["text"]  # type: ignore[index]
        assert unicode_content in text

    def test_gif_image_auto_converted_to_png(self):
        """Test that a GIF document gets auto-converted to PNG in the prompt."""
        # Create a minimal 1x1 GIF image
        buf = io.BytesIO()
        Image.new("RGB", (1, 1), color=(255, 0, 0)).save(buf, format="GIF")
        gif_content = buf.getvalue()

        doc = ConcreteDocument(name="animation.gif", content=gif_content)
        assert doc.mime_type == "image/gif"

        parts = AIMessages.document_to_prompt(doc)

        # Should return 3 parts: header, image, closing
        assert len(parts) == 3
        assert parts[1]["type"] == "image_url"
        image_data = parts[1]["image_url"]  # type: ignore[index]
        # The data URI must use image/png, not image/gif
        assert image_data["url"].startswith("data:image/png;base64,")

    def test_bmp_image_auto_converted_to_png(self):
        """Test that a BMP document gets auto-converted to PNG in the prompt."""
        buf = io.BytesIO()
        Image.new("RGB", (1, 1), color=(0, 255, 0)).save(buf, format="BMP")
        bmp_content = buf.getvalue()

        doc = ConcreteDocument(name="image.bmp", content=bmp_content)
        assert doc.mime_type == "image/bmp"

        parts = AIMessages.document_to_prompt(doc)

        assert len(parts) == 3
        image_data = parts[1]["image_url"]  # type: ignore[index]
        assert image_data["url"].startswith("data:image/png;base64,")

    def test_supported_image_not_converted(self):
        """Test that PNG images are passed through without conversion."""
        # Create a minimal 1x1 PNG
        buf = io.BytesIO()
        Image.new("RGB", (1, 1), color=(0, 0, 255)).save(buf, format="PNG")
        png_content = buf.getvalue()

        doc = ConcreteDocument(name="image.png", content=png_content)

        parts = AIMessages.document_to_prompt(doc)

        assert len(parts) == 3
        image_data = parts[1]["image_url"]  # type: ignore[index]
        # Should still be PNG (passed through, not re-encoded)
        assert image_data["url"].startswith("data:image/png;base64,")
        # Verify original bytes are used (not re-encoded)
        expected_b64 = base64.b64encode(png_content).decode("utf-8")
        assert expected_b64 in image_data["url"]


class TestPdfExtensionWithTextContent:
    """Test handling of documents with .pdf extension but text content.

    This fixes the issue where URLs ending in .pdf (e.g., fetched web pages
    that redirect to HTML) would be incorrectly sent as binary PDF files.
    """

    def test_pdf_extension_with_markdown_content_sent_as_text(self):
        """Document named .pdf but containing markdown should be sent as text."""
        markdown_content = b"# Research Report\n\nThis is a **markdown** document.\n\n## Section 1\n\nSome content here."
        doc = ConcreteDocument(name="fetched_example.com_report.pdf", content=markdown_content)

        # Verify MIME type is PDF based on extension
        assert doc.mime_type == "application/pdf"
        assert doc.is_pdf is True
        assert doc.is_text is False

        parts = AIMessages.document_to_prompt(doc)

        # Should be sent as text (1 part), not as PDF file (3 parts)
        assert len(parts) == 1
        assert parts[0]["type"] == "text"

        text = parts[0]["text"]  # type: ignore[index]
        assert "<document>" in text
        assert "</document>" in text
        assert "# Research Report" in text
        assert "**markdown**" in text
        # Should NOT contain base64 encoding
        assert "base64" not in text

    def test_pdf_extension_with_html_content_sent_as_text(self):
        """Document named .pdf but containing HTML should be sent as text."""
        html_content = b"<!DOCTYPE html>\n<html>\n<head><title>Report</title></head>\n<body>\n<h1>Report</h1>\n</body>\n</html>"
        doc = ConcreteDocument(name="report.pdf", content=html_content)

        assert doc.is_pdf is True

        parts = AIMessages.document_to_prompt(doc)

        # Should be sent as text
        assert len(parts) == 1
        assert parts[0]["type"] == "text"
        assert "<!DOCTYPE html>" in parts[0]["text"]

    def test_real_pdf_sent_as_binary(self):
        """Document with actual PDF content should be sent as binary file."""
        pdf_content = b"%PDF-1.4\n%\xd3\xeb\xe9\xe1\n1 0 obj\n<</Type/Catalog>>\nendobj"
        doc = ConcreteDocument(name="document.pdf", content=pdf_content)

        assert doc.is_pdf is True

        parts = AIMessages.document_to_prompt(doc)

        # Should be sent as file (3 parts: header, file, closing)
        assert len(parts) == 3
        assert parts[1]["type"] == "file"
        assert "file" in parts[1]
        assert parts[1]["file"]["file_data"].startswith("data:application/pdf;base64,")  # type: ignore[index]

    def test_pdf_with_leading_whitespace_sent_as_binary(self):
        """PDF content with leading whitespace should still be detected as real PDF."""
        # Some PDFs have whitespace before the magic bytes
        pdf_content = b"   \n\n%PDF-1.4\n%\xd3\xeb\xe9\xe1\n1 0 obj\n<</Type/Catalog>>\nendobj"
        doc = ConcreteDocument(name="document.pdf", content=pdf_content)

        parts = AIMessages.document_to_prompt(doc)

        # Should still be sent as binary file
        assert len(parts) == 3
        assert parts[1]["type"] == "file"

    def test_pdf_extension_with_json_content_sent_as_text(self):
        """Document named .pdf but containing JSON should be sent as text."""
        json_content = b'{"error": "Not found", "status": 404}'
        doc = ConcreteDocument(name="api_response.pdf", content=json_content)

        parts = AIMessages.document_to_prompt(doc)

        assert len(parts) == 1
        assert parts[0]["type"] == "text"
        assert '"error"' in parts[0]["text"]

    def test_pdf_extension_with_plain_text_sent_as_text(self):
        """Document named .pdf but containing plain text should be sent as text."""
        text_content = b"This PDF could not be downloaded.\nPlease try again later."
        doc = ConcreteDocument(name="broken.pdf", content=text_content)

        parts = AIMessages.document_to_prompt(doc)

        assert len(parts) == 1
        assert parts[0]["type"] == "text"
        assert "could not be downloaded" in parts[0]["text"]

    def test_binary_content_with_pdf_extension_sent_as_binary(self):
        """Binary content (with null bytes) should be sent as binary even without PDF signature."""
        # Binary content that doesn't start with %PDF but has null bytes
        binary_content = b"\x00\x01\x02\x03\x04\x05"
        doc = ConcreteDocument(name="data.pdf", content=binary_content)

        parts = AIMessages.document_to_prompt(doc)

        # Should be sent as file since it contains null bytes (binary)
        assert len(parts) == 3
        assert parts[1]["type"] == "file"

    def test_unicode_text_with_pdf_extension_sent_as_text(self):
        """Unicode text with .pdf extension should be sent as text."""
        unicode_content = "# 研究报告\n\nこれは日本語のテキストです。\n\n🎉 Emoji support!".encode()
        doc = ConcreteDocument(name="research_报告.pdf", content=unicode_content)

        parts = AIMessages.document_to_prompt(doc)

        assert len(parts) == 1
        assert parts[0]["type"] == "text"
        assert "研究报告" in parts[0]["text"]
        assert "日本語" in parts[0]["text"]

    def test_empty_pdf_sent_as_text(self):
        """Empty document with .pdf extension should be sent as text."""
        doc = ConcreteDocument(name="empty.pdf", content=b"")

        parts = AIMessages.document_to_prompt(doc)

        # Empty content is text
        assert len(parts) == 1
        assert parts[0]["type"] == "text"


class TestPdfAttachmentWithTextContent:
    """Test handling of attachments with .pdf extension but text content."""

    def test_pdf_attachment_with_markdown_content_sent_as_text(self):
        """Attachment named .pdf but containing markdown should be sent as text."""
        from ai_pipeline_core.documents.attachment import Attachment

        markdown_content = b"# Attachment Report\n\nThis is markdown in an attachment."
        att = Attachment(name="attachment.pdf", content=markdown_content)
        doc = ConcreteDocument(
            name="main.txt",
            content=b"Main content",
            attachments=(att,),
        )

        parts = AIMessages.document_to_prompt(doc)

        # Collect all text parts
        texts = [p["text"] for p in parts if p["type"] == "text"]  # type: ignore[index]
        combined = "".join(texts)

        # Attachment should be rendered as text, not as file
        assert '<attachment name="attachment.pdf">' in combined
        assert "# Attachment Report" in combined
        assert "</attachment>" in combined

        # Should NOT have a file part for the attachment
        file_parts = [p for p in parts if p["type"] == "file"]
        assert len(file_parts) == 0

    def test_real_pdf_attachment_sent_as_binary(self):
        """Attachment with actual PDF content should be sent as binary."""
        from ai_pipeline_core.documents.attachment import Attachment

        pdf_content = b"%PDF-1.4\n%\xd3\xeb\xe9\xe1\n1 0 obj\n<</Type/Catalog>>\nendobj"
        att = Attachment(name="real.pdf", content=pdf_content)
        doc = ConcreteDocument(
            name="main.txt",
            content=b"Main content",
            attachments=(att,),
        )

        parts = AIMessages.document_to_prompt(doc)

        # Should have a file part for the PDF attachment
        file_parts = [p for p in parts if p["type"] == "file"]
        assert len(file_parts) == 1
        assert file_parts[0]["file"]["file_data"].startswith("data:application/pdf;base64,")  # type: ignore[index]

    def test_mixed_pdf_attachments(self):
        """Mix of real PDF and fake PDF attachments handled correctly."""
        from ai_pipeline_core.documents.attachment import Attachment

        real_pdf = Attachment(
            name="real.pdf",
            content=b"%PDF-1.4\n%\xd3\xeb\xe9\xe1\n1 0 obj\n<</Type/Catalog>>\nendobj",
        )
        fake_pdf = Attachment(
            name="fake.pdf",
            content=b"This is just text pretending to be a PDF",
        )
        doc = ConcreteDocument(
            name="main.txt",
            content=b"Main",
            attachments=(real_pdf, fake_pdf),
        )

        parts = AIMessages.document_to_prompt(doc)

        # One file part for real PDF
        file_parts = [p for p in parts if p["type"] == "file"]
        assert len(file_parts) == 1

        # Fake PDF should be in text
        texts = [p["text"] for p in parts if p["type"] == "text"]  # type: ignore[index]
        combined = "".join(texts)
        assert "pretending to be a PDF" in combined
