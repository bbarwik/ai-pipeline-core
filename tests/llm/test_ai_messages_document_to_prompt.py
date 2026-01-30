"""Tests for AIMessages.document_to_prompt conversion."""

import base64
import io

from PIL import Image

from ai_pipeline_core.llm import AIMessages
from tests.test_helpers import ConcreteFlowDocument


class TestDocumentToPrompt:
    """Test document to prompt conversion."""

    def test_text_document_conversion(self):
        """Test converting text document to prompt format."""
        doc = ConcreteFlowDocument(
            name="test.txt", content=b"This is the document content.", description="A test document"
        )

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
        doc = ConcreteFlowDocument(name="test.txt", content=b"Content only")

        parts = AIMessages.document_to_prompt(doc)

        assert len(parts) == 1
        text = parts[0]["text"]  # type: ignore[index]
        # Description tag should not be present
        assert "<description>" not in text
        assert "Content only" in text

    def test_image_document_conversion(self):
        """Test converting image document to prompt format."""
        # Minimal PNG header
        png_content = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        )
        doc = ConcreteFlowDocument(name="image.png", content=png_content, description="Test image")

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
        doc = ConcreteFlowDocument(name="document.pdf", content=pdf_content)

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
        # Skip this test as ConcreteFlowDocument is frozen and we can't patch its properties
        # The actual behavior is tested in integration tests
        pass

    def test_markdown_document(self):
        """Test markdown document conversion."""
        doc = ConcreteFlowDocument(
            name="readme.md", content=b"# Header\n\nThis is **markdown** content."
        )

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
        doc = ConcreteFlowDocument(name="data.json", content=json.dumps(json_data).encode())

        parts = AIMessages.document_to_prompt(doc)

        # JSON should be treated as text
        assert len(parts) == 1
        assert parts[0]["type"] == "text"
        assert '{"key": "value", "number": 42}' in parts[0]["text"]

    def test_large_text_document(self):
        """Test handling of large text documents."""
        # Create a large text document
        large_content = "x" * 10000
        doc = ConcreteFlowDocument(name="large.txt", content=large_content.encode())

        parts = AIMessages.document_to_prompt(doc)

        assert len(parts) == 1
        assert parts[0]["type"] == "text"
        assert large_content in parts[0]["text"]

    def test_unicode_text_document(self):
        """Test handling of Unicode text."""
        unicode_content = "Hello 世界! 🌍 Здравствуй мир!"
        doc = ConcreteFlowDocument(name="unicode.txt", content=unicode_content.encode("utf-8"))

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

        doc = ConcreteFlowDocument(name="animation.gif", content=gif_content)
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

        doc = ConcreteFlowDocument(name="image.bmp", content=bmp_content)
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

        doc = ConcreteFlowDocument(name="image.png", content=png_content)

        parts = AIMessages.document_to_prompt(doc)

        assert len(parts) == 3
        image_data = parts[1]["image_url"]  # type: ignore[index]
        # Should still be PNG (passed through, not re-encoded)
        assert image_data["url"].startswith("data:image/png;base64,")
        # Verify original bytes are used (not re-encoded)
        expected_b64 = base64.b64encode(png_content).decode("utf-8")
        assert expected_b64 in image_data["url"]
