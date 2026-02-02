"""Tests for attachment rendering in document_to_prompt and attachment tiling."""

import io

from PIL import Image

from ai_pipeline_core.documents.attachment import Attachment
from ai_pipeline_core.llm import AIMessages
from ai_pipeline_core.llm.client import _prepare_images_for_model
from tests.support.helpers import ConcreteDocument


def _make_image_bytes(width: int, height: int, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(100, 150, 200)).save(buf, format=fmt)
    return buf.getvalue()


def _make_text_attachment(
    name: str = "notes.txt",
    text: str = "hello",
    description: str | None = None,
) -> Attachment:
    return Attachment(name=name, content=text.encode("utf-8"), description=description)


def _make_image_attachment(
    name: str = "screenshot.png",
    width: int = 100,
    height: int = 100,
    description: str | None = None,
) -> Attachment:
    return Attachment(name=name, content=_make_image_bytes(width, height), description=description)


def _make_pdf_attachment(name: str = "file.pdf", description: str | None = None) -> Attachment:
    return Attachment(
        name=name,
        content=b"%PDF-1.4\n%\xd3\xeb\xe9\xe1\n1 0 obj\n<</Type/Catalog>>\nendobj",
        description=description,
    )


# ---------------------------------------------------------------------------
# document_to_prompt — attachment rendering
# ---------------------------------------------------------------------------


class TestDocumentToPromptAttachments:
    def test_text_document_with_text_attachment(self):
        att = _make_text_attachment(description="side note")
        doc = ConcreteDocument(
            name="report.txt",
            content=b"Main content",
            attachments=(att,),
        )
        parts = AIMessages.document_to_prompt(doc)

        # Should have: text content + text attachment, merged into parts ending with </document>
        texts = [p["text"] for p in parts if p["type"] == "text"]  # type: ignore[index]
        combined = "".join(texts)
        assert "<content>\nMain content\n</content>" in combined
        assert '<attachment name="notes.txt" description="side note">' in combined
        assert "hello" in combined
        assert "</attachment>" in combined
        assert combined.endswith("</document>\n")

    def test_text_document_with_image_attachment(self):
        img_att = _make_image_attachment()
        doc = ConcreteDocument(
            name="report.txt",
            content=b"Main content",
            attachments=(img_att,),
        )
        parts = AIMessages.document_to_prompt(doc)

        # Should contain image_url part for the attachment
        types = [p["type"] for p in parts]
        assert "image_url" in types
        # First text part has the document content
        assert "<content>" in parts[0]["text"]  # type: ignore[index]

    def test_text_document_with_pdf_attachment(self):
        pdf_att = _make_pdf_attachment()
        doc = ConcreteDocument(
            name="report.txt",
            content=b"Main content",
            attachments=(pdf_att,),
        )
        parts = AIMessages.document_to_prompt(doc)

        types = [p["type"] for p in parts]
        assert "file" in types
        # Verify PDF data URI
        file_parts = [p for p in parts if p["type"] == "file"]
        assert file_parts[0]["file"]["file_data"].startswith("data:application/pdf;base64,")  # type: ignore[index]

    def test_image_document_with_text_attachment(self):
        img_bytes = _make_image_bytes(50, 50)
        att = _make_text_attachment(text="Caption for the image")
        doc = ConcreteDocument(
            name="photo.png",
            content=img_bytes,
            attachments=(att,),
        )
        parts = AIMessages.document_to_prompt(doc)

        types = [p["type"] for p in parts]
        # Should have: text header, image_url (doc content), text (</content>),
        # text (attachment), text (</document>)
        assert "image_url" in types
        texts = [p["text"] for p in parts if p["type"] == "text"]  # type: ignore[index]
        combined = "".join(texts)
        assert "Caption for the image" in combined
        assert "</attachment>" in combined
        assert combined.endswith("</document>\n")

    def test_mixed_attachments(self):
        text_att = _make_text_attachment(name="notes.txt", text="note text")
        img_att = _make_image_attachment(name="screen.png")
        pdf_att = _make_pdf_attachment(name="ref.pdf")
        doc = ConcreteDocument(
            name="report.txt",
            content=b"Main",
            attachments=(text_att, img_att, pdf_att),
        )
        parts = AIMessages.document_to_prompt(doc)

        types = [p["type"] for p in parts]
        assert "image_url" in types
        assert "file" in types
        texts = [p["text"] for p in parts if p["type"] == "text"]  # type: ignore[index]
        combined = "".join(texts)
        assert '<attachment name="notes.txt">' in combined
        assert '<attachment name="screen.png">' in combined
        assert '<attachment name="ref.pdf">' in combined
        assert combined.count("</attachment>") == 3

    def test_no_attachments_unchanged(self):
        """Documents without attachments produce identical structure to pre-attachment code."""
        # Text document
        text_doc = ConcreteDocument(name="test.txt", content=b"Hello")
        text_parts = AIMessages.document_to_prompt(text_doc)
        assert len(text_parts) == 1
        assert text_parts[0]["text"].endswith("</content>\n</document>\n")  # type: ignore[index]

        # Image document
        img_bytes = _make_image_bytes(50, 50)
        img_doc = ConcreteDocument(name="img.png", content=img_bytes)
        img_parts = AIMessages.document_to_prompt(img_doc)
        assert len(img_parts) == 3
        assert img_parts[2]["text"] == "</content>\n</document>\n"  # type: ignore[index]

        # PDF document
        pdf_doc = ConcreteDocument(
            name="doc.pdf",
            content=b"%PDF-1.4\n%\xd3\xeb\xe9\xe1\n1 0 obj\n<</Type/Catalog>>\nendobj",
        )
        pdf_parts = AIMessages.document_to_prompt(pdf_doc)
        assert len(pdf_parts) == 3
        assert pdf_parts[2]["text"] == "</content>\n</document>\n"  # type: ignore[index]

    def test_attachment_description_attribute(self):
        att_with_desc = _make_text_attachment(description="important")
        att_without_desc = _make_text_attachment(name="other.txt")

        doc = ConcreteDocument(
            name="report.txt",
            content=b"Main",
            attachments=(att_with_desc, att_without_desc),
        )
        parts = AIMessages.document_to_prompt(doc)
        texts = [p["text"] for p in parts if p["type"] == "text"]  # type: ignore[index]
        combined = "".join(texts)

        assert 'description="important"' in combined
        # The second attachment should NOT have a description attribute
        # Find the second attachment tag
        idx = combined.index('<attachment name="other.txt">')
        segment = combined[idx : idx + 50]
        assert "description=" not in segment

    def test_unsupported_image_format_attachment(self):
        """BMP attachment image gets converted to PNG."""
        bmp_bytes = _make_image_bytes(10, 10, fmt="BMP")
        att = Attachment(name="capture.bmp", content=bmp_bytes)
        doc = ConcreteDocument(
            name="report.txt",
            content=b"Main",
            attachments=(att,),
        )
        parts = AIMessages.document_to_prompt(doc)

        img_parts = [p for p in parts if p["type"] == "image_url"]
        assert len(img_parts) == 1
        # Should be converted to PNG
        assert img_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")  # type: ignore[index]

    def test_unsupported_attachment_type_skipped(self):
        """Attachment with unsupported MIME type is skipped."""
        att = Attachment(name="data.bin", content=b"\x00\x01\x02\x03")
        doc = ConcreteDocument(
            name="report.txt",
            content=b"Main",
            attachments=(att,),
        )
        parts = AIMessages.document_to_prompt(doc)

        # The attachment should be skipped — only the document content + close
        texts = [p["text"] for p in parts if p["type"] == "text"]  # type: ignore[index]
        combined = "".join(texts)
        assert '<attachment name="data.bin">' not in combined
        assert combined.endswith("</document>\n")

    def test_attachment_rendering_order(self):
        """Attachments render in tuple order."""
        att_a = _make_text_attachment(name="a.txt", text="first")
        att_b = _make_text_attachment(name="b.txt", text="second")
        att_c = _make_text_attachment(name="c.txt", text="third")
        doc = ConcreteDocument(
            name="report.txt",
            content=b"Main",
            attachments=(att_a, att_b, att_c),
        )
        parts = AIMessages.document_to_prompt(doc)
        texts = [p["text"] for p in parts if p["type"] == "text"]  # type: ignore[index]
        combined = "".join(texts)

        idx_a = combined.index('<attachment name="a.txt">')
        idx_b = combined.index('<attachment name="b.txt">')
        idx_c = combined.index('<attachment name="c.txt">')
        assert idx_a < idx_b < idx_c


# ---------------------------------------------------------------------------
# _prepare_images_for_model — attachment tiling
# ---------------------------------------------------------------------------


class TestPrepareImagesAttachmentTiling:
    def test_oversized_image_attachment_tiled(self):
        """Document with an oversized image attachment gets tiled."""
        large_att = _make_image_attachment(name="big.png", width=1000, height=3000)
        doc = ConcreteDocument(
            name="report.txt",
            content=b"Main",
            attachments=(large_att,),
        )
        msgs = AIMessages([doc])
        result = _prepare_images_for_model(msgs, "gpt-5.1")

        assert result is not msgs
        result_doc = result[0]
        assert isinstance(result_doc, ConcreteDocument)
        assert len(result_doc.attachments) > 1
        for att in result_doc.attachments:
            assert att.name.endswith(".jpg")

    def test_small_image_attachment_unchanged(self):
        """Small image attachment is not tiled."""
        small_att = _make_image_attachment(name="small.png", width=100, height=100)
        doc = ConcreteDocument(
            name="report.txt",
            content=b"Main",
            attachments=(small_att,),
        )
        msgs = AIMessages([doc])
        result = _prepare_images_for_model(msgs, "gpt-5.1")

        # No change needed since the attachment image is small
        assert result is msgs

    def test_text_attachment_unchanged(self):
        """Text-only attachments are not affected."""
        text_att = _make_text_attachment()
        doc = ConcreteDocument(
            name="report.txt",
            content=b"Main",
            attachments=(text_att,),
        )
        msgs = AIMessages([doc])
        result = _prepare_images_for_model(msgs, "gpt-5.1")

        assert result is msgs

    def test_mixed_attachments_only_images_tiled(self):
        """Only oversized image attachments are tiled; text and PDF preserved."""
        text_att = _make_text_attachment(name="notes.txt", text="note")
        large_img_att = _make_image_attachment(name="big.png", width=1000, height=3000)
        pdf_att = _make_pdf_attachment(name="ref.pdf")
        doc = ConcreteDocument(
            name="report.txt",
            content=b"Main",
            attachments=(text_att, large_img_att, pdf_att),
        )
        msgs = AIMessages([doc])
        result = _prepare_images_for_model(msgs, "gpt-5.1")

        assert result is not msgs
        result_doc = result[0]
        assert isinstance(result_doc, ConcreteDocument)

        # First attachment: text (unchanged)
        assert result_doc.attachments[0].name == "notes.txt"
        assert result_doc.attachments[0].is_text

        # Middle attachments: tiled images
        tiled = [a for a in result_doc.attachments if a.name.startswith("big")]
        assert len(tiled) > 1

        # Last attachment: PDF (unchanged)
        assert result_doc.attachments[-1].name == "ref.pdf"
        assert result_doc.attachments[-1].is_pdf

    def test_no_attachments_unchanged(self):
        """Document without attachments works as before."""
        doc = ConcreteDocument(name="report.txt", content=b"Main")
        msgs = AIMessages([doc])
        result = _prepare_images_for_model(msgs, "gpt-5.1")
        assert result is msgs

    def test_tiled_attachment_naming(self):
        """Tiled attachment names follow {prefix}_01_of_NN.jpg convention."""
        large_att = _make_image_attachment(name="screenshot.png", width=1000, height=3000)
        doc = ConcreteDocument(
            name="report.txt",
            content=b"Main",
            attachments=(large_att,),
        )
        msgs = AIMessages([doc])
        result = _prepare_images_for_model(msgs, "gpt-5.1")

        result_doc = result[0]
        assert isinstance(result_doc, ConcreteDocument)
        names = [a.name for a in result_doc.attachments]
        assert all(n.startswith("screenshot_") for n in names)
        assert any("01_of_" in n for n in names)
        assert all(n.endswith(".jpg") for n in names)

    def test_tiled_attachment_description(self):
        """Tiled attachment descriptions include part label."""
        large_att = _make_image_attachment(
            name="screenshot.png",
            width=1000,
            height=3000,
            description="Main screenshot",
        )
        doc = ConcreteDocument(
            name="report.txt",
            content=b"Main",
            attachments=(large_att,),
        )
        msgs = AIMessages([doc])
        result = _prepare_images_for_model(msgs, "gpt-5.1")

        result_doc = result[0]
        assert isinstance(result_doc, ConcreteDocument)
        for att in result_doc.attachments:
            assert att.description is not None
            assert "Main screenshot" in att.description

    def test_model_specific_config(self):
        """Gemini uses 3000x3000 config; 2000px image fits Gemini but not default."""
        att = _make_image_attachment(name="medium.png", width=2000, height=2000)
        doc = ConcreteDocument(
            name="report.txt",
            content=b"Main",
            attachments=(att,),
        )
        msgs = AIMessages([doc])

        # Gemini: no split (2000 < 3000)
        gemini_result = _prepare_images_for_model(msgs, "gemini-3-flash")
        assert gemini_result is msgs

        # Default: split (2000 > 1000)
        default_result = _prepare_images_for_model(msgs, "gpt-5.1")
        assert default_result is not msgs
        result_doc = default_result[0]
        assert isinstance(result_doc, ConcreteDocument)
        assert len(result_doc.attachments) > 1

    def test_failed_image_open_keeps_original(self):
        """If PIL fails to open attachment image, original is kept."""
        corrupt_att = Attachment(name="corrupt.png", content=b"not a real image")
        doc = ConcreteDocument(
            name="report.txt",
            content=b"Main",
            attachments=(corrupt_att,),
        )
        msgs = AIMessages([doc])
        result = _prepare_images_for_model(msgs, "gpt-5.1")

        # Corrupt image attachment is kept as-is — no crash
        assert result is msgs

    def test_document_fields_preserved_after_tiling(self):
        """After attachment tiling, Document fields are preserved."""
        large_att = _make_image_attachment(name="big.png", width=1000, height=3000)
        doc = ConcreteDocument(
            name="report.txt",
            content=b"Main content",
            description="Test report",
            attachments=(large_att,),
        )
        msgs = AIMessages([doc])
        result = _prepare_images_for_model(msgs, "gpt-5.1")

        result_doc = result[0]
        assert isinstance(result_doc, ConcreteDocument)
        assert result_doc.name == "report.txt"
        assert result_doc.content == b"Main content"
        assert result_doc.description == "Test report"
        assert len(result_doc.attachments) > 1
