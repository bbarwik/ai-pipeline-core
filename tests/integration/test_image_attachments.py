"""Integration tests for image/PDF attachments on text documents.

Creates images with text rendered on them and verifies the LLM can
read and extract the text — proving attachments actually reach the model.
"""

from io import BytesIO

import pytest
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from ai_pipeline_core.documents import Attachment
from ai_pipeline_core.llm import Conversation, ModelOptions
from ai_pipeline_core.settings import settings
from tests.support.helpers import ConcreteDocument


pytestmark = pytest.mark.integration

HAS_API_KEYS = bool(settings.openai_api_key and settings.openai_base_url)

MODEL = "gemini-3-flash"
OPTIONS = ModelOptions(max_completion_tokens=500)


def _make_text_image(text: str, width: int = 400, height: int = 200, bg: str = "white", fg: str = "black") -> bytes:
    """Create a JPEG image with the given text rendered large and centered."""
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except OSError:
        font = ImageFont.load_default(size=48)
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (width - (bbox[2] - bbox[0])) // 2
    y = (height - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), text, fill=fg, font=font)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _make_tall_text_image(texts: list[str], width: int = 600, tile_height: int = 400) -> bytes:
    """Create a tall image with each text on a separate tile (forces image splitting)."""
    total_height = tile_height * len(texts)
    img = Image.new("RGB", (width, total_height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
    except OSError:
        font = ImageFont.load_default(size=72)
    for i, text in enumerate(texts):
        y_center = i * tile_height + tile_height // 2
        bbox = draw.textbbox((0, 0), text, font=font)
        x = (width - (bbox[2] - bbox[0])) // 2
        y = y_center - (bbox[3] - bbox[1]) // 2
        draw.text((x, y), text, fill="black", font=font)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


class ExtractedWords(BaseModel):
    """Extracted words from image attachments."""

    words: list[str]


@pytest.mark.skipif(not HAS_API_KEYS, reason="API keys not configured")
class TestTextDocImageAttachmentIntegration:
    """Verify that image attachments on text documents actually reach the LLM."""

    @pytest.mark.asyncio
    async def test_single_screenshot_readable(self):
        """LLM must be able to read text from a single image attachment on a text doc."""
        screenshot = _make_text_image("ALPHA")
        doc = ConcreteDocument.create(
            name="page.md",
            content="# Website\n\nThis page has a screenshot attached.",
            attachments=(Attachment(name="screenshot.jpg", content=screenshot),),
        )

        conv = Conversation(model=MODEL, model_options=OPTIONS)
        conv = conv.with_document(doc)
        conv = await conv.send("What word is written in the screenshot image? Reply with just the word.")

        assert "ALPHA" in conv.content.upper()

    @pytest.mark.asyncio
    async def test_multiple_image_attachments_readable(self):
        """LLM must read text from multiple image attachments on a text doc."""
        img1 = _make_text_image("BRAVO", bg="lightyellow")
        img2 = _make_text_image("CHARLIE", bg="lightblue")
        img3 = _make_text_image("DELTA", bg="lightgreen")
        doc = ConcreteDocument.create(
            name="page.md",
            content="# Multi-image page\n\nThree screenshots attached.",
            attachments=(
                Attachment(name="img1.jpg", content=img1),
                Attachment(name="img2.jpg", content=img2),
                Attachment(name="img3.jpg", content=img3),
            ),
        )

        conv = Conversation(model=MODEL, model_options=OPTIONS)
        conv = conv.with_document(doc)
        conv = await conv.send_structured(
            "List all words shown in the image attachments.",
            response_format=ExtractedWords,
        )

        assert conv.parsed is not None
        words_upper = [w.upper() for w in conv.parsed.words]
        assert "BRAVO" in words_upper
        assert "CHARLIE" in words_upper
        assert "DELTA" in words_upper

    @pytest.mark.asyncio
    async def test_mixed_text_and_image_attachments(self):
        """LLM must see both text attachments and image attachments."""
        img = _make_text_image("FOXTROT")
        doc = ConcreteDocument.create(
            name="report.md",
            content="# Research Report",
            attachments=(
                Attachment(name="notes.txt", content=b"The secret code is ECHO."),
                Attachment(name="chart.jpg", content=img),
            ),
        )

        conv = Conversation(model=MODEL, model_options=OPTIONS)
        conv = conv.with_document(doc)
        conv = await conv.send_structured(
            "What is the secret code from the text attachment, and what word is shown in the image?",
            response_format=ExtractedWords,
        )

        assert conv.parsed is not None
        words_upper = [w.upper() for w in conv.parsed.words]
        assert "ECHO" in words_upper
        assert "FOXTROT" in words_upper


@pytest.mark.skipif(not HAS_API_KEYS, reason="API keys not configured")
class TestImageSplittingIntegration:
    """Verify that large images that get split into tiles are still readable by the LLM."""

    @pytest.mark.asyncio
    async def test_tall_image_split_content_readable(self):
        """LLM must read content from an image that exceeds model limits and gets split."""
        # Create an image taller than 3000px (Gemini limit) to force splitting
        tall_img = _make_tall_text_image(["GOLF", "HOTEL", "INDIA", "JULIET", "KILO", "LIMA", "MIKE", "NOVEMBER"])
        doc = ConcreteDocument.create(name="screenshot.jpg", content=tall_img)

        conv = Conversation(model=MODEL, model_options=OPTIONS)
        conv = conv.with_document(doc)
        conv = await conv.send_structured(
            "List all words visible in this image. The image was split into parts.",
            response_format=ExtractedWords,
        )

        assert conv.parsed is not None
        words_upper = [w.upper() for w in conv.parsed.words]
        # Must read at least most of the words across split tiles
        found = sum(1 for w in ["GOLF", "HOTEL", "INDIA", "JULIET", "KILO", "LIMA", "MIKE", "NOVEMBER"] if w in words_upper)
        assert found >= 6, f"Expected at least 6 of 8 words readable, got {found}: {words_upper}"

    @pytest.mark.asyncio
    async def test_tall_image_as_attachment_split_readable(self):
        """Tall image as attachment on text doc must also be split and readable."""
        tall_img = _make_tall_text_image(["OSCAR", "PAPA", "QUEBEC", "ROMEO", "SIERRA", "TANGO", "UNIFORM", "VICTOR"])
        doc = ConcreteDocument.create(
            name="page.md",
            content="# Full page capture with tall screenshot.",
            attachments=(Attachment(name="full_page.jpg", content=tall_img),),
        )

        conv = Conversation(model=MODEL, model_options=OPTIONS)
        conv = conv.with_document(doc)
        conv = await conv.send_structured(
            "List all words visible in the attached screenshot image. It may be split into parts.",
            response_format=ExtractedWords,
        )

        assert conv.parsed is not None
        words_upper = [w.upper() for w in conv.parsed.words]
        found = sum(1 for w in ["OSCAR", "PAPA", "QUEBEC", "ROMEO", "SIERRA", "TANGO", "UNIFORM", "VICTOR"] if w in words_upper)
        assert found >= 6, f"Expected at least 6 of 8 words readable, got {found}: {words_upper}"
