"""Tests for automatic image splitting at the LLM boundary."""

import io

from PIL import Image

from ai_pipeline_core.documents import Document, TemporaryDocument
from ai_pipeline_core.llm import AIMessages
from ai_pipeline_core.llm.client import (
    _DEFAULT_IMAGE_CONFIG,
    _GEMINI_IMAGE_CONFIG,
    _get_image_config,
    _prepare_images_for_model,
)
from tests.test_helpers import ConcreteFlowDocument, create_test_model_response


def _make_image(width: int, height: int, fmt: str = "PNG") -> bytes:
    """Create a synthetic image of given dimensions."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(100, 150, 200)).save(buf, format=fmt)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _get_image_config
# ---------------------------------------------------------------------------


class TestGetImageConfig:
    """Test model-to-config mapping."""

    def test_gemini_model(self):
        assert _get_image_config("gemini-3-flash") is _GEMINI_IMAGE_CONFIG

    def test_gemini_pro_model(self):
        assert _get_image_config("gemini-3-pro") is _GEMINI_IMAGE_CONFIG

    def test_gemini_case_insensitive(self):
        assert _get_image_config("Gemini-3-Flash") is _GEMINI_IMAGE_CONFIG

    def test_gpt_model_gets_default(self):
        assert _get_image_config("gpt-5.1") is _DEFAULT_IMAGE_CONFIG

    def test_claude_model_gets_default(self):
        assert _get_image_config("claude-opus") is _DEFAULT_IMAGE_CONFIG

    def test_grok_model_gets_default(self):
        assert _get_image_config("grok-4.1-fast") is _DEFAULT_IMAGE_CONFIG

    def test_unknown_model_gets_default(self):
        assert _get_image_config("custom-llm-v2") is _DEFAULT_IMAGE_CONFIG

    def test_config_values_gemini(self):
        cfg = _GEMINI_IMAGE_CONFIG
        assert cfg.max_dimension == 3000
        assert cfg.max_pixels == 9_000_000

    def test_config_values_default(self):
        cfg = _DEFAULT_IMAGE_CONFIG
        assert cfg.max_dimension == 1000
        assert cfg.max_pixels == 1_000_000


# ---------------------------------------------------------------------------
# _prepare_images_for_model
# ---------------------------------------------------------------------------


class TestPrepareImagesForModel:
    """Test automatic image splitting."""

    def test_no_images_returns_same_instance(self):
        """When there are no image documents, return the exact same object."""
        msgs = AIMessages(["hello", "world"])
        result = _prepare_images_for_model(msgs, "gpt-5.1")
        assert result is msgs

    def test_small_image_passes_through(self):
        """Images within limits are not split."""
        small_png = _make_image(500, 500)
        doc = ConcreteFlowDocument(name="small.png", content=small_png)
        msgs = AIMessages([doc])

        result = _prepare_images_for_model(msgs, "gpt-5.1")

        assert result is msgs  # no change, same instance
        assert len(result) == 1
        assert result[0] is doc

    def test_large_image_gets_split_default(self):
        """An image exceeding 1000x1000 default gets split for non-Gemini models."""
        large_png = _make_image(1000, 3000)
        doc = ConcreteFlowDocument(name="tall.png", content=large_png)
        msgs = AIMessages([doc])

        result = _prepare_images_for_model(msgs, "gpt-5.1")

        assert result is not msgs
        assert len(result) > 1
        for tile in result:
            assert isinstance(tile, TemporaryDocument)
            assert tile.mime_type == "image/jpeg"

    def test_large_image_fits_gemini_not_default(self):
        """A 2000x2000 image fits Gemini (3000px) but exceeds default (1000px)."""
        img_bytes = _make_image(2000, 2000)
        doc = ConcreteFlowDocument(name="medium.png", content=img_bytes)
        msgs = AIMessages([doc])

        # Gemini: no split (2000 < 3000, 4M < 9M)
        gemini_result = _prepare_images_for_model(msgs, "gemini-3-flash")
        assert gemini_result is msgs

        # Default: split (2000 > 1000)
        default_result = _prepare_images_for_model(msgs, "gpt-5.1")
        assert default_result is not msgs
        assert len(default_result) > 1

    def test_gemini_at_exact_3000x3000_not_split(self):
        """Image exactly at 3000x3000 fits Gemini limits — no split."""
        exact_png = _make_image(3000, 3000)
        doc = ConcreteFlowDocument(name="exact.png", content=exact_png)
        msgs = AIMessages([doc])

        result = _prepare_images_for_model(msgs, "gemini-3-flash")

        assert result is msgs

    def test_gemini_one_pixel_over_3000_gets_reprocessed(self):
        """Image at 3001x3000 exceeds Gemini max_dimension — gets reprocessed."""
        over_png = _make_image(3001, 3000)
        doc = ConcreteFlowDocument(name="over.png", content=over_png)
        msgs = AIMessages([doc])

        result = _prepare_images_for_model(msgs, "gemini-3-flash")

        assert result is not msgs
        assert len(result) >= 1
        for tile in result:
            assert isinstance(tile, TemporaryDocument)

    def test_gemini_splits_tall_image(self):
        """Gemini splits when image height exceeds 3000px."""
        huge_png = _make_image(3000, 6000)
        doc = ConcreteFlowDocument(name="huge.png", content=huge_png)
        msgs = AIMessages([doc])

        result = _prepare_images_for_model(msgs, "gemini-3-flash")

        assert result is not msgs
        assert len(result) > 1

    def test_mixed_content_preserved(self):
        """Strings, small images, and model responses pass through; only large images split."""
        small_png = _make_image(200, 200)
        large_png = _make_image(1000, 3000)
        small_doc = ConcreteFlowDocument(name="small.png", content=small_png)
        large_doc = ConcreteFlowDocument(name="large.png", content=large_png)
        text_doc = ConcreteFlowDocument(name="readme.md", content=b"# Hello")

        msgs = AIMessages(["question", small_doc, large_doc, text_doc, "followup"])

        result = _prepare_images_for_model(msgs, "gpt-5.1")

        assert result is not msgs  # large image was split
        # First element: string
        assert result[0] == "question"
        # Second element: small image, unchanged
        assert result[1] is small_doc
        # Middle elements: tiles from large image
        tile_count = len(result) - 4  # subtract string, small_doc, text_doc, string
        assert tile_count > 1
        for i in range(2, 2 + tile_count):
            assert isinstance(result[i], TemporaryDocument)
        # After tiles: text doc, then followup string
        assert result[2 + tile_count] is text_doc
        assert result[3 + tile_count] == "followup"

    def test_frozen_messages_not_mutated(self):
        """Frozen AIMessages are never mutated; a new instance is returned."""
        large_png = _make_image(1000, 3000)
        doc = ConcreteFlowDocument(name="big.png", content=large_png)
        msgs = AIMessages([doc], frozen=True)

        result = _prepare_images_for_model(msgs, "gpt-5.1")

        assert result is not msgs
        assert len(result) > 1
        # Original still has exactly 1 element
        assert len(msgs) == 1

    def test_non_image_document_passes_through(self):
        """PDF and text documents are never split."""
        pdf_content = b"%PDF-1.4\n%\xd3\xeb\xe9\xe1\n1 0 obj\n<</Type/Catalog>>\nendobj"
        pdf_doc = ConcreteFlowDocument(name="doc.pdf", content=pdf_content)
        text_doc = ConcreteFlowDocument(name="notes.txt", content=b"some text")
        msgs = AIMessages([pdf_doc, text_doc])

        result = _prepare_images_for_model(msgs, "gpt-5.1")

        assert result is msgs

    def test_gif_large_image_gets_split(self):
        """Large GIF images are split into JPEG tiles."""
        gif_bytes = _make_image(1500, 1500, fmt="GIF")
        doc = ConcreteFlowDocument(name="anim.gif", content=gif_bytes)
        msgs = AIMessages([doc])

        result = _prepare_images_for_model(msgs, "gpt-5.1")

        assert result is not msgs
        assert len(result) > 1
        for tile in result:
            assert isinstance(tile, TemporaryDocument)
            assert tile.mime_type == "image/jpeg"

    def test_small_gif_passes_through(self):
        """A small GIF within limits passes through without conversion."""
        gif_bytes = _make_image(500, 500, fmt="GIF")
        doc = ConcreteFlowDocument(name="small.gif", content=gif_bytes)
        msgs = AIMessages([doc])

        result = _prepare_images_for_model(msgs, "gpt-5.1")

        assert result is msgs
        assert result[0] is doc

    def test_tile_names_preserve_original_name(self):
        """Split tiles derive their name prefix from the original document."""
        large_png = _make_image(1000, 3000)
        doc = ConcreteFlowDocument(name="screenshot.png", content=large_png)
        msgs = AIMessages([doc])

        result = _prepare_images_for_model(msgs, "gpt-5.1")

        assert len(result) > 1
        assert all(isinstance(tile, Document) for tile in result)
        names = [tile.name for tile in result if isinstance(tile, Document)]
        # Names should derive from original "screenshot" prefix
        assert all(n.startswith("screenshot_") for n in names)
        assert any("01_of_" in n for n in names)
        assert any("02_of_" in n for n in names)

    def test_empty_messages(self):
        """Empty AIMessages returns same instance."""
        msgs = AIMessages()
        result = _prepare_images_for_model(msgs, "gpt-5.1")
        assert result is msgs

    def test_model_response_passes_through(self):
        """ModelResponse messages pass through unchanged."""
        response = create_test_model_response(
            id="test",
            object="chat.completion",
            created=0,
            model="gpt-5.1",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
        )
        msgs = AIMessages([response])
        result = _prepare_images_for_model(msgs, "gpt-5.1")
        assert result is msgs

    def test_at_exact_limit_not_split(self):
        """Image exactly at 1000x1000 should NOT be split for default config."""
        exact_png = _make_image(1000, 1000)
        doc = ConcreteFlowDocument(name="exact.png", content=exact_png)
        msgs = AIMessages([doc])

        result = _prepare_images_for_model(msgs, "gpt-5.1")

        assert result is msgs

    def test_one_pixel_over_dimension_gets_reprocessed(self):
        """Image at 1001x1000 exceeds max_dimension — width trimmed, single JPEG tile."""
        over_png = _make_image(1001, 1000)
        doc = ConcreteFlowDocument(name="over.png", content=over_png)
        msgs = AIMessages([doc])

        result = _prepare_images_for_model(msgs, "gpt-5.1")

        assert result is not msgs
        # Width trim to 1000, height 1000 fits in one tile
        assert len(result) == 1
        tile = result[0]
        assert isinstance(tile, TemporaryDocument)
        assert tile.mime_type == "image/jpeg"

    def test_corrupt_image_passes_through(self):
        """Corrupt image bytes don't crash — document passes through unchanged."""
        doc = ConcreteFlowDocument(name="corrupt.png", content=b"not a real image")
        # Force image MIME type via extension detection
        assert doc.is_image  # .png extension → image/png
        msgs = AIMessages([doc])

        result = _prepare_images_for_model(msgs, "gpt-5.1")

        assert result is msgs
        assert result[0] is doc

    def test_multiple_large_images_all_split(self):
        """Multiple large images in one AIMessages are all split independently."""
        img_a = _make_image(1000, 3000)
        img_b = _make_image(1000, 2000)
        doc_a = ConcreteFlowDocument(name="page_a.png", content=img_a)
        doc_b = ConcreteFlowDocument(name="page_b.png", content=img_b)
        msgs = AIMessages([doc_a, doc_b])

        result = _prepare_images_for_model(msgs, "gpt-5.1")

        assert result is not msgs
        assert len(result) > 2  # both images produced tiles
        # All tiles are TemporaryDocuments
        for tile in result:
            assert isinstance(tile, TemporaryDocument)
        # Tiles from different sources have different name prefixes
        names = [tile.name for tile in result if isinstance(tile, Document)]
        assert any("page_a" in n for n in names)
        assert any("page_b" in n for n in names)

    def test_rgba_png_gets_split(self):
        """RGBA PNG with alpha channel is split into JPEG tiles without error."""
        buf = io.BytesIO()
        Image.new("RGBA", (1500, 1500), color=(255, 0, 0, 128)).save(buf, format="PNG")
        rgba_bytes = buf.getvalue()
        doc = ConcreteFlowDocument(name="transparent.png", content=rgba_bytes)
        msgs = AIMessages([doc])

        result = _prepare_images_for_model(msgs, "gpt-5.1")

        assert result is not msgs
        assert len(result) > 1
        for tile in result:
            assert isinstance(tile, TemporaryDocument)
            assert tile.mime_type == "image/jpeg"
