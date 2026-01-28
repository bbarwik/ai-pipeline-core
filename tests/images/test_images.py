"""Tests for ai_pipeline_core.images public API — process_image and process_image_to_documents."""

from io import BytesIO

import pytest
from PIL import Image

from ai_pipeline_core import (
    ImagePart,
    ImagePreset,
    ImageProcessingConfig,
    ImageProcessingError,
    ProcessedImage,
    TemporaryDocument,
    process_image,
    process_image_to_documents,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_image_bytes(
    width: int, height: int, fmt: str = "PNG", color: tuple[int, int, int] = (128, 128, 128)
) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format=fmt)
    return buf.getvalue()


def make_document(width: int, height: int) -> TemporaryDocument:
    return TemporaryDocument.create(name="screenshot.png", content=make_image_bytes(width, height))


# ---------------------------------------------------------------------------
# ImagePreset / ImageProcessingConfig
# ---------------------------------------------------------------------------


class TestConfig:
    """Tests for configuration and presets."""

    def test_all_presets_exist(self):
        for preset in ImagePreset:
            config = ImageProcessingConfig.for_preset(preset)
            assert config.max_dimension > 0
            assert config.max_pixels > 0
            assert config.jpeg_quality > 0

    def test_gemini_preset_values(self):
        config = ImageProcessingConfig.for_preset(ImagePreset.GEMINI)
        assert config.max_dimension == 3000
        assert config.max_pixels == 9_000_000
        assert config.jpeg_quality == 75

    def test_claude_preset_values(self):
        config = ImageProcessingConfig.for_preset(ImagePreset.CLAUDE)
        assert config.max_dimension == 1568
        assert config.max_pixels == 1_150_000
        assert config.jpeg_quality == 60

    def test_gpt4v_preset_values(self):
        config = ImageProcessingConfig.for_preset(ImagePreset.GPT4V)
        assert config.max_dimension == 2048
        assert config.max_pixels == 4_000_000
        assert config.jpeg_quality == 70

    def test_custom_config(self):
        config = ImageProcessingConfig(max_dimension=2000, max_pixels=3_000_000, jpeg_quality=80)
        assert config.max_dimension == 2000
        assert config.max_pixels == 3_000_000
        assert config.jpeg_quality == 80
        assert config.overlap_fraction == 0.20
        assert config.max_parts == 20

    def test_config_is_frozen(self):
        config = ImageProcessingConfig.for_preset(ImagePreset.GEMINI)
        with pytest.raises(Exception):
            config.max_dimension = 1000  # type: ignore[misc]


# ---------------------------------------------------------------------------
# process_image — small images (no split)
# ---------------------------------------------------------------------------


class TestProcessImageSmall:
    """Tests for process_image with images that don't need splitting."""

    def test_small_rgb_png(self):
        raw = make_image_bytes(800, 600)
        result = process_image(raw)
        assert len(result) == 1
        assert result.original_width == 800
        assert result.original_height == 600
        assert result.original_bytes == len(raw)
        assert result.output_bytes > 0
        assert not result.was_trimmed
        assert result.warnings == []

    def test_single_part_properties(self):
        result = process_image(make_image_bytes(800, 600))
        part = result[0]
        assert part.index == 0
        assert part.total == 1
        assert part.label == "Full image"
        assert part.width == 800
        assert part.height == 600
        assert part.source_y == 0

    def test_compression_ratio(self):
        # Use a larger, non-uniform image so JPEG is actually smaller than PNG
        import random

        random.seed(42)
        img = Image.new("RGB", (1200, 900))
        img.putdata([
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for _ in range(1200 * 900)
        ])
        buf = BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()
        result = process_image(raw)
        assert 0.0 < result.compression_ratio < 1.0  # JPEG smaller than noisy PNG

    def test_output_is_jpeg(self):
        result = process_image(make_image_bytes(800, 600))
        img = Image.open(BytesIO(result[0].data))
        assert img.format == "JPEG"

    def test_iterable_protocol(self):
        result = process_image(make_image_bytes(800, 600))
        parts = list(result)
        assert len(parts) == 1
        assert isinstance(parts[0], ImagePart)

    def test_getitem(self):
        result = process_image(make_image_bytes(800, 600))
        assert result[0].index == 0


# ---------------------------------------------------------------------------
# process_image — tall images (splitting)
# ---------------------------------------------------------------------------


class TestProcessImageTall:
    """Tests for process_image with tall images that get split."""

    def test_splits_tall_image(self):
        raw = make_image_bytes(1000, 7000)
        result = process_image(raw)
        assert len(result) > 1

    def test_part_labels_sequential(self):
        result = process_image(make_image_bytes(1000, 7000))
        total = len(result)
        for i, part in enumerate(result):
            assert part.index == i
            assert part.total == total
            assert part.label == f"Part {i + 1}/{total}"

    def test_all_parts_valid_jpeg(self):
        result = process_image(make_image_bytes(1000, 7000))
        for part in result:
            img = Image.open(BytesIO(part.data))
            assert img.format == "JPEG"
            assert img.size[0] == part.width
            assert img.size[1] == part.height

    def test_coverage_first_to_last(self):
        result = process_image(make_image_bytes(1000, 7000))
        # First part starts at top
        assert result[0].source_y == 0
        # Last part reaches bottom
        last = result[-1]
        assert last.source_y + last.source_height == 7000

    def test_parts_monotonically_ordered(self):
        result = process_image(make_image_bytes(1000, 7000))
        for i in range(1, len(result)):
            assert result[i].source_y >= result[i - 1].source_y


# ---------------------------------------------------------------------------
# process_image — wide images (trimming)
# ---------------------------------------------------------------------------


class TestProcessImageWide:
    """Tests for width trimming."""

    def test_trims_wide_image(self):
        result = process_image(make_image_bytes(5000, 600))
        assert result.was_trimmed
        assert result[0].width <= 3000

    def test_narrow_not_trimmed(self):
        result = process_image(make_image_bytes(800, 600))
        assert not result.was_trimmed


# ---------------------------------------------------------------------------
# process_image — presets and custom config
# ---------------------------------------------------------------------------


class TestProcessImagePresets:
    """Tests for preset and custom config handling."""

    def test_default_preset_is_gemini(self):
        result = process_image(make_image_bytes(800, 600))
        # Default is GEMINI (3000px) - 800px image shouldn't be trimmed
        assert not result.was_trimmed

    def test_claude_preset_trims_1920(self):
        result = process_image(make_image_bytes(1920, 1080), preset=ImagePreset.CLAUDE)
        assert result.was_trimmed

    def test_custom_config_overrides_preset(self):
        config = ImageProcessingConfig(max_dimension=500, max_pixels=250_000, jpeg_quality=50)
        result = process_image(make_image_bytes(800, 600), config=config)
        assert result.was_trimmed
        assert result[0].width <= 500


# ---------------------------------------------------------------------------
# process_image — Document input
# ---------------------------------------------------------------------------


class TestProcessImageDocument:
    """Tests for Document input support."""

    def test_accepts_document(self):
        doc = make_document(800, 600)
        result = process_image(doc)
        assert len(result) == 1
        assert result.original_width == 800

    def test_accepts_bytes(self):
        result = process_image(make_image_bytes(800, 600))
        assert len(result) == 1


# ---------------------------------------------------------------------------
# process_image — error handling
# ---------------------------------------------------------------------------


class TestProcessImageErrors:
    """Tests for error cases."""

    def test_empty_bytes(self):
        with pytest.raises(ImageProcessingError, match="Empty image data"):
            process_image(b"")

    def test_invalid_bytes(self):
        with pytest.raises(ImageProcessingError, match="Failed to decode"):
            process_image(b"not an image at all")

    def test_invalid_type(self):
        with pytest.raises(ImageProcessingError, match="Unsupported"):
            process_image(12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# process_image_to_documents
# ---------------------------------------------------------------------------


class TestProcessImageToDocuments:
    """Tests for the TemporaryDocument convenience wrapper."""

    def test_single_part_naming(self):
        docs = process_image_to_documents(make_image_bytes(800, 600))
        assert len(docs) == 1
        assert docs[0].name == "image.jpg"
        assert docs[0].description is None

    def test_multi_part_naming(self):
        docs = process_image_to_documents(make_image_bytes(1000, 7000))
        assert len(docs) > 1
        for i, doc in enumerate(docs):
            expected = f"image_{i + 1:02d}_of_{len(docs):02d}.jpg"
            assert doc.name == expected
            assert doc.description is not None

    def test_custom_prefix(self):
        docs = process_image_to_documents(make_image_bytes(800, 600), name_prefix="screenshot")
        assert docs[0].name == "screenshot.jpg"

    def test_returns_temporary_documents(self):
        docs = process_image_to_documents(make_image_bytes(800, 600))
        assert all(isinstance(d, TemporaryDocument) for d in docs)

    def test_sources_propagated(self):
        docs = process_image_to_documents(make_image_bytes(800, 600), sources=["ref1", "ref2"])
        for doc in docs:
            assert doc.sources is not None
            assert "ref1" in doc.sources
            assert "ref2" in doc.sources

    def test_document_input_adds_sha256(self):
        doc = make_document(800, 600)
        docs = process_image_to_documents(doc)
        for d in docs:
            assert d.sources is not None
            assert doc.sha256 in d.sources

    def test_preset_forwarded(self):
        docs_gemini = process_image_to_documents(
            make_image_bytes(1920, 1080), preset=ImagePreset.GEMINI
        )
        docs_claude = process_image_to_documents(
            make_image_bytes(1920, 1080), preset=ImagePreset.CLAUDE
        )
        # Claude trims 1920px to 1568px, Gemini doesn't
        gemini_img = Image.open(BytesIO(docs_gemini[0].content))
        claude_img = Image.open(BytesIO(docs_claude[0].content))
        assert gemini_img.size[0] == 1920
        assert claude_img.size[0] <= 1568


# ---------------------------------------------------------------------------
# ProcessedImage / ImagePart models
# ---------------------------------------------------------------------------


class TestModels:
    """Tests for result model properties."""

    def test_processed_image_frozen(self):
        result = process_image(make_image_bytes(800, 600))
        with pytest.raises(Exception):
            result.original_width = 999  # type: ignore[misc]

    def test_image_part_frozen(self):
        result = process_image(make_image_bytes(800, 600))
        with pytest.raises(Exception):
            result[0].width = 999  # type: ignore[misc]

    def test_compression_ratio_with_zero_bytes(self):
        # Edge case: original_bytes <= 0
        result = ProcessedImage(
            parts=[],
            original_width=0,
            original_height=0,
            original_bytes=0,
            output_bytes=0,
            was_trimmed=False,
        )
        assert result.compression_ratio == 1.0
