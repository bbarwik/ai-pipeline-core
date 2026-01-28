"""Image processing utilities for LLM vision models.

@public

Splits large images, compresses to JPEG, and respects model-specific constraints.
Designed for website screenshots, document pages, and other visual content
sent to vision-capable LLMs.

Quick Start:
    >>> from ai_pipeline_core.images import process_image, ImagePreset
    >>>
    >>> result = process_image(screenshot_bytes)
    >>> for part in result:
    ...     send_to_llm(part.data, context=part.label)
    >>>
    >>> result = process_image(screenshot_bytes, preset=ImagePreset.GEMINI)
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from ai_pipeline_core.documents import Document, TemporaryDocument

from ._processing import execute_split, load_and_normalize, plan_split

__all__ = [
    "ImagePreset",
    "ImageProcessingConfig",
    "ImagePart",
    "ProcessedImage",
    "ImageProcessingError",
    "process_image",
    "process_image_to_documents",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ImagePreset(StrEnum):
    """Presets for LLM vision model constraints.

    @public
    """

    GEMINI = "gemini"
    CLAUDE = "claude"
    GPT4V = "gpt4v"


class ImageProcessingConfig(BaseModel):
    """Configuration for image processing.

    @public

    Use ``for_preset`` for standard configurations or construct directly for
    custom constraints.

    Example:
        >>> config = ImageProcessingConfig.for_preset(ImagePreset.GEMINI)
        >>> config = ImageProcessingConfig(max_dimension=2000, jpeg_quality=80)
    """

    model_config = {"frozen": True}

    max_dimension: int = Field(
        default=3000,
        ge=100,
        le=8192,
        description="Maximum width AND height in pixels",
    )
    max_pixels: int = Field(
        default=9_000_000,
        ge=10_000,
        description="Maximum total pixels per output image part",
    )
    overlap_fraction: float = Field(
        default=0.20,
        ge=0.0,
        le=0.5,
        description="Overlap between adjacent vertical parts (0.0-0.5)",
    )
    max_parts: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of output image parts",
    )
    jpeg_quality: int = Field(
        default=60,
        ge=10,
        le=95,
        description="JPEG compression quality (10-95)",
    )

    @classmethod
    def for_preset(cls, preset: ImagePreset) -> "ImageProcessingConfig":
        """Create configuration from a model preset.

        @public
        """
        return _PRESETS[preset]


_PRESETS: dict[ImagePreset, ImageProcessingConfig] = {
    ImagePreset.GEMINI: ImageProcessingConfig(
        max_dimension=3000,
        max_pixels=9_000_000,
        jpeg_quality=75,
    ),
    ImagePreset.CLAUDE: ImageProcessingConfig(
        max_dimension=1568,
        max_pixels=1_150_000,
        jpeg_quality=60,
    ),
    ImagePreset.GPT4V: ImageProcessingConfig(
        max_dimension=2048,
        max_pixels=4_000_000,
        jpeg_quality=70,
    ),
}


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class ImagePart(BaseModel):
    """A single processed image part.

    @public
    """

    model_config = {"frozen": True}

    data: bytes = Field(repr=False)
    width: int
    height: int
    index: int = Field(ge=0, description="0-indexed position")
    total: int = Field(ge=1, description="Total number of parts")
    source_y: int = Field(ge=0, description="Y offset in original image")
    source_height: int = Field(ge=1, description="Height of region in original")

    @property
    def label(self) -> str:
        """Human-readable label for LLM context, 1-indexed.

        @public
        """
        if self.total == 1:
            return "Full image"
        return f"Part {self.index + 1}/{self.total}"


class ProcessedImage(BaseModel):
    """Result of image processing.

    @public

    Iterable: ``for part in result`` iterates over parts.
    """

    model_config = {"frozen": True}

    parts: list[ImagePart]
    original_width: int
    original_height: int
    original_bytes: int
    output_bytes: int
    was_trimmed: bool = Field(description="True if width was trimmed to fit")
    warnings: list[str] = Field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        """Output size / input size (lower means more compression).

        @public
        """
        if self.original_bytes <= 0:
            return 1.0
        return self.output_bytes / self.original_bytes

    def __len__(self) -> int:
        return len(self.parts)

    def __iter__(self):  # type: ignore[override]
        return iter(self.parts)

    def __getitem__(self, idx: int) -> ImagePart:
        return self.parts[idx]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ImageProcessingError(Exception):
    """Image processing failed.

    @public
    """


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_image(
    image: bytes | Document,
    preset: ImagePreset = ImagePreset.GEMINI,
    config: ImageProcessingConfig | None = None,
) -> ProcessedImage:
    """Process an image for LLM vision models.

    @public

    Splits tall images vertically with overlap, trims width if needed, and
    compresses to JPEG.  The default preset is **GEMINI** (3 000 px, 9 M pixels).

    Args:
        image: Raw image bytes or a Document whose content is an image.
        preset: Model preset (ignored when *config* is provided).
        config: Custom configuration that overrides the preset.

    Returns:
        A ``ProcessedImage`` containing one or more ``ImagePart`` objects.

    Raises:
        ImageProcessingError: If the image cannot be decoded or processed.

    Example:
        >>> result = process_image(screenshot_bytes)
        >>> for part in result:
        ...     print(part.label, len(part.data))
    """
    effective = config if config is not None else ImageProcessingConfig.for_preset(preset)

    # Resolve input bytes
    raw: bytes
    if isinstance(image, Document):
        raw = image.content
    elif isinstance(image, bytes):  # type: ignore[reportUnnecessaryIsInstance]
        raw = image
    else:
        raise ImageProcessingError(f"Unsupported image input type: {type(image)}")

    if not raw:
        raise ImageProcessingError("Empty image data")

    original_bytes = len(raw)

    # Load & normalise
    try:
        img = load_and_normalize(raw)
    except Exception as exc:
        raise ImageProcessingError(f"Failed to decode image: {exc}") from exc

    original_width, original_height = img.size

    # Plan
    plan = plan_split(
        width=original_width,
        height=original_height,
        max_dimension=effective.max_dimension,
        max_pixels=effective.max_pixels,
        overlap_fraction=effective.overlap_fraction,
        max_parts=effective.max_parts,
    )

    # Execute
    raw_parts = execute_split(img, plan, effective.jpeg_quality)

    # Build result
    parts: list[ImagePart] = []
    total = len(raw_parts)
    total_output = 0

    for idx, (data, w, h, sy, sh) in enumerate(raw_parts):
        total_output += len(data)
        parts.append(
            ImagePart(
                data=data,
                width=w,
                height=h,
                index=idx,
                total=total,
                source_y=sy,
                source_height=sh,
            )
        )

    return ProcessedImage(
        parts=parts,
        original_width=original_width,
        original_height=original_height,
        original_bytes=original_bytes,
        output_bytes=total_output,
        was_trimmed=plan.trim_width is not None,
        warnings=plan.warnings,
    )


def process_image_to_documents(
    image: bytes | Document,
    preset: ImagePreset = ImagePreset.GEMINI,
    config: ImageProcessingConfig | None = None,
    name_prefix: str = "image",
    sources: list[str] | None = None,
) -> list[TemporaryDocument]:
    """Process an image and return parts as ``TemporaryDocument`` list.

    @public

    Convenience wrapper around ``process_image`` for direct integration
    with ``AIMessages``.

    Args:
        image: Raw image bytes or a Document.
        preset: Model preset (ignored when *config* is provided).
        config: Custom configuration.
        name_prefix: Prefix for generated document names.
        sources: Optional provenance references attached to each document.

    Returns:
        List of ``TemporaryDocument`` instances with JPEG image data.

    Example:
        >>> docs = process_image_to_documents(screenshot_bytes)
        >>> messages = AIMessages(docs)
    """
    result = process_image(image, preset=preset, config=config)

    # Resolve sources
    doc_sources: list[str] = list(sources or [])
    if isinstance(image, Document):
        doc_sources.append(image.sha256)

    documents: list[TemporaryDocument] = []
    for part in result.parts:
        if len(result.parts) == 1:
            name = f"{name_prefix}.jpg"
            desc = None
        else:
            name = f"{name_prefix}_{part.index + 1:02d}_of_{part.total:02d}.jpg"
            desc = part.label

        documents.append(
            TemporaryDocument.create(
                name=name,
                content=part.data,
                description=desc,
                sources=doc_sources or None,
            )
        )

    return documents
