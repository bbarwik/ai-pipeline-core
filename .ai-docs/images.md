# MODULE: images
# CLASSES: ImageDocument, ImagePreset, ImageProcessingConfig, ImagePart, ProcessedImage, ImageProcessingError
# DEPENDS: BaseModel, Exception, StrEnum
# SIZE: ~9KB

# === DEPENDENCIES (Resolved) ===

class BaseModel:
    """Pydantic base model. Fields are typed class attributes."""
    ...

class Exception:
    """External base class (not fully documented)."""
    ...

class StrEnum:
    """String enumeration base class."""
    ...

# === PUBLIC API ===

class ImageDocument(Document):
    """Concrete document for processed image parts."""
    # [Inherited from Document]
    # __init__, __init_subclass__, approximate_tokens_count, as_json, as_pydantic_model, as_yaml, canonical_name, content_sha256, create, from_dict, get_expected_files, has_source, id, is_image, is_pdf, is_text, mime_type, model_convert, parse, serialize_content, serialize_model, sha256, size, source_documents, source_references, text, validate_content, validate_file_name, validate_name, validate_no_source_origin_overlap, validate_origins, validate_sources, validate_total_size


class ImagePreset(StrEnum):
    """Presets for LLM vision model constraints."""
    GEMINI = 'gemini'
    CLAUDE = 'claude'
    GPT4V = 'gpt4v'


class ImageProcessingConfig(BaseModel):
    """Configuration for image processing.

Use ``for_preset`` for standard configurations or construct directly for
custom constraints."""
    model_config = {'frozen': True}
    max_dimension: int = Field(default=3000, ge=100, le=8192, description='Maximum width AND height in pixels')
    max_pixels: int = Field(default=9000000, ge=10000, description='Maximum total pixels per output image part')
    overlap_fraction: float = Field(default=0.2, ge=0.0, le=0.5, description='Overlap between adjacent vertical parts (0.0-0.5)')
    max_parts: int = Field(default=20, ge=1, le=100, description='Maximum number of output image parts')
    jpeg_quality: int = Field(default=60, ge=10, le=95, description='JPEG compression quality (10-95)')

    @classmethod
    def for_preset(cls, preset: ImagePreset) -> "ImageProcessingConfig":
        """Create configuration from a model preset."""
        return _PRESETS[preset]


class ImagePart(BaseModel):
    """A single processed image part."""
    model_config = {'frozen': True}
    data: bytes = Field(repr=False)
    width: int
    height: int
    index: int = Field(ge=0, description='0-indexed position')
    total: int = Field(ge=1, description='Total number of parts')
    source_y: int = Field(ge=0, description='Y offset in original image')
    source_height: int = Field(ge=1, description='Height of region in original')

    @property
    def label(self) -> str:
        """Human-readable label for LLM context, 1-indexed."""
        if self.total == 1:
            return "Full image"
        return f"Part {self.index + 1}/{self.total}"


class ProcessedImage(BaseModel):
    """Result of image processing.

Iterable: ``for part in result`` iterates over parts."""
    model_config = {'frozen': True}
    parts: list[ImagePart]
    original_width: int
    original_height: int
    original_bytes: int
    output_bytes: int
    was_trimmed: bool = Field(description='True if width was trimmed to fit')
    warnings: list[str] = Field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        """Output size / input size (lower means more compression)."""
        if self.original_bytes <= 0:
            return 1.0
        return self.output_bytes / self.original_bytes

    def __getitem__(self, idx: int) -> ImagePart:
        return self.parts[idx]

    def __iter__(self):  # type: ignore[override]
        return iter(self.parts)

    def __len__(self) -> int:
        return len(self.parts)


class ImageProcessingError(Exception):
    """Image processing failed."""

# === FUNCTIONS ===

def process_image(  # noqa: RUF067
    image: bytes | Document,
    preset: ImagePreset = ImagePreset.GEMINI,
    config: ImageProcessingConfig | None = None,
) -> ProcessedImage:
    """Process an image for LLM vision models.

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

    """
    effective = config if config is not None else ImageProcessingConfig.for_preset(preset)

    # Resolve input bytes
    raw: bytes
    if isinstance(image, Document):
        raw = image.content
    elif isinstance(image, bytes):  # type: ignore[reportUnnecessaryIsInstance]
        raw = image
    else:
        raise ImageProcessingError(f"Unsupported image input type: {type(image)}")  # pyright: ignore[reportUnreachable]

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

def process_image_to_documents(  # noqa: RUF067
    image: bytes | Document,
    preset: ImagePreset = ImagePreset.GEMINI,
    config: ImageProcessingConfig | None = None,
    name_prefix: str = "image",
    sources: tuple[str, ...] | None = None,
) -> list[ImageDocument]:
    """Process an image and return parts as ImageDocument list.

    Convenience wrapper around ``process_image`` for direct integration
    with ``AIMessages``.
    """
    result = process_image(image, preset=preset, config=config)

    source_list: list[str] = list(sources or ())
    if isinstance(image, Document):
        source_list.append(image.sha256)
    doc_sources = tuple(source_list) if source_list else None

    documents: list[ImageDocument] = []
    for part in result.parts:
        if len(result.parts) == 1:
            name = f"{name_prefix}.jpg"
            desc = None
        else:
            name = f"{name_prefix}_{part.index + 1:02d}_of_{part.total:02d}.jpg"
            desc = part.label

        documents.append(
            ImageDocument.create(
                name=name,
                content=part.data,
                description=desc,
                sources=doc_sources,
            )
        )

    return documents

# === EXAMPLES (from tests/) ===

# Example: Preset forwarded
# Source: tests/images/test_images.py:310
def test_preset_forwarded(self):
    docs_gemini = process_image_to_documents(make_image_bytes(1920, 1080), preset=ImagePreset.GEMINI)
    docs_claude = process_image_to_documents(make_image_bytes(1920, 1080), preset=ImagePreset.CLAUDE)
    # Claude trims 1920px to 1568px, Gemini doesn't
    gemini_img = Image.open(BytesIO(docs_gemini[0].content))
    claude_img = Image.open(BytesIO(docs_claude[0].content))
    assert gemini_img.size[0] == 1920
    assert claude_img.size[0] <= 1568

# Example: Accepts bytes
# Source: tests/images/test_images.py:240
def test_accepts_bytes(self):
    result = process_image(make_image_bytes(800, 600))
    assert len(result) == 1

# Example: Accepts document
# Source: tests/images/test_images.py:234
def test_accepts_document(self):
    doc = make_document(800, 600)
    result = process_image(doc)
    assert len(result) == 1
    assert result.original_width == 800

# Example: All parts valid jpeg
# Source: tests/images/test_images.py:162
def test_all_parts_valid_jpeg(self):
    result = process_image(make_image_bytes(1000, 7000))
    for part in result:
        img = Image.open(BytesIO(part.data))
        assert img.format == "JPEG"
        assert img.size[0] == part.width
        assert img.size[1] == part.height

# === ERROR EXAMPLES (What NOT to Do) ===

# Error: Config is frozen
# Source: tests/images/test_images.py:77
def test_config_is_frozen(self):
    config = ImageProcessingConfig.for_preset(ImagePreset.GEMINI)
    with pytest.raises(Exception):
        config.max_dimension = 1000  # type: ignore[misc]

# Error: Empty bytes
# Source: tests/images/test_images.py:253
def test_empty_bytes(self):
    with pytest.raises(ImageProcessingError, match="Empty image data"):
        process_image(b"")

# Error: Image part frozen
# Source: tests/images/test_images.py:333
def test_image_part_frozen(self):
    result = process_image(make_image_bytes(800, 600))
    with pytest.raises(Exception):
        result[0].width = 999  # type: ignore[misc]

# Error: Invalid bytes
# Source: tests/images/test_images.py:257
def test_invalid_bytes(self):
    with pytest.raises(ImageProcessingError, match="Failed to decode"):
        process_image(b"not an image at all")
