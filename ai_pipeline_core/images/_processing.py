"""Internal image processing logic: planning, splitting, encoding."""

from dataclasses import dataclass
from io import BytesIO
from math import ceil

from PIL import Image, ImageOps

PIL_MAX_PIXELS = 100_000_000  # 100MP security limit


@dataclass(frozen=True)
class SplitPlan:
    """Describes how to split an image into parts."""

    tile_width: int
    tile_height: int
    step_y: int
    num_parts: int
    trim_width: int | None  # None = no trim needed
    warnings: list[str]


def plan_split(  # noqa: PLR0917
    width: int,
    height: int,
    max_dimension: int,
    max_pixels: int,
    overlap_fraction: float,
    max_parts: int,
) -> SplitPlan:
    """Calculate how to split an image. Pure function, no side effects.

    Returns a SplitPlan describing tile size, step, and number of parts.
    """
    warnings: list[str] = []

    # Effective tile size respecting both max_dimension and max_pixels
    tile_size = max_dimension
    while tile_size * tile_size > max_pixels and tile_size > 100:
        tile_size -= 10

    # Width: trim if needed (left-aligned, web content is left-aligned)
    trim_width = tile_size if width > tile_size else None

    effective_width = min(width, tile_size)

    # If single-tile pixel budget is still exceeded by width * tile_height, reduce tile_height
    tile_h = tile_size
    while effective_width * tile_h > max_pixels and tile_h > 100:
        tile_h -= 10

    # No vertical split needed
    if height <= tile_h:
        return SplitPlan(
            tile_width=effective_width,
            tile_height=height,
            step_y=0,
            num_parts=1,
            trim_width=trim_width,
            warnings=warnings,
        )

    # Vertical split with overlap
    overlap_px = int(tile_h * overlap_fraction)
    step = tile_h - overlap_px
    if step <= 0:
        step = 1

    num_parts = 1 + ceil((height - tile_h) / step)

    # Auto-reduce if exceeds max_parts
    if num_parts > max_parts:
        warnings.append(f"Image requires {num_parts} parts but max is {max_parts}. Reducing to {max_parts} parts with larger step.")
        num_parts = max_parts
        if num_parts > 1:
            step = (height - tile_h) // (num_parts - 1)
        else:
            step = 0

    return SplitPlan(
        tile_width=effective_width,
        tile_height=tile_h,
        step_y=step,
        num_parts=num_parts,
        trim_width=trim_width,
        warnings=warnings,
    )


def load_and_normalize(data: bytes) -> Image.Image:
    """Load image from bytes, apply EXIF orientation, validate size."""
    img = Image.open(BytesIO(data))
    img.load()

    if img.width * img.height > PIL_MAX_PIXELS:
        raise ValueError(f"Image too large: {img.width}x{img.height} = {img.width * img.height:,} pixels (limit: {PIL_MAX_PIXELS:,})")

    # Fix EXIF orientation (important for mobile photos)
    img = ImageOps.exif_transpose(img)
    return img


def encode_jpeg(img: Image.Image, quality: int) -> bytes:
    """Encode PIL Image as JPEG bytes."""
    # Convert to RGB if needed (JPEG doesn't support alpha)
    if img.mode not in {"RGB", "L"}:
        img = img.convert("RGB")

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def execute_split(
    img: Image.Image,
    plan: SplitPlan,
    jpeg_quality: int,
) -> list[tuple[bytes, int, int, int, int]]:
    """Execute a split plan on an image.

    Returns list of (data, width, height, source_y, source_height) tuples.
    """
    width, height = img.size

    # Trim width if needed (left-aligned crop)
    if plan.trim_width is not None and width > plan.trim_width:
        img = img.crop((0, 0, plan.trim_width, height))
        width = plan.trim_width

    # Convert to RGB once for JPEG
    if img.mode not in {"RGB", "L"}:
        img = img.convert("RGB")

    parts: list[tuple[bytes, int, int, int, int]] = []

    for i in range(plan.num_parts):
        if plan.num_parts == 1:
            y = 0
        else:
            y = i * plan.step_y
            # Clamp so last tile aligns to bottom
            y = min(y, max(0, height - plan.tile_height))

        h = min(plan.tile_height, height - y)
        tile = img.crop((0, y, width, y + h))

        data = encode_jpeg(tile, jpeg_quality)
        parts.append((data, width, h, y, h))

    return parts
