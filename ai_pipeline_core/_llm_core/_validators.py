"""Shared content validation helpers used by both _llm_core and llm layers."""

from io import BytesIO

from PIL import Image


def validate_image_content(data: bytes, name: str = "image") -> str | None:
    """Validate image content via PIL. Returns error message or None if valid."""
    if not data:
        return f"empty image content in '{name}'"
    try:
        with Image.open(BytesIO(data)) as img:
            img.verify()
        return None
    except (OSError, ValueError, Image.DecompressionBombError) as e:
        return f"invalid image in '{name}': {e}"
