"""Content validation for LLM inputs.

Validates image, PDF, and text content before sending to LLM.
Single source of truth for all content validation across _llm_core and llm layers.
"""

from io import BytesIO

from PIL import Image
from pypdf import PdfReader

from ai_pipeline_core.logger import get_pipeline_logger

logger = get_pipeline_logger(__name__)


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


def validate_pdf(data: bytes, name: str) -> str | None:
    """Validate PDF with page count check. Returns error message or None if valid."""
    if not data:
        return f"empty PDF content in '{name}'"
    if not data.lstrip().startswith(b"%PDF-"):
        return f"invalid PDF header in '{name}'"
    try:
        reader = PdfReader(BytesIO(data))
        if len(reader.pages) == 0:
            return f"PDF has no pages in '{name}'"
    except Exception as e:
        return f"corrupted PDF in '{name}': {e}"
    return None


def validate_text(data: bytes, name: str) -> str | None:
    """Validate text content. Returns error message or None if valid."""
    if not data:
        return f"empty text content in '{name}'"
    if b"\x00" in data:
        return f"binary content (null bytes) in text '{name}'"
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as e:
        return f"invalid UTF-8 encoding in '{name}': {e}"
    return None
