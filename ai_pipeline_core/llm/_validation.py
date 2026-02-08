"""Document validation for LLM inputs.

Validates documents and attachments before sending to LLM.
All validation happens in llm/ layer, not _llm_core/.
"""

from io import BytesIO

from pypdf import PdfReader

from ai_pipeline_core._llm_core._validators import validate_image_content as validate_image  # noqa: F401  # pyright: ignore[reportUnusedImport]
from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)


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
