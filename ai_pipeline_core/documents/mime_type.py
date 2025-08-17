"""MIME type detection utilities for documents"""

import magic

from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)


def detect_mime_type(content: bytes, name: str) -> str:
    """Detect MIME type from content using python-magic"""

    try:
        if name.endswith(".md") and content.decode("utf-8"):
            return "text/markdown"
    except UnicodeDecodeError:
        pass

    if len(content) <= 4:
        return "application/x-empty"

    try:
        mime = magic.from_buffer(content[:1024], mime=True)
        return mime
    except (AttributeError, OSError, magic.MagicException) as e:
        logger.warning(f"MIME detection failed for {name}: {e}, falling back to extension")
        return mime_type_from_extension(name)
    except Exception as e:
        logger.error(f"Unexpected error in MIME detection for {name}: {e}")
        return mime_type_from_extension(name)


def mime_type_from_extension(name: str) -> str:
    """Get MIME type based on file extension"""
    ext = name.lower().split(".")[-1] if "." in name else ""

    mime_map = {
        "md": "text/markdown",
        "txt": "text/plain",
        "pdf": "application/pdf",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "bmp": "image/bmp",
        "webp": "image/webp",
        "json": "application/json",
        "yaml": "application/yaml",
        "yml": "application/yaml",
        "xml": "text/xml",
        "html": "text/html",
        "htm": "text/html",
    }

    return mime_map.get(ext, "application/octet-stream")


def is_text_mime_type(mime_type: str) -> bool:
    """Check if MIME type represents text content"""
    text_types = [
        "text/",
        "application/json",
        "application/xml",
        "application/javascript",
        "application/yaml",
        "application/x-yaml",
    ]
    return any(mime_type.startswith(t) for t in text_types)


def is_pdf_mime_type(mime_type: str) -> bool:
    """Check if MIME type is PDF"""
    return mime_type == "application/pdf"


def is_image_mime_type(mime_type: str) -> bool:
    """Check if MIME type is an image"""
    return mime_type.startswith("image/")
