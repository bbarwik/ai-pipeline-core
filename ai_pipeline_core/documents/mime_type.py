"""MIME type detection utilities for documents"""

import magic

from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)

# Extension to MIME type mapping for common formats
# These are formats where extension-based detection is more reliable
EXTENSION_MIME_MAP = {
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
    "py": "text/x-python",
    "css": "text/css",
    "js": "application/javascript",
    "ts": "application/typescript",
    "tsx": "application/typescript",
    "jsx": "application/javascript",
}


def detect_mime_type(content: bytes, name: str) -> str:
    """Detect MIME type from content and filename

    Uses a hybrid approach:
    1. Check for empty content
    2. Try extension-based detection for known formats
    3. Fall back to magic content detection
    4. Final fallback to application/octet-stream
    """

    # Check for empty content
    if len(content) == 0:
        return "application/x-empty"

    # Try extension-based detection first for known formats
    # This is more reliable for text formats that magic might misidentify
    ext = name.lower().split(".")[-1] if "." in name else ""
    if ext in EXTENSION_MIME_MAP:
        return EXTENSION_MIME_MAP[ext]

    # Try content-based detection with magic
    try:
        mime = magic.from_buffer(content[:1024], mime=True)
        # If magic returns a valid mime type, use it
        if mime and mime != "application/octet-stream":
            return mime
    except (AttributeError, OSError, magic.MagicException) as e:
        logger.warning(f"MIME detection failed for {name}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in MIME detection for {name}: {e}")

    # Final fallback based on extension or default
    return EXTENSION_MIME_MAP.get(ext, "application/octet-stream")


def mime_type_from_extension(name: str) -> str:
    """Get MIME type based on file extension

    Legacy function kept for compatibility
    """
    ext = name.lower().split(".")[-1] if "." in name else ""
    return EXTENSION_MIME_MAP.get(ext, "application/octet-stream")


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


def is_json_mime_type(mime_type: str) -> bool:
    """Check if MIME type is JSON"""
    return mime_type == "application/json"


def is_yaml_mime_type(mime_type: str) -> bool:
    """Check if MIME type is YAML"""
    return mime_type == "application/yaml" or mime_type == "application/x-yaml"


def is_pdf_mime_type(mime_type: str) -> bool:
    """Check if MIME type is PDF"""
    return mime_type == "application/pdf"


def is_image_mime_type(mime_type: str) -> bool:
    """Check if MIME type is an image"""
    return mime_type.startswith("image/")
