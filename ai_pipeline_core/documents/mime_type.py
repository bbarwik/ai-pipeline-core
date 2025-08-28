"""@internal MIME type detection utilities for documents.

This module provides functions for detecting and validating MIME types
from document content and filenames. It uses a hybrid approach combining
extension-based detection for known formats and content analysis via
python-magic for unknown files.
"""

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
    r"""Detect MIME type from document content and filename.

    Uses a multi-stage detection strategy for maximum accuracy:
    1. Returns 'application/x-empty' for empty content
    2. Uses extension-based detection for known formats (most reliable)
    3. Falls back to python-magic content analysis
    4. Final fallback to extension or 'application/octet-stream'

    Args:
        content: Document content as bytes.
        name: Filename with extension.

    Returns:
        MIME type string (e.g., 'text/plain', 'application/json').
        Never returns None or empty string.

    Fallback behavior:
        - Empty content: 'application/x-empty'
        - Unknown extension with binary content: 'application/octet-stream'
        - Magic library failure: Falls back to extension or 'application/octet-stream'

    Performance:
        Only the first 1024 bytes are analyzed for content detection.
        Extension-based detection is O(1) lookup.

    Note:
        Extension-based detection is preferred for text formats as
        content analysis can sometimes misidentify structured text.

    Example:
        >>> detect_mime_type(b'{"key": "value"}', "data.json")
        'application/json'
        >>> detect_mime_type(b'Hello World', "text.txt")
        'text/plain'
        >>> detect_mime_type(b'', "empty.txt")
        'application/x-empty'
        >>> detect_mime_type(b'\\x89PNG', "image.xyz")
        'image/png'  # Magic detects PNG despite wrong extension
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
    """Get MIME type based solely on file extension.

    Simple extension-based MIME type detection without content analysis.
    This is a legacy function maintained for backward compatibility.

    Args:
        name: Filename with extension.

    Returns:
        MIME type based on extension, or 'application/octet-stream'
        if extension is unknown.

    Note:
        Prefer detect_mime_type() for more accurate detection.
        This function only checks the file extension.

    Example:
        >>> mime_type_from_extension("document.pdf")
        'application/pdf'
        >>> mime_type_from_extension("unknown.xyz")
        'application/octet-stream'
    """
    ext = name.lower().split(".")[-1] if "." in name else ""
    return EXTENSION_MIME_MAP.get(ext, "application/octet-stream")


def is_text_mime_type(mime_type: str) -> bool:
    """Check if MIME type represents text-based content.

    Determines if content can be safely decoded as text.
    Includes common text formats and structured text like JSON/YAML.

    Args:
        mime_type: MIME type string to check.

    Returns:
        True if MIME type indicates text content, False otherwise.

    Recognized as text:
        - Any type starting with 'text/'
        - application/json
        - application/xml
        - application/javascript
        - application/yaml
        - application/x-yaml

    Example:
        >>> is_text_mime_type('text/plain')
        True
        >>> is_text_mime_type('application/json')
        True
        >>> is_text_mime_type('image/png')
        False
    """
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
    """Check if MIME type is JSON.

    Args:
        mime_type: MIME type string to check.

    Returns:
        True if MIME type is 'application/json', False otherwise.

    Note:
        Only matches exact 'application/json', not variants like
        'application/ld+json' or 'application/vnd.api+json'.

    Example:
        >>> is_json_mime_type('application/json')
        True
        >>> is_json_mime_type('text/json')  # Not standard JSON MIME
        False
    """
    return mime_type == "application/json"


def is_yaml_mime_type(mime_type: str) -> bool:
    """Check if MIME type is YAML.

    Recognizes both standard YAML MIME types.

    Args:
        mime_type: MIME type string to check.

    Returns:
        True if MIME type is YAML, False otherwise.

    Recognized types:
        - application/yaml (standard)
        - application/x-yaml (legacy)

    Example:
        >>> is_yaml_mime_type('application/yaml')
        True
        >>> is_yaml_mime_type('application/x-yaml')
        True
    """
    return mime_type == "application/yaml" or mime_type == "application/x-yaml"


def is_pdf_mime_type(mime_type: str) -> bool:
    """Check if MIME type is PDF.

    Args:
        mime_type: MIME type string to check.

    Returns:
        True if MIME type is 'application/pdf', False otherwise.

    Note:
        PDF documents require special handling in the LLM module
        and are supported by certain vision-capable models.

    Example:
        >>> is_pdf_mime_type('application/pdf')
        True
        >>> is_pdf_mime_type('text/plain')
        False
    """
    return mime_type == "application/pdf"


def is_image_mime_type(mime_type: str) -> bool:
    """Check if MIME type represents an image.

    Args:
        mime_type: MIME type string to check.

    Returns:
        True if MIME type starts with 'image/', False otherwise.

    Recognized formats:
        Any MIME type starting with 'image/' including:
        - image/png
        - image/jpeg
        - image/gif
        - image/webp
        - image/svg+xml

    Note:
        Image documents are automatically encoded for vision-capable
        LLM models in the AIMessages.document_to_prompt() method.

    Example:
        >>> is_image_mime_type('image/png')
        True
        >>> is_image_mime_type('application/pdf')
        False
    """
    return mime_type.startswith("image/")
