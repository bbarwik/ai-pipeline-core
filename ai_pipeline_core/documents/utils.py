"""Utility functions for document handling.

Provides helper functions for URL sanitization, naming conventions,
hash validation, and shared constants used throughout the document system.
"""

import re
from urllib.parse import urlparse

# Regex for detecting data URIs (RFC 2397): data:<mime>;base64,<payload>
DATA_URI_PATTERN = re.compile(r"^data:[a-zA-Z0-9.+/-]+;base64,")


def sanitize_url(url: str) -> str:
    """Sanitize URL or query string for use in filenames.

    Removes or replaces characters that are invalid in filenames.

    Args:
        url: The URL or query string to sanitize.

    Returns:
        A sanitized string safe for use as a filename.
    """
    # Remove protocol if it's a URL
    if url.startswith(("http://", "https://")):
        parsed = urlparse(url)
        # Use domain + path
        url = parsed.netloc + parsed.path

    # Replace invalid filename characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", url)

    # Replace multiple underscores with single one
    sanitized = re.sub(r"_+", "_", sanitized)

    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip("_.")

    # Limit length to prevent too long filenames
    if len(sanitized) > 100:
        sanitized = sanitized[:100]

    # Ensure we have something
    if not sanitized:
        sanitized = "unnamed"

    return sanitized


def camel_to_snake(name: str) -> str:
    """Convert CamelCase (incl. acronyms) to snake_case.

    Args:
        name: The CamelCase string to convert.

    Returns:
        The converted snake_case string.
    """
    s1 = re.sub(r"(.)([A-Z][a-z0-9]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.replace("__", "_").strip("_").lower()


def is_document_sha256(value: str) -> bool:
    """Check if a string is a valid base32-encoded SHA256 hash with proper entropy.

    This function validates that a string is not just formatted like a SHA256 hash,
    but actually has the entropy characteristics of a real hash. It checks:
    1. Correct length (52 characters without padding)
    2. Valid base32 characters (A-Z, 2-7)
    3. Sufficient entropy (at least 8 unique characters)

    The entropy check prevents false positives like 'AAAAAAA...AAA' from being
    identified as valid document hashes.

    Args:
        value: String to check if it's a document SHA256 hash.

    Returns:
        True if the string appears to be a real base32-encoded SHA256 hash,
        False otherwise.

    Examples:
        >>> # Real SHA256 hash
        >>> is_document_sha256("P3AEMA2PSYILKFYVBUALJLMIYWVZIS2QDI3S5VTMD2X7SOODF2YQ")
        True

        >>> # Too uniform - lacks entropy
        >>> is_document_sha256("A" * 52)
        False

        >>> # Wrong length
        >>> is_document_sha256("ABC123")
        False

        >>> # Invalid characters
        >>> is_document_sha256("a" * 52)  # lowercase
        False
    """
    # Check basic format: exactly 52 uppercase base32 characters
    try:
        if not value or len(value) != 52:
            return False
    except (TypeError, AttributeError):
        return False

    # Check if all characters are valid base32 (A-Z, 2-7)
    try:
        if not re.match(r"^[A-Z2-7]{52}$", value):
            return False
    except TypeError:
        # re.match raises TypeError for non-string types like bytes
        return False

    # Check entropy: real SHA256 hashes have high entropy
    # Require at least 8 unique characters (out of 32 possible in base32)
    # This prevents patterns like "AAAAAAA..." from being identified as real hashes
    unique_chars = len(set(value))
    return unique_chars >= 8
