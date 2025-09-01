"""Utility functions for document handling.

Provides helper functions for URL sanitization, naming conventions,
canonical key generation, and hash validation used throughout the document system.
"""

import re
from typing import Any, Iterable, Type
from urllib.parse import urlparse


def sanitize_url(url: str) -> str:
    """Sanitize URL or query string for use in filenames.

    @public

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


def canonical_name_key(
    obj_or_name: Type[Any] | str,
    *,
    max_parent_suffixes: int = 3,
    extra_suffixes: Iterable[str] = (),
) -> str:
    """Produce a canonical snake_case key from a class or name.

    @public

    Process:
      1) Starting with the class name (or given string),
      2) Stripping any trailing parent class names (up to `max_parent_suffixes` from the MRO),
      3) Stripping any `extra_suffixes`,
      4) Converting to snake_case.

    Args:
        obj_or_name: A class or string to convert.
        max_parent_suffixes: Maximum number of parent classes to consider for suffix removal.
        extra_suffixes: Additional suffixes to strip.

    Returns:
        The canonical snake_case name.

    Examples:
        FinalReportDocument(WorkflowDocument -> Document) -> 'final_report'
        FooWorkflowDocument(WorkflowDocument -> Document) -> 'foo'
        BarFlow(Config -> Base -> Flow) -> 'bar'
    """
    name = obj_or_name.__name__ if isinstance(obj_or_name, type) else str(obj_or_name)

    # From MRO, collect up to N parent names to consider as removable suffixes
    suffixes: list[str] = []
    if isinstance(obj_or_name, type):
        for base in obj_or_name.mro()[1 : 1 + max_parent_suffixes]:
            if base is object:
                continue
            suffixes.append(base.__name__)

    # Add any custom suffixes the caller wants to strip (e.g., 'Config')
    suffixes.extend(extra_suffixes)

    # Iteratively trim the longest matching suffix first
    trimmed = True
    while trimmed and suffixes:
        trimmed = False
        for sfx in sorted(set(suffixes), key=len, reverse=True):
            if sfx and name.endswith(sfx):
                name = name[: -len(sfx)]
                trimmed = True
                break

    return camel_to_snake(name)


def is_document_sha256(value: str) -> bool:
    """Check if a string is a valid base32-encoded SHA256 hash with proper entropy.

    @public

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
    if unique_chars < 8:
        return False

    return True
