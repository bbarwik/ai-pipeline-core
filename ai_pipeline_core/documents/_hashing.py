"""Document hashing utilities for store implementations.

Computes document_sha256 and content_sha256 as defined in the document store
design: length-prefixed fields with null-byte separators, BASE32 encoded.
"""

import hashlib
from base64 import b32encode
from typing import Any, Protocol


class _Hashable(Protocol):
    """Protocol for objects whose identity hash can be computed."""

    name: str
    content: bytes
    sources: tuple[str, ...]
    origins: tuple[str, ...]
    attachments: Any


def compute_document_sha256(doc: _Hashable) -> str:
    """Compute the document identity hash including provenance.

    Fields included: name, content, sources, origins, attachments.
    Uses length-prefixed fields with null-byte separators for collision resistance.
    Groups (sources, origins, attachments) are count-prefixed for unambiguous boundaries.
    Result is BASE32 encoded (uppercase, no padding), consistent with Document.sha256.

    Excluded from hash: description, class_name.
    """
    h = hashlib.sha256()

    _hash_field(h, doc.name.encode("utf-8"))
    _hash_field(h, doc.content)

    # Sources group (sorted for order-independence)
    sorted_sources = sorted(doc.sources)
    _hash_field(h, str(len(sorted_sources)).encode("ascii"))
    for source in sorted_sources:
        _hash_field(h, source.encode("utf-8"))

    # Origins group (sorted for order-independence)
    sorted_origins = sorted(doc.origins)
    _hash_field(h, str(len(sorted_origins)).encode("ascii"))
    for origin in sorted_origins:
        _hash_field(h, origin.encode("utf-8"))

    # Attachments group (sorted by name)
    sorted_atts = sorted(doc.attachments, key=lambda a: a.name)
    _hash_field(h, str(len(sorted_atts)).encode("ascii"))
    for att in sorted_atts:
        _hash_field(h, att.name.encode("utf-8"))
        _hash_field(h, att.content)

    return b32encode(h.digest()).decode("ascii").upper().rstrip("=")


def compute_content_sha256(content: bytes) -> str:
    """Compute SHA256 of raw content bytes, BASE32 encoded."""
    return b32encode(hashlib.sha256(content).digest()).decode("ascii").upper().rstrip("=")


def _hash_field(h: Any, data: bytes) -> None:
    """Append a length-prefixed, null-separated field to the hash."""
    h.update(str(len(data)).encode("ascii"))
    h.update(b"\x00")
    h.update(data)
