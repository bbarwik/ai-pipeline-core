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
    attachments: Any


def compute_document_sha256(doc: _Hashable) -> str:
    """Compute the document identity hash: hash(name + content + sorted_attachments).

    Uses length-prefixed fields with null-byte separators for collision resistance.
    Attachments are sorted by name. Result is BASE32 encoded (uppercase, no padding),
    consistent with Document.sha256.

    Excluded from hash: description, sources, origins, class_name.
    """
    h = hashlib.sha256()

    name_bytes = doc.name.encode("utf-8")
    _hash_field(h, name_bytes)
    _hash_field(h, doc.content)

    for att in sorted(doc.attachments, key=lambda a: a.name):
        att_name_bytes = att.name.encode("utf-8")
        _hash_field(h, att_name_bytes)
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
