"""Document hashing utilities for store implementations.

Computes document_sha256 and content_sha256 as defined in the document store
design: length-prefixed fields with null-byte separators, BASE32 encoded.
"""

import hashlib
from base64 import b32encode
from typing import Any, Protocol

from ai_pipeline_core.documents._context import DocumentSha256


class _Hashable(Protocol):
    """Protocol for objects whose identity hash can be computed."""

    name: str
    description: str
    content: bytes
    derived_from: tuple[str, ...]
    triggered_by: tuple[DocumentSha256, ...]
    attachments: Any


def compute_document_sha256(doc: _Hashable) -> DocumentSha256:
    """Compute the document identity hash including provenance.

    Included (identity-bearing): name, description, content, derived_from, triggered_by,
    attachments (name, description, content). Changing any of these changes the hash.

    Excluded (non-identity): summary, mime_type, class_name.
    Summary can be updated without changing document identity.

    Uses length-prefixed fields with null-byte separators for collision resistance.
    Groups (derived_from, triggered_by, attachments) are count-prefixed for unambiguous boundaries.
    Result is BASE32 encoded (uppercase, no padding), consistent with Document.sha256.
    """
    h = hashlib.sha256()

    _hash_field(h, doc.name.encode("utf-8"))
    _hash_field(h, doc.description.encode("utf-8"))
    _hash_field(h, doc.content)

    # derived_from group (sorted for order-independence)
    sorted_derived = sorted(doc.derived_from)
    _hash_field(h, str(len(sorted_derived)).encode("ascii"))
    for ref in sorted_derived:
        _hash_field(h, ref.encode("utf-8"))

    # triggered_by group (sorted for order-independence)
    sorted_triggers = sorted(doc.triggered_by)
    _hash_field(h, str(len(sorted_triggers)).encode("ascii"))
    for trigger in sorted_triggers:
        _hash_field(h, trigger.encode("utf-8"))

    # Attachments group (sorted by name, includes description)
    sorted_atts = sorted(doc.attachments, key=lambda a: a.name)
    _hash_field(h, str(len(sorted_atts)).encode("ascii"))
    for att in sorted_atts:
        _hash_field(h, att.name.encode("utf-8"))
        _hash_field(h, att.description.encode("utf-8"))
        _hash_field(h, att.content)

    return DocumentSha256(b32encode(h.digest()).decode("ascii").upper().rstrip("=")[:26])


def compute_content_sha256(content: bytes) -> str:
    """Compute SHA256 of raw content bytes, BASE32 encoded."""
    return b32encode(hashlib.sha256(content).digest()).decode("ascii").upper().rstrip("=")[:26]


def _hash_field(h: Any, data: bytes) -> None:
    """Append a length-prefixed, null-separated field to the hash."""
    h.update(str(len(data)).encode("ascii"))
    h.update(b"\x00")
    h.update(data)
