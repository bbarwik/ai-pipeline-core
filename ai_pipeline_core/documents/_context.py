"""Domain types for the document system."""

from typing import NewType

__all__ = [
    "DocumentSha256",
]

DocumentSha256 = NewType("DocumentSha256", str)
"""BASE32-encoded SHA256 identity hash of a Document (name + content + derived_from + triggered_by + attachments)."""
