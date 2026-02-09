"""Domain-specific types for the document system."""

from typing import NewType

DocumentSha256 = NewType("DocumentSha256", str)
"""BASE32-encoded SHA256 identity hash of a Document (name + content + sources + origins + attachments)."""

RunScope = NewType("RunScope", str)
"""Scoping identifier for a pipeline run, used to partition documents in the store."""

__all__ = ["DocumentSha256", "RunScope"]
