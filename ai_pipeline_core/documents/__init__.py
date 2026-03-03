"""Document system for AI pipeline flows.

Provides the Document base class (immutable, content-addressed), Attachment for
binary sub-documents, and RunContext for pipeline run context.
"""

from ._context import (
    DocumentSha256,
    RunContext,
    RunScope,
)
from .attachment import Attachment
from .document import Document
from .utils import ensure_extension, find_document, is_document_sha256, replace_extension, sanitize_url

__all__ = [
    "Attachment",
    "Document",
    "DocumentSha256",
    "RunContext",
    "RunScope",
    "ensure_extension",
    "find_document",
    "is_document_sha256",
    "replace_extension",
    "sanitize_url",
]
