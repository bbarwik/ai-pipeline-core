"""Document system for AI pipeline flows.

Provides the Document base class (immutable, content-addressed), Attachment for
binary sub-documents, and RunContext/TaskDocumentContext for document lifecycle
management within pipeline tasks.
"""

from ._context import (
    DocumentSha256,
    RunContext,
    RunScope,
    TaskDocumentContext,
    get_run_context,
    reset_run_context,
    set_run_context,
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
    "TaskDocumentContext",
    "ensure_extension",
    "find_document",
    "get_run_context",
    "is_document_sha256",
    "replace_extension",
    "reset_run_context",
    "sanitize_url",
    "set_run_context",
]
