"""Document system for AI pipeline flows.

Provides the Document base class (immutable, content-addressed), Attachment for
binary sub-documents, and RunContext/TaskDocumentContext for document lifecycle
management within pipeline tasks.
"""

from .attachment import Attachment
from .context import RunContext, TaskDocumentContext, get_run_context, reset_run_context, set_run_context
from .document import Document
from .utils import canonical_name_key, is_document_sha256, sanitize_url

__all__ = [
    "Attachment",
    "Document",
    "RunContext",
    "TaskDocumentContext",
    "canonical_name_key",
    "get_run_context",
    "is_document_sha256",
    "reset_run_context",
    "sanitize_url",
    "set_run_context",
]
