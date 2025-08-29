"""Document abstraction system for AI pipeline flows.

@public

The documents package provides immutable, type-safe data structures for handling
various content types in AI pipelines, including text, images, PDFs, and other
binary data with automatic MIME type detection.
"""

from .document import ContentInput, Document
from .document_list import DocumentList
from .flow_document import FlowDocument
from .task_document import TaskDocument
from .temporary_document import TemporaryDocument
from .utils import canonical_name_key, sanitize_url

__all__ = [
    "ContentInput",
    "Document",
    "DocumentList",
    "FlowDocument",
    "TaskDocument",
    "TemporaryDocument",
    "canonical_name_key",
    "sanitize_url",
]
