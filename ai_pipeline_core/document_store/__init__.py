"""Document store protocol and backends for AI pipeline flows."""

from ._summary import SummaryGenerator
from .factory import create_document_store
from .protocol import DocumentStore, get_document_store, set_document_store

__all__ = [
    "DocumentStore",
    "SummaryGenerator",
    "create_document_store",
    "get_document_store",
    "set_document_store",
]
