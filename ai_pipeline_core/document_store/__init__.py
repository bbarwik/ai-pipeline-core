"""Document store protocol and backends for AI pipeline flows."""

from .protocol import DocumentReader, get_document_store

__all__ = [
    "DocumentReader",
    "get_document_store",
]
