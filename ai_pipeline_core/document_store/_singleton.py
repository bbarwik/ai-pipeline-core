"""@internal Process-global document store singleton.

Separated to avoid circular imports between protocol.py and _protocol.py.
"""

from typing import Any

__all__ = [
    "get_store",
    "set_store",
]

_document_store: Any = None


def get_store() -> Any:
    """Get the process-global document store singleton (untyped to avoid circular imports)."""
    return _document_store


def set_store(store: Any) -> None:
    """Set the process-global document store singleton."""
    global _document_store
    _document_store = store
