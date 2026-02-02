"""Tests for DocumentStore protocol and singleton management."""

import pytest

from ai_pipeline_core.document_store import (
    DocumentStore,
    get_document_store,
    set_document_store,
)
from ai_pipeline_core.documents.document import Document


@pytest.fixture(autouse=True)
def _reset_store():
    """Reset the document store singleton after each test."""
    yield
    set_document_store(None)


class _DummyStore:
    """Minimal class implementing the DocumentStore protocol."""

    async def save(self, document: Document, run_scope: str) -> None:
        pass

    async def save_batch(self, documents: list[Document], run_scope: str) -> None:
        pass

    async def load(self, run_scope: str, document_types: list[type[Document]]) -> list[Document]:
        return []

    async def has_documents(self, run_scope: str, document_type: type[Document]) -> bool:
        return False

    async def check_existing(self, sha256s: list[str]) -> set[str]:
        return set()

    async def update_summary(self, run_scope: str, document_sha256: str, summary: str) -> None:
        pass

    async def load_summaries(self, run_scope: str, document_sha256s: list[str]) -> dict[str, str]:
        return {}

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class _IncompleteStore:
    """Class missing required protocol methods."""

    async def save(self, document: Document, run_scope: str) -> None:
        pass


def test_document_store_is_runtime_checkable_protocol():
    """A class implementing all methods satisfies the protocol isinstance check."""
    store = _DummyStore()
    assert isinstance(store, DocumentStore)


def test_incomplete_class_fails_isinstance():
    """A class missing methods does not satisfy the protocol."""
    store = _IncompleteStore()
    assert not isinstance(store, DocumentStore)


def test_get_document_store_returns_none_by_default():
    """Before any set call, the store is None."""
    assert get_document_store() is None


def test_set_and_get_document_store():
    """Setting a store makes it retrievable."""
    store = _DummyStore()
    set_document_store(store)
    assert get_document_store() is store


def test_set_document_store_to_none():
    """Store can be reset to None."""
    store = _DummyStore()
    set_document_store(store)
    assert get_document_store() is store
    set_document_store(None)
    assert get_document_store() is None
