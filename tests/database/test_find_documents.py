"""Tests for find_latest_documents_by_derived_from and store_document."""

from datetime import UTC, datetime, timedelta
from typing import ClassVar
from uuid import uuid4

import pytest

from ai_pipeline_core.database import DocumentRecord, MemoryDatabase
from ai_pipeline_core.database._documents import store_document
from ai_pipeline_core.documents import Document


class StoreTestDoc(Document):
    """Document for store_document test."""

    publicly_visible: ClassVar[bool] = True


def _make_document(**kwargs: object) -> DocumentRecord:
    defaults: dict[str, object] = {
        "document_sha256": f"doc-{uuid4().hex}",
        "content_sha256": f"blob-{uuid4().hex}",
        "document_type": "TestDocument",
        "name": "test.md",
        "mime_type": "text/markdown",
        "size_bytes": 10,
        "derived_from": (),
        "triggered_by": (),
        "created_at": datetime(2026, 3, 14, 12, 0, tzinfo=UTC),
    }
    defaults.update(kwargs)
    return DocumentRecord(**defaults)


@pytest.mark.asyncio
async def test_find_empty_values_returns_empty() -> None:
    db = MemoryDatabase()
    result = await db.find_latest_documents_by_derived_from([])
    assert result == {}


@pytest.mark.asyncio
async def test_find_no_matches_returns_empty() -> None:
    db = MemoryDatabase()
    await db.save_document(_make_document(derived_from=("https://a.com",)))
    result = await db.find_latest_documents_by_derived_from(["https://b.com"])
    assert result == {}


@pytest.mark.asyncio
async def test_find_single_match() -> None:
    db = MemoryDatabase()
    doc = _make_document(derived_from=("https://a.com",))
    await db.save_document(doc)
    result = await db.find_latest_documents_by_derived_from(["https://a.com"])
    assert "https://a.com" in result
    assert result["https://a.com"].document_sha256 == doc.document_sha256


@pytest.mark.asyncio
async def test_find_returns_newest() -> None:
    db = MemoryDatabase()
    old = _make_document(derived_from=("https://a.com",), created_at=datetime(2026, 1, 1, tzinfo=UTC))
    new = _make_document(derived_from=("https://a.com",), created_at=datetime(2026, 3, 1, tzinfo=UTC))
    await db.save_document(old)
    await db.save_document(new)
    result = await db.find_latest_documents_by_derived_from(["https://a.com"])
    assert result["https://a.com"].document_sha256 == new.document_sha256


@pytest.mark.asyncio
async def test_find_filters_by_document_type() -> None:
    db = MemoryDatabase()
    doc_a = _make_document(document_type="TypeA", derived_from=("https://a.com",))
    doc_b = _make_document(document_type="TypeB", derived_from=("https://a.com",))
    await db.save_document(doc_a)
    await db.save_document(doc_b)

    result = await db.find_latest_documents_by_derived_from(["https://a.com"], document_type="TypeA")
    assert result["https://a.com"].document_type == "TypeA"


@pytest.mark.asyncio
async def test_find_filters_by_max_age() -> None:
    db = MemoryDatabase()
    old = _make_document(derived_from=("https://a.com",), created_at=datetime(2020, 1, 1, tzinfo=UTC))
    await db.save_document(old)

    result = await db.find_latest_documents_by_derived_from(["https://a.com"], max_age=timedelta(days=30))
    assert result == {}


@pytest.mark.asyncio
async def test_find_multiple_values() -> None:
    db = MemoryDatabase()
    doc_a = _make_document(derived_from=("https://a.com",))
    doc_b = _make_document(derived_from=("https://b.com",))
    await db.save_document(doc_a)
    await db.save_document(doc_b)

    result = await db.find_latest_documents_by_derived_from(["https://a.com", "https://b.com", "https://c.com"])
    assert len(result) == 2
    assert "https://a.com" in result
    assert "https://b.com" in result
    assert "https://c.com" not in result


@pytest.mark.asyncio
async def test_publicly_visible_field_on_document_record() -> None:
    doc = _make_document(publicly_visible=True)
    assert doc.publicly_visible is True

    doc_default = _make_document()
    assert doc_default.publicly_visible is False


@pytest.mark.asyncio
async def test_store_document_roundtrip() -> None:
    db = MemoryDatabase()
    doc = StoreTestDoc.create_root(reason="test", name="test.md", content="hello")
    await store_document(db, doc)

    record = await db.get_document(doc.sha256)
    assert record is not None
    assert record.document_type == "StoreTestDoc"
    assert record.publicly_visible is True

    blob = await db.get_blob(record.content_sha256)
    assert blob is not None
    assert blob.content == b"hello"
