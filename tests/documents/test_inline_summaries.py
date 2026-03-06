"""Tests for inline summary registry functions."""

from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents._context import _suppress_document_registration
from ai_pipeline_core.documents.document import get_inline_summary, pop_inline_summary, set_inline_summary


class _SummaryDoc(Document):
    """Test document for inline summary tests."""


def test_set_and_get_inline_summary() -> None:
    set_inline_summary("sha256_abc", "A summary")
    assert get_inline_summary("sha256_abc") == "A summary"
    pop_inline_summary("sha256_abc")  # cleanup


def test_pop_inline_summary_removes_it() -> None:
    set_inline_summary("sha256_pop", "Will be popped")
    result = pop_inline_summary("sha256_pop")
    assert result == "Will be popped"
    assert get_inline_summary("sha256_pop") is None


def test_get_nonexistent_returns_none() -> None:
    assert get_inline_summary("sha256_nonexistent") is None


def test_pop_nonexistent_returns_none() -> None:
    assert pop_inline_summary("sha256_nonexistent") is None


def test_set_overwrites_existing() -> None:
    set_inline_summary("sha256_overwrite", "first")
    set_inline_summary("sha256_overwrite", "second")
    assert get_inline_summary("sha256_overwrite") == "second"
    pop_inline_summary("sha256_overwrite")  # cleanup


def test_create_with_summary_stores_inline() -> None:
    source = _SummaryDoc.create_root(name="source.txt", content=b"source", reason="test input")
    with _suppress_document_registration():
        doc = _SummaryDoc.create(
            name="test.txt",
            content="hello",
            derived_from=(source.sha256,),
            summary="Auto-generated summary",
        )
    assert get_inline_summary(doc.sha256) == "Auto-generated summary"
    pop_inline_summary(doc.sha256)  # cleanup


def test_create_without_summary_does_not_store_inline() -> None:
    source = _SummaryDoc.create_root(name="source.txt", content=b"source", reason="test input")
    with _suppress_document_registration():
        doc = _SummaryDoc.create(
            name="test.txt",
            content="hello",
            derived_from=(source.sha256,),
        )
    assert get_inline_summary(doc.sha256) is None
