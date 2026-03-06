"""Tests for RunContext and _TaskDocumentContext."""

import pytest

from ai_pipeline_core.documents import Document, RunScope
from ai_pipeline_core.documents._context import (
    RunContext,
    _TaskDocumentContext,
    _get_run_context,
    _suppress_document_registration,
    _set_run_context,
)


# --- Concrete document subclass for testing ---


class SampleDoc(Document):
    pass


def _make_doc(
    name: str,
    content: str = "test",
    derived_from: tuple[str, ...] | None = None,
    triggered_by: tuple[str, ...] | None = None,
) -> SampleDoc:
    with _suppress_document_registration():
        if derived_from or triggered_by:
            return SampleDoc.create(name=name, content=content, derived_from=derived_from, triggered_by=triggered_by)
        return SampleDoc.create_root(name=name, content=content, reason="test fixture")


# ===== RunContext tests =====


class TestRunContext:
    def test_creation(self):
        ctx = RunContext(run_scope=RunScope("project/flow/run123"))
        assert ctx.run_scope == "project/flow/run123"

    def test_frozen(self):
        ctx = RunContext(run_scope=RunScope("test"))
        with pytest.raises(AttributeError):
            ctx.run_scope = "changed"  # type: ignore[misc]

    def test_get_returns_none_by_default(self):
        assert _get_run_context() is None

    def test_set_and_get(self):
        ctx = RunContext(run_scope=RunScope("my-run"))
        token = _set_run_context(ctx)
        try:
            assert _get_run_context() is ctx
        finally:
            from ai_pipeline_core.documents._context import _run_context

            _run_context.reset(token)

    def test_token_restores_previous(self):
        ctx1 = RunContext(run_scope=RunScope("first"))
        ctx2 = RunContext(run_scope=RunScope("second"))
        token1 = _set_run_context(ctx1)
        token2 = _set_run_context(ctx2)
        assert _get_run_context() is ctx2

        from ai_pipeline_core.documents._context import _run_context

        _run_context.reset(token2)
        assert _get_run_context() is ctx1
        _run_context.reset(token1)
        assert _get_run_context() is None

    def test_execution_id_defaults_to_none(self):
        ctx = RunContext(run_scope=RunScope("test"))
        assert ctx.execution_id is None

    def test_execution_id_stored(self):
        from uuid import uuid4

        uid = uuid4()
        ctx = RunContext(run_scope=RunScope("test"), execution_id=uid)
        assert ctx.execution_id == uid


# ===== _TaskDocumentContext.register_created =====


class TestRegistration:
    def test_register_created(self):
        ctx = _TaskDocumentContext()
        doc = _make_doc("a.txt")
        ctx.register_created(doc)
        assert doc.sha256 in ctx.created

    def test_register_multiple(self):
        ctx = _TaskDocumentContext()
        doc_a = _make_doc("a.txt", "aaa")
        doc_b = _make_doc("b.txt", "bbb")
        ctx.register_created(doc_a)
        ctx.register_created(doc_b)
        assert len(ctx.created) == 2


# ===== validate_provenance =====


class TestValidateProvenance:
    def test_derived_from_in_existing_set(self):
        """derived_from reference exists in the store — no warning."""
        parent = _make_doc("parent.txt", "parent")
        doc = _make_doc("a.txt", "aaa", derived_from=(parent.sha256,))
        ctx = _TaskDocumentContext()
        warnings = ctx.validate_provenance([doc], existing_sha256s={parent.sha256})
        assert warnings == []

    def test_missing_derived_from(self):
        """Document references a SHA256 that doesn't exist in the store."""
        phantom = _make_doc("phantom.txt", "ghost")
        doc = _make_doc("a.txt", "aaa", derived_from=(phantom.sha256,))
        ctx = _TaskDocumentContext()
        warnings = ctx.validate_provenance([doc], existing_sha256s=set())
        assert len(warnings) == 1
        assert "does not exist" in warnings[0]

    def test_accepts_url_derived_from(self):
        """URLs in derived_from are accepted and not validated as SHA256."""
        doc = _make_doc("a.txt", "aaa", derived_from=("https://example.com",))
        ctx = _TaskDocumentContext()
        warnings = ctx.validate_provenance([doc], existing_sha256s=set())
        assert all("does not exist" not in w for w in warnings)

    def test_rejects_invalid_derived_from_strings(self):
        """Non-SHA256, non-URL strings are rejected at document creation."""
        with pytest.raises(Exception):
            _make_doc("a.txt", "aaa", derived_from=("not-a-hash",))
        with pytest.raises(Exception):
            _make_doc("a.txt", "aaa", derived_from=("short",))

    def test_missing_triggered_by(self):
        """Document references a triggered_by SHA256 that doesn't exist."""
        phantom = _make_doc("phantom.txt", "ghost")
        doc = _make_doc("a.txt", "aaa", triggered_by=(phantom.sha256,))
        ctx = _TaskDocumentContext()
        warnings = ctx.validate_provenance([doc], existing_sha256s=set())
        assert len(warnings) == 1
        assert "triggered_by" in warnings[0]
        assert "does not exist" in warnings[0]

    def test_triggered_by_in_existing_set(self):
        """triggered_by reference exists in the store — no warning."""
        parent = _make_doc("parent.txt", "parent")
        doc = _make_doc("a.txt", "aaa", triggered_by=(parent.sha256,))
        ctx = _TaskDocumentContext()
        warnings = ctx.validate_provenance([doc], existing_sha256s={parent.sha256})
        assert warnings == []

    def test_same_task_derived_from_interdep(self):
        """derived_from SHA256 created in the same task produces a warning."""
        doc_a = _make_doc("a.txt", "aaa")
        doc_b = _make_doc("b.txt", "bbb", derived_from=(doc_a.sha256,))
        ctx = _TaskDocumentContext()
        ctx.register_created(doc_a)
        ctx.register_created(doc_b)
        warnings = ctx.validate_provenance([doc_b], existing_sha256s=set())
        assert len(warnings) == 1
        assert "same task" in warnings[0]

    def test_same_task_triggered_by_interdep(self):
        """triggered_by SHA256 created in the same task produces a warning."""
        doc_a = _make_doc("a.txt", "aaa")
        doc_b = _make_doc("b.txt", "bbb", triggered_by=(doc_a.sha256,))
        ctx = _TaskDocumentContext()
        ctx.register_created(doc_a)
        ctx.register_created(doc_b)
        warnings = ctx.validate_provenance([doc_b], existing_sha256s=set())
        assert len(warnings) == 1
        assert "same task" in warnings[0]
        assert "triggered_by" in warnings[0]

    def test_no_provenance_warning(self):
        """Document with no derived_from and no triggered_by gets a warning."""
        doc = _make_doc("a.txt", "aaa")
        ctx = _TaskDocumentContext()
        warnings = ctx.validate_provenance([doc], existing_sha256s=set())
        assert len(warnings) == 1
        assert "no provenance" in warnings[0]

    def test_url_derived_from_no_provenance_warning(self):
        """Document with URL derived_from has provenance — no warning."""
        doc = _make_doc("a.txt", "aaa", derived_from=("https://example.com",))
        ctx = _TaskDocumentContext()
        warnings = ctx.validate_provenance([doc], existing_sha256s=set())
        assert warnings == []

    def test_mixed_valid_and_invalid(self):
        """Multiple documents with different provenance issues."""
        parent = _make_doc("parent.txt", "parent")
        valid = _make_doc("valid.txt", "valid", derived_from=(parent.sha256,))
        orphan = _make_doc("orphan.txt", "orphan")  # no provenance
        ctx = _TaskDocumentContext()
        warnings = ctx.validate_provenance([valid, orphan], existing_sha256s={parent.sha256})
        assert len(warnings) == 1
        assert "no provenance" in warnings[0]


# ===== deduplicate =====


class TestDeduplicate:
    def test_empty_list(self):
        assert _TaskDocumentContext.deduplicate([]) == []

    def test_no_duplicates(self):
        docs = [_make_doc("a.txt", "aaa"), _make_doc("b.txt", "bbb")]
        result = _TaskDocumentContext.deduplicate(docs)
        assert len(result) == 2

    def test_removes_duplicates(self):
        doc = _make_doc("a.txt", "aaa")
        result = _TaskDocumentContext.deduplicate([doc, doc])
        assert len(result) == 1
        assert result[0].name == "a.txt"

    def test_preserves_first_occurrence_order(self):
        doc_a = _make_doc("a.txt", "aaa")
        doc_b = _make_doc("b.txt", "bbb")
        doc_c = _make_doc("c.txt", "ccc")
        result = _TaskDocumentContext.deduplicate([doc_c, doc_a, doc_b, doc_a, doc_c])
        assert [d.name for d in result] == ["c.txt", "a.txt", "b.txt"]

    def test_same_content_different_names(self):
        """Documents with identical content but different names have the same SHA256."""
        doc1 = _make_doc("first.txt", "same content")
        doc2 = _make_doc("first.txt", "same content")
        result = _TaskDocumentContext.deduplicate([doc1, doc2])
        assert len(result) == 1


# ===== finalize (orphan detection) =====


class TestFinalize:
    def test_no_orphans_when_all_returned(self):
        ctx = _TaskDocumentContext()
        doc_a = _make_doc("a.txt", "aaa")
        doc_b = _make_doc("b.txt", "bbb")
        ctx.register_created(doc_a)
        ctx.register_created(doc_b)
        orphans = ctx.finalize([doc_a, doc_b])
        assert orphans == []

    def test_detects_orphaned_documents(self):
        ctx = _TaskDocumentContext()
        doc_a = _make_doc("a.txt", "aaa")
        doc_b = _make_doc("b.txt", "bbb")
        ctx.register_created(doc_a)
        ctx.register_created(doc_b)
        orphans = ctx.finalize([doc_a])  # doc_b not returned
        assert len(orphans) == 1
        assert orphans[0] == doc_b.sha256

    def test_all_orphaned(self):
        ctx = _TaskDocumentContext()
        doc = _make_doc("a.txt", "aaa")
        ctx.register_created(doc)
        orphans = ctx.finalize([])
        assert len(orphans) == 1
        assert orphans[0] == doc.sha256

    def test_empty_context_no_orphans(self):
        ctx = _TaskDocumentContext()
        orphans = ctx.finalize([])
        assert orphans == []

    def test_returns_sorted_sha256s(self):
        ctx = _TaskDocumentContext()
        docs = [_make_doc(f"{i}.txt", f"content-{i}") for i in range(5)]
        for d in docs:
            ctx.register_created(d)
        orphans = ctx.finalize([docs[2]])  # return only one
        assert orphans == sorted(orphans)
        assert len(orphans) == 4


# ===== suppression flag =====
