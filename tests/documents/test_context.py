"""Tests for RunContext, TaskDocumentContext, and suppression flag."""

import pytest

from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents._types import RunScope
from ai_pipeline_core.documents.context import (
    RunContext,
    TaskDocumentContext,
    get_run_context,
    is_registration_suppressed,
    set_run_context,
    suppress_registration,
)


# --- Concrete document subclass for testing ---


class SampleDoc(Document):
    pass


def _make_doc(
    name: str,
    content: str = "test",
    sources: tuple[str, ...] | None = None,
    origins: tuple[str, ...] | None = None,
) -> SampleDoc:
    return SampleDoc.create(name=name, content=content, sources=sources, origins=origins)


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
        assert get_run_context() is None

    def test_set_and_get(self):
        ctx = RunContext(run_scope=RunScope("my-run"))
        token = set_run_context(ctx)
        try:
            assert get_run_context() is ctx
        finally:
            from ai_pipeline_core.documents._context_vars import _run_context

            _run_context.reset(token)

    def test_token_restores_previous(self):
        ctx1 = RunContext(run_scope=RunScope("first"))
        ctx2 = RunContext(run_scope=RunScope("second"))
        token1 = set_run_context(ctx1)
        token2 = set_run_context(ctx2)
        assert get_run_context() is ctx2

        from ai_pipeline_core.documents._context_vars import _run_context

        _run_context.reset(token2)
        assert get_run_context() is ctx1
        _run_context.reset(token1)
        assert get_run_context() is None


# ===== TaskDocumentContext.register_created =====


class TestRegistration:
    def test_register_created(self):
        ctx = TaskDocumentContext()
        doc = _make_doc("a.txt")
        ctx.register_created(doc)
        assert doc.sha256 in ctx.created

    def test_register_multiple(self):
        ctx = TaskDocumentContext()
        doc_a = _make_doc("a.txt", "aaa")
        doc_b = _make_doc("b.txt", "bbb")
        ctx.register_created(doc_a)
        ctx.register_created(doc_b)
        assert len(ctx.created) == 2


# ===== validate_provenance =====


class TestValidateProvenance:
    def test_source_in_existing_set(self):
        """Source exists in the store — no warning."""
        parent = _make_doc("parent.txt", "parent")
        doc = _make_doc("a.txt", "aaa", sources=(parent.sha256,))
        ctx = TaskDocumentContext()
        warnings = ctx.validate_provenance([doc], existing_sha256s={parent.sha256})
        assert warnings == []

    def test_missing_source(self):
        """Document references a SHA256 that doesn't exist in the store."""
        phantom = _make_doc("phantom.txt", "ghost")
        doc = _make_doc("a.txt", "aaa", sources=(phantom.sha256,))
        ctx = TaskDocumentContext()
        warnings = ctx.validate_provenance([doc], existing_sha256s=set())
        assert len(warnings) == 1
        assert "does not exist" in warnings[0]

    def test_accepts_url_sources(self):
        """URLs in sources are accepted and not validated as SHA256."""
        doc = _make_doc("a.txt", "aaa", sources=("https://example.com",))
        ctx = TaskDocumentContext()
        warnings = ctx.validate_provenance([doc], existing_sha256s=set())
        # URL sources have provenance, no "does not exist" warning for URLs
        assert all("does not exist" not in w for w in warnings)

    def test_rejects_invalid_source_strings(self):
        """Non-SHA256, non-URL strings are rejected at document creation."""
        with pytest.raises(Exception):
            _make_doc("a.txt", "aaa", sources=("not-a-hash",))
        with pytest.raises(Exception):
            _make_doc("a.txt", "aaa", sources=("short",))

    def test_missing_origin(self):
        """Document references an origin SHA256 that doesn't exist."""
        phantom = _make_doc("phantom.txt", "ghost")
        doc = _make_doc("a.txt", "aaa", origins=(phantom.sha256,))
        ctx = TaskDocumentContext()
        warnings = ctx.validate_provenance([doc], existing_sha256s=set())
        assert len(warnings) == 1
        assert "origin" in warnings[0]
        assert "does not exist" in warnings[0]

    def test_origin_in_existing_set(self):
        """Origin exists in the store — no warning."""
        parent = _make_doc("parent.txt", "parent")
        doc = _make_doc("a.txt", "aaa", origins=(parent.sha256,))
        ctx = TaskDocumentContext()
        warnings = ctx.validate_provenance([doc], existing_sha256s={parent.sha256})
        assert warnings == []

    def test_same_task_source_interdep(self):
        """Source SHA256 created in the same task produces a warning."""
        doc_a = _make_doc("a.txt", "aaa")
        doc_b = _make_doc("b.txt", "bbb", sources=(doc_a.sha256,))
        ctx = TaskDocumentContext()
        ctx.register_created(doc_a)
        ctx.register_created(doc_b)
        warnings = ctx.validate_provenance([doc_b], existing_sha256s=set())
        assert len(warnings) == 1
        assert "same task" in warnings[0]

    def test_same_task_origin_interdep(self):
        """Origin SHA256 created in the same task produces a warning."""
        doc_a = _make_doc("a.txt", "aaa")
        doc_b = _make_doc("b.txt", "bbb", origins=(doc_a.sha256,))
        ctx = TaskDocumentContext()
        ctx.register_created(doc_a)
        ctx.register_created(doc_b)
        warnings = ctx.validate_provenance([doc_b], existing_sha256s=set())
        assert len(warnings) == 1
        assert "same task" in warnings[0]
        assert "origin" in warnings[0]

    def test_no_provenance_warning(self):
        """Document with no sources and no origins gets a warning."""
        doc = _make_doc("a.txt", "aaa")
        ctx = TaskDocumentContext()
        warnings = ctx.validate_provenance([doc], existing_sha256s=set())
        assert len(warnings) == 1
        assert "no provenance" in warnings[0]

    def test_source_url_no_provenance_warning(self):
        """Document with URL source has provenance — no warning."""
        doc = _make_doc("a.txt", "aaa", sources=("https://example.com",))
        ctx = TaskDocumentContext()
        warnings = ctx.validate_provenance([doc], existing_sha256s=set())
        assert warnings == []

    def test_mixed_valid_and_invalid(self):
        """Multiple documents with different provenance issues."""
        parent = _make_doc("parent.txt", "parent")
        valid = _make_doc("valid.txt", "valid", sources=(parent.sha256,))
        orphan = _make_doc("orphan.txt", "orphan")  # no provenance
        ctx = TaskDocumentContext()
        warnings = ctx.validate_provenance([valid, orphan], existing_sha256s={parent.sha256})
        assert len(warnings) == 1
        assert "no provenance" in warnings[0]

    def test_check_created_warns_on_foreign_doc(self):
        """With check_created=True, returning a doc not created in this context warns."""
        foreign = _make_doc("foreign.txt", "foreign")
        ctx = TaskDocumentContext()
        # foreign was NOT registered via register_created
        warnings = ctx.validate_provenance([foreign], existing_sha256s=set(), check_created=True)
        assert any("was not created in this task" in w for w in warnings)

    def test_check_created_no_warning_when_created(self):
        """With check_created=True, returning a doc created in this context is fine."""
        doc = _make_doc("mine.txt", "mine")
        ctx = TaskDocumentContext()
        ctx.register_created(doc)
        warnings = ctx.validate_provenance([doc], existing_sha256s=set(), check_created=True)
        # Only the "no provenance" warning, not the "not created" warning
        assert all("was not created" not in w for w in warnings)

    def test_check_created_disabled_by_default(self):
        """Without check_created, returning a foreign doc does not warn about creation."""
        foreign = _make_doc("foreign.txt", "foreign")
        ctx = TaskDocumentContext()
        warnings = ctx.validate_provenance([foreign], existing_sha256s=set())
        assert all("was not created" not in w for w in warnings)


# ===== finalize =====


class TestFinalize:
    def test_no_warnings_when_all_returned(self):
        ctx = TaskDocumentContext()
        doc_a = _make_doc("a.txt", "aaa")
        doc_b = _make_doc("b.txt", "bbb")
        ctx.register_created(doc_a)
        ctx.register_created(doc_b)
        warnings = ctx.finalize([doc_a, doc_b])
        assert warnings == []

    def test_warns_on_created_not_returned(self):
        ctx = TaskDocumentContext()
        doc_a = _make_doc("a.txt", "aaa")
        doc_b = _make_doc("b.txt", "bbb")
        ctx.register_created(doc_a)
        ctx.register_created(doc_b)
        warnings = ctx.finalize([doc_a])  # doc_b not returned
        assert len(warnings) == 1
        assert "created but not returned" in warnings[0]

    def test_empty_context(self):
        ctx = TaskDocumentContext()
        warnings = ctx.finalize([])
        assert warnings == []


# ===== deduplicate =====


class TestDeduplicate:
    def test_empty_list(self):
        assert TaskDocumentContext.deduplicate([]) == []

    def test_no_duplicates(self):
        docs = [_make_doc("a.txt", "aaa"), _make_doc("b.txt", "bbb")]
        result = TaskDocumentContext.deduplicate(docs)
        assert len(result) == 2

    def test_removes_duplicates(self):
        doc = _make_doc("a.txt", "aaa")
        result = TaskDocumentContext.deduplicate([doc, doc])
        assert len(result) == 1
        assert result[0].name == "a.txt"

    def test_preserves_first_occurrence_order(self):
        doc_a = _make_doc("a.txt", "aaa")
        doc_b = _make_doc("b.txt", "bbb")
        doc_c = _make_doc("c.txt", "ccc")
        result = TaskDocumentContext.deduplicate([doc_c, doc_a, doc_b, doc_a, doc_c])
        assert [d.name for d in result] == ["c.txt", "a.txt", "b.txt"]

    def test_same_content_different_names(self):
        """Documents with identical content but different names have the same SHA256."""
        doc1 = _make_doc("first.txt", "same content")
        doc2 = _make_doc("first.txt", "same content")
        result = TaskDocumentContext.deduplicate([doc1, doc2])
        assert len(result) == 1


# ===== suppression flag =====


class TestSuppression:
    def test_default_not_suppressed(self):
        assert is_registration_suppressed() is False

    def test_suppress_registration_sets_flag(self):
        with suppress_registration():
            assert is_registration_suppressed() is True
        assert is_registration_suppressed() is False

    def test_nested_suppression(self):
        with suppress_registration():
            assert is_registration_suppressed() is True
            with suppress_registration():
                assert is_registration_suppressed() is True
            assert is_registration_suppressed() is True
        assert is_registration_suppressed() is False
