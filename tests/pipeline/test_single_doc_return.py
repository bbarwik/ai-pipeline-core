"""Prove single-Document return annotations pass validation but shouldn't.

_task_return_annotation_error accepts bare Document subclass (-> OutputDoc),
but _normalize_result_documents always wraps single results into a tuple.
The annotation misleads callers about the actual return type.
"""

# pyright: reportPrivateUsage=false, reportUnusedClass=false

import pytest

from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.pipeline._task import PipelineTask
from ai_pipeline_core.pipeline._type_validation import (
    _task_return_annotation_error,
    validate_task_return_annotation,
)


class SDInputDoc(Document):
    """Input document for single-doc return tests."""


class SDOutputDoc(Document):
    """Output document for single-doc return tests."""


# ── Proving tests: PASS on current code, demonstrate the bug ─────────


class TestValidationRejectsSingleDocument:
    """Prove _task_return_annotation_error rejects bare Document subclass."""

    def test_single_document_annotation_returns_error(self) -> None:
        """_task_return_annotation_error returns error for bare Document subclass."""
        error = _task_return_annotation_error(SDOutputDoc)
        assert error is not None
        assert "tuple" in error.lower()

    def test_validate_task_return_rejects_single_document(self) -> None:
        """validate_task_return_annotation raises for -> SDOutputDoc."""
        with pytest.raises(TypeError, match="must not use bare"):
            validate_task_return_annotation(SDOutputDoc, task_name="BugTask")

    def test_task_class_with_single_return_rejected(self) -> None:
        """PipelineTask with -> SDOutputDoc fails __init_subclass__."""
        with pytest.raises(TypeError, match="must not use bare"):

            class SingleReturnTask(PipelineTask):
                @classmethod
                async def run(cls, documents: tuple[SDInputDoc, ...]) -> SDOutputDoc:
                    _ = cls
                    return SDOutputDoc.create_root(name="out.md", content="x", reason="test")


# ── xfail tests: should FAIL now, PASS after fix ────────────────────


class TestSingleDocAnnotationRejected:
    """These tests will pass once bare Document return is rejected."""

    def test_bare_document_returns_error(self) -> None:
        """_task_return_annotation_error rejects single Document."""
        error = _task_return_annotation_error(SDOutputDoc)
        assert error is not None
        assert "tuple" in error.lower()

    def test_task_definition_raises_typeerror(self) -> None:
        """PipelineTask with -> OutputDoc raises TypeError."""
        with pytest.raises(TypeError, match="must not use bare"):

            class BadSingleTask(PipelineTask):
                @classmethod
                async def run(cls, documents: tuple[SDInputDoc, ...]) -> SDOutputDoc:
                    _ = cls
