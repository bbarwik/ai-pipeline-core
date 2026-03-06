"""Tests for strict document creation enforcement.

These tests verify that documents can only be created within proper
pipeline task/flow context. They live in tests/documents/ (not tests/pipeline/)
because the pipeline conftest.py has an autouse fixture that suppresses
registration — which is exactly what these tests need to verify.
"""

import pytest

from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents._context import TaskContext, reset_task_context, set_task_context


class _EnforcementDoc(Document):
    """Test document for enforcement tests."""


def test_document_creation_outside_context_raises() -> None:
    with pytest.raises(RuntimeError, match="outside pipeline task context"):
        _EnforcementDoc(name="test.txt", content=b"data")


def test_create_root_outside_context_succeeds() -> None:
    doc = _EnforcementDoc.create_root(name="root.txt", content=b"root data", reason="deployment input")
    assert doc.name == "root.txt"
    assert doc.content == b"root data"


def test_create_root_inside_task_context_raises() -> None:
    ctx = TaskContext(task_class_name="SomeTask")
    token = set_task_context(ctx)
    try:
        with pytest.raises(RuntimeError, match="cannot be called inside pipeline task/flow execution"):
            _EnforcementDoc.create_root(name="root.txt", content=b"data", reason="should fail")
    finally:
        reset_task_context(token)


def test_create_root_requires_nonempty_reason() -> None:
    with pytest.raises(ValueError, match="requires a non-empty reason"):
        _EnforcementDoc.create_root(name="root.txt", content=b"data", reason="")


def test_create_requires_provenance() -> None:
    ctx = TaskContext(task_class_name="SomeTask")
    token = set_task_context(ctx)
    try:
        with pytest.raises(ValueError, match="requires derived_from or triggered_by"):
            _EnforcementDoc.create(name="test.txt", content="data")
    finally:
        reset_task_context(token)


def test_document_creation_inside_task_context_succeeds() -> None:
    source = _EnforcementDoc.create_root(name="source.txt", content=b"source", reason="test provenance")
    ctx = TaskContext(task_class_name="SomeTask")
    token = set_task_context(ctx)
    try:
        doc = _EnforcementDoc.create(
            name="task_output.txt",
            content="output data",
            derived_from=(source.sha256,),
        )
        assert doc.name == "task_output.txt"
        assert doc.sha256 in ctx.created
    finally:
        reset_task_context(token)


def test_document_instantiate_base_class_raises() -> None:
    with pytest.raises(TypeError, match="Cannot instantiate Document directly"):
        Document(name="test.txt", content=b"data")
