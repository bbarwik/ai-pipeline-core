"""Trace trim configuration tests for class-based tasks and flows.

Tests trace_trim_documents ClassVar behavior on PipelineTask.
"""

from typing import Any

import pytest

from ai_pipeline_core import Document
from ai_pipeline_core.pipeline import PipelineTask, pipeline_test_context


class InDoc(Document):
    pass


class OutDoc(Document):
    pass


def test_default_trace_trim_documents_enabled() -> None:
    class DefaultTask(PipelineTask):
        @classmethod
        async def run(cls, documents: list[InDoc]) -> list[OutDoc]:
            _ = (cls, documents)
            return []

    assert DefaultTask.trace_trim_documents is True


def test_trace_trim_documents_can_be_disabled() -> None:
    class UntrimmedTask(PipelineTask):
        trace_trim_documents = False

        @classmethod
        async def run(cls, documents: list[InDoc]) -> list[OutDoc]:
            _ = (cls, documents)
            return []

    assert UntrimmedTask.trace_trim_documents is False


def test_trace_trim_documents_explicit_true() -> None:
    class TrimmedTask(PipelineTask):
        trace_trim_documents = True

        @classmethod
        async def run(cls, documents: list[InDoc]) -> list[OutDoc]:
            _ = (cls, documents)
            return []

    assert TrimmedTask.trace_trim_documents is True


def test_trace_trim_documents_inherited() -> None:
    """Subclass inherits parent's trace_trim_documents setting."""

    class _BaseTask(PipelineTask):
        trace_trim_documents = False

        @classmethod
        async def run(cls, documents: list[InDoc]) -> list[OutDoc]:
            _ = (cls, documents)
            return []

    class DerivedTask(_BaseTask):
        pass

    assert DerivedTask.trace_trim_documents is False


def test_trace_trim_documents_override_in_subclass() -> None:
    """Subclass can override parent's trace_trim_documents."""

    class _BaseTask(PipelineTask):
        trace_trim_documents = False

        @classmethod
        async def run(cls, documents: list[InDoc]) -> list[OutDoc]:
            _ = (cls, documents)
            return []

    class OverrideTask(_BaseTask):
        trace_trim_documents = True

    assert OverrideTask.trace_trim_documents is True


@pytest.mark.asyncio
async def test_task_functional_with_trim_enabled() -> None:
    """Task with trace_trim_documents=True executes correctly."""

    class TrimTask(PipelineTask):
        trace_trim_documents = True

        @classmethod
        async def run(cls, doc: InDoc) -> OutDoc:
            return OutDoc.derive(
                from_documents=(doc,),
                name=f"processed_{doc.name}",
                content="ok",
            )

    doc = InDoc.create_root(name="input.txt", content="x" * 1000, reason="test input")
    with pipeline_test_context():
        results: list[Any] = await TrimTask.run(doc)
    assert results[0].name == "processed_input.txt"


@pytest.mark.asyncio
async def test_task_functional_with_trim_disabled() -> None:
    """Task with trace_trim_documents=False executes correctly."""

    class UntrimmedTask(PipelineTask):
        trace_trim_documents = False

        @classmethod
        async def run(cls, doc: InDoc) -> OutDoc:
            return OutDoc.derive(
                from_documents=(doc,),
                name=f"full_{doc.name}",
                content="full output",
            )

    doc = InDoc.create_root(name="input.txt", content="y" * 1000, reason="test input")
    with pipeline_test_context():
        results: list[Any] = await UntrimmedTask.run(doc)
    assert results[0].name == "full_input.txt"


@pytest.mark.asyncio
async def test_tasks_with_different_trim_settings() -> None:
    """Multiple tasks with different trim settings work together."""

    class TrimTask(PipelineTask):
        trace_trim_documents = True

        @classmethod
        async def run(cls, doc: InDoc) -> OutDoc:
            return OutDoc.derive(
                from_documents=(doc,),
                name=f"trimmed_{doc.name}",
                content="trimmed",
            )

    class FullTask(PipelineTask):
        trace_trim_documents = False

        @classmethod
        async def run(cls, doc: InDoc) -> OutDoc:
            return OutDoc.derive(
                from_documents=(doc,),
                name=f"full_{doc.name}",
                content="full",
            )

    doc = InDoc.create_root(name="input.txt", content="data", reason="test input")
    with pipeline_test_context():
        trimmed: list[Any] = await TrimTask.run(doc)
        full: list[Any] = await FullTask.run(doc)
    assert trimmed[0].name == "trimmed_input.txt"
    assert full[0].name == "full_input.txt"
