"""PipelineTask persistence tests.

Tests document auto-persistence, deduplication, and behavior
without a database configured.
"""

import pytest

from ai_pipeline_core import Document
from ai_pipeline_core.pipeline import PipelineTask, pipeline_test_context


class InputDoc(Document):
    pass


class OutputDoc(Document):
    pass


class PersistTask(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[InputDoc, ...]) -> tuple[OutputDoc, ...]:
        _ = cls
        source = documents[0]
        return (OutputDoc.derive(from_documents=(source,), name="out.txt", content="ok"),)


@pytest.mark.asyncio
async def test_task_returns_documents() -> None:
    doc = InputDoc.create_root(name="in.txt", content="hello", reason="test input")
    with pipeline_test_context():
        outputs = await PersistTask.run((doc,))
    assert len(outputs) == 1
    assert outputs[0].name == "out.txt"


@pytest.mark.asyncio
async def test_task_returns_correct_document_type() -> None:
    """Task returns documents with correct type."""
    doc = InputDoc.create_root(name="input.txt", content="test input", reason="test input")
    with pipeline_test_context():
        result = await PersistTask.run((doc,))
    assert len(result) == 1
    assert isinstance(result[0], OutputDoc)


@pytest.mark.asyncio
async def test_task_deduplicates_returned_documents() -> None:
    """Task deduplicates returned documents by SHA256."""

    class DedupTask(PipelineTask):
        @classmethod
        async def run(cls, source: InputDoc) -> tuple[OutputDoc, ...]:
            _ = cls
            doc = OutputDoc.derive(from_documents=(source,), name="output.txt", content="test output")
            return (doc, doc)

    doc = InputDoc.create_root(name="input.txt", content="test input", reason="test input")
    with pipeline_test_context():
        result = await DedupTask.run(doc)
    # Deduplication by SHA256: identical documents collapse to one
    assert len(result) == 1


@pytest.mark.asyncio
async def test_task_multiple_outputs_returned() -> None:
    """Task with multiple distinct outputs returns all of them."""

    class MultiTask(PipelineTask):
        @classmethod
        async def run(cls, documents: tuple[InputDoc, ...]) -> tuple[OutputDoc, ...]:
            _ = cls
            return tuple(OutputDoc.derive(from_documents=(doc,), name=f"out_{doc.name}", content=f"processed: {doc.text}") for doc in documents)

    doc1 = InputDoc.create_root(name="a.txt", content="alpha", reason="test input")
    doc2 = InputDoc.create_root(name="b.txt", content="beta", reason="test input")
    with pipeline_test_context():
        results = await MultiTask.run((doc1, doc2))
    assert len(results) == 2
    assert {d.name for d in results} == {"out_a.txt", "out_b.txt"}
