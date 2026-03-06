"""PipelineTask persistence tests.

Tests document auto-persistence, deduplication, and behavior
with and without a DocumentStore configured.
"""

import pytest

from ai_pipeline_core import Document
from ai_pipeline_core.document_store._memory import MemoryDocumentStore
from ai_pipeline_core.pipeline import PipelineTask, pipeline_test_context


class InputDoc(Document):
    pass


class OutputDoc(Document):
    pass


class PersistTask(PipelineTask):
    @classmethod
    async def run(cls, documents: list[InputDoc]) -> list[OutputDoc]:
        _ = cls
        source = documents[0]
        return [OutputDoc.derive(from_documents=(source,), name="out.txt", content="ok")]


@pytest.mark.asyncio
async def test_task_auto_persists_returned_documents() -> None:
    store = MemoryDocumentStore()
    doc = InputDoc.create_root(name="in.txt", content="hello", reason="test input")
    with pipeline_test_context(store=store) as ctx:
        outputs = await PersistTask.run([doc])
        loaded = await store.load(ctx.run_scope, [OutputDoc])
    assert outputs
    assert len(loaded) == 1
    assert loaded[0].name == "out.txt"


@pytest.mark.asyncio
async def test_task_returns_documents_with_store_configured() -> None:
    """Task returns documents correctly when a store is configured."""
    store = MemoryDocumentStore()
    doc = InputDoc.create_root(name="input.txt", content="test input", reason="test input")
    with pipeline_test_context(store=store):
        result = await PersistTask.run([doc])
    assert len(result) == 1
    assert isinstance(result[0], OutputDoc)


@pytest.mark.asyncio
async def test_task_deduplicates_returned_documents() -> None:
    """Task deduplicates returned documents by SHA256."""

    class DedupTask(PipelineTask):
        @classmethod
        async def run(cls, source: InputDoc) -> list[OutputDoc]:
            _ = cls
            doc = OutputDoc.derive(from_documents=(source,), name="output.txt", content="test output")
            return [doc, doc]  # Same document twice

    store = MemoryDocumentStore()
    doc = InputDoc.create_root(name="input.txt", content="test input", reason="test input")
    with pipeline_test_context(store=store) as ctx:
        await DedupTask.run(doc)
        loaded = await store.load(ctx.run_scope, [OutputDoc])
    assert len(loaded) == 1


@pytest.mark.asyncio
async def test_task_multiple_outputs_saved() -> None:
    """Task with multiple distinct outputs saves all of them."""

    class MultiTask(PipelineTask):
        @classmethod
        async def run(cls, documents: list[InputDoc]) -> list[OutputDoc]:
            _ = cls
            return [OutputDoc.derive(from_documents=(doc,), name=f"out_{doc.name}", content=f"processed: {doc.text}") for doc in documents]

    store = MemoryDocumentStore()
    doc1 = InputDoc.create_root(name="a.txt", content="alpha", reason="test input")
    doc2 = InputDoc.create_root(name="b.txt", content="beta", reason="test input")
    with pipeline_test_context(store=store) as ctx:
        results = await MultiTask.run([doc1, doc2])
        loaded = await store.load(ctx.run_scope, [OutputDoc])
    assert len(results) == 2
    assert len(loaded) == 2
    assert {d.name for d in loaded} == {"out_a.txt", "out_b.txt"}
