"""PipelineTask and PipelineFlow entry-point tests.

Tests estimated_minutes validation, document persistence to store,
persistence failure handling, and annotation extraction.
"""

from unittest.mock import AsyncMock

import pytest

from ai_pipeline_core import Document
from ai_pipeline_core.document_store._memory import MemoryDocumentStore
from ai_pipeline_core.pipeline import PipelineFlow, PipelineTask, collect_tasks, pipeline_test_context
from ai_pipeline_core.pipeline.options import FlowOptions


class InputDoc(Document):
    pass


class OutputDoc(Document):
    pass


class AltInputDoc(Document):
    pass


# --------------------------------------------------------------------------- #
# Basic task lifecycle tests
# --------------------------------------------------------------------------- #


class EchoTask(PipelineTask):
    @classmethod
    async def run(cls, documents: list[InputDoc]) -> list[OutputDoc]:
        _ = cls
        return [
            OutputDoc.derive(
                from_documents=(doc,),
                name=f"out_{doc.name}",
                content=doc.content,
            )
            for doc in documents
        ]


@pytest.mark.asyncio
async def test_task_run_is_awaitable() -> None:
    doc = InputDoc.create_root(name="a.txt", content="hello", reason="test input")
    with pipeline_test_context():
        outputs = await EchoTask.run([doc])
    assert len(outputs) == 1
    assert outputs[0].name == "out_a.txt"


@pytest.mark.asyncio
async def test_task_handles_can_be_collected() -> None:
    first = InputDoc.create_root(name="a.txt", content="a", reason="test input")
    second = InputDoc.create_root(name="b.txt", content="b", reason="test input")
    with pipeline_test_context():
        batch = await collect_tasks(EchoTask.run([first]), EchoTask.run([second]))
    flattened = [doc for docs in batch.completed for doc in docs]
    assert batch.incomplete == []
    assert {doc.name for doc in flattened} == {"out_a.txt", "out_b.txt"}


# --------------------------------------------------------------------------- #
# Estimated minutes tests
# --------------------------------------------------------------------------- #


class TestEstimatedMinutes:
    """Test estimated_minutes ClassVar validation."""

    def test_task_stores_estimated_minutes(self):
        class MinutesTask(PipelineTask):
            estimated_minutes = 5

            @classmethod
            async def run(cls) -> None:
                pass

        assert MinutesTask.estimated_minutes == 5

    def test_task_default_estimated_minutes(self):
        class DefaultTask(PipelineTask):
            @classmethod
            async def run(cls) -> None:
                pass

        assert DefaultTask.estimated_minutes == 1

    def test_flow_stores_estimated_minutes(self):
        class MinutesFlow(PipelineFlow):
            estimated_minutes = 30

            async def run(self, run_id: str, documents: list[InputDoc], options: FlowOptions) -> list[OutputDoc]:
                return []

        assert MinutesFlow.estimated_minutes == 30

    def test_flow_default_estimated_minutes(self):
        class DefaultFlow(PipelineFlow):
            async def run(self, run_id: str, documents: list[InputDoc], options: FlowOptions) -> list[OutputDoc]:
                return []

        assert DefaultFlow.estimated_minutes == 1

    def test_task_rejects_zero(self):
        with pytest.raises(TypeError, match="estimated_minutes"):

            class BadTask(PipelineTask):
                estimated_minutes = 0

                @classmethod
                async def run(cls) -> None:
                    pass

    def test_flow_rejects_zero(self):
        with pytest.raises(TypeError, match="estimated_minutes"):

            class BadFlow(PipelineFlow):
                estimated_minutes = 0

                async def run(self, run_id: str, documents: list[InputDoc], options: FlowOptions) -> list[OutputDoc]:
                    return []


# --------------------------------------------------------------------------- #
# Flow annotation extraction tests
# --------------------------------------------------------------------------- #


class TestFlowAnnotationExtraction:
    """Test that PipelineFlow extracts document types from annotations."""

    def test_extracts_input_types(self):
        class ExtractFlow(PipelineFlow):
            async def run(self, run_id: str, documents: list[InputDoc], options: FlowOptions) -> list[OutputDoc]:
                return []

        assert ExtractFlow.input_document_types == [InputDoc]

    def test_extracts_output_types(self):
        class ExtractFlow(PipelineFlow):
            async def run(self, run_id: str, documents: list[InputDoc], options: FlowOptions) -> list[OutputDoc]:
                return []

        assert ExtractFlow.output_document_types == [OutputDoc]

    def test_extracts_union_input_types(self):
        class UnionFlow(PipelineFlow):
            async def run(self, run_id: str, documents: list[InputDoc | AltInputDoc], options: FlowOptions) -> list[OutputDoc]:
                return []

        assert set(UnionFlow.input_document_types) == {InputDoc, AltInputDoc}


# --------------------------------------------------------------------------- #
# Document auto-save tests
# --------------------------------------------------------------------------- #


class TestDocumentAutoSave:
    """Test document persistence via PipelineTask."""

    @pytest.mark.asyncio
    async def test_documents_saved_to_store(self):
        store = MemoryDocumentStore()

        class SaveTask(PipelineTask):
            @classmethod
            async def run(cls, source: InputDoc) -> list[OutputDoc]:
                return [OutputDoc.derive(from_documents=(source,), name="out.txt", content="output")]

        source = InputDoc.create_root(name="in.txt", content="input", reason="test input")
        with pipeline_test_context(store=store) as ctx:
            result = await SaveTask.run(source)
            assert len(result) == 1

            loaded = await store.load(ctx.run_scope, [OutputDoc])
            assert len(loaded) == 1
            assert loaded[0].name == "out.txt"

    @pytest.mark.asyncio
    async def test_no_store_works(self):
        """Task works when no store is configured."""

        class NoStoreTask(PipelineTask):
            @classmethod
            async def run(cls, source: InputDoc) -> list[OutputDoc]:
                return [OutputDoc.derive(from_documents=(source,), name="out.txt", content="output")]

        source = InputDoc.create_root(name="in.txt", content="input", reason="test input")
        with pipeline_test_context():
            result = await NoStoreTask.run(source)
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_single_document_return_saved(self):
        store = MemoryDocumentStore()

        class SingleTask(PipelineTask):
            @classmethod
            async def run(cls, source: InputDoc) -> OutputDoc:
                return OutputDoc.derive(from_documents=(source,), name="single.txt", content="data")

        source = InputDoc.create_root(name="in.txt", content="input", reason="test input")
        with pipeline_test_context(store=store) as ctx:
            await SingleTask.run(source)
            loaded = await store.load(ctx.run_scope, [OutputDoc])
            assert len(loaded) == 1

    @pytest.mark.asyncio
    async def test_none_return_saves_nothing(self):
        store = MemoryDocumentStore()

        class NoneTask(PipelineTask):
            @classmethod
            async def run(cls, source: InputDoc) -> None:
                pass

        source = InputDoc.create_root(name="in.txt", content="input", reason="test input")
        with pipeline_test_context(store=store) as ctx:
            result = await NoneTask.run(source)
            assert result == []
            loaded = await store.load(ctx.run_scope, [OutputDoc])
            assert len(loaded) == 0


# --------------------------------------------------------------------------- #
# Persistence failure graceful degradation
# --------------------------------------------------------------------------- #


class TestPersistenceGracefulDegradation:
    """Test that persistence failures don't crash tasks."""

    @pytest.mark.asyncio
    async def test_store_save_failure_logs_warning(self):
        """A store that raises on save_batch should not crash the task."""
        broken_store = AsyncMock(spec=MemoryDocumentStore)
        broken_store.save_batch.side_effect = RuntimeError("store broken")

        class FailStoreTask(PipelineTask):
            @classmethod
            async def run(cls, source: InputDoc) -> list[OutputDoc]:
                return [OutputDoc.derive(from_documents=(source,), name="out.txt", content="output")]

        source = InputDoc.create_root(name="in.txt", content="input", reason="test input")
        with pipeline_test_context(store=broken_store):
            result = await FailStoreTask.run(source)
            assert len(result) == 1


# Sync function rejection: see test_static_validation.py for canonical tests.
