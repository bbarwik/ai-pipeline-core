"""Tests for trace_trim_documents parameter in pipeline decorators."""

from unittest.mock import MagicMock, patch

from ai_pipeline_core.documents import Document
from ai_pipeline_core.pipeline import pipeline_flow, pipeline_task
from ai_pipeline_core.pipeline.options import FlowOptions


class WorkTaskDoc(Document):
    """Task document for testing tasks."""


class InputFlowDoc(Document):
    """Input document for testing flows."""


class OutputFlowDoc(Document):
    """Output document for testing flows."""


class TestPipelineTraceTrimDocuments:
    """Test trace_trim_documents parameter in pipeline decorators."""

    @patch("ai_pipeline_core.pipeline.decorators.trace")
    async def test_pipeline_task_with_trace_trim_documents_true(self, mock_trace):
        mock_decorator = MagicMock()
        mock_trace.return_value = mock_decorator
        mock_decorator.return_value = MagicMock()

        @pipeline_task(trace_trim_documents=True, name="test_task")
        async def task_func(doc: WorkTaskDoc) -> WorkTaskDoc:  # pyright: ignore[reportUnusedFunction]
            return doc

        mock_trace.assert_called_once()
        call_kwargs = mock_trace.call_args.kwargs
        assert call_kwargs["trim_documents"] is True
        assert call_kwargs["name"] == "test_task"

    @patch("ai_pipeline_core.pipeline.decorators.trace")
    async def test_pipeline_task_with_trace_trim_documents_false(self, mock_trace):
        mock_decorator = MagicMock()
        mock_trace.return_value = mock_decorator
        mock_decorator.return_value = MagicMock()

        @pipeline_task(trace_trim_documents=False, name="test_task")
        async def task_func(doc: WorkTaskDoc) -> WorkTaskDoc:  # pyright: ignore[reportUnusedFunction]
            return doc

        mock_trace.assert_called_once()
        call_kwargs = mock_trace.call_args.kwargs
        assert call_kwargs["trim_documents"] is False

    @patch("ai_pipeline_core.pipeline.decorators.trace")
    async def test_pipeline_task_default_trace_trim_documents(self, mock_trace):
        mock_decorator = MagicMock()
        mock_trace.return_value = mock_decorator
        mock_decorator.return_value = MagicMock()

        @pipeline_task(name="test_task")
        async def task_func(doc: WorkTaskDoc) -> WorkTaskDoc:  # pyright: ignore[reportUnusedFunction]
            return doc

        mock_trace.assert_called_once()
        call_kwargs = mock_trace.call_args.kwargs
        assert call_kwargs["trim_documents"] is True

    @patch("ai_pipeline_core.pipeline.decorators.trace")
    async def test_pipeline_flow_with_trace_trim_documents_true(self, mock_trace):
        mock_decorator = MagicMock()
        mock_trace.return_value = mock_decorator

        async def mock_wrapper(*args, **kwargs):
            return list([OutputFlowDoc.create_root(name="out.txt", content="test")], reason="test input")

        mock_decorator.return_value = mock_wrapper

        @pipeline_flow(trace_trim_documents=True, name="test_flow")
        async def flow_func(  # pyright: ignore[reportUnusedFunction]
            run_id: str,
            documents: list[InputFlowDoc],
            flow_options: FlowOptions,
        ) -> list[OutputFlowDoc]:
            return [OutputFlowDoc.create_root(name=f"output_{doc.name}", content=doc.content, reason="test input") for doc in documents]

        mock_trace.assert_called_once()
        call_kwargs = mock_trace.call_args.kwargs
        assert call_kwargs["trim_documents"] is True
        assert call_kwargs["name"] == "test_flow"

    @patch("ai_pipeline_core.pipeline.decorators.trace")
    async def test_pipeline_flow_with_trace_trim_documents_false(self, mock_trace):
        mock_decorator = MagicMock()
        mock_trace.return_value = mock_decorator

        async def mock_wrapper(*args, **kwargs):
            return list([OutputFlowDoc.create_root(name="out.txt", content="test")], reason="test input")

        mock_decorator.return_value = mock_wrapper

        @pipeline_flow(trace_trim_documents=False, name="test_flow")
        async def flow_func(  # pyright: ignore[reportUnusedFunction]
            run_id: str,
            documents: list[InputFlowDoc],
            flow_options: FlowOptions,
        ) -> list[OutputFlowDoc]:
            return [OutputFlowDoc.create_root(name=f"output_{doc.name}", content=doc.content, reason="test input") for doc in documents]

        mock_trace.assert_called_once()
        call_kwargs = mock_trace.call_args.kwargs
        assert call_kwargs["trim_documents"] is False

    @patch("ai_pipeline_core.pipeline.decorators.trace")
    async def test_pipeline_flow_default_trace_trim_documents(self, mock_trace):
        mock_decorator = MagicMock()
        mock_trace.return_value = mock_decorator

        async def mock_wrapper(*args, **kwargs):
            return list([OutputFlowDoc.create_root(name="out.txt", content="test")], reason="test input")

        mock_decorator.return_value = mock_wrapper

        @pipeline_flow(name="test_flow")
        async def flow_func(  # pyright: ignore[reportUnusedFunction]
            run_id: str,
            documents: list[InputFlowDoc],
            flow_options: FlowOptions,
        ) -> list[OutputFlowDoc]:
            return [OutputFlowDoc.create_root(name=f"output_{doc.name}", content=doc.content, reason="test input") for doc in documents]

        mock_trace.assert_called_once()
        call_kwargs = mock_trace.call_args.kwargs
        assert call_kwargs["trim_documents"] is True

    async def test_pipeline_task_functional_with_trim(self):
        @pipeline_task(trace_trim_documents=True)
        async def process_doc(doc: WorkTaskDoc) -> WorkTaskDoc:
            return WorkTaskDoc.create(
                name=f"processed_{doc.name}",
                content=f"Processed: {doc.text[:50]}",
                derived_from=(doc.sha256,),
            )

        doc = WorkTaskDoc.create_root(name="input.txt", content="x" * 1000, reason="test input")
        result = await process_doc(doc)

        assert result.name == "processed_input.txt"
        assert "Processed:" in result.text
        assert doc.sha256 in result.derived_from

    async def test_pipeline_flow_functional_with_trim(self):
        @pipeline_flow(trace_trim_documents=True)
        async def process_flow(
            run_id: str,
            documents: list[InputFlowDoc],
            flow_options: FlowOptions,
        ) -> list[OutputFlowDoc]:
            return [
                OutputFlowDoc.create(
                    name=f"{run_id}_{doc.name}",
                    content=f"Flow processed: {doc.text[:100]}",
                    derived_from=(doc.sha256,),
                )
                for doc in documents
            ]

        input_doc = InputFlowDoc.create_root(name="input.txt", content="y" * 1000, reason="test input")
        docs = [input_doc]
        options = FlowOptions()

        result = await process_flow("test_project", docs, options)

        assert len(result) == 1
        assert result[0].name == "test_project_input.txt"
        assert "Flow processed:" in result[0].text
        assert input_doc.sha256 in result[0].derived_from

    async def test_pipeline_decorators_combined(self):
        @pipeline_task(trace_trim_documents=True)
        async def trim_task(doc: WorkTaskDoc) -> WorkTaskDoc:
            return WorkTaskDoc.create_root(name=f"trimmed_{doc.name}", content=doc.content[:100], reason="test input")

        @pipeline_flow(trace_trim_documents=False)
        async def full_flow(
            run_id: str,
            documents: list[InputFlowDoc],
            flow_options: FlowOptions,
        ) -> list[OutputFlowDoc]:
            task_doc = WorkTaskDoc.create_root(name="task.txt", content="a" * 500, reason="test input")
            trimmed = await trim_task(task_doc)

            return [
                OutputFlowDoc.create_root(
                    name=f"output_{doc.name}",
                    content=f"{doc.text}\nTask result: {trimmed.text}",
                    reason="test input",
                )
                for doc in documents
            ]

        input_doc = InputFlowDoc.create_root(name="input.txt", content="Original content", reason="test input")
        docs = [input_doc]
        options = FlowOptions()

        result = await full_flow("test", docs, options)

        assert len(result) == 1
        assert "Task result: " in result[0].text
        assert "Original content" in result[0].text
