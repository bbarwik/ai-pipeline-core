"""Tests for trace_trim_documents parameter in pipeline decorators."""

from unittest.mock import MagicMock, patch

from ai_pipeline_core import (
    DocumentList,
    FlowConfig,
    FlowDocument,
    FlowOptions,
    TaskDocument,
    pipeline_flow,
    pipeline_task,
)


class InputFlowDoc(FlowDocument):
    """Input document for testing flows."""

    pass


class OutputFlowDoc(FlowDocument):
    """Output document for testing flows."""

    pass


class WorkTaskDoc(TaskDocument):
    """Task document for testing tasks."""

    pass


class PipelineFlowConfig(FlowConfig):
    """Test flow configuration."""

    INPUT_DOCUMENT_TYPES = [InputFlowDoc]
    OUTPUT_DOCUMENT_TYPE = OutputFlowDoc


class TestPipelineTraceTrimDocuments:
    """Test trace_trim_documents parameter in pipeline decorators."""

    @patch("ai_pipeline_core.pipeline.trace")
    async def test_pipeline_task_with_trace_trim_documents_true(self, mock_trace):
        """Test pipeline_task with trace_trim_documents=True passes parameter to trace."""
        # Setup mock
        mock_decorator = MagicMock()
        mock_trace.return_value = mock_decorator
        mock_decorator.return_value = MagicMock()

        @pipeline_task(trace_trim_documents=True, name="test_task")
        async def task_func(doc: WorkTaskDoc) -> WorkTaskDoc:  # pyright: ignore[reportUnusedFunction]
            return doc

        # Verify trace was called with trim_documents=True
        mock_trace.assert_called_once()
        call_kwargs = mock_trace.call_args.kwargs
        assert call_kwargs["trim_documents"] is True
        assert call_kwargs["name"] == "test_task"

    @patch("ai_pipeline_core.pipeline.trace")
    async def test_pipeline_task_with_trace_trim_documents_false(self, mock_trace):
        """Test pipeline_task with trace_trim_documents=False passes parameter to trace."""
        # Setup mock
        mock_decorator = MagicMock()
        mock_trace.return_value = mock_decorator
        mock_decorator.return_value = MagicMock()

        @pipeline_task(trace_trim_documents=False, name="test_task")
        async def task_func(doc: WorkTaskDoc) -> WorkTaskDoc:  # pyright: ignore[reportUnusedFunction]
            return doc

        # Verify trace was called with trim_documents=False
        mock_trace.assert_called_once()
        call_kwargs = mock_trace.call_args.kwargs
        assert call_kwargs["trim_documents"] is False

    @patch("ai_pipeline_core.pipeline.trace")
    async def test_pipeline_task_default_trace_trim_documents(self, mock_trace):
        """Test pipeline_task defaults trace_trim_documents to True."""
        # Setup mock
        mock_decorator = MagicMock()
        mock_trace.return_value = mock_decorator
        mock_decorator.return_value = MagicMock()

        @pipeline_task(name="test_task")
        async def task_func(doc: WorkTaskDoc) -> WorkTaskDoc:  # pyright: ignore[reportUnusedFunction]
            return doc

        # Verify trace was called with trim_documents=True (default)
        mock_trace.assert_called_once()
        call_kwargs = mock_trace.call_args.kwargs
        assert call_kwargs["trim_documents"] is True

    @patch("ai_pipeline_core.pipeline.trace")
    async def test_pipeline_flow_with_trace_trim_documents_true(self, mock_trace):
        """Test pipeline_flow with trace_trim_documents=True passes parameter to trace."""
        # Setup mock
        mock_decorator = MagicMock()
        mock_trace.return_value = mock_decorator

        # Mock the decorated function to avoid actual flow execution
        async def mock_wrapper(*args, **kwargs):
            return DocumentList([OutputFlowDoc.create(name="out.txt", content="test")])

        mock_decorator.return_value = mock_wrapper

        @pipeline_flow(config=PipelineFlowConfig, trace_trim_documents=True, name="test_flow")
        async def flow_func(  # pyright: ignore[reportUnusedFunction]
            project_name: str,
            documents: DocumentList,
            flow_options: FlowOptions,
        ) -> DocumentList:
            outputs = DocumentList()
            for doc in documents:
                output = OutputFlowDoc.create(name=f"output_{doc.name}", content=doc.content)
                outputs.append(output)
            return outputs

        # Verify trace was called with trim_documents=True
        mock_trace.assert_called_once()
        call_kwargs = mock_trace.call_args.kwargs
        assert call_kwargs["trim_documents"] is True
        assert call_kwargs["name"] == "test_flow"

    @patch("ai_pipeline_core.pipeline.trace")
    async def test_pipeline_flow_with_trace_trim_documents_false(self, mock_trace):
        """Test pipeline_flow with trace_trim_documents=False passes parameter to trace."""
        # Setup mock
        mock_decorator = MagicMock()
        mock_trace.return_value = mock_decorator

        # Mock the decorated function
        async def mock_wrapper(*args, **kwargs):
            return DocumentList([OutputFlowDoc.create(name="out.txt", content="test")])

        mock_decorator.return_value = mock_wrapper

        @pipeline_flow(config=PipelineFlowConfig, trace_trim_documents=False, name="test_flow")
        async def flow_func(  # pyright: ignore[reportUnusedFunction]
            project_name: str,
            documents: DocumentList,
            flow_options: FlowOptions,
        ) -> DocumentList:
            outputs = DocumentList()
            for doc in documents:
                output = OutputFlowDoc.create(name=f"output_{doc.name}", content=doc.content)
                outputs.append(output)
            return outputs

        # Verify trace was called with trim_documents=False
        mock_trace.assert_called_once()
        call_kwargs = mock_trace.call_args.kwargs
        assert call_kwargs["trim_documents"] is False

    @patch("ai_pipeline_core.pipeline.trace")
    async def test_pipeline_flow_default_trace_trim_documents(self, mock_trace):
        """Test pipeline_flow defaults trace_trim_documents to True."""
        # Setup mock
        mock_decorator = MagicMock()
        mock_trace.return_value = mock_decorator

        # Mock the decorated function
        async def mock_wrapper(*args, **kwargs):
            return DocumentList([OutputFlowDoc.create(name="out.txt", content="test")])

        mock_decorator.return_value = mock_wrapper

        @pipeline_flow(config=PipelineFlowConfig, name="test_flow")
        async def flow_func(  # pyright: ignore[reportUnusedFunction]
            project_name: str,
            documents: DocumentList,
            flow_options: FlowOptions,
        ) -> DocumentList:
            outputs = DocumentList()
            for doc in documents:
                output = OutputFlowDoc.create(name=f"output_{doc.name}", content=doc.content)
                outputs.append(output)
            return outputs

        # Verify trace was called with trim_documents=True (default)
        mock_trace.assert_called_once()
        call_kwargs = mock_trace.call_args.kwargs
        assert call_kwargs["trim_documents"] is True

    async def test_pipeline_task_functional_with_trim(self, prefect_test_fixture):
        """Functional test that pipeline_task works with trace_trim_documents."""

        @pipeline_task(trace_trim_documents=True)
        async def process_doc(doc: WorkTaskDoc) -> WorkTaskDoc:
            # Process and return document
            return WorkTaskDoc.create(
                name=f"processed_{doc.name}",
                content=f"Processed: {doc.text[:50]}",
                sources=[doc.sha256],
            )

        # Test execution
        doc = WorkTaskDoc.create(name="input.txt", content="x" * 1000)
        result = await process_doc(doc)

        assert result.name == "processed_input.txt"
        assert "Processed:" in result.text
        assert doc.sha256 in result.sources

    async def test_pipeline_flow_functional_with_trim(self, prefect_test_fixture):
        """Functional test that pipeline_flow works with trace_trim_documents."""

        @pipeline_flow(config=PipelineFlowConfig, trace_trim_documents=True)
        async def process_flow(
            project_name: str,
            documents: DocumentList,
            flow_options: FlowOptions,
        ) -> DocumentList:
            outputs = DocumentList()
            for doc in documents:
                output = OutputFlowDoc.create(
                    name=f"{project_name}_{doc.name}",
                    content=f"Flow processed: {doc.text[:100]}",
                    sources=[doc.sha256],
                )
                outputs.append(output)
            return outputs

        # Test execution
        input_doc = InputFlowDoc.create(name="input.txt", content="y" * 1000)
        docs = DocumentList([input_doc])
        options = FlowOptions()

        result = await process_flow("test_project", docs, options)

        assert len(result) == 1
        assert result[0].name == "test_project_input.txt"
        assert "Flow processed:" in result[0].text
        assert input_doc.sha256 in result[0].sources

    async def test_pipeline_decorators_combined(self, prefect_test_fixture):
        """Test using both pipeline_task and pipeline_flow with trace_trim_documents."""

        @pipeline_task(trace_trim_documents=True)
        async def trim_task(doc: WorkTaskDoc) -> WorkTaskDoc:
            return WorkTaskDoc.create(name=f"trimmed_{doc.name}", content=doc.content[:100])

        @pipeline_flow(config=PipelineFlowConfig, trace_trim_documents=False)
        async def full_flow(
            project_name: str,
            documents: DocumentList,
            flow_options: FlowOptions,
        ) -> DocumentList:
            # Use the task within the flow
            task_doc = WorkTaskDoc.create(name="task.txt", content="a" * 500)
            trimmed = await trim_task(task_doc)

            outputs = DocumentList()
            for doc in documents:
                output = OutputFlowDoc.create(
                    name=f"output_{doc.name}",
                    content=f"{doc.text}\nTask result: {trimmed.text}",
                )
                outputs.append(output)
            return outputs

        # Test execution
        input_doc = InputFlowDoc.create(name="input.txt", content="Original content")
        docs = DocumentList([input_doc])
        options = FlowOptions()

        result = await full_flow("test", docs, options)

        assert len(result) == 1
        assert "Task result: " in result[0].text
        assert "Original content" in result[0].text
