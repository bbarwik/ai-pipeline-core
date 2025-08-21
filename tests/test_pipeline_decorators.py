"""Comprehensive tests for pipeline.py decorators with 100% coverage."""

import asyncio
import datetime
from typing import Any

import pytest
from prefect.cache_policies import NONE
from prefect.context import TaskRunContext
from prefect.flows import Flow
from prefect.tasks import Task

from ai_pipeline_core.documents import DocumentList, FlowDocument
from ai_pipeline_core.flow.options import FlowOptions
from ai_pipeline_core.pipeline import (
    pipeline_flow,
    pipeline_task,
)


class TestPipelineTaskDecorator:
    """Test the pipeline_task decorator functionality."""

    def test_task_bare_decorator(self):
        """Test @pipeline_task without parentheses."""

        @pipeline_task
        def my_task(x: int) -> int:
            return x * 2

        assert isinstance(my_task, Task)
        assert hasattr(my_task, "submit")
        assert hasattr(my_task, "map")
        result = my_task(5)
        assert result == 10

    def test_task_with_parentheses(self):
        """Test @pipeline_task() with parentheses but no arguments."""

        @pipeline_task()
        def my_task(x: int) -> int:
            return x + 1

        assert isinstance(my_task, Task)
        result = my_task(5)
        assert result == 6

    def test_task_with_trace_level_off(self):
        """Test @pipeline_task with trace_level='off'."""

        @pipeline_task(trace_level="off")
        def my_task(x: int) -> int:
            return x * 3

        assert isinstance(my_task, Task)
        result = my_task(5)
        assert result == 15

    def test_task_with_trace_parameters(self):
        """Test @pipeline_task with various trace parameters."""

        def format_input(*args, **kwargs) -> str:
            return f"Input: {args}"

        def format_output(result) -> str:
            return f"Output: {result}"

        @pipeline_task(
            trace_level="debug",
            trace_ignore_input=True,
            trace_ignore_output=True,
            trace_ignore_inputs=["secret"],
            trace_input_formatter=format_input,
            trace_output_formatter=format_output,
        )
        def my_task(x: int, secret: str) -> int:
            return x * 2

        assert isinstance(my_task, Task)
        result = my_task(5, "password")
        assert result == 10

    def test_task_with_prefect_parameters(self):
        """Test @pipeline_task with Prefect parameters."""

        @pipeline_task(
            name="custom_task",
            description="A test task",
            tags=["test", "pipeline"],
            version="1.0.0",
            retries=3,
            timeout_seconds=60,
            log_prints=True,
        )
        def my_task(x: int) -> int:
            print(f"Processing {x}")
            return x * 2

        assert isinstance(my_task, Task)
        assert my_task.name == "custom_task"
        assert my_task.description == "A test task"
        assert my_task.retries == 3

    def test_task_with_cache_parameters(self):
        """Test @pipeline_task with cache-related parameters."""

        def cache_key(context: TaskRunContext, parameters: dict[str, Any]) -> str:
            return f"key-{parameters.get('x', 0)}"

        @pipeline_task(
            cache_policy=NONE,
            cache_key_fn=cache_key,
            cache_expiration=datetime.timedelta(minutes=10),
            refresh_cache=True,
            cache_result_in_memory=False,
        )
        def my_task(x: int) -> int:
            return x * 2

        assert isinstance(my_task, Task)

    def test_task_with_result_parameters(self):
        """Test @pipeline_task with result storage parameters."""

        @pipeline_task(
            persist_result=True,
            result_storage="local-file-system",
            result_serializer="json",
            result_storage_key="my-result",
        )
        def my_task(x: int) -> int:
            return x * 2

        assert isinstance(my_task, Task)

    def test_task_with_retry_parameters(self):
        """Test @pipeline_task with retry parameters."""

        def should_retry(task, task_run, state) -> bool:
            return True

        @pipeline_task(
            retries=3,
            retry_delay_seconds=[1, 2, 3],
            retry_jitter_factor=0.1,
            retry_condition_fn=should_retry,
        )
        def my_task(x: int) -> int:
            return x * 2

        assert isinstance(my_task, Task)

    def test_task_with_hooks(self):
        """Test @pipeline_task with state hooks."""

        def on_complete(task, task_run, state):
            print("Task completed")

        def on_fail(task, task_run, state):
            print("Task failed")

        @pipeline_task(on_completion=[on_complete], on_failure=[on_fail])
        def my_task(x: int) -> int:
            return x * 2

        assert isinstance(my_task, Task)

    def test_task_with_viz_and_assets(self):
        """Test @pipeline_task with viz_return_value and asset_deps."""

        @pipeline_task(viz_return_value=True, asset_deps=[])
        def my_task(x: int) -> int:
            return x * 2

        assert isinstance(my_task, Task)

    def test_task_async_function(self):
        """Test @pipeline_task with async function."""

        @pipeline_task(trace_level="always")
        async def my_async_task(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        assert isinstance(my_async_task, Task)
        result = asyncio.run(my_async_task(5))
        assert result == 10

    def test_task_with_task_run_name(self):
        """Test @pipeline_task with task_run_name parameter."""

        @pipeline_task(task_run_name="my-task-run")
        def my_task(x: int) -> int:
            return x

        assert isinstance(my_task, Task)

        @pipeline_task(task_run_name=lambda: "dynamic-name")
        def my_task2(x: int) -> int:
            return x

        assert isinstance(my_task2, Task)


class TestPipelineFlowDecorator:
    """Test the pipeline_flow decorator functionality."""

    def test_flow_bare_decorator(self):
        """Test @pipeline_flow without parentheses."""

        class TestDocument(FlowDocument):
            pass

        @pipeline_flow
        def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            # Use TestDocument to avoid unused warning
            if TestDocument:
                pass
            return documents

        assert isinstance(my_flow, Flow)
        docs = DocumentList([TestDocument(name="test.txt", content=b"test")])
        options = FlowOptions()
        result = my_flow("project", docs, options)
        assert result == docs

    def test_flow_with_parentheses(self):
        """Test @pipeline_flow() with parentheses."""

        class TestDocument(FlowDocument):
            pass

        @pipeline_flow()
        def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            return DocumentList([TestDocument(name="modified.txt", content=b"modified")])

        assert isinstance(my_flow, Flow)
        docs = DocumentList([TestDocument(name="test.txt", content=b"test")])
        options = FlowOptions()
        result = my_flow("project", docs, options)
        assert len(result) == 1

    def test_flow_with_trace_parameters(self):
        """Test @pipeline_flow with trace parameters."""

        class TestDocument(FlowDocument):
            pass

        @pipeline_flow(
            trace_level="debug",
            trace_ignore_input=True,
            trace_ignore_output=False,
        )
        def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            # Use TestDocument to avoid unused warning
            if TestDocument:
                pass
            return documents

        assert isinstance(my_flow, Flow)

    def test_flow_with_prefect_parameters(self):
        """Test @pipeline_flow with Prefect parameters."""

        class TestDocument(FlowDocument):
            pass

        @pipeline_flow(
            name="docs_flow",
            version="1.0",
            description="Process documents",
            retries=1,
            timeout_seconds=600,
        )
        def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            # Use TestDocument to avoid unused warning
            if TestDocument:
                pass
            return documents

        assert isinstance(my_flow, Flow)
        assert my_flow.name == "docs_flow"
        assert my_flow.version == "1.0"

    def test_flow_with_extra_parameters(self):
        """Test @pipeline_flow with additional parameters."""

        class TestDocument(FlowDocument):
            pass

        @pipeline_flow
        def my_flow(
            project_name: str,
            documents: DocumentList,
            flow_options: FlowOptions,
            extra_param: int = 10,
        ) -> DocumentList:
            # Use TestDocument to avoid unused warning
            if TestDocument:
                pass
            return documents

        assert isinstance(my_flow, Flow)
        docs = DocumentList([TestDocument(name="test.txt", content=b"test")])
        options = FlowOptions()
        result = my_flow("project", docs, options, 20)
        assert result == docs

    def test_flow_async(self):
        """Test @pipeline_flow with async function."""

        class TestDocument(FlowDocument):
            pass

        @pipeline_flow
        async def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            await asyncio.sleep(0.01)
            # Use TestDocument to avoid unused warning
            if TestDocument:
                pass
            return documents

        assert isinstance(my_flow, Flow)
        docs = DocumentList([TestDocument(name="test.txt", content=b"test")])
        options = FlowOptions()
        result = asyncio.run(my_flow("project", docs, options))  # type: ignore[arg-type]
        assert result == docs

    def test_flow_invalid_signature(self):
        """Test @pipeline_flow with invalid signature."""

        with pytest.raises(TypeError, match="must accept at least 3 arguments"):

            @pipeline_flow  # type: ignore[arg-type]
            def my_flow(project_name: str) -> DocumentList:  # type: ignore[misc]
                return DocumentList([])

    def test_flow_invalid_return_type(self):
        """Test @pipeline_flow with invalid return type."""

        class TestDocument(FlowDocument):
            pass

        @pipeline_flow
        def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            # Use TestDocument to avoid unused warning
            if TestDocument:
                pass
            return "invalid"  # type: ignore

        docs = DocumentList([TestDocument(name="test.txt", content=b"test")])
        options = FlowOptions()

        with pytest.raises(TypeError, match="must return a DocumentList"):
            my_flow("project", docs, options)

    def test_flow_with_custom_options(self):
        """Test @pipeline_flow with custom FlowOptions subclass."""

        class TestDocument(FlowDocument):
            pass

        class CustomOptions(FlowOptions):
            custom_field: str = "custom"

        @pipeline_flow  # type: ignore[arg-type]
        def my_flow(
            project_name: str, documents: DocumentList, flow_options: CustomOptions
        ) -> DocumentList:
            assert flow_options.custom_field == "custom"
            # Use TestDocument to avoid unused warning
            if TestDocument:
                pass
            return documents

        assert isinstance(my_flow, Flow)
        docs = DocumentList([TestDocument(name="test.txt", content=b"test")])
        options = CustomOptions()
        result = my_flow("project", docs, options)
        assert result == docs

    def test_flow_parameter_name_warning(self, capsys):
        """Test @pipeline_flow warns about non-standard parameter names."""

        class TestDocument(FlowDocument):
            pass

        @pipeline_flow
        def my_flow(
            proj: str,  # Non-standard name
            docs: DocumentList,  # Non-standard name
            opts: FlowOptions,  # Non-standard name
        ) -> DocumentList:
            # Use TestDocument to avoid unused warning
            if TestDocument:
                pass
            return docs

        # Assert flow was created successfully
        assert isinstance(my_flow, Flow)

        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "convention suggests 'project_name'" in captured.out

    def test_flow_with_all_parameters(self):
        """Test @pipeline_flow with all possible parameters."""

        class TestDocument(FlowDocument):
            pass

        def format_input(*args, **kwargs) -> str:
            return "input"

        def format_output(result) -> str:
            return "output"

        def on_complete(flow, flow_run, state):
            pass

        @pipeline_flow(  # type: ignore[arg-type]
            trace_level="off",
            trace_ignore_input=True,
            trace_ignore_output=True,
            trace_ignore_inputs=["secret"],
            trace_input_formatter=format_input,
            trace_output_formatter=format_output,
            name="full_flow",
            version="3.0",
            flow_run_name="test-run",
            retries=2,
            retry_delay_seconds=5.5,
            description="Full test",
            timeout_seconds=120.5,
            validate_parameters=False,
            persist_result=True,
            result_storage="local-file-system",
            result_serializer="json",
            cache_result_in_memory=False,
            log_prints=True,
            on_completion=[on_complete],
        )
        def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            # Use TestDocument to avoid unused warning
            if TestDocument:
                pass
            return documents

        assert isinstance(my_flow, Flow)
        assert my_flow.name == "full_flow"

    def test_flow_with_nested_task(self):
        """Test @pipeline_flow with @pipeline_task inside."""

        @pipeline_task(trace_level="debug")
        def add_one(x: int) -> int:
            return x + 1

        class TestDocument(FlowDocument):
            pass

        @pipeline_flow(trace_level="always")
        def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            # Just call the task to test integration
            add_one(5)
            # Use TestDocument to avoid unused warning
            if TestDocument:
                pass
            return documents

        assert isinstance(my_flow, Flow)
        docs = DocumentList([TestDocument(name="test.txt", content=b"test")])
        options = FlowOptions()
        result = my_flow("project", docs, options)
        assert result == docs
