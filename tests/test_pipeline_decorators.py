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
from ai_pipeline_core.flow.config import FlowConfig
from ai_pipeline_core.flow.options import FlowOptions
from ai_pipeline_core.pipeline import (
    pipeline_flow,
    pipeline_task,
)


class TestPipelineTaskDecorator:
    """Test the pipeline_task decorator functionality."""

    async def test_task_bare_decorator(self):
        """Test @pipeline_task without parentheses."""

        @pipeline_task
        async def my_task(x: int) -> int:
            return x * 2

        assert isinstance(my_task, Task)
        assert hasattr(my_task, "submit")
        assert hasattr(my_task, "map")
        result = await my_task(5)
        assert result == 10

    async def test_task_with_parentheses(self):
        """Test @pipeline_task() with parentheses but no arguments."""

        @pipeline_task()
        async def my_task(x: int) -> int:
            return x + 1

        assert isinstance(my_task, Task)
        result = await my_task(5)
        assert result == 6

    async def test_task_with_trace_level_off(self):
        """Test @pipeline_task with trace_level='off'."""

        @pipeline_task(trace_level="off")
        async def my_task(x: int) -> int:
            return x * 3

        assert isinstance(my_task, Task)
        result = await my_task(5)
        assert result == 15

    async def test_task_with_trace_parameters(self):
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
        async def my_task(x: int, secret: str) -> int:
            return x * 2

        assert isinstance(my_task, Task)
        result = await my_task(5, "password")
        assert result == 10

    async def test_task_with_prefect_parameters(self):
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
        async def my_task(x: int) -> int:
            print(f"Processing {x}")
            return x * 2

        assert isinstance(my_task, Task)
        assert my_task.name == "custom_task"
        assert my_task.description == "A test task"
        assert my_task.retries == 3

    async def test_task_with_cache_parameters(self):
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
        async def my_task(x: int) -> int:
            return x * 2

        assert isinstance(my_task, Task)

    async def test_task_with_result_parameters(self):
        """Test @pipeline_task with result storage parameters."""

        @pipeline_task(
            persist_result=True,
            result_storage="local-file-system",
            result_serializer="json",
            result_storage_key="my-result",
        )
        async def my_task(x: int) -> int:
            return x * 2

        assert isinstance(my_task, Task)

    async def test_task_with_retry_parameters(self):
        """Test @pipeline_task with retry parameters."""

        def should_retry(task, task_run, state) -> bool:
            return True

        @pipeline_task(
            retries=3,
            retry_delay_seconds=[1, 2, 3],
            retry_jitter_factor=0.1,
            retry_condition_fn=should_retry,
        )
        async def my_task(x: int) -> int:
            return x * 2

        assert isinstance(my_task, Task)

    async def test_task_with_hooks(self):
        """Test @pipeline_task with state hooks."""

        def on_complete(task, task_run, state):
            print("Task completed")

        def on_fail(task, task_run, state):
            print("Task failed")

        @pipeline_task(on_completion=[on_complete], on_failure=[on_fail])
        async def my_task(x: int) -> int:
            return x * 2

        assert isinstance(my_task, Task)

    async def test_task_with_viz_and_assets(self):
        """Test @pipeline_task with viz_return_value and asset_deps."""

        @pipeline_task(viz_return_value=True, asset_deps=[])
        async def my_task(x: int) -> int:
            return x * 2

        assert isinstance(my_task, Task)

    async def test_task_async_function(self):
        """Test @pipeline_task with async function."""

        @pipeline_task(trace_level="always")
        async def my_async_task(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        assert isinstance(my_async_task, Task)
        result = await my_async_task(5)
        assert result == 10

    async def test_task_with_task_run_name(self):
        """Test @pipeline_task with task_run_name parameter."""

        @pipeline_task(task_run_name="my-task-run")
        async def my_task(x: int) -> int:
            return x

        assert isinstance(my_task, Task)

        @pipeline_task(task_run_name=lambda: "dynamic-name")
        async def my_task2(x: int) -> int:
            return x

        assert isinstance(my_task2, Task)

    async def test_task_with_trace_cost(self):
        """Test @pipeline_task with trace_cost parameter."""
        from unittest.mock import patch

        with patch("ai_pipeline_core.pipeline.set_trace_cost") as mock_set_cost:

            @pipeline_task(trace_cost=0.025)
            async def my_task(x: int) -> int:
                return x * 2

            assert isinstance(my_task, Task)
            result = await my_task(5)
            assert result == 10

            # Verify that set_trace_cost was called with the correct value
            mock_set_cost.assert_called_once_with(0.025)


class TestPipelineFlowDecorator:
    """Test the pipeline_flow decorator functionality."""

    async def test_flow_bare_decorator(self):
        """Test @pipeline_flow without parentheses."""

        class InputDocument(FlowDocument):
            pass

        class OutputDocument(FlowDocument):
            pass

        class TestConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = [InputDocument]
            OUTPUT_DOCUMENT_TYPE = OutputDocument

        @pipeline_flow(config=TestConfig)
        async def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            # Use InputDocument to avoid unused warning
            if InputDocument:
                pass
            return DocumentList([OutputDocument(name="output.txt", content=b"output")])

        assert isinstance(my_flow, Flow)
        docs = DocumentList([InputDocument(name="test.txt", content=b"test")])
        options = FlowOptions()
        result = await my_flow("project", docs, options)
        assert len(result) == 1
        assert result[0].name == "output.txt"
        assert result[0].content == b"output"

    async def test_flow_with_parentheses(self):
        """Test @pipeline_flow() with parentheses."""

        class InputDocument(FlowDocument):
            pass

        class OutputDocument(FlowDocument):
            pass

        class TestConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = [InputDocument]
            OUTPUT_DOCUMENT_TYPE = OutputDocument

        @pipeline_flow(config=TestConfig)
        async def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            return DocumentList([OutputDocument(name="modified.txt", content=b"modified")])

        assert isinstance(my_flow, Flow)
        docs = DocumentList([InputDocument(name="test.txt", content=b"test")])
        options = FlowOptions()
        result = await my_flow("project", docs, options)
        assert len(result) == 1

    async def test_flow_with_trace_parameters(self):
        """Test @pipeline_flow with trace parameters."""

        class InputDocument(FlowDocument):
            pass

        class OutputDocument(FlowDocument):
            pass

        class TestConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = [InputDocument]
            OUTPUT_DOCUMENT_TYPE = OutputDocument

        @pipeline_flow(
            config=TestConfig,
            trace_level="debug",
            trace_ignore_input=True,
            trace_ignore_output=False,
        )
        async def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            # Use InputDocument to avoid unused warning
            if InputDocument:
                pass
            return DocumentList([OutputDocument(name="output.txt", content=b"output")])

        assert isinstance(my_flow, Flow)

    async def test_flow_with_prefect_parameters(self):
        """Test @pipeline_flow with Prefect parameters."""

        class InputDocument(FlowDocument):
            pass

        class OutputDocument(FlowDocument):
            pass

        class TestConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = [InputDocument]
            OUTPUT_DOCUMENT_TYPE = OutputDocument

        @pipeline_flow(
            config=TestConfig,
            name="docs_flow",
            version="1.0",
            description="Process documents",
            retries=1,
            timeout_seconds=600,
        )
        async def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            # Use InputDocument to avoid unused warning
            if InputDocument:
                pass
            return DocumentList([OutputDocument(name="output.txt", content=b"output")])

        assert isinstance(my_flow, Flow)
        assert my_flow.name == "docs_flow"
        assert my_flow.version == "1.0"

    async def test_flow_with_extra_parameters(self):
        """Test @pipeline_flow with additional parameters."""

        class InputDocument(FlowDocument):
            pass

        class OutputDocument(FlowDocument):
            pass

        class TestConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = [InputDocument]
            OUTPUT_DOCUMENT_TYPE = OutputDocument

        @pipeline_flow(config=TestConfig)
        async def my_flow(
            project_name: str,
            documents: DocumentList,
            flow_options: FlowOptions,
        ) -> DocumentList:
            # Test that we can access all three parameters
            assert project_name == "project"
            assert len(documents) == 1
            assert documents[0].name == "test.txt"
            assert flow_options is not None
            # Use InputDocument to avoid unused warning
            if InputDocument:
                pass
            return DocumentList([OutputDocument(name="output.txt", content=b"output")])

        assert isinstance(my_flow, Flow)
        docs = DocumentList([InputDocument(name="test.txt", content=b"test")])
        options = FlowOptions()
        result = await my_flow("project", docs, options)
        assert len(result) == 1
        assert result[0].name == "output.txt"
        assert result[0].content == b"output"

    async def test_flow_async(self):
        """Test @pipeline_flow with async function."""

        class InputDocument(FlowDocument):
            pass

        class OutputDocument(FlowDocument):
            pass

        class TestConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = [InputDocument]
            OUTPUT_DOCUMENT_TYPE = OutputDocument

        @pipeline_flow(config=TestConfig)
        async def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            await asyncio.sleep(0.01)
            # Use InputDocument to avoid unused warning
            if InputDocument:
                pass
            return DocumentList([OutputDocument(name="output.txt", content=b"output")])

        assert isinstance(my_flow, Flow)
        docs = DocumentList([InputDocument(name="test.txt", content=b"test")])
        options = FlowOptions()
        result = await my_flow("project", docs, options)  # type: ignore[arg-type]
        assert len(result) == 1
        assert result[0].name == "output.txt"
        assert result[0].content == b"output"

    async def test_flow_invalid_signature(self):
        """Test @pipeline_flow with invalid signature."""

        class InputDocument(FlowDocument):
            pass

        class OutputDocument(FlowDocument):
            pass

        class TestConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = [InputDocument]
            OUTPUT_DOCUMENT_TYPE = OutputDocument

        with pytest.raises(TypeError, match="must accept"):

            @pipeline_flow(config=TestConfig)  # type: ignore[arg-type]
            async def my_flow(project_name: str) -> DocumentList:  # type: ignore[misc]
                return DocumentList([])

    async def test_flow_invalid_return_type(self):
        """Test @pipeline_flow with invalid return type."""

        class InputDocument(FlowDocument):
            pass

        class OutputDocument(FlowDocument):
            pass

        class TestConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = [InputDocument]
            OUTPUT_DOCUMENT_TYPE = OutputDocument

        @pipeline_flow(config=TestConfig)
        async def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            # Use InputDocument to avoid unused warning
            if InputDocument:
                pass
            return "invalid"  # type: ignore

        docs = DocumentList([InputDocument(name="test.txt", content=b"test")])
        options = FlowOptions()

        with pytest.raises(TypeError, match="must return DocumentList"):
            await my_flow("project", docs, options)

    async def test_flow_with_custom_options(self):
        """Test @pipeline_flow with custom FlowOptions subclass."""

        class InputDocument(FlowDocument):
            pass

        class OutputDocument(FlowDocument):
            pass

        class TestConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = [InputDocument]
            OUTPUT_DOCUMENT_TYPE = OutputDocument

        class CustomOptions(FlowOptions):
            custom_field: str = "custom"

        @pipeline_flow(config=TestConfig)  # type: ignore[arg-type]
        async def my_flow(
            project_name: str, documents: DocumentList, flow_options: CustomOptions
        ) -> DocumentList:
            assert flow_options.custom_field == "custom"
            # Use InputDocument to avoid unused warning
            if InputDocument:
                pass
            return DocumentList([OutputDocument(name="output.txt", content=b"output")])

        assert isinstance(my_flow, Flow)
        docs = DocumentList([InputDocument(name="test.txt", content=b"test")])
        options = CustomOptions()
        result = await my_flow("project", docs, options)
        assert len(result) == 1
        assert result[0].name == "output.txt"
        assert result[0].content == b"output"

    async def test_flow_with_all_parameters(self):
        """Test @pipeline_flow with all possible parameters."""

        class InputDocument(FlowDocument):
            pass

        class OutputDocument(FlowDocument):
            pass

        class TestConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = [InputDocument]
            OUTPUT_DOCUMENT_TYPE = OutputDocument

        def format_input(*args, **kwargs) -> str:
            return "input"

        def format_output(result) -> str:
            return "output"

        def on_complete(flow, flow_run, state):
            pass

        @pipeline_flow(  # type: ignore[arg-type]
            config=TestConfig,
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
        async def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            # Use InputDocument to avoid unused warning
            if InputDocument:
                pass
            return DocumentList([OutputDocument(name="output.txt", content=b"output")])

        assert isinstance(my_flow, Flow)
        assert my_flow.name == "full_flow"

    async def test_flow_with_nested_task(self):
        """Test @pipeline_flow with @pipeline_task inside."""

        @pipeline_task(trace_level="debug")
        async def add_one(x: int) -> int:
            return x + 1

        class InputDocument(FlowDocument):
            pass

        class OutputDocument(FlowDocument):
            pass

        class TestConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = [InputDocument]
            OUTPUT_DOCUMENT_TYPE = OutputDocument

        @pipeline_flow(config=TestConfig, trace_level="always")
        async def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            # Just call the task to test integration
            await add_one(5)
            # Use InputDocument to avoid unused warning
            if InputDocument:
                pass
            return DocumentList([OutputDocument(name="output.txt", content=b"output")])

        assert isinstance(my_flow, Flow)
        docs = DocumentList([InputDocument(name="test.txt", content=b"test")])
        options = FlowOptions()
        result = await my_flow("project", docs, options)
        assert len(result) == 1
        assert result[0].name == "output.txt"
        assert result[0].content == b"output"

    async def test_flow_with_trace_cost(self):
        """Test @pipeline_flow with trace_cost parameter."""
        from unittest.mock import patch

        class InputDocument(FlowDocument):
            pass

        class OutputDocument(FlowDocument):
            pass

        class TestConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = [InputDocument]
            OUTPUT_DOCUMENT_TYPE = OutputDocument

        with patch("ai_pipeline_core.pipeline.set_trace_cost") as mock_set_cost:

            @pipeline_flow(config=TestConfig, trace_cost=0.15)
            async def my_flow(
                project_name: str, documents: DocumentList, flow_options: FlowOptions
            ) -> DocumentList:
                return DocumentList([OutputDocument(name="output.txt", content=b"output")])

            assert isinstance(my_flow, Flow)
            docs = DocumentList([InputDocument(name="test.txt", content=b"test")])
            options = FlowOptions()
            result = await my_flow("project", docs, options)
            assert len(result) == 1
            assert result[0].name == "output.txt"
            assert result[0].content == b"output"

            # Verify that set_trace_cost was called with the correct value
            mock_set_cost.assert_called_once_with(0.15)


class TestSyncFunctionRejection:
    """Test rejection of sync functions with pipeline decorators."""

    def test_sync_function_with_pipeline_task_raises_error(self):
        """Test that @pipeline_task on sync function raises TypeError."""
        from typing import Any, cast

        with pytest.raises(TypeError, match="must be 'async def'"):

            @cast(Any, pipeline_task)
            def sync_task(x: int) -> int:  # pyright: ignore[reportUnusedFunction]
                return x * 2

    def test_sync_function_with_pipeline_task_with_params_raises_error(self):
        """Test that @pipeline_task with params on sync function raises TypeError."""
        from typing import Any, cast

        with pytest.raises(TypeError, match="must be 'async def'"):

            @cast(Any, pipeline_task(retries=3, trace_level="debug"))
            def sync_task(x: int) -> int:  # pyright: ignore[reportUnusedFunction]
                return x * 2

    def test_sync_function_with_pipeline_flow_raises_error(self):
        """Test that @pipeline_flow on sync function raises TypeError."""
        from typing import Any, cast

        class InputDocument(FlowDocument):
            pass

        class OutputDocument(FlowDocument):
            pass

        class TestConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = [InputDocument]
            OUTPUT_DOCUMENT_TYPE = OutputDocument

        with pytest.raises(TypeError, match="must be declared with 'async def'"):

            @cast(Any, pipeline_flow(config=TestConfig))
            def sync_flow(  # pyright: ignore[reportUnusedFunction]
                project_name: str, documents: DocumentList, flow_options: FlowOptions
            ) -> DocumentList:
                return DocumentList([OutputDocument(name="output.txt", content=b"output")])

    def test_sync_function_with_pipeline_flow_with_params_raises_error(self):
        """Test that @pipeline_flow with params on sync function raises TypeError."""
        from typing import Any, cast

        class InputDocument(FlowDocument):
            pass

        class OutputDocument(FlowDocument):
            pass

        class TestConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = [InputDocument]
            OUTPUT_DOCUMENT_TYPE = OutputDocument

        with pytest.raises(TypeError, match="must be declared with 'async def'"):

            @cast(Any, pipeline_flow(config=TestConfig, name="sync_flow", retries=2))
            def sync_flow(  # pyright: ignore[reportUnusedFunction]
                project_name: str, documents: DocumentList, flow_options: FlowOptions
            ) -> DocumentList:
                return DocumentList([OutputDocument(name="output.txt", content=b"output")])


class TestDoubleTracingDetection:
    """Test detection of double tracing with @trace and pipeline decorators."""

    def test_trace_then_pipeline_task_raises_error(self):
        """Test that @trace followed by @pipeline_task raises an error."""
        from ai_pipeline_core import trace

        with pytest.raises(TypeError, match="already decorated.*with @trace"):

            @pipeline_task
            @trace
            async def my_task(x: int) -> int:  # pyright: ignore[reportUnusedFunction]
                return x * 2

    def test_pipeline_task_then_trace_raises_error(self):
        """Test that @pipeline_task followed by @trace raises an error."""
        from ai_pipeline_core import trace

        # When trace is applied after pipeline_task, it should detect the marker
        with pytest.raises(TypeError, match="already decorated with @pipeline_task"):

            @trace
            @pipeline_task
            async def my_task(x: int) -> int:  # pyright: ignore[reportUnusedFunction]
                return x * 2

    def test_trace_then_pipeline_flow_raises_error(self):
        """Test that @trace followed by @pipeline_flow raises an error."""
        from ai_pipeline_core import trace

        class InputDocument(FlowDocument):
            pass

        class OutputDocument(FlowDocument):
            pass

        class TestConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = [InputDocument]
            OUTPUT_DOCUMENT_TYPE = OutputDocument

        with pytest.raises(TypeError, match="already decorated.*with @trace"):

            @pipeline_flow(config=TestConfig)
            @trace
            async def my_flow(  # pyright: ignore[reportUnusedFunction]
                project_name: str, documents: DocumentList, flow_options: FlowOptions
            ) -> DocumentList:
                return DocumentList([OutputDocument(name="output.txt", content=b"output")])

    def test_pipeline_flow_then_trace_raises_error(self):
        """Test that @pipeline_flow followed by @trace raises an error."""
        from ai_pipeline_core import trace

        class InputDocument(FlowDocument):
            pass

        class OutputDocument(FlowDocument):
            pass

        class TestConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = [InputDocument]
            OUTPUT_DOCUMENT_TYPE = OutputDocument

        with pytest.raises(TypeError, match="already decorated with @pipeline"):

            @trace
            @pipeline_flow(config=TestConfig)
            async def my_flow(  # pyright: ignore[reportUnusedFunction]
                project_name: str, documents: DocumentList, flow_options: FlowOptions
            ) -> DocumentList:
                return DocumentList([OutputDocument(name="output.txt", content=b"output")])

    def test_multiple_trace_then_pipeline_task_raises_error(self):
        """Test that multiple @trace decorators followed by @pipeline_task raises an error."""
        from ai_pipeline_core import trace

        with pytest.raises(TypeError, match="already decorated.*with @trace"):

            @pipeline_task
            @trace
            @trace
            async def my_task(x: int) -> int:  # pyright: ignore[reportUnusedFunction]
                return x * 2

    def test_trace_with_params_then_pipeline_task_raises_error(self):
        """Test that @trace with parameters followed by @pipeline_task raises an error."""
        from ai_pipeline_core import trace

        with pytest.raises(TypeError, match="already decorated.*with @trace"):

            @pipeline_task
            @trace(level="always")  # Use "always" to ensure trace is applied
            async def my_task(x: int) -> int:  # pyright: ignore[reportUnusedFunction]
                return x * 2

    async def test_normal_pipeline_task_still_works(self):
        """Test that normal @pipeline_task without @trace still works."""

        @pipeline_task
        async def my_task(x: int) -> int:
            return x * 3

        result = await my_task(5)
        assert result == 15

    async def test_normal_pipeline_flow_still_works(self):
        """Test that normal @pipeline_flow without @trace still works."""

        class InputDoc(FlowDocument):
            pass

        class OutputDoc(FlowDocument):
            pass

        class TestConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = [InputDoc]
            OUTPUT_DOCUMENT_TYPE = OutputDoc

        @pipeline_flow(config=TestConfig)
        async def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            return DocumentList([OutputDoc(name="output.txt", content=b"output")])

        docs = DocumentList([InputDoc(name="test.txt", content=b"test")])
        options = FlowOptions()
        result = await my_flow("project", docs, options)
        assert len(result) == 1
        assert result[0].name == "output.txt"
        assert result[0].content == b"output"
