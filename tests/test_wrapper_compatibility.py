"""Test that our wrappers maintain proper compatibility."""

import inspect
from typing import get_args

from ai_pipeline_core.pipeline import pipeline_flow, pipeline_task
from ai_pipeline_core.prefect import flow, task
from ai_pipeline_core.tracing import TraceLevel, trace


def test_task_wrapper_creates_prefect_task():
    """Test that task wrapper creates a proper Prefect Task."""

    @task
    def my_task(x: int) -> int:
        return x * 2

    # Check it's a Prefect Task
    assert hasattr(my_task, "__wrapped__")
    assert hasattr(my_task, "submit")
    assert hasattr(my_task, "map")


def test_flow_wrapper_creates_prefect_flow():
    """Test that flow wrapper creates a proper Prefect Flow."""

    @flow
    def my_flow(x: int) -> int:
        return x * 2

    # Check it's a Prefect Flow
    assert hasattr(my_flow, "__wrapped__")
    assert hasattr(my_flow, "serve")


async def test_pipeline_task_creates_prefect_task():
    """Test that pipeline_task creates a proper Prefect Task with tracing."""

    @pipeline_task
    async def my_task(x: int) -> int:
        return x * 2

    # Check it's a Prefect Task
    assert hasattr(my_task, "__wrapped__")
    assert hasattr(my_task, "submit")
    assert hasattr(my_task, "map")


# Note: pipeline_flow now enforces document processing signature,
# so simple function tests are moved to test_pipeline_decorators.py


def test_pipeline_flow_with_documents():
    """Test that pipeline_flow creates a proper flow for document processing."""
    from ai_pipeline_core.documents import DocumentList, FlowDocument
    from ai_pipeline_core.flow.config import FlowConfig
    from ai_pipeline_core.flow.options import FlowOptions

    class InputDoc(FlowDocument):
        pass

    class OutputDoc(FlowDocument):
        pass

    class TestConfig(FlowConfig):
        INPUT_DOCUMENT_TYPES = [InputDoc]
        OUTPUT_DOCUMENT_TYPE = OutputDoc

    @pipeline_flow(config=TestConfig)
    async def my_doc_flow(
        project_name: str, documents: DocumentList, flow_options: FlowOptions
    ) -> DocumentList:
        return DocumentList([OutputDoc.create(name="output.txt", content=b"output")])

    # Check it's a Prefect Flow
    assert hasattr(my_doc_flow, "__wrapped__")
    assert hasattr(my_doc_flow, "serve")


def test_trace_with_level():
    """Test trace decorator with level control."""

    @trace(level="debug")
    def func1():
        return 1

    @trace(level="debug")
    def func2():
        return 2

    @trace(level="debug")
    def func3():
        return 3

    # All should execute normally
    assert func1() == 1
    assert func2() == 2
    assert func3() == 3


async def test_pipeline_task_with_trace_params():
    """Test that pipeline_task decorator accepts trace parameters."""

    @pipeline_task(
        trace_level="debug",
        trace_ignore_input=True,
        trace_ignore_output=True,
        trace_ignore_inputs=["password"],
        name="my_task",
    )
    async def my_task(x: int, password: str) -> int:
        return x * 2

    # Should create a valid task
    assert hasattr(my_task, "__wrapped__")


# Note: pipeline_flow trace parameter tests are in test_pipeline_decorators.py
# since pipeline_flow now enforces document processing signature


def test_trace_level_type():
    """Test that TraceLevel is properly typed."""
    assert "always" in get_args(TraceLevel)
    assert "debug" in get_args(TraceLevel)
    assert "off" in get_args(TraceLevel)


def test_wrapper_signatures_include_necessary_params():
    """Test that our wrappers have the necessary parameters."""
    # Test clean wrappers (no trace params)
    task_sig = inspect.signature(task)
    flow_sig = inspect.signature(flow)

    # Check task has key parameters (but no trace params)
    task_params = set(task_sig.parameters.keys())
    assert "name" in task_params
    assert "retries" in task_params
    assert "cache_policy" in task_params
    assert "trace_level" not in task_params  # Clean wrapper should not have trace params

    # Check flow has key parameters (but no trace params)
    flow_params = set(flow_sig.parameters.keys())
    assert "name" in flow_params
    assert "version" in flow_params
    assert "task_runner" in flow_params
    assert "trace_level" not in flow_params  # Clean wrapper should not have trace params

    # Test pipeline wrappers (with trace params)
    pipeline_task_sig = inspect.signature(pipeline_task)
    pipeline_flow_sig = inspect.signature(pipeline_flow)

    # Check pipeline_task has both trace and prefect parameters
    pipeline_task_params = set(pipeline_task_sig.parameters.keys())
    assert "trace_level" in pipeline_task_params
    assert "name" in pipeline_task_params
    assert "retries" in pipeline_task_params
    assert "cache_policy" in pipeline_task_params

    # Check pipeline_flow has both trace and prefect parameters
    pipeline_flow_params = set(pipeline_flow_sig.parameters.keys())
    assert "trace_level" in pipeline_flow_params
    assert "name" in pipeline_flow_params
    assert "version" in pipeline_flow_params
    assert "task_runner" in pipeline_flow_params
