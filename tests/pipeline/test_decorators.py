"""Comprehensive tests for pipeline.py decorators."""

import asyncio
import datetime
from typing import Any

import pytest
from prefect.cache_policies import NONE
from prefect.context import TaskRunContext
from prefect.flows import Flow
from prefect.tasks import Task
from pydantic import BaseModel

from ai_pipeline_core.document_store import set_document_store
from ai_pipeline_core.document_store.memory import MemoryDocumentStore
from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents.context import RunContext, get_run_context, reset_run_context, set_run_context
from ai_pipeline_core.pipeline import (
    pipeline_flow,
    pipeline_task,
)
from ai_pipeline_core.pipeline.decorators import (
    _extract_documents,
    _flatten_union,
    _parse_document_types_from_annotation,
)
from ai_pipeline_core.pipeline.options import FlowOptions


class InputDocument(Document):
    """Input document for flow testing."""


class OutputDocument(Document):
    """Output document for flow testing."""


class AltInputDocument(Document):
    """Alternative input document for union type testing."""


# --------------------------------------------------------------------------- #
# Annotation parsing tests
# --------------------------------------------------------------------------- #
class TestAnnotationParsing:
    """Test annotation extraction from type hints."""

    def test_single_type(self):
        parsed = _parse_document_types_from_annotation(list[InputDocument])
        assert parsed == [InputDocument]

    def test_pipe_union(self):
        parsed = _parse_document_types_from_annotation(list[InputDocument | AltInputDocument])
        assert set(parsed) == {InputDocument, AltInputDocument}

    def test_typing_union(self):
        parsed = _parse_document_types_from_annotation(list[InputDocument | AltInputDocument])
        assert set(parsed) == {InputDocument, AltInputDocument}

    def test_base_document(self):
        parsed = _parse_document_types_from_annotation(list[Document])
        assert parsed == [Document]

    def test_non_list_returns_empty(self):
        parsed = _parse_document_types_from_annotation(dict[str, InputDocument])
        assert parsed == []

    def test_plain_list_returns_empty(self):
        parsed = _parse_document_types_from_annotation(list)
        assert parsed == []

    def test_non_document_types_ignored(self):
        # list[str] has no Document subclasses
        parsed = _parse_document_types_from_annotation(list[str])
        assert parsed == []

    def test_flatten_union_simple(self):
        result = _flatten_union(InputDocument)
        assert result == [InputDocument]

    def test_flatten_union_pipe(self):
        result = _flatten_union(InputDocument | AltInputDocument)
        assert set(result) == {InputDocument, AltInputDocument}


# --------------------------------------------------------------------------- #
# Document extraction tests
# --------------------------------------------------------------------------- #
class TestDocumentExtraction:
    """Test _extract_documents recursive walker."""

    def test_single_document(self):
        doc = InputDocument(name="a.txt", content=b"a")
        assert _extract_documents(doc) == [doc]

    def test_list_of_documents(self):
        d1 = InputDocument(name="a.txt", content=b"a")
        d2 = OutputDocument(name="b.txt", content=b"b")
        result = _extract_documents([d1, d2])
        assert result == [d1, d2]

    def test_tuple_of_documents(self):
        d1 = InputDocument(name="a.txt", content=b"a")
        result = _extract_documents((d1,))
        assert result == [d1]

    def test_dict_values(self):
        d1 = InputDocument(name="a.txt", content=b"a")
        result = _extract_documents({"key": d1})
        assert result == [d1]

    def test_pydantic_model_fields(self):
        d1 = InputDocument(name="a.txt", content=b"a")

        class Result(BaseModel):
            model_config = {"arbitrary_types_allowed": True}
            report: InputDocument
            count: int = 5

        r = Result(report=d1)
        result = _extract_documents(r)
        assert result == [d1]

    def test_nested_structures(self):
        d1 = InputDocument(name="a.txt", content=b"a")
        d2 = OutputDocument(name="b.txt", content=b"b")
        result = _extract_documents({"items": [d1, (d2,)]})
        assert set(id(d) for d in result) == {id(d1), id(d2)}

    def test_non_document_returns_empty(self):
        assert _extract_documents("hello") == []
        assert _extract_documents(42) == []
        assert _extract_documents(None) == []

    def test_mixed_types(self):
        d1 = InputDocument(name="a.txt", content=b"a")
        result = _extract_documents([d1, "string", 42, None])
        assert result == [d1]


# --------------------------------------------------------------------------- #
# Estimated minutes tests
# --------------------------------------------------------------------------- #
class TestEstimatedMinutes:
    """Test estimated_minutes parameter."""

    def test_task_stores_estimated_minutes(self):
        @pipeline_task(estimated_minutes=5, persist=False)
        async def my_task(x: int) -> int:
            return x

        assert my_task.estimated_minutes == 5  # type: ignore[attr-defined]

    def test_task_default_estimated_minutes(self):
        @pipeline_task(persist=False)
        async def my_task(x: int) -> int:
            return x

        assert my_task.estimated_minutes == 1  # type: ignore[attr-defined]

    def test_flow_stores_estimated_minutes(self):
        @pipeline_flow(estimated_minutes=30)
        async def my_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
            return []

        assert my_flow.estimated_minutes == 30  # type: ignore[attr-defined]

    def test_flow_default_estimated_minutes(self):
        @pipeline_flow()
        async def my_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
            return []

        assert my_flow.estimated_minutes == 1  # type: ignore[attr-defined]

    def test_task_rejects_zero(self):
        with pytest.raises(ValueError, match="estimated_minutes must be >= 1"):

            @pipeline_task(estimated_minutes=0, persist=False)
            async def my_task(x: int) -> int:
                return x

    def test_flow_rejects_zero(self):
        with pytest.raises(ValueError, match="estimated_minutes must be >= 1"):

            @pipeline_flow(estimated_minutes=0)
            async def my_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
                return []


# --------------------------------------------------------------------------- #
# Flow annotation extraction tests
# --------------------------------------------------------------------------- #
class TestFlowAnnotationExtraction:
    """Test that @pipeline_flow extracts document types from annotations."""

    def test_extracts_input_types(self):
        @pipeline_flow()
        async def my_flow(project_name: str, documents: list[InputDocument], flow_options: FlowOptions) -> list[OutputDocument]:
            return []

        assert my_flow.input_document_types == [InputDocument]  # type: ignore[attr-defined]

    def test_extracts_output_types(self):
        @pipeline_flow()
        async def my_flow(project_name: str, documents: list[InputDocument], flow_options: FlowOptions) -> list[OutputDocument]:
            return []

        assert my_flow.output_document_types == [OutputDocument]  # type: ignore[attr-defined]

    def test_extracts_union_input_types(self):
        @pipeline_flow()
        async def my_flow(project_name: str, documents: list[InputDocument | AltInputDocument], flow_options: FlowOptions) -> list[OutputDocument]:
            return []

        assert set(my_flow.input_document_types) == {InputDocument, AltInputDocument}  # type: ignore[attr-defined]

    def test_base_document_annotation(self):
        @pipeline_flow()
        async def my_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
            return []

        assert my_flow.input_document_types == [Document]  # type: ignore[attr-defined]
        assert my_flow.output_document_types == [Document]  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Document auto-save tests
# --------------------------------------------------------------------------- #
class TestDocumentAutoSave:
    """Test document persistence via @pipeline_task."""

    async def test_documents_saved_to_store(self):
        store = MemoryDocumentStore()
        set_document_store(store)
        token = set_run_context(RunContext(run_scope="test/flow"))
        try:

            @pipeline_task(persist=True)
            async def my_task() -> list[Document]:
                return [OutputDocument(name="out.txt", content=b"output")]

            result = await my_task()
            assert len(result) == 1

            loaded = await store.load("test/flow", [OutputDocument])
            assert len(loaded) == 1
            assert loaded[0].name == "out.txt"
        finally:
            reset_run_context(token)
            set_document_store(None)

    async def test_persist_false_skips_save(self):
        store = MemoryDocumentStore()
        set_document_store(store)
        token = set_run_context(RunContext(run_scope="test/flow"))
        try:

            @pipeline_task(persist=False)
            async def my_task() -> list[Document]:
                return [OutputDocument(name="out.txt", content=b"output")]

            await my_task()
            loaded = await store.load("test/flow", [OutputDocument])
            assert len(loaded) == 0
        finally:
            reset_run_context(token)
            set_document_store(None)

    async def test_no_store_skips_save(self):
        set_document_store(None)
        token = set_run_context(RunContext(run_scope="test/flow"))
        try:

            @pipeline_task
            async def my_task() -> list[Document]:
                return [OutputDocument(name="out.txt", content=b"output")]

            result = await my_task()
            assert len(result) == 1  # task still works
        finally:
            reset_run_context(token)

    async def test_no_run_context_skips_save(self):
        store = MemoryDocumentStore()
        set_document_store(store)
        try:

            @pipeline_task
            async def my_task() -> list[Document]:
                return [OutputDocument(name="out.txt", content=b"output")]

            result = await my_task()
            assert len(result) == 1
            loaded = await store.load("any_scope", [OutputDocument])
            assert len(loaded) == 0
        finally:
            set_document_store(None)

    async def test_single_document_return_saved(self):
        store = MemoryDocumentStore()
        set_document_store(store)
        token = set_run_context(RunContext(run_scope="test/flow"))
        try:

            @pipeline_task
            async def my_task() -> Document:
                return OutputDocument(name="single.txt", content=b"data")

            await my_task()
            loaded = await store.load("test/flow", [OutputDocument])
            assert len(loaded) == 1
        finally:
            reset_run_context(token)
            set_document_store(None)


# --------------------------------------------------------------------------- #
# Flow RunContext tests
# --------------------------------------------------------------------------- #
class TestFlowRunContext:
    """Test that @pipeline_flow sets RunContext."""

    async def test_flow_sets_run_context(self):
        captured_scope: list[str] = []

        @pipeline_flow()
        async def my_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
            ctx = get_run_context()
            assert ctx is not None
            captured_scope.append(ctx.run_scope)
            return []

        await my_flow("myproject", [], FlowOptions())
        assert captured_scope == ["myproject/my_flow"]

    async def test_run_context_cleared_after_flow(self):
        @pipeline_flow()
        async def my_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
            return []

        await my_flow("myproject", [], FlowOptions())
        assert get_run_context() is None

    async def test_nested_flow_preserves_outer_context(self):
        """When outer RunContext exists (set by deployment), flow defers to it."""
        outer_token = set_run_context(RunContext(run_scope="outer/scope"))
        try:
            captured: list[str] = []

            @pipeline_flow()
            async def inner_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
                ctx = get_run_context()
                assert ctx is not None
                captured.append(ctx.run_scope)
                return []

            await inner_flow("inner", [], FlowOptions())
            # Flow should see the outer (deployment-level) context
            assert captured == ["outer/scope"]
            # Outer context must still be intact
            assert get_run_context() is not None
            assert get_run_context().run_scope == "outer/scope"  # type: ignore[union-attr]
        finally:
            reset_run_context(outer_token)

    async def test_run_scope_uses_name_override(self):
        """When name= is provided, run_scope should use it instead of function name."""
        captured_scope: list[str] = []

        @pipeline_flow(name="custom_name")
        async def my_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
            ctx = get_run_context()
            assert ctx is not None
            captured_scope.append(ctx.run_scope)
            return []

        await my_flow("proj", [], FlowOptions())
        assert captured_scope == ["proj/custom_name"]

    async def test_run_context_cleared_on_exception(self):
        """RunContext should be cleared even if the flow raises."""

        @pipeline_flow()
        async def failing_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await failing_flow("proj", [], FlowOptions())

        assert get_run_context() is None


# --------------------------------------------------------------------------- #
# Persistence failure graceful degradation
# --------------------------------------------------------------------------- #
class TestPersistenceGracefulDegradation:
    """Test that persistence failures don't crash tasks."""

    async def test_store_save_failure_logs_warning(self):
        """A store that raises on save_batch should not crash the task."""
        from unittest.mock import AsyncMock, MagicMock

        broken_store = MagicMock()
        broken_store.save_batch = AsyncMock(side_effect=RuntimeError("store broken"))
        broken_store.check_existing = AsyncMock(return_value=set())
        set_document_store(broken_store)
        token = set_run_context(RunContext(run_scope="test/flow"))
        try:

            @pipeline_task(persist=True)
            async def my_task() -> list[Document]:
                return [OutputDocument(name="out.txt", content=b"output")]

            # Should not raise
            result = await my_task()
            assert len(result) == 1
        finally:
            reset_run_context(token)
            set_document_store(None)


# --------------------------------------------------------------------------- #
# Document extraction edge cases
# --------------------------------------------------------------------------- #
class TestDocumentExtractionEdgeCases:
    """Additional edge cases for _extract_documents."""

    def test_duplicate_instance_deduplication(self):
        """Same instance appearing multiple times is collected only once."""
        doc = InputDocument(name="a.txt", content=b"a")
        result = _extract_documents([doc, doc, doc])
        assert len(result) == 1
        assert result[0] is doc

    def test_empty_structures(self):
        assert _extract_documents([]) == []
        assert _extract_documents({}) == []
        assert _extract_documents(()) == []


# --------------------------------------------------------------------------- #
# Original test cases (preserved from previous version)
# --------------------------------------------------------------------------- #
class TestPipelineTaskDecorator:
    """Test the pipeline_task decorator functionality."""

    async def test_task_bare_decorator(self):
        @pipeline_task(persist=False)
        async def my_task(x: int) -> int:
            return x * 2

        assert isinstance(my_task, Task)
        assert hasattr(my_task, "submit")
        assert hasattr(my_task, "map")
        result = await my_task(5)
        assert result == 10

    async def test_task_with_parentheses(self):
        @pipeline_task(persist=False)
        async def my_task(x: int) -> int:
            return x + 1

        assert isinstance(my_task, Task)
        result = await my_task(5)
        assert result == 6

    async def test_task_with_trace_level_off(self):
        @pipeline_task(trace_level="off", persist=False)
        async def my_task(x: int) -> int:
            return x * 3

        assert isinstance(my_task, Task)
        result = await my_task(5)
        assert result == 15

    async def test_task_with_trace_parameters(self):
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
            persist=False,
        )
        async def my_task(x: int, secret: str) -> int:
            return x * 2

        assert isinstance(my_task, Task)
        result = await my_task(5, "password")
        assert result == 10

    async def test_task_with_prefect_parameters(self):
        @pipeline_task(
            name="custom_task",
            description="A test task",
            tags=["test", "pipeline"],
            version="1.0.0",
            retries=3,
            timeout_seconds=60,
            log_prints=True,
            persist=False,
        )
        async def my_task(x: int) -> int:
            print(f"Processing {x}")
            return x * 2

        assert isinstance(my_task, Task)
        assert my_task.name == "custom_task"
        assert my_task.description == "A test task"
        assert my_task.retries == 3

    async def test_task_with_cache_parameters(self):
        def cache_key(context: TaskRunContext, parameters: dict[str, Any]) -> str:
            return f"key-{parameters.get('x', 0)}"

        @pipeline_task(
            cache_policy=NONE,
            cache_key_fn=cache_key,
            cache_expiration=datetime.timedelta(minutes=10),
            refresh_cache=True,
            cache_result_in_memory=False,
            persist=False,
        )
        async def my_task(x: int) -> int:
            return x * 2

        assert isinstance(my_task, Task)

    async def test_task_with_result_parameters(self):
        @pipeline_task(
            persist_result=True,
            result_storage="local-file-system",
            result_serializer="json",
            result_storage_key="my-result",
            persist=False,
        )
        async def my_task(x: int) -> int:
            return x * 2

        assert isinstance(my_task, Task)

    async def test_task_with_retry_parameters(self):
        def should_retry(task, task_run, state) -> bool:
            return True

        @pipeline_task(
            retries=3,
            retry_delay_seconds=[1, 2, 3],
            retry_jitter_factor=0.1,
            retry_condition_fn=should_retry,
            persist=False,
        )
        async def my_task(x: int) -> int:
            return x * 2

        assert isinstance(my_task, Task)

    async def test_task_with_hooks(self):
        def on_complete(task, task_run, state):
            print("Task completed")

        def on_fail(task, task_run, state):
            print("Task failed")

        @pipeline_task(on_completion=[on_complete], on_failure=[on_fail], persist=False)
        async def my_task(x: int) -> int:
            return x * 2

        assert isinstance(my_task, Task)

    async def test_task_with_viz_and_assets(self):
        @pipeline_task(viz_return_value=True, asset_deps=[], persist=False)
        async def my_task(x: int) -> int:
            return x * 2

        assert isinstance(my_task, Task)

    async def test_task_async_function(self):
        @pipeline_task(trace_level="always", persist=False)
        async def my_async_task(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        assert isinstance(my_async_task, Task)
        result = await my_async_task(5)
        assert result == 10

    async def test_task_with_task_run_name(self):
        @pipeline_task(task_run_name="my-task-run", persist=False)
        async def my_task(x: int) -> int:
            return x

        assert isinstance(my_task, Task)

        @pipeline_task(task_run_name=lambda: "dynamic-name", persist=False)
        async def my_task2(x: int) -> int:
            return x

        assert isinstance(my_task2, Task)

    async def test_task_with_trace_cost(self):
        from unittest.mock import patch

        with patch("ai_pipeline_core.pipeline.decorators.set_trace_cost") as mock_set_cost:

            @pipeline_task(trace_cost=0.025, persist=False)
            async def my_task(x: int) -> int:
                return x * 2

            assert isinstance(my_task, Task)
            result = await my_task(5)
            assert result == 10
            mock_set_cost.assert_called_once_with(0.025)


class TestPipelineFlowDecorator:
    """Test the pipeline_flow decorator functionality."""

    async def test_flow_bare_decorator(self):
        @pipeline_flow()
        async def my_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
            return list([OutputDocument(name="output.txt", content=b"output")])

        assert isinstance(my_flow, Flow)
        docs = list([InputDocument(name="test.txt", content=b"test")])
        options = FlowOptions()
        result = await my_flow("project", docs, options)
        assert len(result) == 1
        assert result[0].name == "output.txt"
        assert result[0].content == b"output"

    async def test_flow_with_parentheses(self):
        @pipeline_flow()
        async def my_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
            return list([OutputDocument(name="modified.txt", content=b"modified")])

        assert isinstance(my_flow, Flow)
        docs = list([InputDocument(name="test.txt", content=b"test")])
        options = FlowOptions()
        result = await my_flow("project", docs, options)
        assert len(result) == 1

    async def test_flow_with_trace_parameters(self):
        @pipeline_flow(
            trace_level="debug",
            trace_ignore_input=True,
            trace_ignore_output=False,
        )
        async def my_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
            return list([OutputDocument(name="output.txt", content=b"output")])

        assert isinstance(my_flow, Flow)

    async def test_flow_with_prefect_parameters(self):
        @pipeline_flow(
            name="docs_flow",
            version="1.0",
            description="Process documents",
            retries=1,
            timeout_seconds=600,
        )
        async def my_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
            return list([OutputDocument(name="output.txt", content=b"output")])

        assert isinstance(my_flow, Flow)
        assert my_flow.name == "docs_flow"
        assert my_flow.version == "1.0"

    async def test_flow_with_extra_parameters(self):
        @pipeline_flow()
        async def my_flow(
            project_name: str,
            documents: list[Document],
            flow_options: FlowOptions,
        ) -> list[Document]:
            assert project_name == "project"
            assert len(documents) == 1
            assert documents[0].name == "test.txt"
            assert flow_options is not None
            return list([OutputDocument(name="output.txt", content=b"output")])

        assert isinstance(my_flow, Flow)
        docs = list([InputDocument(name="test.txt", content=b"test")])
        options = FlowOptions()
        result = await my_flow("project", docs, options)
        assert len(result) == 1
        assert result[0].name == "output.txt"
        assert result[0].content == b"output"

    async def test_flow_async(self):
        @pipeline_flow()
        async def my_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
            await asyncio.sleep(0.01)
            return list([OutputDocument(name="output.txt", content=b"output")])

        assert isinstance(my_flow, Flow)
        docs = list([InputDocument(name="test.txt", content=b"test")])
        options = FlowOptions()
        result = await my_flow("project", docs, options)
        assert len(result) == 1
        assert result[0].name == "output.txt"
        assert result[0].content == b"output"

    async def test_flow_invalid_signature(self):
        with pytest.raises(TypeError, match="must have exactly 3 parameters"):

            @pipeline_flow()  # type: ignore[arg-type]
            async def my_flow(project_name: str) -> list[Document]:  # type: ignore[misc]
                return list([])

    async def test_flow_invalid_return_type(self):
        @pipeline_flow()
        async def my_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
            return "invalid"  # type: ignore

        docs = list([InputDocument(name="test.txt", content=b"test")])
        options = FlowOptions()

        with pytest.raises(TypeError, match=r"must return list\[Document\]"):
            await my_flow("project", docs, options)

    async def test_flow_with_custom_options(self):
        class CustomOptions(FlowOptions):
            custom_field: str = "custom"

        @pipeline_flow()  # type: ignore[arg-type]
        async def my_flow(project_name: str, documents: list[Document], flow_options: CustomOptions) -> list[Document]:
            assert flow_options.custom_field == "custom"
            return list([OutputDocument(name="output.txt", content=b"output")])

        assert isinstance(my_flow, Flow)
        docs = list([InputDocument(name="test.txt", content=b"test")])
        options = CustomOptions()
        result = await my_flow("project", docs, options)
        assert len(result) == 1
        assert result[0].name == "output.txt"
        assert result[0].content == b"output"

    async def test_flow_with_all_parameters(self):
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
        async def my_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
            return list([OutputDocument(name="output.txt", content=b"output")])

        assert isinstance(my_flow, Flow)
        assert my_flow.name == "full_flow"

    async def test_flow_with_nested_task(self):
        @pipeline_task(trace_level="debug", persist=False)
        async def add_one(x: int) -> int:
            return x + 1

        @pipeline_flow(trace_level="always")
        async def my_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
            await add_one(5)
            return list([OutputDocument(name="output.txt", content=b"output")])

        assert isinstance(my_flow, Flow)
        docs = list([InputDocument(name="test.txt", content=b"test")])
        options = FlowOptions()
        result = await my_flow("project", docs, options)
        assert len(result) == 1
        assert result[0].name == "output.txt"
        assert result[0].content == b"output"

    async def test_flow_with_trace_cost(self):
        from unittest.mock import patch

        with patch("ai_pipeline_core.pipeline.decorators.set_trace_cost") as mock_set_cost:

            @pipeline_flow(trace_cost=0.15)
            async def my_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
                return list([OutputDocument(name="output.txt", content=b"output")])

            assert isinstance(my_flow, Flow)
            docs = list([InputDocument(name="test.txt", content=b"test")])
            options = FlowOptions()
            result = await my_flow("project", docs, options)
            assert len(result) == 1
            assert result[0].name == "output.txt"
            assert result[0].content == b"output"

            mock_set_cost.assert_called_once_with(0.15)


class TestSyncFunctionRejection:
    """Test rejection of sync functions with pipeline decorators."""

    def test_sync_function_with_pipeline_task_raises_error(self):
        from typing import Any, cast

        with pytest.raises(TypeError, match="must be 'async def'"):

            @cast(Any, pipeline_task)
            def sync_task(x: int) -> int:  # pyright: ignore[reportUnusedFunction]
                return x * 2

    def test_sync_function_with_pipeline_task_with_params_raises_error(self):
        from typing import Any, cast

        with pytest.raises(TypeError, match="must be 'async def'"):

            @cast(Any, pipeline_task(retries=3, trace_level="debug"))
            def sync_task(x: int) -> int:  # pyright: ignore[reportUnusedFunction]
                return x * 2

    def test_sync_function_with_pipeline_flow_raises_error(self):
        from typing import Any, cast

        with pytest.raises(TypeError, match="must be declared with 'async def'"):

            @cast(Any, pipeline_flow())
            def sync_flow(  # pyright: ignore[reportUnusedFunction]
                project_name: str, documents: list[Document], flow_options: FlowOptions
            ) -> list[Document]:
                return list([OutputDocument(name="output.txt", content=b"output")])

    def test_sync_function_with_pipeline_flow_with_params_raises_error(self):
        from typing import Any, cast

        with pytest.raises(TypeError, match="must be declared with 'async def'"):

            @cast(Any, pipeline_flow(name="sync_flow", retries=2))
            def sync_flow(  # pyright: ignore[reportUnusedFunction]
                project_name: str, documents: list[Document], flow_options: FlowOptions
            ) -> list[Document]:
                return list([OutputDocument(name="output.txt", content=b"output")])


class TestDoubleTracingDetection:
    """Test detection of double tracing with @trace and pipeline decorators."""

    def test_trace_then_pipeline_task_raises_error(self):
        from ai_pipeline_core import trace

        with pytest.raises(TypeError, match=r"already decorated.*with @trace"):

            @pipeline_task
            @trace
            async def my_task(x: int) -> int:  # pyright: ignore[reportUnusedFunction]
                return x * 2

    def test_pipeline_task_then_trace_raises_error(self):
        from ai_pipeline_core import trace

        with pytest.raises(TypeError, match=r"already decorated with @pipeline_task"):

            @trace
            @pipeline_task(persist=False)
            async def my_task(x: int) -> int:  # pyright: ignore[reportUnusedFunction]
                return x * 2

    def test_trace_then_pipeline_flow_raises_error(self):
        from ai_pipeline_core import trace

        with pytest.raises(TypeError, match=r"already decorated.*with @trace"):

            @pipeline_flow()
            @trace
            async def my_flow(  # pyright: ignore[reportUnusedFunction]
                project_name: str, documents: list[Document], flow_options: FlowOptions
            ) -> list[Document]:
                return list([OutputDocument(name="output.txt", content=b"output")])

    def test_pipeline_flow_then_trace_raises_error(self):
        from ai_pipeline_core import trace

        with pytest.raises(TypeError, match=r"already decorated with @pipeline"):

            @trace
            @pipeline_flow()
            async def my_flow(  # pyright: ignore[reportUnusedFunction]
                project_name: str, documents: list[Document], flow_options: FlowOptions
            ) -> list[Document]:
                return list([OutputDocument(name="output.txt", content=b"output")])

    def test_multiple_trace_then_pipeline_task_raises_error(self):
        from ai_pipeline_core import trace

        with pytest.raises(TypeError, match=r"already decorated.*with @trace"):

            @pipeline_task
            @trace
            @trace
            async def my_task(x: int) -> int:  # pyright: ignore[reportUnusedFunction]
                return x * 2

    def test_trace_with_params_then_pipeline_task_raises_error(self):
        from ai_pipeline_core import trace

        with pytest.raises(TypeError, match=r"already decorated.*with @trace"):

            @pipeline_task
            @trace(level="always")
            async def my_task(x: int) -> int:  # pyright: ignore[reportUnusedFunction]
                return x * 2

    async def test_normal_pipeline_task_still_works(self):
        @pipeline_task(persist=False)
        async def my_task(x: int) -> int:
            return x * 3

        result = await my_task(5)
        assert result == 15

    async def test_normal_pipeline_flow_still_works(self):
        @pipeline_flow()
        async def my_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
            return list([OutputDocument(name="output.txt", content=b"output")])

        docs = list([InputDocument(name="test.txt", content=b"test")])
        options = FlowOptions()
        result = await my_flow("project", docs, options)
        assert len(result) == 1
        assert result[0].name == "output.txt"
        assert result[0].content == b"output"
