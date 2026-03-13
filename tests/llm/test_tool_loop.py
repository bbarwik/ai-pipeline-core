"""Unit tests for llm/_tool_loop.py."""

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from uuid import uuid7

from pydantic import BaseModel, Field

from ai_pipeline_core._llm_core.model_response import ModelResponse
from ai_pipeline_core._llm_core.types import CoreMessage, Role
from ai_pipeline_core.database import SpanKind
from ai_pipeline_core.database._memory import MemoryDatabase
from ai_pipeline_core.deployment._types import _NoopPublisher
from ai_pipeline_core.llm._tool_loop import _execute_single_tool, execute_tool_loop
from ai_pipeline_core.llm.tools import Tool, ToolOutput
from ai_pipeline_core.pipeline._execution_context import ExecutionContext, set_execution_context
from ai_pipeline_core.pipeline._runtime_sinks import build_runtime_sinks
from ai_pipeline_core.pipeline.limits import _SharedStatus
from ai_pipeline_core.settings import settings

from .conftest import make_response, make_tool_call


class SearchTool(Tool):
    """Search the web."""

    class Input(BaseModel):
        query: str = Field(description="Search query")

    async def execute(self, input: Input) -> ToolOutput:
        return ToolOutput(content=f"Results for: {input.query}")


class FailingTool(Tool):
    """A tool that always raises."""

    class Input(BaseModel):
        reason: str = Field(description="Failure reason")

    async def execute(self, input: Input) -> ToolOutput:
        raise RuntimeError(f"Intentional: {input.reason}")


class SlowTool(Tool):
    """Tool that takes too long."""

    class Input(BaseModel):
        delay: float = Field(description="Delay in seconds")

    async def execute(self, input: Input) -> ToolOutput:
        import asyncio

        await asyncio.sleep(input.delay)
        return ToolOutput(content="done")


@dataclass(frozen=True)
class _FakeToolResultMsg:
    tool_call_id: str
    function_name: str
    content: str


def _build_msg(tid: str, fn: str, content: str) -> _FakeToolResultMsg:
    return _FakeToolResultMsg(tool_call_id=tid, function_name=fn, content=content)


class _RecordingSpanDatabase(MemoryDatabase):
    def __init__(self) -> None:
        super().__init__()
        self.inserted_spans: list[object] = []

    async def insert_span(self, span: object) -> None:
        self.inserted_spans.append(span)
        await super().insert_span(span)  # type: ignore[arg-type]


def _make_context(database: MemoryDatabase) -> ExecutionContext:
    deployment_id = uuid7()
    span_id = uuid7()
    return ExecutionContext(
        run_id="tool-loop-test",
        execution_id=None,
        publisher=_NoopPublisher(),
        limits=MappingProxyType({}),
        limits_status=_SharedStatus(),
        database=database,
        sinks=build_runtime_sinks(database=database, settings_obj=settings),
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        deployment_name="tool-loop-test",
        span_id=span_id,
        current_span_id=span_id,
        flow_span_id=span_id,
    )


def _finished_spans(database: _RecordingSpanDatabase, kind: str) -> list[object]:
    return [span for span in database.inserted_spans if getattr(span, "kind", None) == kind and getattr(span, "output_json", "")]


async def test_execute_single_tool_success_returns_record_output_and_span() -> None:
    tool = SearchTool()
    tc = make_tool_call("c1", "search", '{"query": "test"}')
    database = _RecordingSpanDatabase()

    with set_execution_context(_make_context(database)):
        record, output = await _execute_single_tool(tool, tc, round_num=1)

    assert record is not None
    assert record.tool is SearchTool
    assert "Results for: test" in output.content
    tool_call_span = _finished_spans(database, SpanKind.TOOL_CALL)[0]
    tool_meta = json.loads(tool_call_span.meta_json)
    assert tool_meta["tool_name"] == "search_tool"
    assert tool_meta["round_index"] == 1


async def test_execute_single_tool_validation_error_records_failed_tool_call() -> None:
    tool = SearchTool()
    tc = make_tool_call("c1", "search", "not valid json")
    database = _RecordingSpanDatabase()

    with set_execution_context(_make_context(database)):
        record, output = await _execute_single_tool(tool, tc, round_num=1)

    assert record is None
    assert "Invalid arguments" in output.content
    tool_call_span = _finished_spans(database, SpanKind.TOOL_CALL)[0]
    assert tool_call_span.error_type == ""
    tool_meta = json.loads(tool_call_span.meta_json)
    assert tool_meta["tool_call_id"] == "c1"
    assert tool_meta["round_index"] == 1


async def test_execute_single_tool_timeout_records_failed_tool_call() -> None:
    import ai_pipeline_core.llm._tool_loop as tool_loop

    original = tool_loop.TOOL_EXECUTION_TIMEOUT_SECONDS
    database = _RecordingSpanDatabase()
    tool_loop.TOOL_EXECUTION_TIMEOUT_SECONDS = 0.01
    try:
        tool = SlowTool()
        tc = make_tool_call("c1", "slow", '{"delay": 10}')
        with set_execution_context(_make_context(database)):
            record, output = await _execute_single_tool(tool, tc, round_num=2)
    finally:
        tool_loop.TOOL_EXECUTION_TIMEOUT_SECONDS = original

    assert record is not None
    assert "timed out" in output.content
    tool_call_span = _finished_spans(database, SpanKind.TOOL_CALL)[0]
    assert tool_call_span.error_type == ""
    assert json.loads(tool_call_span.meta_json)["round_index"] == 2


async def test_execute_tool_loop_records_rounds_and_tool_calls_in_order() -> None:
    call_count = 0

    async def invoke_llm(**kwargs: Any) -> ModelResponse[Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return make_response(
                content="",
                tool_calls=(make_tool_call("c1", "search", '{"query": "test"}'),),
            )
        return make_response(content="Found results")

    database = _RecordingSpanDatabase()
    with set_execution_context(_make_context(database)):
        msgs, resp, records = await execute_tool_loop(
            invoke_llm=invoke_llm,
            tool_schemas=[{"type": "function", "function": {"name": "search"}}],
            tool_lookup={"search": SearchTool()},
            tool_choice="auto",
            max_tool_rounds=5,
            purpose="test",
            expected_cost=None,
            core_messages=[CoreMessage(role=Role.USER, content="search for test")],
            context_count=0,
            effective_options=None,
            substitutor=None,
            build_tool_result_message=_build_msg,
        )

    assert resp.content == "Found results"
    assert len(records) == 1
    assert len(msgs) == 3
    assert [span.kind for span in _finished_spans(database, SpanKind.TOOL_CALL)] == [SpanKind.TOOL_CALL]


async def test_execute_tool_loop_unknown_tool_records_tool_call_span() -> None:
    call_count = 0

    async def invoke_llm(**kwargs: Any) -> ModelResponse[Any]:
        nonlocal call_count
        _ = kwargs
        call_count += 1
        return make_response(content="", tool_calls=(make_tool_call("c1", "missing_tool", "{}"),)) if call_count == 1 else make_response(content="done")

    database = _RecordingSpanDatabase()
    with set_execution_context(_make_context(database)):
        msgs, resp, records = await execute_tool_loop(
            invoke_llm=invoke_llm,
            tool_schemas=[],
            tool_lookup={"search": SearchTool()},
            tool_choice="auto",
            max_tool_rounds=5,
            purpose="test",
            expected_cost=None,
            core_messages=[CoreMessage(role=Role.USER, content="hi")],
            context_count=0,
            effective_options=None,
            substitutor=None,
            build_tool_result_message=_build_msg,
        )

    assert resp.content == "done"
    assert records == ()
    assert any("Unknown tool" in msg.content for msg in msgs if isinstance(msg, _FakeToolResultMsg))
