"""Bug-proving tests for tool loop forced-final response issues.

Bugs B1, B2 from NEW-BUGS-REPORT.md:
- B1: Forced final response sends tools it shouldn't
- B2: Forced final response failure loses all tool results
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from uuid import uuid7

from pydantic import BaseModel, Field

from ai_pipeline_core._llm_core.model_response import ModelResponse
from ai_pipeline_core._llm_core.types import CoreMessage, Role
from ai_pipeline_core.database._memory import _MemoryDatabase
from ai_pipeline_core.deployment._types import _NoopPublisher
from ai_pipeline_core.exceptions import LLMError
from ai_pipeline_core.llm._tool_loop import execute_tool_loop
from ai_pipeline_core.llm.tools import Tool
from ai_pipeline_core.pipeline._execution_context import ExecutionContext, set_execution_context
from ai_pipeline_core.pipeline._runtime_sinks import build_runtime_sinks
from ai_pipeline_core.pipeline.limits import _SharedStatus
from ai_pipeline_core.settings import settings

from .conftest import make_response, make_tool_call


class SearchTool(Tool):
    """Search the web."""

    class Input(BaseModel):
        query: str = Field(description="Search query")

    class Output(BaseModel):
        results: str

    async def run(self, input: Input) -> Output:
        return self.Output(results=f"Results for: {input.query}")


@dataclass(frozen=True)
class _FakeToolResultMsg:
    tool_call_id: str
    function_name: str
    content: str


def _build_msg(tid: str, fn: str, content: str) -> _FakeToolResultMsg:
    return _FakeToolResultMsg(tool_call_id=tid, function_name=fn, content=content)


def _make_context(database: _MemoryDatabase) -> ExecutionContext:
    deployment_id = uuid7()
    span_id = uuid7()
    return ExecutionContext(
        run_id="tool-loop-bug-test",
        execution_id=None,
        publisher=_NoopPublisher(),
        limits=MappingProxyType({}),
        limits_status=_SharedStatus(),
        database=database,
        sinks=build_runtime_sinks(database=database, settings_obj=settings),
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        deployment_name="tool-loop-bug-test",
        span_id=span_id,
        current_span_id=span_id,
        flow_span_id=span_id,
    )


TOOL_SCHEMAS = [{"type": "function", "function": {"name": "search_tool", "parameters": {}}}]


# ── B1: Forced final response sends tools it shouldn't ─────────────────────


async def test_forced_final_omits_tools() -> None:
    """When max_tool_rounds is exhausted, the forced final LLM call must NOT include
    tool schemas or tool_choice. Currently it sends tools=tool_schemas, tool_choice='none'.
    """
    recorded_calls: list[dict[str, Any]] = []
    round_count = 0

    async def mock_invoke(**kwargs: Any) -> ModelResponse[Any]:
        nonlocal round_count
        recorded_calls.append(kwargs)
        round_count += 1
        if round_count == 1:
            return make_response(
                content="",
                tool_calls=(make_tool_call("c1", "search_tool", '{"query": "test"}'),),
            )
        return make_response(content="Final synthesis")

    database = _MemoryDatabase()
    with set_execution_context(_make_context(database)):
        await execute_tool_loop(
            invoke_llm=mock_invoke,
            tool_schemas=TOOL_SCHEMAS,
            tool_lookup={"search_tool": SearchTool()},
            tool_choice="auto",
            max_tool_rounds=1,
            purpose="test",
            expected_cost=None,
            core_messages=[CoreMessage(role=Role.USER, content="search for test")],
            context_count=0,
            effective_options=None,
            substitutor=None,
            build_tool_result_message=_build_msg,
        )

    # The final call (forced final) should have empty tools and no tool_choice
    forced_final_call = recorded_calls[-1]
    assert forced_final_call["tools"] in ([], None), f"Forced final should not send tool schemas, got: {forced_final_call['tools']}"
    assert forced_final_call["tool_choice"] is None, f"Forced final should not send tool_choice, got: {forced_final_call['tool_choice']}"
    assert forced_final_call.get("tool_schemas") in ([], None), (
        f"Forced final tool_schemas for span metadata should be empty, got: {forced_final_call.get('tool_schemas')}"
    )


# ── B2: Forced final response failure loses all tool results ───────────────


async def test_forced_final_failure_preserves_tool_records() -> None:
    """When the forced final LLM call fails (e.g., empty response after retries),
    the accumulated tool_call_records from prior rounds must still be returned.
    Currently they are lost because the exception propagates out of execute_tool_loop.
    """
    round_count = 0

    async def mock_invoke(**kwargs: Any) -> ModelResponse[Any]:
        nonlocal round_count
        round_count += 1
        if round_count == 1:
            return make_response(
                content="",
                tool_calls=(make_tool_call("c1", "search_tool", '{"query": "deep research"}'),),
            )
        # Forced final fails
        raise LLMError("Empty response content — model returned no text and no tool calls.")

    database = _MemoryDatabase()
    with set_execution_context(_make_context(database)):
        msgs, response, records = await execute_tool_loop(
            invoke_llm=mock_invoke,
            tool_schemas=TOOL_SCHEMAS,
            tool_lookup={"search_tool": SearchTool()},
            tool_choice="auto",
            max_tool_rounds=1,
            purpose="test",
            expected_cost=None,
            core_messages=[CoreMessage(role=Role.USER, content="research")],
            context_count=0,
            effective_options=None,
            substitutor=None,
            build_tool_result_message=_build_msg,
        )

    # Tool records from prior rounds must be preserved
    assert len(records) >= 1, "Tool call records from successful rounds should be preserved"
    assert records[0].tool is SearchTool
    # Response should exist (synthetic fallback), not raise
    assert response is not None
    assert response.content  # some content, even if synthetic


async def test_forced_final_failure_preserves_accumulated_messages() -> None:
    """Accumulated messages (assistant responses, tool results) from successful
    rounds must survive forced-final failure.
    """
    round_count = 0

    async def mock_invoke(**kwargs: Any) -> ModelResponse[Any]:
        nonlocal round_count
        round_count += 1
        if round_count <= 2:
            return make_response(
                content="",
                tool_calls=(make_tool_call(f"c{round_count}", "search_tool", '{"query": "q"}'),),
            )
        raise LLMError("Forced final failed")

    database = _MemoryDatabase()
    with set_execution_context(_make_context(database)):
        msgs, response, records = await execute_tool_loop(
            invoke_llm=mock_invoke,
            tool_schemas=TOOL_SCHEMAS,
            tool_lookup={"search_tool": SearchTool()},
            tool_choice="auto",
            max_tool_rounds=2,
            purpose="test",
            expected_cost=None,
            core_messages=[CoreMessage(role=Role.USER, content="multi-round")],
            context_count=0,
            effective_options=None,
            substitutor=None,
            build_tool_result_message=_build_msg,
        )

    # Messages from 2 successful tool rounds must be present
    assert len(msgs) >= 4, f"Expected messages from 2 tool rounds + forced final, got {len(msgs)}"
    assert len(records) == 2, f"Expected 2 tool records from 2 rounds, got {len(records)}"
