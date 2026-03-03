"""Unit tests for llm/_tool_loop.py: execute_tool_loop and _execute_single_tool."""

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel, Field

from ai_pipeline_core._llm_core.model_response import ModelResponse
from ai_pipeline_core._llm_core.types import CoreMessage, ModelOptions, Role
from ai_pipeline_core.llm._tool_loop import _execute_single_tool, execute_tool_loop
from ai_pipeline_core.llm.tools import Tool, ToolOutput

from .conftest import make_response, make_tool_call


# ── Test tools ────────────────────────────────────────────────────────────────


class SearchTool(Tool):
    """Search the web."""

    class Input(BaseModel):
        query: str = Field(description="Search query")

    async def execute(self, input: BaseModel) -> ToolOutput:
        return ToolOutput(content=f"Results for: {input.query}")  # type: ignore[attr-defined]


class FailingTool(Tool):
    """A tool that always raises."""

    class Input(BaseModel):
        reason: str = Field(description="Failure reason")

    async def execute(self, input: BaseModel) -> ToolOutput:
        raise RuntimeError(f"Intentional: {input.reason}")  # type: ignore[attr-defined]


class BadReturnTool(Tool):
    """Tool that returns wrong type."""

    class Input(BaseModel):
        x: int = Field(description="Some value")

    async def execute(self, input: BaseModel) -> ToolOutput:
        return "not a ToolOutput"  # type: ignore[return-value]


class SlowTool(Tool):
    """Tool that takes too long."""

    class Input(BaseModel):
        delay: float = Field(description="Delay in seconds")

    async def execute(self, input: BaseModel) -> ToolOutput:
        import asyncio

        await asyncio.sleep(input.delay)  # type: ignore[attr-defined]
        return ToolOutput(content="done")


# ── Helpers ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _FakeToolResultMsg:
    tool_call_id: str
    function_name: str
    content: str


# ── _execute_single_tool ─────────────────────────────────────────────────────


async def test_execute_single_tool_success() -> None:
    tool = SearchTool()
    tc = make_tool_call("c1", "search", '{"query": "test"}')
    record, output = await _execute_single_tool(tool, tc, round_num=1)
    assert record is not None
    assert record.tool is SearchTool
    assert record.round == 1
    assert "Results for: test" in output.content


async def test_execute_single_tool_invalid_json() -> None:
    """Malformed JSON arguments produce error ToolOutput with no record."""
    tool = SearchTool()
    tc = make_tool_call("c1", "search", "not valid json")
    record, output = await _execute_single_tool(tool, tc, round_num=1)
    assert record is None
    assert "Invalid arguments" in output.content


async def test_execute_single_tool_validation_error() -> None:
    """Missing required field produces error ToolOutput with no record."""
    tool = SearchTool()
    tc = make_tool_call("c1", "search", '{"wrong_field": 123}')
    record, output = await _execute_single_tool(tool, tc, round_num=1)
    assert record is None
    assert "Invalid arguments" in output.content


async def test_execute_single_tool_execution_raises() -> None:
    """Tool execute() raising produces error ToolOutput with no record."""
    tool = FailingTool()
    tc = make_tool_call("c1", "failing_tool", '{"reason": "test failure"}')
    record, output = await _execute_single_tool(tool, tc, round_num=1)
    assert record is None
    assert "failed" in output.content
    assert "Intentional" in output.content


async def test_execute_single_tool_wrong_return_type() -> None:
    """execute() returning non-ToolOutput raises TypeError."""
    tool = BadReturnTool()
    tc = make_tool_call("c1", "bad", '{"x": 1}')
    with pytest.raises(TypeError, match="must return ToolOutput"):
        await _execute_single_tool(tool, tc, round_num=1)


async def test_execute_single_tool_timeout() -> None:
    """Tool exceeding timeout produces error ToolOutput."""
    import ai_pipeline_core.llm._tool_loop as tl

    original = tl.TOOL_EXECUTION_TIMEOUT_SECONDS
    tl.TOOL_EXECUTION_TIMEOUT_SECONDS = 0.01  # 10ms timeout for test
    try:
        tool = SlowTool()
        tc = make_tool_call("c1", "slow", '{"delay": 10}')
        record, output = await _execute_single_tool(tool, tc, round_num=1)
        assert record is None
        assert "timed out" in output.content
    finally:
        tl.TOOL_EXECUTION_TIMEOUT_SECONDS = original


# ── execute_tool_loop ─────────────────────────────────────────────────────────


def _build_msg(tid: str, fn: str, content: str) -> _FakeToolResultMsg:
    return _FakeToolResultMsg(tool_call_id=tid, function_name=fn, content=content)


async def test_loop_no_tool_calls_immediate_return() -> None:
    """LLM responds without tool calls — loop exits immediately."""
    final = make_response(content="Direct answer")

    async def invoke_llm(**kwargs: Any) -> ModelResponse[Any]:
        return final

    msgs, resp, records = await execute_tool_loop(
        invoke_llm=invoke_llm,
        tool_schemas=[],
        tool_lookup={},
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
    assert resp.content == "Direct answer"
    assert records == ()
    assert len(msgs) == 1  # just the final response


async def test_loop_single_tool_call_and_final() -> None:
    """LLM calls tool once, then produces final response."""
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
    assert records[0].tool is SearchTool
    # accumulated: tool_call response + tool result msg + final response
    assert len(msgs) == 3


async def test_loop_parallel_tool_calls() -> None:
    """LLM returns multiple tool calls — all execute in parallel, results in order."""
    call_count = 0

    async def invoke_llm(**kwargs: Any) -> ModelResponse[Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return make_response(
                content="",
                tool_calls=(
                    make_tool_call("c1", "search", '{"query": "first"}'),
                    make_tool_call("c2", "search", '{"query": "second"}'),
                ),
            )
        return make_response(content="Both done")

    msgs, resp, records = await execute_tool_loop(
        invoke_llm=invoke_llm,
        tool_schemas=[{"type": "function", "function": {"name": "search"}}],
        tool_lookup={"search": SearchTool()},
        tool_choice="auto",
        max_tool_rounds=5,
        purpose="test",
        expected_cost=None,
        core_messages=[CoreMessage(role=Role.USER, content="two searches")],
        context_count=0,
        effective_options=None,
        substitutor=None,
        build_tool_result_message=_build_msg,
    )
    assert len(records) == 2
    assert "first" in records[0].output.content
    assert "second" in records[1].output.content
    # accumulated: tool_call response + 2 tool result msgs + final response
    assert len(msgs) == 4


async def test_loop_unknown_tool() -> None:
    """Unknown tool produces error message listing available tools."""
    call_count = 0

    async def invoke_llm(**kwargs: Any) -> ModelResponse[Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return make_response(
                content="",
                tool_calls=(make_tool_call("c1", "nonexistent_tool", "{}"),),
            )
        return make_response(content="OK")

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
    assert len(records) == 0  # no successful records for unknown tools
    # The error message should have been sent back to LLM
    tool_result_msgs = [m for m in msgs if isinstance(m, _FakeToolResultMsg)]
    assert any("Unknown tool" in m.content for m in tool_result_msgs)
    assert any("search" in m.content for m in tool_result_msgs)  # lists available


async def test_loop_mixed_known_and_unknown_tools() -> None:
    """Mix of valid and unknown tools — valid executes, unknown gets error."""
    call_count = 0

    async def invoke_llm(**kwargs: Any) -> ModelResponse[Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return make_response(
                content="",
                tool_calls=(
                    make_tool_call("c1", "search", '{"query": "valid"}'),
                    make_tool_call("c2", "nonexistent", "{}"),
                ),
            )
        return make_response(content="Done")

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
    assert len(records) == 1  # only the valid tool
    assert records[0].tool is SearchTool
    tool_result_msgs = [m for m in msgs if isinstance(m, _FakeToolResultMsg)]
    assert len(tool_result_msgs) == 2  # one success, one error


async def test_loop_max_rounds_exhaustion() -> None:
    """Loop always gets tool calls — max_tool_rounds forces final with tool_choice='none'."""
    invocations: list[dict[str, Any]] = []

    async def invoke_llm(**kwargs: Any) -> ModelResponse[Any]:
        invocations.append(kwargs)
        if kwargs.get("tool_choice") == "none":
            return make_response(content="Forced final answer")
        return make_response(
            content="",
            tool_calls=(make_tool_call("c1", "search", '{"query": "again"}'),),
        )

    msgs, resp, records = await execute_tool_loop(
        invoke_llm=invoke_llm,
        tool_schemas=[{"type": "function", "function": {"name": "search"}}],
        tool_lookup={"search": SearchTool()},
        tool_choice="auto",
        max_tool_rounds=2,
        purpose="test",
        expected_cost=None,
        core_messages=[CoreMessage(role=Role.USER, content="hi")],
        context_count=0,
        effective_options=None,
        substitutor=None,
        build_tool_result_message=_build_msg,
    )
    assert resp.content == "Forced final answer"
    assert len(records) == 2  # 2 rounds of tool calls
    # Last invocation should have tool_choice="none"
    assert invocations[-1]["tool_choice"] == "none"
    # Should have been called 3 times: 2 rounds + 1 forced final
    assert len(invocations) == 3


async def test_loop_tool_choice_only_first_round() -> None:
    """tool_choice is applied on first round, None for subsequent rounds."""
    invocations: list[dict[str, Any]] = []

    async def invoke_llm(**kwargs: Any) -> ModelResponse[Any]:
        invocations.append(kwargs)
        if len(invocations) <= 2:
            return make_response(
                content="",
                tool_calls=(make_tool_call("c1", "search", '{"query": "test"}'),),
            )
        return make_response(content="Final")

    await execute_tool_loop(
        invoke_llm=invoke_llm,
        tool_schemas=[{"type": "function", "function": {"name": "search"}}],
        tool_lookup={"search": SearchTool()},
        tool_choice="required",
        max_tool_rounds=5,
        purpose="test",
        expected_cost=None,
        core_messages=[CoreMessage(role=Role.USER, content="hi")],
        context_count=0,
        effective_options=None,
        substitutor=None,
        build_tool_result_message=_build_msg,
    )
    assert invocations[0]["tool_choice"] == "required"  # first round
    assert invocations[1]["tool_choice"] is None  # second round
    assert invocations[2]["tool_choice"] is None  # third round


async def test_loop_stop_sequences_preserved_throughout() -> None:
    """Stop sequences are preserved for all rounds including tool rounds and forced final."""
    invocations: list[dict[str, Any]] = []

    async def invoke_llm(**kwargs: Any) -> ModelResponse[Any]:
        invocations.append(kwargs)
        if kwargs.get("tool_choice") == "none":
            return make_response(content="Final")
        return make_response(
            content="",
            tool_calls=(make_tool_call("c1", "search", '{"query": "test"}'),),
        )

    opts = ModelOptions(stop=("</result>",))
    await execute_tool_loop(
        invoke_llm=invoke_llm,
        tool_schemas=[{"type": "function", "function": {"name": "search"}}],
        tool_lookup={"search": SearchTool()},
        tool_choice="auto",
        max_tool_rounds=1,
        purpose="test",
        expected_cost=None,
        core_messages=[CoreMessage(role=Role.USER, content="hi")],
        context_count=0,
        effective_options=opts,
        substitutor=None,
        build_tool_result_message=_build_msg,
    )
    # All rounds should preserve stop sequences
    assert invocations[0]["effective_options"].stop == ("</result>",)
    assert invocations[1]["effective_options"].stop == ("</result>",)


async def test_loop_stop_sequence_active_on_natural_final() -> None:
    """When LLM produces a natural final answer (no tool calls) after tool rounds,
    stop sequences must be active so </result> extraction works correctly."""
    invocations: list[dict[str, Any]] = []
    call_count = 0

    async def invoke_llm(**kwargs: Any) -> ModelResponse[Any]:
        nonlocal call_count
        invocations.append(kwargs)
        call_count += 1
        if call_count == 1:
            # First round: LLM calls a tool
            return make_response(
                content="",
                tool_calls=(make_tool_call("c1", "search", '{"query": "test"}'),),
            )
        # Second round: LLM produces natural final answer (no tool calls)
        return make_response(content="<result>Here is the answer</result>")

    opts = ModelOptions(stop=("</result>",))
    await execute_tool_loop(
        invoke_llm=invoke_llm,
        tool_schemas=[{"type": "function", "function": {"name": "search"}}],
        tool_lookup={"search": SearchTool()},
        tool_choice="auto",
        max_tool_rounds=5,
        purpose="test",
        expected_cost=None,
        core_messages=[CoreMessage(role=Role.USER, content="hi")],
        context_count=0,
        effective_options=opts,
        substitutor=None,
        build_tool_result_message=_build_msg,
    )
    # Both rounds should have stop sequences active
    assert invocations[0]["effective_options"].stop == ("</result>",)
    assert invocations[1]["effective_options"].stop == ("</result>",)


async def test_loop_substitutor_applied_to_tool_result() -> None:
    """URLSubstitutor is applied to tool results before sending to LLM."""

    class FakeSubstitutor:
        def substitute(self, text: str) -> str:
            return text.replace("http://example.com", "[URL_1]")

    call_count = 0
    core_messages_captured: list[CoreMessage] = []

    async def invoke_llm(**kwargs: Any) -> ModelResponse[Any]:
        nonlocal call_count
        call_count += 1
        # Capture core_messages to inspect substitution
        core_messages_captured.extend(kwargs["core_messages"])
        if call_count == 1:
            return make_response(
                content="",
                tool_calls=(make_tool_call("c1", "search", '{"query": "url test"}'),),
            )
        return make_response(content="Final")

    class URLSearchTool(Tool):
        """Returns a URL."""

        class Input(BaseModel):
            query: str = Field(description="Query")

        async def execute(self, input: BaseModel) -> ToolOutput:
            return ToolOutput(content="Found at http://example.com")

    core_msgs: list[CoreMessage] = [CoreMessage(role=Role.USER, content="find url")]
    msgs, resp, records = await execute_tool_loop(
        invoke_llm=invoke_llm,
        tool_schemas=[{"type": "function", "function": {"name": "search"}}],
        tool_lookup={"search": URLSearchTool()},
        tool_choice="auto",
        max_tool_rounds=5,
        purpose="test",
        expected_cost=None,
        core_messages=core_msgs,
        context_count=0,
        effective_options=None,
        substitutor=FakeSubstitutor(),
        build_tool_result_message=_build_msg,
    )
    # Core messages should have substituted content
    tool_core_msgs = [m for m in core_msgs if m.role == Role.TOOL]
    assert len(tool_core_msgs) == 1
    assert "[URL_1]" in tool_core_msgs[0].content  # type: ignore[operator]

    # Accumulated messages should have ORIGINAL content
    result_msgs = [m for m in msgs if isinstance(m, _FakeToolResultMsg)]
    assert len(result_msgs) == 1
    assert "http://example.com" in result_msgs[0].content


# ── Regression tests ──────────────────────────────────────────────────────────


async def test_loop_max_tool_rounds_zero_raises() -> None:
    """max_tool_rounds=0 is invalid — must be >= 1."""

    async def invoke_llm(**kwargs: Any) -> ModelResponse[Any]:
        return make_response(content="ok")

    with pytest.raises(ValueError, match="max_tool_rounds must be >= 1"):
        await execute_tool_loop(
            invoke_llm=invoke_llm,
            tool_schemas=[],
            tool_lookup={},
            tool_choice="auto",
            max_tool_rounds=0,
            purpose="test",
            expected_cost=None,
            core_messages=[CoreMessage(role=Role.USER, content="hi")],
            context_count=0,
            effective_options=None,
            substitutor=None,
            build_tool_result_message=_build_msg,
        )


async def test_loop_negative_max_tool_rounds_raises() -> None:
    """Negative max_tool_rounds is invalid."""

    async def invoke_llm(**kwargs: Any) -> ModelResponse[Any]:
        return make_response(content="ok")

    with pytest.raises(ValueError, match="max_tool_rounds must be >= 1"):
        await execute_tool_loop(
            invoke_llm=invoke_llm,
            tool_schemas=[],
            tool_lookup={},
            tool_choice="auto",
            max_tool_rounds=-1,
            purpose="test",
            expected_cost=None,
            core_messages=[CoreMessage(role=Role.USER, content="hi")],
            context_count=0,
            effective_options=None,
            substitutor=None,
            build_tool_result_message=_build_msg,
        )


async def test_loop_programming_error_propagates() -> None:
    """TypeError from tool execution propagates instead of being sent to LLM."""

    async def invoke_llm(**kwargs: Any) -> ModelResponse[Any]:
        return make_response(
            content="",
            tool_calls=(make_tool_call("c1", "bad_return_tool", '{"x": 1}'),),
        )

    with pytest.raises(TypeError, match="must return ToolOutput"):
        await execute_tool_loop(
            invoke_llm=invoke_llm,
            tool_schemas=[{"type": "function", "function": {"name": "bad_return_tool"}}],
            tool_lookup={"bad_return_tool": BadReturnTool()},
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
