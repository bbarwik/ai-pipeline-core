"""Tool execution loop for Conversation.

Implements the auto-loop that calls the LLM, executes requested tools, and
re-sends results until the LLM produces a final answer or max rounds is reached.
"""

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import Any

from pydantic import BaseModel, ValidationError

from ai_pipeline_core._llm_core.model_response import ModelResponse
from ai_pipeline_core._llm_core.types import CoreMessage, ModelOptions, RawToolCall, Role
from ai_pipeline_core.logging import get_pipeline_logger

from ._substitutor import URLSubstitutor
from .tools import Tool, ToolCallRecord, ToolOutput, to_snake_case

__all__: list[str] = []

logger = get_pipeline_logger(__name__)

TOOL_EXECUTION_TIMEOUT_SECONDS = 120


async def _execute_single_tool(
    tool: Tool,
    tool_call: RawToolCall,
    round_num: int,
) -> tuple[ToolCallRecord | None, ToolOutput]:
    """Execute a single tool call with error handling and timeout."""
    tool_cls = type(tool)
    snake_name = to_snake_case(tool_cls.__name__)
    try:
        parsed_input = tool_cls.Input.model_validate_json(tool_call.arguments)
    except (ValidationError, json.JSONDecodeError) as e:
        logger.warning("Tool input validation failed for %s: %s", tool_cls.__name__, e)
        return None, ToolOutput(content=f"Error: Invalid arguments for tool '{snake_name}': {e}")

    try:
        result = await asyncio.wait_for(
            tool.execute(parsed_input),
            timeout=TOOL_EXECUTION_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        logger.warning("Tool execution timed out: %s", tool_cls.__name__)
        output = ToolOutput(content=f"Error: Tool '{snake_name}' timed out after {TOOL_EXECUTION_TIMEOUT_SECONDS}s")
        return ToolCallRecord(tool=tool_cls, input=parsed_input, output=output, round=round_num), output
    except Exception as e:
        logger.warning("Tool execution failed for %s: %s", tool_cls.__name__, e)
        output = ToolOutput(content=f"Error: Tool '{snake_name}' failed: {e}")
        return ToolCallRecord(tool=tool_cls, input=parsed_input, output=output, round=round_num), output

    if not isinstance(result, ToolOutput):
        raise TypeError(
            f"Tool '{tool_cls.__name__}'.execute() must return ToolOutput (or subclass), got {type(result).__name__}. "
            f"This is a programming error in the tool implementation."
        )
    record = ToolCallRecord(tool=tool_cls, input=parsed_input, output=result, round=round_num)
    return record, result


InvokeLLMFn = Callable[..., Coroutine[Any, Any, ModelResponse[Any]]]


async def _execute_all_tool_calls(
    tool_calls: tuple[RawToolCall, ...],
    tool_lookup: dict[str, Tool],
    round_num: int,
) -> list[tuple[RawToolCall, ToolCallRecord | None, ToolOutput]]:
    """Execute all tool calls in parallel, returning results in original order."""

    async def _execute_one(tc: RawToolCall) -> tuple[RawToolCall, ToolCallRecord | None, ToolOutput]:
        if tc.function_name not in tool_lookup:
            available = ", ".join(sorted(tool_lookup.keys()))
            return tc, None, ToolOutput(content=f"Error: Unknown tool '{tc.function_name}'. Available tools: {available}")
        return (tc, *(await _execute_single_tool(tool_lookup[tc.function_name], tc, round_num)))

    results = await asyncio.gather(*(_execute_one(tc) for tc in tool_calls), return_exceptions=True)
    merged: list[tuple[RawToolCall, ToolCallRecord | None, ToolOutput]] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            if isinstance(result, (TypeError, AssertionError, KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                raise result
            logger.warning("Unexpected error executing tool: %s", result)
            merged.append((tool_calls[i], None, ToolOutput(content=f"Error: {result}")))
        else:
            merged.append(result)
    return merged


async def execute_tool_loop(
    *,
    invoke_llm: InvokeLLMFn,
    tool_schemas: list[dict[str, Any]],
    tool_lookup: dict[str, Tool],
    tool_choice: str | None,
    max_tool_rounds: int,
    purpose: str | None,
    expected_cost: float | None,
    core_messages: list[CoreMessage],
    context_count: int,
    effective_options: ModelOptions | None,
    substitutor: URLSubstitutor | None,
    build_tool_result_message: Callable[[str, str, str], Any],
    response_format: type[BaseModel] | None = None,
) -> tuple[list[Any], ModelResponse[Any], tuple[ToolCallRecord, ...]]:
    """Execute the tool auto-loop.

    Returns (accumulated_new_messages, final_response, tool_call_records).
    """
    if max_tool_rounds < 1:
        raise ValueError(f"max_tool_rounds must be >= 1, got {max_tool_rounds}")

    accumulated_messages: list[Any] = []
    all_records: list[ToolCallRecord] = []

    for round_num in range(1, max_tool_rounds + 1):
        response: ModelResponse[Any] = await invoke_llm(
            core_messages=core_messages,
            effective_options=effective_options,
            context_count=context_count,
            tools=tool_schemas,
            tool_choice=tool_choice if round_num == 1 else None,
            response_format=response_format,
            purpose=f"{purpose}:tool_round_{round_num}" if purpose else f"tool_round_{round_num}",
            expected_cost=expected_cost,
        )

        if not response.has_tool_calls:
            accumulated_messages.append(response)
            return accumulated_messages, response, tuple(all_records)

        accumulated_messages.append(response)
        core_messages.append(CoreMessage(role=Role.ASSISTANT, content=response.content or "", tool_calls=response.tool_calls))

        merged = await _execute_all_tool_calls(response.tool_calls, tool_lookup, round_num)

        for tc, record, output in merged:
            if record:
                all_records.append(record)
            result_content = substitutor.substitute(output.content) if substitutor else output.content
            accumulated_messages.append(build_tool_result_message(tc.id, tc.function_name, output.content))
            core_messages.append(CoreMessage(role=Role.TOOL, content=result_content, tool_call_id=tc.id, name=tc.function_name))

    # max_tool_rounds exhausted — force a final text response
    logger.warning("Tool loop reached max_tool_rounds=%d. Forcing final response with tool_choice='none'.", max_tool_rounds)
    response = await invoke_llm(
        core_messages=core_messages,
        effective_options=effective_options,
        context_count=context_count,
        tools=tool_schemas,
        tool_choice="none",
        response_format=response_format,
        purpose=f"{purpose}:forced_final" if purpose else "forced_final",
        expected_cost=expected_cost,
    )
    accumulated_messages.append(response)
    return accumulated_messages, response, tuple(all_records)
