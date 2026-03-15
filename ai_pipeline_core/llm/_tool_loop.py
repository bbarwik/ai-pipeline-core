"""Tool execution loop for Conversation.

Implements the auto-loop that calls the LLM, executes requested tools, and
re-sends results until the LLM produces a final answer or max rounds is reached.
"""

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from ai_pipeline_core._codec import UniversalCodec
from ai_pipeline_core._llm_core.model_response import ModelResponse
from ai_pipeline_core._llm_core.types import CoreMessage, ModelOptions, RawToolCall, Role, TokenUsage
from ai_pipeline_core.database import SpanKind
from ai_pipeline_core.logger import get_pipeline_logger
from ai_pipeline_core.pipeline._execution_context import get_execution_context, get_sinks
from ai_pipeline_core.pipeline._track_span import track_span

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
    execution_ctx = get_execution_context()
    tool_target = f"instance_method:{tool_cls.__module__}:{tool_cls.__qualname__}.execute"
    receiver_payload = {"mode": "constructor_args", "value": _serialize_tool_constructor_args(tool)}

    async with track_span(
        SpanKind.TOOL_CALL,
        snake_name,
        tool_target,
        sinks=get_sinks(),
        encode_receiver=receiver_payload,
        encode_input={"input": _parse_tool_arguments(tool_call.arguments), "tool_call_id": tool_call.id, "round_index": round_num},
        db=execution_ctx.database if execution_ctx is not None else None,
        input_preview={"tool_name": snake_name, "arguments": _parse_tool_arguments(tool_call.arguments)},
    ) as span_ctx:
        try:
            parsed_input = tool_cls.Input.model_validate_json(tool_call.arguments)
        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning("Tool input validation failed for %s: %s", tool_cls.__name__, e)
            output = ToolOutput(content=f"Error: Invalid arguments for tool '{snake_name}': {e}")
            span_ctx.set_meta(
                tool_name=snake_name,
                tool_class_path=f"{tool_cls.__module__}:{tool_cls.__qualname__}",
                tool_call_id=tool_call.id,
                round_index=round_num,
            )
            span_ctx.set_output_preview(output.model_dump(mode="json"))
            span_ctx._set_output_value(output)
            return None, output
        try:
            result = await asyncio.wait_for(tool.execute(parsed_input), timeout=TOOL_EXECUTION_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.warning("Tool execution timed out: %s", tool_cls.__name__)
            output = ToolOutput(content=f"Error: Tool '{snake_name}' timed out after {TOOL_EXECUTION_TIMEOUT_SECONDS}s")
            span_ctx.set_meta(
                tool_name=snake_name,
                tool_class_path=f"{tool_cls.__module__}:{tool_cls.__qualname__}",
                tool_call_id=tool_call.id,
                round_index=round_num,
            )
            span_ctx.set_output_preview(output.model_dump(mode="json"))
            span_ctx._set_output_value(output)
            return (ToolCallRecord(tool=tool_cls, input=parsed_input, output=output, round=round_num), output)
        except Exception as e:
            logger.warning("Tool execution failed for %s: %s", tool_cls.__name__, e)
            output = ToolOutput(content=f"Error: Tool '{snake_name}' failed: {e}")
            span_ctx.set_meta(
                tool_name=snake_name,
                tool_class_path=f"{tool_cls.__module__}:{tool_cls.__qualname__}",
                tool_call_id=tool_call.id,
                round_index=round_num,
            )
            span_ctx.set_output_preview(output.model_dump(mode="json"))
            span_ctx._set_output_value(output)
            return (ToolCallRecord(tool=tool_cls, input=parsed_input, output=output, round=round_num), output)

        validated_result = cast(Any, result)
        if not isinstance(validated_result, ToolOutput):
            raise TypeError(
                f"Tool '{tool_cls.__name__}'.execute() must return ToolOutput (or subclass), got {type(validated_result).__name__}. "
                f"This is a programming error in the tool implementation."
            )
        record = ToolCallRecord(tool=tool_cls, input=parsed_input, output=validated_result, round=round_num)
        span_ctx.set_meta(
            tool_name=snake_name,
            tool_class_path=f"{tool_cls.__module__}:{tool_cls.__qualname__}",
            tool_call_id=tool_call.id,
            round_index=round_num,
        )
        span_ctx.set_output_preview(validated_result.model_dump(mode="json"))
        span_ctx._set_output_value(validated_result)
        return record, validated_result


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
            output = ToolOutput(content=f"Error: Unknown tool '{tc.function_name}'. Available tools: {available}")
            return tc, None, output
        record, output = await _execute_single_tool(tool_lookup[tc.function_name], tc, round_num)
        return tc, record, output

    results = await asyncio.gather(*(_execute_one(tc) for tc in tool_calls), return_exceptions=True)
    merged: list[tuple[RawToolCall, ToolCallRecord | None, ToolOutput]] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            if isinstance(result, (TypeError, AssertionError, KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                raise result
            logger.warning("Unexpected error executing tool: %s", result)
            output = ToolOutput(content=f"Error: {result}")
            merged.append((tool_calls[i], None, output))
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
        logger.info("Tool loop round %d/%d (purpose=%s)", round_num, max_tool_rounds, purpose or "unspecified")
        response: ModelResponse[Any] = await invoke_llm(
            core_messages=core_messages,
            effective_options=effective_options,
            context_count=context_count,
            tools=tool_schemas,
            tool_choice=tool_choice if round_num == 1 else None,
            response_format=response_format,
            purpose=f"{purpose}:tool_round_{round_num}" if purpose else f"tool_round_{round_num}",
            expected_cost=expected_cost,
            round_index=round_num,
            tool_schemas=tool_schemas,
        )

        if not response.has_tool_calls:
            accumulated_messages.append(response)
            return accumulated_messages, response, tuple(all_records)

        accumulated_messages.append(response)
        core_messages.append(CoreMessage(role=Role.ASSISTANT, content=response.content or "", tool_calls=response.tool_calls))

        tool_names = ", ".join(tc.function_name for tc in response.tool_calls)
        logger.info("Tool loop round %d: executing %d tool(s): %s", round_num, len(response.tool_calls), tool_names)
        merged = await _execute_all_tool_calls(response.tool_calls, tool_lookup, round_num)

        for tc, record, output in merged:
            if record:
                all_records.append(record)
            result_content = substitutor.substitute(output.content) if substitutor else output.content
            accumulated_messages.append(build_tool_result_message(tc.id, tc.function_name, output.content))
            core_messages.append(CoreMessage(role=Role.TOOL, content=result_content, tool_call_id=tc.id, name=tc.function_name))

    # max_tool_rounds exhausted — force a final text response without tools
    logger.warning("Tool loop reached max_tool_rounds=%d. Forcing final response without tools.", max_tool_rounds)
    try:
        response = await invoke_llm(
            core_messages=core_messages,
            effective_options=effective_options,
            context_count=context_count,
            tools=[],
            tool_choice=None,
            response_format=response_format,
            purpose=f"{purpose}:forced_final" if purpose else "forced_final",
            expected_cost=expected_cost,
            round_index=max_tool_rounds + 1,
            tool_schemas=[],
        )
    except Exception:
        logger.warning(
            "Forced final response failed after max_tool_rounds=%d. Returning accumulated tool results (%d records) without final synthesis.",
            max_tool_rounds,
            len(all_records),
        )
        response = ModelResponse[Any](
            content="[Final synthesis failed after max tool rounds. Tool results are preserved in tool_call_records.]",
            parsed="[Final synthesis failed after max tool rounds.]",
            usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model="",
        )
    accumulated_messages.append(response)
    return accumulated_messages, response, tuple(all_records)


def _parse_tool_arguments(arguments: str) -> Any:
    """Best-effort parse of tool call arguments for tracing payloads."""
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return {"_raw": arguments}


def _serialize_tool_constructor_args(tool: Tool) -> dict[str, Any]:
    """Serialize tool constructor state into JSON-friendly values."""
    return cast(dict[str, Any], UniversalCodec().encode(tool.__dict__).value)
