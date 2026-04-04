"""Tool execution loop for Conversation.

Implements the auto-loop that calls the LLM, executes requested tools, and
re-sends results until the LLM produces a final answer or max rounds is reached.
"""

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from pydantic import BaseModel, ValidationError

from ai_pipeline_core._llm_core.model_response import ModelResponse
from ai_pipeline_core._llm_core.types import CoreMessage, ModelOptions, RawToolCall, Role, TokenUsage
from ai_pipeline_core.database import SpanKind
from ai_pipeline_core.pipeline._execution_context import get_execution_context, get_sinks
from ai_pipeline_core.pipeline._track_span import track_span

from ._substitutor import URLSubstitutor
from .tools import Tool, ToolCallRecord, ToolOutput

__all__: list[str] = []

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_UNKNOWN_TOOL_ROUNDS = 3
"""Abort the tool loop after this many consecutive rounds where ALL tool calls targeted
non-existent tools. Prevents burning through max_tool_rounds with zero useful work while
still allowing the model to recover from occasional hallucinations."""

_FORCED_FINAL_MAX_ROUNDS_MSG = (
    "You have reached the maximum number of tool-call rounds. "
    "No more tools are available in this conversation. "
    "Do not call any tools or functions. "
    "Use only the information already present in the conversation and the tool results already returned "
    "to produce the final answer now. "
    "If a structured response format was requested earlier, follow it exactly. "
    "If the available information is insufficient, explain the limitation instead of requesting another tool."
)

_FORCED_FINAL_UNKNOWN_TOOLS_MSG = (
    "Tool calling has been stopped because your recent requests targeted tools that do not exist. "
    "No more tools are available in this conversation. "
    "Do not call any tools or functions. "
    "Use only the information already present in the conversation and the tool results already returned "
    "to produce the final answer now. "
    "If a structured response format was requested earlier, follow it exactly. "
    "If the available information is insufficient, explain the limitation instead of requesting another tool."
)


async def _execute_single_tool(
    tool: Tool,
    tool_call: RawToolCall,
    round_num: int,
) -> tuple[ToolCallRecord | None, ToolOutput]:
    """Execute a single tool call with error handling and timeout."""
    tool_cls = type(tool)
    snake_name = tool_cls.name
    execution_ctx = get_execution_context()
    tool_target = f"tool_call:{tool_cls.__module__}:{tool_cls.__qualname__}"
    receiver_payload = {
        "mode": "tool_ref",
        "value": {"name": tool_cls.name, "class_path": f"{tool_cls.__module__}:{tool_cls.__qualname__}"},
    }

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
            result = await tool.execute(parsed_input)
        except TypeError, AssertionError:
            raise
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

        record = ToolCallRecord(tool=tool_cls, input=parsed_input, output=result, round=round_num)
        span_ctx.set_meta(
            tool_name=snake_name,
            tool_class_path=f"{tool_cls.__module__}:{tool_cls.__qualname__}",
            tool_call_id=tool_call.id,
            round_index=round_num,
        )
        span_ctx.set_output_preview(result.model_dump(mode="json"))
        span_ctx._set_output_value(result)
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
    _list_item_type: type[BaseModel] | None = None,
) -> tuple[list[Any], ModelResponse[Any], tuple[ToolCallRecord, ...]]:
    """Execute the tool auto-loop.

    Returns (accumulated_new_messages, final_response, tool_call_records).
    """
    if max_tool_rounds < 1:
        raise ValueError(f"max_tool_rounds must be >= 1, got {max_tool_rounds}")

    accumulated_messages: list[Any] = []
    all_records: list[ToolCallRecord] = []
    consecutive_unknown_rounds = 0
    round_num = 0

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
            _list_item_type=_list_item_type,
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

        # Track consecutive rounds where ALL calls targeted non-existent tools.
        # Tool results are already appended above so the forced final has full error context.
        if all(tc.function_name not in tool_lookup for tc in response.tool_calls):
            consecutive_unknown_rounds += 1
            logger.warning(
                "Tool loop round %d: all %d tool call(s) targeted unknown tools (%d/%d consecutive). Available: [%s]. Requested: [%s].",
                round_num,
                len(response.tool_calls),
                consecutive_unknown_rounds,
                MAX_CONSECUTIVE_UNKNOWN_TOOL_ROUNDS,
                ", ".join(sorted(tool_lookup.keys())),
                tool_names,
            )
            if consecutive_unknown_rounds >= MAX_CONSECUTIVE_UNKNOWN_TOOL_ROUNDS:
                logger.warning(
                    "Tool loop aborting after %d consecutive rounds of unknown tool calls. Forcing final response.",
                    consecutive_unknown_rounds,
                )
                break
        else:
            consecutive_unknown_rounds = 0

    # max_tool_rounds exhausted or unknown-tool limit reached — force a final text response.
    forced_by_unknown_tools = consecutive_unknown_rounds >= MAX_CONSECUTIVE_UNKNOWN_TOOL_ROUNDS
    steering_msg = _FORCED_FINAL_UNKNOWN_TOOLS_MSG if forced_by_unknown_tools else _FORCED_FINAL_MAX_ROUNDS_MSG
    logger.warning(
        "Tool loop forcing final response (reason=%s, max_tool_rounds=%d, consecutive_unknown_rounds=%d).",
        "unknown_tool_limit" if forced_by_unknown_tools else "max_rounds",
        max_tool_rounds,
        consecutive_unknown_rounds,
    )

    # Inject USER message so the LLM knows tools are no longer available.
    # Without this, the model generates tool calls against tools=[] causing ValueError
    # in client.py and burning all retry attempts before the fallback triggers.
    # Only added to core_messages (for the immediate forced final call), NOT to
    # accumulated_messages — persisting it in conversation history would suppress
    # valid tool calls in follow-up send() calls.
    core_messages.append(CoreMessage(role=Role.USER, content=steering_msg))

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
            round_index=round_num + 1,
            tool_schemas=[],
            _list_item_type=_list_item_type,
        )
    except Exception as forced_exc:
        logger.warning(
            "Forced final response failed after max_tool_rounds=%d (consecutive_unknown_rounds=%d). "
            "Returning accumulated tool results (%d records) without final synthesis.",
            max_tool_rounds,
            consecutive_unknown_rounds,
            len(all_records),
            exc_info=forced_exc,
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
