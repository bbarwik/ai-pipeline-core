"""Shared test helpers for llm/ tests."""

from typing import Any

from ai_pipeline_core._llm_core.model_response import ModelResponse
from ai_pipeline_core._llm_core.types import RawToolCall, TokenUsage

ZERO_USAGE = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


def make_response(content: str = "", tool_calls: tuple[RawToolCall, ...] = ()) -> ModelResponse[Any]:
    """Build a minimal ModelResponse for unit tests."""
    return ModelResponse(
        content=content,
        parsed=content,
        model="test",
        usage=ZERO_USAGE,
        tool_calls=tool_calls,
    )


def make_tool_call(id: str, fn: str, args: str = "{}") -> RawToolCall:
    """Build a RawToolCall for unit tests."""
    return RawToolCall(id=id, function_name=fn, arguments=args)
