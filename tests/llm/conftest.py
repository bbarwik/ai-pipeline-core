"""Shared fixtures for llm tests."""

import pytest

from ai_pipeline_core._llm_core.model_response import ModelResponse
from ai_pipeline_core._llm_core.types import RawToolCall, TokenUsage
from ai_pipeline_core.documents._context import _suppress_document_registration


@pytest.fixture(autouse=True)
def _suppress_registration():
    with _suppress_document_registration():
        yield


def make_tool_call(call_id: str, function_name: str, arguments: str) -> RawToolCall:
    """Build a RawToolCall for tool-loop tests."""
    return RawToolCall(id=call_id, function_name=function_name, arguments=arguments)


def make_response(
    content: str,
    *,
    tool_calls: tuple[RawToolCall, ...] = (),
    usage: TokenUsage | None = None,
    model: str = "test-model",
) -> ModelResponse[str]:
    """Build a minimal ModelResponse for conversation/tool-loop tests."""
    return ModelResponse[str](
        content=content,
        parsed=content,
        usage=usage or TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        cost=0.0,
        model=model,
        response_id="resp-test",
        metadata={},
        tool_calls=tool_calls,
    )
