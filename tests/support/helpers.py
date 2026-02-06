"""Test helper classes for concrete document implementations."""

from typing import Any

from pydantic import BaseModel

from ai_pipeline_core.documents import Document
from ai_pipeline_core.llm import ModelResponse, TokenUsage


class ConcreteDocument(Document):
    """Concrete Document implementation for testing."""


class ConcreteDocument2(Document):
    """Second concrete Document subclass for testing."""


def create_test_model_response(
    content: str = "Test response",
    reasoning_content: str = "",
    model: str = "test-model",
    response_id: str = "test-response-id",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    cost: float | None = None,
) -> ModelResponse[str]:
    """Create a ModelResponse[str] for testing."""
    return ModelResponse[str](
        content=content,
        parsed=content,
        reasoning_content=reasoning_content,
        citations=(),
        usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        cost=cost,
        model=model,
        response_id=response_id,
        metadata={},
    )


def create_test_structured_model_response(
    parsed: BaseModel,
    content: str | None = None,
    reasoning_content: str = "",
    model: str = "test-model",
    response_id: str = "test-response-id",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    cost: float | None = None,
) -> ModelResponse[Any]:
    """Create a ModelResponse with structured output for testing."""
    if content is None:
        content = parsed.model_dump_json()
    return ModelResponse(
        content=content,
        parsed=parsed,
        reasoning_content=reasoning_content,
        citations=(),
        usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        cost=cost,
        model=model,
        response_id=response_id,
        metadata={},
    )
