"""Test helper classes for concrete document implementations."""

from typing import Any

from openai.types.chat import ChatCompletion

from ai_pipeline_core.documents import Document
from ai_pipeline_core.llm import ModelResponse, StructuredModelResponse


class ConcreteDocument(Document):
    """Concrete Document implementation for testing."""


class ConcreteDocument2(Document):
    """Second concrete Document subclass for testing."""


def create_test_model_response(**kwargs: Any) -> ModelResponse:
    """Create a ModelResponse for testing."""
    completion = ChatCompletion(**kwargs)
    return ModelResponse(
        chat_completion=completion,
        model_options={},
        metadata={},
    )


def create_test_structured_model_response(**kwargs: Any) -> StructuredModelResponse[Any]:
    """Create a StructuredModelResponse for testing."""
    completion = ChatCompletion(**kwargs)
    model_response = ModelResponse(
        chat_completion=completion,
        model_options={},
        metadata={},
    )
    return StructuredModelResponse.from_model_response(model_response)
