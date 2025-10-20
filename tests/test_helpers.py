"""Test helper classes for concrete document implementations."""

from typing import Any

from openai.types.chat import ChatCompletion

from ai_pipeline_core.documents import FlowDocument, TaskDocument
from ai_pipeline_core.llm import ModelResponse, StructuredModelResponse


class ConcreteFlowDocument(FlowDocument):
    """Concrete FlowDocument implementation for testing."""

    pass


class ConcreteTaskDocument(TaskDocument):
    """Concrete TaskDocument implementation for testing."""

    pass


def create_test_model_response(**kwargs: Any) -> ModelResponse:
    """Create a ModelResponse for testing.

    Args:
        **kwargs: ChatCompletion attributes (id, object, created, model, choices, usage, etc.)

    Returns:
        ModelResponse instance.
    """
    completion = ChatCompletion(**kwargs)
    return ModelResponse(
        chat_completion=completion,
        model_options={},
        metadata={},
    )


def create_test_structured_model_response(**kwargs: Any) -> StructuredModelResponse[Any]:
    """Create a StructuredModelResponse for testing.

    Args:
        **kwargs: ChatCompletion attributes (id, object, created, model, choices, usage, etc.)

    Returns:
        StructuredModelResponse instance.
    """
    completion = ChatCompletion(**kwargs)
    model_response = ModelResponse(
        chat_completion=completion,
        model_options={},
        metadata={},
    )
    return StructuredModelResponse.from_model_response(model_response)
