"""Large Language Model integration via LiteLLM proxy.

This package provides OpenAI API-compatible LLM interactions with built-in retry logic,
LMNR tracing, and structured output generation using Pydantic models.
"""

from .ai_messages import AIMessages, AIMessageType
from .client import (
    generate,
    generate_structured,
    generate_with_retry_for_testing,
    process_messages_for_testing,
)
from .model_options import ModelOptions
from .model_response import ModelResponse, StructuredModelResponse
from .model_types import ModelName

__all__ = [
    "AIMessages",
    "AIMessageType",
    "ModelName",
    "ModelOptions",
    "ModelResponse",
    "StructuredModelResponse",
    "generate",
    "generate_structured",
    # Internal functions exposed for testing only
    "process_messages_for_testing",
    "generate_with_retry_for_testing",
]
