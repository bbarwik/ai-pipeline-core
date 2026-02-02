"""Large Language Model integration via LiteLLM proxy.

This package provides OpenAI API-compatible LLM interactions with built-in retry logic,
LMNR tracing, and structured output generation using Pydantic models. Supports per-call
observability via purpose and expected_cost parameters for span naming and cost tracking.
"""

from .ai_messages import AIMessages, AIMessageType
from .client import (
    generate,
    generate_structured,
)
from .model_options import ModelOptions
from .model_response import Citation, ModelResponse, StructuredModelResponse
from .model_types import ModelName

__all__ = [
    "AIMessageType",
    "AIMessages",
    "Citation",
    "ModelName",
    "ModelOptions",
    "ModelResponse",
    "StructuredModelResponse",
    "generate",
    "generate_structured",
]
