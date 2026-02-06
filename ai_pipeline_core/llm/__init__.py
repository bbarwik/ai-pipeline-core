"""Large Language Model integration via LiteLLM proxy.

This package provides the Conversation class for LLM interactions with built-in
retry logic, LMNR tracing, and structured output generation using Pydantic models.

Primary API:
    Conversation class - Immutable, Document-based, warmup+fork friendly

All primitive types (ModelOptions, ModelName, etc.) are re-exported from _llm_core.
"""

from ai_pipeline_core._llm_core import (
    Citation,
    ModelName,
    ModelOptions,
    ModelResponse,
    TokenUsage,
    generate,
    generate_structured,
)

from ._images import (
    ImagePart,
    ImagePreset,
    ImageProcessingConfig,
    ImageProcessingError,
    ProcessedImage,
    process_image,
)
from ._substitutor import URLSubstitutor
from .conversation import Conversation, ConversationContent, MessageType

__all__ = [
    "Citation",
    "Conversation",
    "ConversationContent",
    "ImagePart",
    "ImagePreset",
    "ImageProcessingConfig",
    "ImageProcessingError",
    "MessageType",
    "ModelName",
    "ModelOptions",
    "ModelResponse",
    "ProcessedImage",
    "TokenUsage",
    "URLSubstitutor",
    "generate",
    "generate_structured",
    "process_image",
]
