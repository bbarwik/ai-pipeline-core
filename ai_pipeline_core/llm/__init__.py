"""Large Language Model integration via LiteLLM proxy.

This package provides the Conversation class for LLM interactions with built-in
retry logic, LMNR tracing, and structured output generation using Pydantic models.

Primary API:
    Conversation class - Immutable, Document-based, warmup+fork friendly

Primitive types (ModelOptions, ModelName, etc.) are re-exported from _llm_core.
"""

from ai_pipeline_core._llm_core.model_response import Citation
from ai_pipeline_core._llm_core.types import ModelName, ModelOptions, TokenUsage

from .conversation import Conversation, ConversationContent

__all__ = [
    "Citation",
    "Conversation",
    "ConversationContent",
    "ModelName",
    "ModelOptions",
    "TokenUsage",
]
