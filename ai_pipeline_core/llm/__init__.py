"""Large Language Model integration via LiteLLM proxy.

This package provides the Conversation class for LLM interactions with built-in
retry logic, LMNR tracing, and structured output generation using Pydantic models.

Primary API:
    Conversation class - Immutable, Document-based, warmup+fork friendly

Primitive types (ModelOptions, ModelName, etc.) are re-exported from llm.types.
"""

from .conversation import Conversation, ConversationContent
from .types import Citation, ModelName, ModelOptions

__all__ = [
    "Citation",
    "Conversation",
    "ConversationContent",
    "ModelName",
    "ModelOptions",
]
