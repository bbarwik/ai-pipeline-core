"""Primitive LLM layer with NO Document dependency.

This internal module provides low-level LLM access for modules that cannot
depend on Documents (database, observability). App code should use
the llm module's Conversation class instead.

Exports:
    Types: Role, TextContent, ImageContent, PDFContent, ContentPart, CoreMessage, TokenUsage
    Model config: ModelOptions, ModelName
    Response types: ModelResponse, Citation
    Functions: generate, generate_structured
"""

from .client import generate, generate_structured
from .model_response import Citation, ModelResponse
from .types import (
    ContentPart,
    CoreMessage,
    ImageContent,
    ModelName,
    ModelOptions,
    PDFContent,
    RawToolCall,
    Role,
    TextContent,
    TokenUsage,
)

__all__ = [
    "Citation",
    "ContentPart",
    "CoreMessage",
    "ImageContent",
    "ModelName",
    "ModelOptions",
    "ModelResponse",
    "PDFContent",
    "RawToolCall",
    "Role",
    "TextContent",
    "TokenUsage",
    "generate",
    "generate_structured",
]
