"""Primitive LLM layer with NO Document dependency.

This internal module provides low-level LLM access for modules that cannot
depend on Documents (document_store, observability). App code should use
the llm module's Conversation class instead.

Exports:
    Types: Role, TextContent, ImageContent, PDFContent, ContentPart, CoreMessage, TokenUsage
    Model config: ModelOptions, ModelName
    Response types: ModelResponse, StructuredModelResponse, Citation
    Functions: generate, generate_structured
"""

from .client import generate, generate_structured
from .model_options import ModelOptions
from .model_response import Citation, ModelResponse
from .model_types import ModelName
from .types import (
    ContentPart,
    CoreMessage,
    ImageContent,
    PDFContent,
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
    "Role",
    "TextContent",
    "TokenUsage",
    "generate",
    "generate_structured",
]
