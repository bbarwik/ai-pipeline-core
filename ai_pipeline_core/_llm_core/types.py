"""Primitive types and constants for LLM interactions.

This module provides low-level types for LLM communication that have NO dependency
on the Document class. These types are used by internal modules (document_store,
observability) that need LLM access but cannot depend on Documents.

All types are frozen Pydantic models for immutability and JSON serialization.
"""

from enum import StrEnum
from typing import Literal

from pydantic import Base64Bytes, BaseModel, ConfigDict

# Token count per image/PDF (per CLAUDE.md §3.6)
TOKENS_PER_IMAGE = 1080


type ModelName = (
    Literal[
        # Core models
        "gemini-3-pro",
        "gpt-5.1",
        # Small models
        "gemini-3-flash",
        "gpt-5-mini",
        "grok-4.1-fast",
        # Search models
        "gemini-3-flash-search",
        "gpt-5-mini-search",
        "grok-4.1-fast-search",
        "sonar-pro-search",
    ]
    | str
)
"""Type-safe model name with IDE autocompletion for common models.

Literal[...] | str provides autocomplete for known models while accepting any string.
Model availability depends on your LiteLLM proxy configuration and provider access.
"""


class Role(StrEnum):
    """Message role in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class TextContent(BaseModel):
    """Plain text content part."""

    model_config = ConfigDict(frozen=True)

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Image content with binary data.

    Uses Base64Bytes for automatic base64 encoding/decoding in JSON serialization.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["image"] = "image"
    data: Base64Bytes
    mime_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


class PDFContent(BaseModel):
    """PDF document content with binary data.

    Uses Base64Bytes for automatic base64 encoding/decoding in JSON serialization.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["pdf"] = "pdf"
    data: Base64Bytes


ContentPart = TextContent | ImageContent | PDFContent
"""Union of all content part types for polymorphic handling."""


class CoreMessage(BaseModel):
    """A single message in a conversation.

    Content can be:
    - str: Plain text (converted to TextContent internally)
    - ContentPart: Single content part (text, image, or PDF)
    - tuple[ContentPart, ...]: Multiple content parts (multimodal message)
    """

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str | ContentPart | tuple[ContentPart, ...]


class TokenUsage(BaseModel):
    """Token usage statistics from an LLM call."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int = 0
    reasoning_tokens: int = 0
