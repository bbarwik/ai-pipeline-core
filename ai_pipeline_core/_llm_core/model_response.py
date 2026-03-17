"""Unified ModelResponse for LLM interactions.

Provides a single generic ModelResponse[T] class for both structured and
unstructured LLM output. Fully serializable with Pydantic.
"""

from collections.abc import Mapping
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from .types import RawToolCall, TokenUsage

T = TypeVar("T", default=str)
"""Type parameter for response output. str for unstructured, BaseModel subclass for structured."""


class Citation(BaseModel):
    """A URL citation returned by search-enabled models.

    The start_index and end_index fields indicate character positions in the response content
    where the citation applies. Note that index behavior varies by model.
    """

    model_config = ConfigDict(frozen=True)

    title: str
    url: str
    start_index: int
    end_index: int


class ModelResponse(BaseModel, Generic[T]):
    """Unified LLM response for both structured and unstructured output.

    Generic[T] provides type hints:
    - ModelResponse[str]: unstructured text output
    - ModelResponse[MyModel]: structured output with typed .parsed

    All fields are serializable. After JSON round-trip, `parsed` becomes a dict
    for structured responses (use MyModel.model_validate(response.parsed) to reconstruct).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # Core response data
    content: str
    """Raw LLM output text (or JSON string for structured)."""

    parsed: T
    """Parsed result: same as content for T=str, typed model for T=BaseModel."""

    reasoning_content: str = ""
    """Reasoning/thinking content from the model (if available)."""

    citations: tuple[Citation, ...] = ()
    """URL citations from search-enabled models."""

    # Usage and cost
    usage: TokenUsage
    """Token usage statistics."""

    cost: float | None = None
    """Generation cost in USD (if available)."""

    # Metadata for observability
    model: str
    """Model identifier used for generation."""

    response_id: str = ""
    """Unique response identifier from the API (empty if not provided)."""

    metadata: Mapping[str, Any] = Field(default_factory=dict)
    """Additional metadata (timing, model options, etc.)."""

    # Tool calling
    tool_calls: tuple[RawToolCall, ...] = ()
    """Tool calls requested by the model. Empty tuple when no tools were called."""

    # Provider-specific fields for multi-turn reasoning
    thinking_blocks: tuple[dict[str, Any], ...] | None = None
    """Structured thinking blocks from the model (if available)."""

    provider_specific_fields: Mapping[str, Any] | None = None
    """Provider-specific fields like Gemini thought_signatures for multi-turn."""

    @property
    def has_tool_calls(self) -> bool:
        """Whether the model requested tool calls in this response."""
        return len(self.tool_calls) > 0

    @field_serializer("parsed", when_used="always")
    def serialize_parsed(self, value: T) -> Any:
        """Serialize parsed value - convert BaseModel to dict."""
        if isinstance(value, BaseModel):
            return value.model_dump()
        return value
