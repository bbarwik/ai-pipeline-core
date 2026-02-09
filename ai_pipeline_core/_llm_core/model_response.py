"""Unified ModelResponse for LLM interactions.

Provides a single generic ModelResponse[T] class for both structured and
unstructured LLM output. Fully serializable with Pydantic.

Usage:
    # Unstructured (T=str)
    response: ModelResponse[str] = await generate(...)
    print(response.content)  # Raw text
    print(response.parsed)   # Same as content

    # Structured (T=MyModel)
    response: ModelResponse[MyModel] = await generate_structured(..., MyModel)
    print(response.parsed)        # MyModel instance
    print(response.parsed.field)  # Type-safe access
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from .types import TokenUsage

T = TypeVar("T")
"""Type parameter for response output. str for unstructured, BaseModel subclass for structured."""


@dataclass(frozen=True, slots=True)
class Citation:
    """A URL citation returned by search-enabled models.

    The start_index and end_index fields indicate character positions in the response content
    where the citation applies. Note that index behavior varies by model.
    """

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

    # Provider-specific fields for multi-turn reasoning
    thinking_blocks: tuple[dict[str, Any], ...] | None = None
    """Structured thinking blocks from the model (if available)."""

    provider_specific_fields: dict[str, Any] | None = None
    """Provider-specific fields like Gemini thought_signatures for multi-turn."""

    @field_serializer("parsed", when_used="always")
    def serialize_parsed(self, value: T) -> Any:
        """Serialize parsed value - convert BaseModel to dict."""
        if isinstance(value, BaseModel):
            return value.model_dump()
        return value

    @field_serializer("citations", when_used="always")
    def serialize_citations(self, value: tuple[Citation, ...]) -> list[dict[str, Any]]:
        """Serialize citations dataclass to dict."""
        return [{"title": c.title, "url": c.url, "start_index": c.start_index, "end_index": c.end_index} for c in value]

    def get_laminar_metadata(self) -> dict[str, str | int | float]:
        """Extract metadata for LMNR (Laminar) observability.

        Attribute names match Laminar's span_attributes.rs constants:
        - gen_ai.usage.input_tokens / output_tokens (not prompt/completion)
        - gen_ai.usage.cache_read_input_tokens (not cached_tokens)
        - gen_ai.usage.cost / input_cost / output_cost
        - gen_ai.request_model for model identification
        """
        result: dict[str, str | int | float] = {
            "gen_ai.response.id": self.response_id,
            "gen_ai.system": "litellm",
            "gen_ai.request_model": self.model,
            "gen_ai.usage.input_tokens": self.usage.prompt_tokens,
            "gen_ai.usage.output_tokens": self.usage.completion_tokens,
            "gen_ai.usage.total_tokens": self.usage.total_tokens,
        }

        if self.usage.cached_tokens:
            result["gen_ai.usage.cache_read_input_tokens"] = self.usage.cached_tokens

        if self.usage.reasoning_tokens:
            result["gen_ai.usage.reasoning_tokens"] = self.usage.reasoning_tokens

        if self.cost is not None:
            result["gen_ai.usage.output_cost"] = self.cost
            result["gen_ai.usage.cost"] = self.cost

        # Include timing metadata
        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float)):
                result[key] = value

        return result
