"""Model response structures for LLM interactions.

Provides enhanced response classes that use OpenAI-compatible base types via LiteLLM
with additional metadata, cost tracking, and structured output support.
"""

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from openai.types.chat import ChatCompletion
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

T = TypeVar(
    "T",
    bound=BaseModel,
)
"""Type parameter for structured response Pydantic models."""


@dataclass(frozen=True)
class Citation:
    """A URL citation returned by search-enabled models (e.g. sonar-pro-search, gemini-3-flash-search)."""

    title: str
    url: str


class ModelResponse(ChatCompletion):
    """Response wrapper for LLM text generation.

    Primary usage is adding to AIMessages for multi-turn conversations:

        >>> response = await llm.generate("gpt-5.1", messages=messages)
        >>> messages.append(response)  # Add assistant response to conversation
        >>> print(response.content)  # Access generated text

    The two main interactions with ModelResponse:
    1. Adding to AIMessages for conversation flow
    2. Accessing .content property for the generated text

    Almost all use cases are covered by these two patterns. Advanced features
    like token usage and cost tracking are available but rarely needed.

    Inherits from OpenAI's ChatCompletion for compatibility.
    Other properties (usage, model, id) should only be accessed
    when absolutely necessary.
    """

    def __init__(
        self,
        chat_completion: ChatCompletion,
        model_options: dict[str, Any],
        metadata: dict[str, Any],
        usage: CompletionUsage | None = None,
    ) -> None:
        """Initialize ModelResponse from ChatCompletion.

        Wraps an OpenAI ChatCompletion object with additional metadata
        and model options for tracking and observability.

        Args:
            chat_completion: ChatCompletion object from the API.
            model_options: Model configuration options used for the request.
                          Stored for metadata extraction and tracing.
            metadata: Custom metadata for tracking (time_taken, first_token_time, etc.).
                     Includes timing information and custom tags.
            usage: Optional usage information from streaming response.

        """
        data = chat_completion.model_dump()

        # fixes issue where the role is "assistantassistant" instead of "assistant"
        valid_finish_reasons = {"stop", "length", "tool_calls", "content_filter", "function_call"}
        for i in range(len(data["choices"])):
            data["choices"][i]["message"]["role"] = "assistant"
            # Only update finish_reason if it's not already a valid value
            current_finish_reason = data["choices"][i].get("finish_reason")
            if current_finish_reason not in valid_finish_reasons:
                data["choices"][i]["finish_reason"] = "stop"
            # Strip annotations with unsupported types (e.g. Grok returns type="file" for PDFs,
            # but OpenAI's ChatCompletion only accepts type="url_citation")
            if annotations := data["choices"][i]["message"].get("annotations"):
                data["choices"][i]["message"]["annotations"] = [a for a in annotations if a.get("type") == "url_citation"]

        super().__init__(**data)

        self._model_options = model_options
        self._metadata = metadata
        if usage:
            self.usage = usage

    @property
    def content(self) -> str:
        """Get the generated text content.

        Primary property for accessing the LLM's response text.
        This is the main property you'll use with ModelResponse.

        Returns:
            Generated text from the model, or empty string if none.

        """
        content = self.choices[0].message.content or ""
        return content.split("</think>")[-1].strip()

    @property
    def reasoning_content(self) -> str:
        """Get the reasoning content.

        Returns:
            The reasoning content from the model, or empty string if none.
        """
        message = self.choices[0].message
        if reasoning_content := getattr(message, "reasoning_content", None):
            return reasoning_content
        if not message.content or "</think>" not in message.content:
            return ""
        return message.content.split("</think>")[0].strip()

    @property
    def citations(self) -> list[Citation]:
        """Get URL citations from search-enabled models.

        Returns:
            List of Citation objects with title and url. Empty list for non-search models.
        """
        annotations = self.choices[0].message.annotations
        if not annotations:
            return []
        return [Citation(title=a.url_citation.title, url=a.url_citation.url) for a in annotations if a.url_citation]

    def get_laminar_metadata(self) -> dict[str, str | int | float]:  # noqa: C901
        """Extract metadata for LMNR (Laminar) observability including cost tracking.

        Collects comprehensive metadata about the generation for tracing,
        monitoring, and cost analysis in the LMNR platform. This method
        provides detailed insights into token usage, caching effectiveness,
        and generation costs.

        Returns:
            Dictionary containing:
            - LiteLLM headers (call ID, costs, model info, etc.)
            - Token usage statistics (input, output, total, cached)
            - Model configuration used for generation
            - Cost information in multiple formats
            - Cached token counts (when context caching enabled)
            - Reasoning token counts (for O1 models)

        Metadata structure:
            - litellm.*: All LiteLLM-specific headers
            - gen_ai.usage.prompt_tokens: Input token count
            - gen_ai.usage.completion_tokens: Output token count
            - gen_ai.usage.total_tokens: Total tokens used
            - gen_ai.usage.cached_tokens: Cached tokens (if applicable)
            - gen_ai.usage.reasoning_tokens: Reasoning tokens (O1 models)
            - gen_ai.usage.output_cost: Generation cost in dollars
            - gen_ai.usage.cost: Alternative cost field (same value)
            - gen_ai.cost: Simple cost field (same value)
            - gen_ai.response.*: Response identifiers
            - model_options.*: Configuration used

        Cost tracking:
            Cost information is extracted from two sources:
            1. x-litellm-response-cost header (primary)
            2. usage.cost attribute (fallback)

            Cost is stored in three fields for observability tool consumption:
            - gen_ai.usage.output_cost (OpenTelemetry GenAI semantic convention)
            - gen_ai.usage.cost (aggregated cost)
            - gen_ai.cost (short-form)

        Cost availability depends on LiteLLM proxy configuration. Not all providers
        return cost information. Cached tokens reduce actual cost but may not be reflected.
        Used internally by tracing but accessible for cost analysis.
        """
        metadata: dict[str, str | int | float] = deepcopy(self._metadata)

        # Add base metadata
        # NOTE: gen_ai.response.model is intentionally omitted — Laminar's UI uses it
        # to override the span display name in the tree view, hiding the actual span name
        # (set via `purpose` parameter). Tracked upstream: Laminar's getSpanDisplayName()
        # in frontend/components/traces/trace-view/utils.ts prefers model over span name
        # for LLM spans. Restore once Laminar shows both or prefers span name.
        metadata.update({
            "gen_ai.response.id": self.id,
            "gen_ai.system": "litellm",
        })

        # Add usage metadata if available
        cost = None
        if self.usage:
            metadata.update({
                "gen_ai.usage.prompt_tokens": self.usage.prompt_tokens,
                "gen_ai.usage.completion_tokens": self.usage.completion_tokens,
                "gen_ai.usage.total_tokens": self.usage.total_tokens,
            })

            # Check for cost in usage object
            if hasattr(self.usage, "cost"):
                # The 'cost' attribute is added by LiteLLM but not in OpenAI types
                cost = float(self.usage.cost)  # type: ignore[attr-defined]

            # Add reasoning tokens if available
            if (completion_details := self.usage.completion_tokens_details) and (reasoning_tokens := completion_details.reasoning_tokens):
                metadata["gen_ai.usage.reasoning_tokens"] = reasoning_tokens

            # Add cached tokens if available
            if (prompt_details := self.usage.prompt_tokens_details) and (cached_tokens := prompt_details.cached_tokens):
                metadata["gen_ai.usage.cached_tokens"] = cached_tokens

        # Add cost metadata if available
        if cost and cost > 0:
            metadata.update({
                "gen_ai.usage.output_cost": cost,
                "gen_ai.usage.cost": cost,
                "gen_ai.cost": cost,
            })

        for key, value in self._model_options.items():
            if "messages" in key:
                continue
            metadata[f"model_options.{key}"] = str(value)

        other_fields = self.__dict__
        for key, value in other_fields.items():
            if key in {"_model_options", "_metadata", "choices"}:
                continue
            try:
                metadata[f"response.raw.{key}"] = json.dumps(value, indent=2, default=str)
            except Exception:
                metadata[f"response.raw.{key}"] = str(value)

        message = self.choices[0].message
        for key, value in message.__dict__.items():
            if key in {"content"}:
                continue
            metadata[f"response.raw.message.{key}"] = json.dumps(value, indent=2, default=str)

        return metadata

    def validate_output(self) -> None:
        """Validate response output content and format.

        Checks that response has non-empty content and validates against
        response_format if structured output was requested.

        Raises:
            ValueError: If response content is empty.
            ValidationError: If content doesn't match response_format schema.
        """
        if not self.content:
            raise ValueError("Empty response content")

        if (response_format := self._model_options.get("response_format")) and isinstance(response_format, BaseModel):
            response_format.model_validate_json(self.content)


class StructuredModelResponse(ModelResponse, Generic[T]):  # noqa: UP046
    """Response wrapper for structured/typed LLM output.

    Primary usage is accessing the .parsed property for the structured data.
    """

    @classmethod
    def from_model_response(cls, model_response: ModelResponse) -> "StructuredModelResponse[T]":
        """Convert a ModelResponse to StructuredModelResponse.

        Takes an existing ModelResponse and converts it to a StructuredModelResponse
        for accessing parsed structured output. Used internally by generate_structured().

        Args:
            model_response: The ModelResponse to convert.

        Returns:
            StructuredModelResponse with lazy parsing support.
        """
        model_response.__class__ = cls
        return model_response  # type: ignore[return-value]

    @property
    def parsed(self) -> T:
        """Get the parsed structured output.

        Lazily parses the JSON content into the specified Pydantic model.
        Result is cached after first access.

        Returns:
            Parsed Pydantic model instance.

        Raises:
            ValidationError: If content doesn't match the response_format schema.
        """
        if not hasattr(self, "_parsed_value"):
            response_format = self._model_options.get("response_format")
            self._parsed_value: T = response_format.model_validate_json(self.content)  # type: ignore[return-value]
        return self._parsed_value
