"""Model response structures for LLM interactions.

@public

Provides enhanced response classes that use OpenAI-compatible base types via LiteLLM
with additional metadata, cost tracking, and structured output support.
"""

import json
from copy import deepcopy
from typing import Any, Generic, TypeVar

from openai.types.chat import ChatCompletion
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

T = TypeVar(
    "T",
    bound=BaseModel,
)
"""Type parameter for structured response Pydantic models."""


class ModelResponse(ChatCompletion):
    """Response wrapper for LLM text generation.

    @public

    Primary usage is adding to AIMessages for multi-turn conversations:

        >>> response = await llm.generate("gpt-5", messages=messages)
        >>> messages.append(response)  # Add assistant response to conversation
        >>> print(response.content)  # Access generated text

    The two main interactions with ModelResponse:
    1. Adding to AIMessages for conversation flow
    2. Accessing .content property for the generated text

    Almost all use cases are covered by these two patterns. Advanced features
    like token usage and cost tracking are available but rarely needed.

    Example:
        >>> from ai_pipeline_core import llm, AIMessages
        >>>
        >>> messages = AIMessages(["Explain quantum computing"])
        >>> response = await llm.generate("gpt-5", messages=messages)
        >>>
        >>> # Primary usage: add to conversation
        >>> messages.append(response)
        >>>
        >>> # Access generated text
        >>> print(response.content)

    Note:
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

        Example:
            >>> # Usually created internally by generate()
            >>> response = ModelResponse(
            ...     chat_completion=completion,
            ...     model_options={"temperature": 0.7, "model": "gpt-4"},
            ...     metadata={"time_taken": 1.5, "first_token_time": 0.3}
            ... )
        """
        data = chat_completion.model_dump()

        # fixes issue where the role is "assistantassistant" instead of "assistant"
        for i in range(len(data["choices"])):
            if role := data["choices"][i]["message"].get("role"):
                if role.startswith("assistant") and role != "assistant":
                    data["choices"][i]["message"]["role"] = "assistant"

        super().__init__(**data)

        self._model_options = model_options
        self._metadata = metadata
        if usage:
            self.usage = usage

    @property
    def content(self) -> str:
        """Get the generated text content.

        @public

        Primary property for accessing the LLM's response text.
        This is the main property you'll use with ModelResponse.

        Returns:
            Generated text from the model, or empty string if none.

        Example:
            >>> response = await generate("gpt-5", messages="Hello")
            >>> text = response.content  # The generated response
            >>>
            >>> # Common pattern: add to messages then use content
            >>> messages.append(response)
            >>> if "error" in response.content.lower():
            ...     # Handle error case
        """
        content = self.choices[0].message.content or ""
        return content.split("</think>")[-1].strip()

    @property
    def reasoning_content(self) -> str:
        """Get the reasoning content.

        @public

        Returns:
            The reasoning content from the model, or empty string if none.
        """
        message = self.choices[0].message
        if reasoning_content := getattr(message, "reasoning_content", None):
            return reasoning_content
        if not message.content or "</think>" not in message.content:
            return ""
        return message.content.split("</think>")[0].strip()

    def get_laminar_metadata(self) -> dict[str, str | int | float]:
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

            Cost is stored in three fields for compatibility:
            - gen_ai.usage.output_cost (standard)
            - gen_ai.usage.cost (alternative)
            - gen_ai.cost (simple)

        Example:
            >>> response = await llm.generate(
            ...     "gpt-5",
            ...     context=large_doc,
            ...     messages="Summarize this"
            ... )
            >>>
            >>> # Get comprehensive metadata
            >>> metadata = response.get_laminar_metadata()
            >>>
            >>> # Track generation cost
            >>> cost = metadata.get('gen_ai.usage.output_cost', 0)
            >>> if cost > 0:
            ...     print(f"Generation cost: ${cost:.4f}")
            >>>
            >>> # Monitor token usage
            >>> print(f"Input: {metadata.get('gen_ai.usage.prompt_tokens', 0)} tokens")
            >>> print(f"Output: {metadata.get('gen_ai.usage.completion_tokens', 0)} tokens")
            >>> print(f"Total: {metadata.get('gen_ai.usage.total_tokens', 0)} tokens")
            >>>
            >>> # Check cache effectiveness
            >>> cached = metadata.get('gen_ai.usage.cached_tokens', 0)
            >>> if cached > 0:
            ...     total = metadata.get('gen_ai.usage.total_tokens', 1)
            ...     savings = (cached / total) * 100
            ...     print(f"Cache hit: {cached} tokens ({savings:.1f}% savings)")
            >>>
            >>> # Calculate cost per token
            >>> if cost > 0 and metadata.get('gen_ai.usage.total_tokens'):
            ...     cost_per_1k = (cost / metadata['gen_ai.usage.total_tokens']) * 1000
            ...     print(f"Cost per 1K tokens: ${cost_per_1k:.4f}")

        Note:
            - Cost availability depends on LiteLLM proxy configuration
            - Not all providers return cost information
            - Cached tokens reduce actual cost but may not be reflected
            - Used internally by tracing but accessible for cost analysis
        """
        metadata: dict[str, str | int | float] = deepcopy(self._metadata)

        # Add base metadata
        metadata.update({
            "gen_ai.response.id": self.id,
            "gen_ai.response.model": self.model,
            "get_ai.system": "litellm",
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
            if completion_details := self.usage.completion_tokens_details:
                if reasoning_tokens := completion_details.reasoning_tokens:
                    metadata["gen_ai.usage.reasoning_tokens"] = reasoning_tokens

            # Add cached tokens if available
            if prompt_details := self.usage.prompt_tokens_details:
                if cached_tokens := prompt_details.cached_tokens:
                    metadata["gen_ai.usage.cached_tokens"] = cached_tokens

        # Add cost metadata if available
        if cost and cost > 0:
            metadata.update({
                "gen_ai.usage.output_cost": cost,
                "gen_ai.usage.cost": cost,
                "get_ai.cost": cost,
            })

        for key, value in self._model_options.items():
            if "messages" in key:
                continue
            metadata[f"model_options.{key}"] = str(value)

        other_fields = self.__dict__
        for key, value in other_fields.items():
            if key in ["_model_options", "_metadata", "choices", "usage"]:
                continue
            try:
                metadata[f"response.raw.{key}"] = json.dumps(value, indent=2, default=str)
            except Exception:
                metadata[f"response.raw.{key}"] = str(value)

        message = self.choices[0].message
        for key, value in message.__dict__.items():
            if key in ["content"]:
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

        if response_format := self._model_options.get("response_format"):
            if isinstance(response_format, BaseModel):
                response_format.model_validate_json(self.content)


class StructuredModelResponse(ModelResponse, Generic[T]):
    """Response wrapper for structured/typed LLM output.

    @public

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
