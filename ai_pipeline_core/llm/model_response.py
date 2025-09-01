"""Model response structures for LLM interactions.

@public

Provides enhanced response classes that wrap OpenAI API responses
with additional metadata, cost tracking, and structured output support.
"""

import copy
from typing import Any, Generic, TypeVar

from openai.types.chat import ChatCompletion, ParsedChatCompletion
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)
"""Type parameter for structured response Pydantic models."""


class ModelResponse(ChatCompletion):
    """Response wrapper for LLM text generation.

    @public

    Primary usage is adding to AIMessages for multi-turn conversations:

        >>> response = await llm.generate(messages=messages)
        >>> messages.add(response)  # Add assistant response to conversation
        >>> print(response.content)  # Access generated text

    The two main interactions with ModelResponse:
    1. Adding to AIMessages for conversation flow
    2. Accessing .content property for the generated text

    Almost all use cases are covered by these two patterns. Advanced features
    like token usage and cost tracking are available but rarely needed.

    Example:
        >>> from ai_pipeline_core.llm import AIMessages, generate
        >>>
        >>> messages = AIMessages("Explain quantum computing")
        >>> response = await generate(messages=messages)
        >>>
        >>> # Primary usage: add to conversation
        >>> messages.add(response)
        >>>
        >>> # Access generated text
        >>> print(response.content)

    Note:
        Inherits from OpenAI's ChatCompletion for compatibility.
        Other properties (usage, model, id) should only be accessed
        when absolutely necessary.
    """

    headers: dict[str, str] = Field(default_factory=dict)
    model_options: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, chat_completion: ChatCompletion | None = None, **kwargs: Any) -> None:
        """Initialize ModelResponse from ChatCompletion or kwargs.

        Can be initialized from an existing ChatCompletion object or
        directly from keyword arguments. Automatically initializes
        headers dict if not provided.

        Args:
            chat_completion: Optional ChatCompletion to wrap.
            **kwargs: Direct initialization parameters if no
                     ChatCompletion provided.

        Example:
            >>> # From ChatCompletion
            >>> response = ModelResponse(chat_completion_obj)
            >>>
            >>> # Direct initialization (mainly for testing)
            >>> response = ModelResponse(
            ...     id="test",
            ...     model="gpt-5",
            ...     choices=[...]
            ... )
        """
        if chat_completion:
            # Copy all attributes from the ChatCompletion instance
            data = chat_completion.model_dump()
            data["headers"] = {}  # Add default headers
            super().__init__(**data)
        else:
            # Initialize from kwargs
            if "headers" not in kwargs:
                kwargs["headers"] = {}
            super().__init__(**kwargs)

    @property
    def content(self) -> str:
        """Get the generated text content.

        @public

        Primary property for accessing the LLM's response text.
        This covers 99% of use cases with ModelResponse.

        Returns:
            Generated text from the model, or empty string if none.

        Example:
            >>> response = await generate(messages="Hello")
            >>> text = response.content  # The generated response
            >>>
            >>> # Common pattern: add to messages then use content
            >>> messages.add(response)
            >>> if "error" in response.content.lower():
            ...     # Handle error case
        """
        return self.choices[0].message.content or ""

    def set_model_options(self, options: dict[str, Any]) -> None:
        """Store the model configuration used for generation.

        Saves a deep copy of the options used for this generation,
        excluding the messages for brevity.

        Args:
            options: Dictionary of model options from the API call.

        Note:
            Messages are removed to avoid storing large prompts.
            Called internally by the generation functions.
        """
        self.model_options = copy.deepcopy(options)
        if "messages" in self.model_options:
            del self.model_options["messages"]

    def set_headers(self, headers: dict[str, str]) -> None:
        """Store HTTP response headers.

        Saves response headers which contain LiteLLM metadata
        including cost information and call IDs.

        Args:
            headers: Dictionary of HTTP headers from the response.

        Headers of interest:
            - x-litellm-response-cost: Generation cost
            - x-litellm-call-id: Unique call identifier
            - x-litellm-model-id: Actual model used
        """
        self.headers = copy.deepcopy(headers)

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
            ...     messages="Summarize this",
            ...     options=ModelOptions(cache_ttl="300s")
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
        metadata: dict[str, str | int | float] = {}

        litellm_id = self.headers.get("x-litellm-call-id")
        cost = float(self.headers.get("x-litellm-response-cost") or 0)

        # Add all x-litellm-* headers
        for header, value in self.headers.items():
            if header.startswith("x-litellm-"):
                header_name = header.replace("x-litellm-", "").lower()
                metadata[f"litellm.{header_name}"] = value

        # Add base metadata
        metadata.update({
            "gen_ai.response.id": litellm_id or self.id,
            "gen_ai.response.model": self.model,
            "get_ai.system": "litellm",
        })

        # Add usage metadata if available
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

        if self.model_options:
            for key, value in self.model_options.items():
                metadata[f"model_options.{key}"] = str(value)

        return metadata


class StructuredModelResponse(ModelResponse, Generic[T]):
    """Response wrapper for structured/typed LLM output.

    @public

    Primary usage is adding to AIMessages and accessing .parsed property:

        >>> class Analysis(BaseModel):
        ...     sentiment: float
        ...     summary: str
        >>>
        >>> response = await generate_structured(
        ...     response_format=Analysis,
        ...     messages="Analyze this text..."
        ... )
        >>>
        >>> # Primary usage: access parsed model
        >>> analysis = response.parsed
        >>> print(f"Sentiment: {analysis.sentiment}")
        >>>
        >>> # Can add to messages for conversation
        >>> messages.add(response)

    The two main interactions:
    1. Accessing .parsed property for the structured data
    2. Adding to AIMessages for conversation continuity

    These patterns cover virtually all use cases. Advanced features exist
    but should only be used when absolutely necessary.

    Type Parameter:
        T: The Pydantic model type for the structured output.

    Note:
        Extends ModelResponse with type-safe parsed data access.
        Other inherited properties should rarely be needed.
    """

    def __init__(
        self,
        chat_completion: ChatCompletion | None = None,
        parsed_value: T | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with ChatCompletion and parsed value.

        Creates a structured response from a base completion and
        optionally a pre-parsed value. Can extract parsed value
        from ParsedChatCompletion automatically.

        Args:
            chat_completion: Base chat completion response.
            parsed_value: Pre-parsed Pydantic model instance.
                         If None, attempts extraction from
                         ParsedChatCompletion.
            **kwargs: Additional ChatCompletion parameters.

        Extraction behavior:
            1. Use provided parsed_value if given
            2. Extract from ParsedChatCompletion if available
            3. Store as None (access will raise ValueError)

        Note:
            Usually created internally by generate_structured().
            The parsed value is validated by Pydantic automatically.
        """
        super().__init__(chat_completion, **kwargs)
        self._parsed_value: T | None = parsed_value

        # Extract parsed value from ParsedChatCompletion if available
        if chat_completion and isinstance(chat_completion, ParsedChatCompletion):
            if chat_completion.choices:  # type: ignore[attr-defined]
                message = chat_completion.choices[0].message  # type: ignore[attr-defined]
                if hasattr(message, "parsed"):  # type: ignore
                    self._parsed_value = message.parsed  # type: ignore[attr-defined]

    @property
    def parsed(self) -> T:
        """Get the parsed Pydantic model instance.

        @public

        Primary property for accessing structured output.
        This is the main reason to use generate_structured().

        Returns:
            Validated instance of the Pydantic model type T.

        Raises:
            ValueError: If no parsed content available (internal error).

        Example:
            >>> class UserInfo(BaseModel):
            ...     name: str
            ...     age: int
            >>>
            >>> response = await generate_structured(
            ...     response_format=UserInfo,
            ...     messages="Extract user info..."
            ... )
            >>>
            >>> # Primary usage: get the parsed model
            >>> user = response.parsed
            >>> print(f"{user.name} is {user.age} years old")
            >>>
            >>> # Can also add to messages
            >>> messages.add(response)

        Note:
            Type-safe with full IDE support. This property covers
            99% of structured response use cases.
        """
        if self._parsed_value is not None:
            return self._parsed_value

        raise ValueError(
            "No parsed content available. This should not happen for StructuredModelResponse."
        )
