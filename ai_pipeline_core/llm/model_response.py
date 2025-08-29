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
    """Enhanced response wrapper for LLM text generation.

    @public

    Structurally compatible with OpenAI ChatCompletion response format. All LLM provider
    responses are normalized to this format by LiteLLM proxy, ensuring consistent
    interface across providers (OpenAI, Anthropic, Google, Grok, etc.).

    Additional Attributes:
        headers: HTTP response headers including cost information. Only populated
                when using our client; will be empty dict if deserializing from JSON.
        model_options: Configuration used for this generation.

    Key Properties:
        content: Quick access to generated text content.
        usage: Token usage statistics (inherited).
        model: Model identifier used (inherited).
        id: Unique response ID (inherited).

    Example:
        >>> from ai_pipeline_core.llm import generate
        >>> response = await generate("gpt-5", messages="Hello")
        >>> print(response.content)  # Generated text
        >>> print(response.usage.total_tokens)  # Token count
        >>> print(response.headers.get("x-litellm-response-cost"))  # Cost

    Note:
        This class maintains full compatibility with ChatCompletion
        while adding pipeline-specific functionality.
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

        Convenience property for accessing the first choice's message
        content. Returns empty string if no content available.

        Returns:
            Generated text from the first choice, or empty string.

        Example:
            >>> response = await generate("gpt-5", messages="Hello")
            >>> text = response.content  # Direct access to generated text
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
        """Extract metadata for LMNR (Laminar) observability.

        Collects comprehensive metadata about the generation for
        tracing and monitoring in the LMNR platform.

        Returns:
            Dictionary containing:
            - LiteLLM headers (call ID, costs, etc.)
            - Token usage statistics
            - Model configuration
            - Cost information
            - Cached token counts
            - Reasoning token counts (for O1 models)

        Metadata structure:
            - litellm.*: All LiteLLM-specific headers
            - gen_ai.usage.*: Token usage statistics
            - gen_ai.response.*: Response identifiers
            - gen_ai.cost: Cost information
            - model_options.*: Configuration used

        Example:
            >>> response = await generate(...)
            >>> metadata = response.get_laminar_metadata()
            >>> print(f"Cost: ${metadata.get('gen_ai.cost', 0)}")
            >>> print(f"Tokens: {metadata.get('gen_ai.usage.total_tokens')}")

        Note:
            Used internally by the tracing system for observability.
            Cost is extracted from headers or usage object.
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

    Structurally compatible with OpenAI ChatCompletion response format. Extends ModelResponse
    with type-safe access to parsed Pydantic model instances.

    Type Parameter:
        T: The Pydantic model type for the structured output.

    Additional Features:
        - Type-safe access to parsed Pydantic model
        - Automatically parses structured JSON output from model response
        - All features of ModelResponse (cost, metadata, etc.)

    Example:
        >>> from pydantic import BaseModel
        >>> from ai_pipeline_core.llm import generate_structured
        >>>
        >>> class Analysis(BaseModel):
        ...     sentiment: float
        ...     summary: str
        >>>
        >>> response = await generate_structured(
        ...     "gpt-5",
        ...     response_format=Analysis,
        ...     messages="Analyze: ..."
        ... )
        >>>
        >>> # Type-safe access
        >>> analysis: Analysis = response.parsed
        >>> print(f"Sentiment: {analysis.sentiment}")
        >>>
        >>> # Still have access to metadata
        >>> print(f"Tokens used: {response.usage.total_tokens}")

    Note:
        The parsed property provides type-safe access to the
        validated Pydantic model instance.
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

        Provides type-safe access to the structured output that was
        generated according to the specified schema.

        Returns:
            Validated instance of the Pydantic model type T.

        Raises:
            ValueError: If no parsed content is available (should
                       not happen in normal operation).

        Example:
            >>> class UserInfo(BaseModel):
            ...     name: str
            ...     age: int
            >>>
            >>> response: StructuredModelResponse[UserInfo] = ...
            >>> user = response.parsed  # Type is UserInfo
            >>> print(f"{user.name} is {user.age} years old")

        Type Safety:
            The return type matches the type parameter T, providing
            full IDE support and type checking.

        Note:
            This property should always return a value for properly
            generated structured responses. ValueError indicates an
            internal error.
        """
        if self._parsed_value is not None:
            return self._parsed_value

        raise ValueError(
            "No parsed content available. This should not happen for StructuredModelResponse."
        )
