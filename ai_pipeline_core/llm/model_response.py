import copy
from typing import Any, Generic, TypeVar

from openai.types.chat import ChatCompletion, ParsedChatCompletion
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class ModelResponse(ChatCompletion):
    """Response from an LLM without structured output."""

    headers: dict[str, str] = Field(default_factory=dict)
    model_options: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, chat_completion: ChatCompletion | None = None, **kwargs: Any) -> None:
        """Initialize ModelResponse from a ChatCompletion."""
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
        """Get the text content of the response."""
        return self.choices[0].message.content or ""

    def set_model_options(self, options: dict[str, Any]) -> None:
        """Set the model options."""
        self.model_options = copy.deepcopy(options)
        if "messages" in self.model_options:
            del self.model_options["messages"]

    def set_headers(self, headers: dict[str, str]) -> None:
        """Set the response headers."""
        self.headers = copy.deepcopy(headers)

    def get_laminar_metadata(self) -> dict[str, str | int | float]:
        """Extract metadata for Laminar observability logging."""
        metadata: dict[str, str | int | float] = {}

        litellm_id = self.headers.get("x-litellm-call-id")
        cost = float(self.headers.get("x-litellm-response-cost") or 0)

        # Add all x-litellm-* headers
        for header, value in self.headers.items():
            if header.startswith("x-litellm-"):
                header_name = header.replace("x-litellm-", "").lower()
                metadata[f"litellm.{header_name}"] = value

        # Add base metadata
        metadata.update(
            {
                "gen_ai.response.id": litellm_id or self.id,
                "gen_ai.response.model": self.model,
                "get_ai.system": "litellm",
            }
        )

        # Add usage metadata if available
        if self.usage:
            metadata.update(
                {
                    "gen_ai.usage.prompt_tokens": self.usage.prompt_tokens,
                    "gen_ai.usage.completion_tokens": self.usage.completion_tokens,
                    "gen_ai.usage.total_tokens": self.usage.total_tokens,
                }
            )

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
            metadata.update(
                {
                    "gen_ai.usage.output_cost": cost,
                    "gen_ai.usage.cost": cost,
                    "get_ai.cost": cost,
                }
            )

        if self.model_options:
            for key, value in self.model_options.items():
                metadata[f"model_options.{key}"] = str(value)

        return metadata


class StructuredModelResponse(ModelResponse, Generic[T]):
    """Response from an LLM with structured output of type T."""

    def __init__(
        self,
        chat_completion: ChatCompletion | None = None,
        parsed_value: T | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize StructuredModelResponse with a parsed value.

        Args:
            chat_completion: The base chat completion
            parsed_value: The parsed structured output
            **kwargs: Additional arguments for ChatCompletion
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
        """Get the parsed structured output.

        Returns:
            The parsed value of type T.

        Raises:
            ValueError: If no parsed content is available.
        """
        if self._parsed_value is not None:
            return self._parsed_value

        raise ValueError(
            "No parsed content available. This should not happen for StructuredModelResponse."
        )
