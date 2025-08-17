from typing import Any, Literal

from pydantic import BaseModel


class ModelOptions(BaseModel):
    system_prompt: str | None = None
    search_context_size: Literal["low", "medium", "high"] | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    retries: int = 3
    retry_delay_seconds: int = 10
    timeout: int = 300
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None
    max_completion_tokens: int | None = None
    response_format: type[BaseModel] | None = None

    def to_openai_completion_kwargs(self) -> dict[str, Any]:
        """Convert ModelOptions to OpenAI completion kwargs."""
        kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            "extra_body": {},
        }

        if self.max_completion_tokens:
            kwargs["max_completion_tokens"] = self.max_completion_tokens

        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        if self.search_context_size:
            kwargs["web_search_options"] = {"search_context_size": self.search_context_size}

        if self.response_format:
            kwargs["response_format"] = self.response_format

        if self.service_tier:
            kwargs["service_tier"] = self.service_tier

        return kwargs
