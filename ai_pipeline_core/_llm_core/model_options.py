"""Configuration options for LLM generation.

Provides the ModelOptions class for configuring model behavior,
retry logic, and advanced features like web search and reasoning.
"""

from collections.abc import Mapping
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, field_validator


class ModelOptions(BaseModel):
    r"""Configuration options for LLM generation requests.

    All fields are optional with sensible defaults. Extra fields are forbidden.

    Non-obvious behaviors:
    - cache_ttl: string format ("60s", "5m", "1h") or None to disable. Default "300s".
    - service_tier: only OpenAI models honor this; other providers silently ignore it.
    - stop: accepts a single string or list; coerced to tuple. Max 4 by most providers.
    - usage_tracking: when True (default), injects {"usage": {"include": True}} into extra_body.
    - extra_body: merged with usage_tracking dict if both are set.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float | None = None
    system_prompt: str | None = None
    search_context_size: Literal["low", "medium", "high"] | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    retries: int = 3
    retry_delay_seconds: int = 20
    timeout: int = 600
    cache_ttl: str | None = "300s"
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None
    max_completion_tokens: int | None = None
    stop: tuple[str, ...] | None = None
    verbosity: Literal["low", "medium", "high"] | None = None
    stream: bool = False
    usage_tracking: bool = True
    user: str | None = None
    metadata: Mapping[str, str] | None = None
    extra_body: Mapping[str, Any] | None = None

    @field_validator("stop", mode="before")
    @classmethod
    def _coerce_stop(cls, v: Any) -> tuple[str, ...] | None:
        if v is None:
            return None
        if isinstance(v, str):
            return (v,)
        if isinstance(v, (list, tuple)):
            return tuple(str(s) for s in cast(list[Any], v))
        return v

    def to_openai_completion_kwargs(self) -> dict[str, Any]:
        """Convert options to OpenAI API completion parameters.

        Only includes non-None values. Framework-only fields (system_prompt,
        retries, retry_delay_seconds, cache_ttl, stream) are excluded.
        """
        kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            "extra_body": dict(self.extra_body) if self.extra_body else {},
        }

        # Direct 1:1 field mappings (field name == API kwarg name)
        for attr in ("temperature", "max_completion_tokens", "reasoning_effort", "service_tier", "verbosity", "user"):
            if (v := getattr(self, attr)) is not None:
                kwargs[attr] = v

        # Fields needing transformation
        if self.stop is not None:
            kwargs["stop"] = list(self.stop)
        if self.search_context_size:
            kwargs["web_search_options"] = {"search_context_size": self.search_context_size}
        if self.metadata:
            kwargs["metadata"] = dict(self.metadata)
        if self.usage_tracking:
            kwargs["extra_body"]["usage"] = {"include": True}
            kwargs["stream_options"] = {"include_usage": True}

        return kwargs
