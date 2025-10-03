"""Configuration options for LLM generation.

Provides the ModelOptions class for configuring model behavior,
retry logic, and advanced features like web search and reasoning.
"""

from typing import Any, Literal

from pydantic import BaseModel


class ModelOptions(BaseModel):
    r"""Configuration options for LLM generation requests.

    ModelOptions encapsulates all configuration parameters for model
    generation, including model behavior settings, retry logic, and
    advanced features. All fields are optional with sensible defaults.

    Attributes:
        temperature: Controls randomness in generation (0.0-2.0).
                    Lower values = more deterministic, higher = more creative.
                    If None, the parameter is omitted from the API call,
                    causing the provider to use its own default (often 1.0).

        system_prompt: System-level instructions for the model.
                      Sets the model's behavior and persona.

        search_context_size: Web search result depth for search-enabled models.
                           Literal["low", "medium", "high"] | None
                           "low": Minimal context (~1-2 results)
                           "medium": Moderate context (~3-5 results)
                           "high": Extensive context (~6+ results)

        reasoning_effort: Reasoning intensity for models that support explicit reasoning.
                         Literal["low", "medium", "high"] | None
                         "low": Quick reasoning
                         "medium": Balanced reasoning
                         "high": Deep, thorough reasoning
                         Note: Availability and effect vary by provider and model. Only models
                         that expose an explicit reasoning control will honor this parameter.

        retries: Number of retry attempts on failure (default: 3).

        retry_delay_seconds: Seconds to wait between retries (default: 10).

        timeout: Maximum seconds to wait for response (default: 300).

        cache_ttl: Cache TTL for context messages (default: "5m").
                   String format like "60s", "5m", or None to disable caching.
                   Applied to the last context message for efficient token reuse.

        service_tier: API tier selection for performance/cost trade-offs.
                     "auto": Let API choose
                     "default": Standard tier
                     "flex": Flexible (cheaper, may be slower)
                     "scale": Scaled performance
                     "priority": Priority processing
                     Note: Service tiers are correct as of Q3 2025. Only OpenAI models
                     support this parameter. Other providers (Anthropic, Google, Grok)
                     silently ignore it.

        max_completion_tokens: Maximum tokens to generate.
                              None uses model default.

        stop: Stop sequences that halt generation when encountered.
             Can be a single string or list of strings.
             When the model generates any of these sequences, it stops immediately.
             Maximum of 4 stop sequences supported by most providers.

        response_format: Pydantic model class for structured output.
                        Pass a Pydantic model; the client converts it to JSON Schema.
                        Set automatically by generate_structured().
                        Structured output support varies by provider and model.

        verbosity: Controls output verbosity for models that support it.
                  Literal["low", "medium", "high"] | None
                  "low": Minimal output
                  "medium": Standard output
                  "high": Detailed output
                  Note: Only some models support verbosity control.

        usage_tracking: Enable token usage tracking in API responses (default: True).
                       When enabled, adds {"usage": {"include": True}} to extra_body.
                       Disable for providers that don't support usage tracking.

        user: User identifier for cost tracking and monitoring.
             A unique identifier representing the end-user, which can help track costs
             and detect abuse. Maximum length is typically 256 characters.
             Useful for multi-tenant applications or per-user billing.

        extra_body: Additional provider-specific parameters to pass in request body.
                   Dictionary of custom parameters not covered by standard options.
                   Merged with usage_tracking if both are set.
                   Useful for beta features or provider-specific capabilities.

    Example:
        >>> # Basic configuration
        >>> options = ModelOptions(
        ...     temperature=0.7,
        ...     max_completion_tokens=1000
        ... )
        >>>
        >>> # With system prompt
        >>> options = ModelOptions(
        ...     system_prompt="You are a helpful coding assistant",
        ...     temperature=0.3  # Lower for code generation
        ... )
        >>>
        >>> # With custom cache TTL
        >>> options = ModelOptions(
        ...     cache_ttl="300s",  # Cache context for 5 minutes
        ...     max_completion_tokens=1000
        ... )
        >>>
        >>> # Disable caching
        >>> options = ModelOptions(
        ...     cache_ttl=None,  # No context caching
        ...     temperature=0.5
        ... )
        >>>
        >>> # For search-enabled models
        >>> options = ModelOptions(
        ...     search_context_size="high",  # Get more search results
        ...     max_completion_tokens=2000
        ... )
        >>>
        >>> # For reasoning models
        >>> options = ModelOptions(
        ...     reasoning_effort="high",  # Deep reasoning
        ...     timeout=600  # More time for complex reasoning
        ... )
        >>>
        >>> # With stop sequences
        >>> options = ModelOptions(
        ...     stop=["STOP", "END", "\n\n"],  # Stop on these sequences
        ...     temperature=0.7
        ... )
        >>>
        >>> # With custom extra_body parameters
        >>> options = ModelOptions(
        ...     extra_body={"custom_param": "value", "beta_feature": True},
        ...     usage_tracking=True  # Still tracks usage alongside custom params
        ... )
        >>>
        >>> # With user tracking for cost monitoring
        >>> options = ModelOptions(
        ...     user="user_12345",  # Track costs per user
        ...     temperature=0.7
        ... )

    Note:
        - Not all options apply to all models
        - search_context_size only works with search models
        - reasoning_effort only works with models that support explicit reasoning
        - response_format is set internally by generate_structured()
        - cache_ttl accepts formats like "120s", "5m" (default), "1h" or None to disable caching
        - stop sequences are limited to 4 by most providers
        - user identifier helps track costs per end-user (max 256 chars)
        - extra_body allows passing provider-specific parameters
        - usage_tracking is enabled by default for cost monitoring
    """

    temperature: float | None = None
    system_prompt: str | None = None
    search_context_size: Literal["low", "medium", "high"] | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    retries: int = 3
    retry_delay_seconds: int = 10
    timeout: int = 600
    cache_ttl: str | None = "5m"
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    response_format: type[BaseModel] | None = None
    verbosity: Literal["low", "medium", "high"] | None = None
    usage_tracking: bool = True
    user: str | None = None
    extra_body: dict[str, Any] | None = None

    def to_openai_completion_kwargs(self) -> dict[str, Any]:
        """Convert options to OpenAI API completion parameters.

        Transforms ModelOptions fields into the format expected by
        the OpenAI completion API. Only includes non-None values.

        Returns:
            Dictionary with OpenAI API parameters:
            - Always includes 'timeout' and 'extra_body'
            - Conditionally includes other parameters if set
            - Maps search_context_size to web_search_options
            - Passes reasoning_effort directly

        API parameter mapping:
            - temperature -> temperature
            - max_completion_tokens -> max_completion_tokens
            - stop -> stop (string or list of strings)
            - reasoning_effort -> reasoning_effort
            - search_context_size -> web_search_options.search_context_size
            - response_format -> response_format
            - service_tier -> service_tier
            - verbosity -> verbosity
            - user -> user (for cost tracking)
            - extra_body -> extra_body (merged with usage tracking)

        Web Search Structure:
            When search_context_size is set, creates:
            {"web_search_options": {"search_context_size": "low|medium|high"}}
            Non-search models silently ignore this parameter.

        Example:
            >>> options = ModelOptions(temperature=0.5, timeout=60)
            >>> kwargs = options.to_openai_completion_kwargs()
            >>> kwargs
            {'timeout': 60, 'extra_body': {}, 'temperature': 0.5}

        Note:
            - system_prompt is handled separately in _process_messages()
            - retries and retry_delay_seconds are used by retry logic
            - extra_body always includes usage tracking for cost monitoring
        """
        kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            "extra_body": {},
        }

        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        if self.temperature:
            kwargs["temperature"] = self.temperature

        if self.max_completion_tokens:
            kwargs["max_completion_tokens"] = self.max_completion_tokens

        if self.stop:
            kwargs["stop"] = self.stop

        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        if self.search_context_size:
            kwargs["web_search_options"] = {"search_context_size": self.search_context_size}

        if self.response_format:
            kwargs["response_format"] = self.response_format

        if self.service_tier:
            kwargs["service_tier"] = self.service_tier

        if self.verbosity:
            kwargs["verbosity"] = self.verbosity

        if self.user:
            kwargs["user"] = self.user

        if self.usage_tracking:
            kwargs["extra_body"]["usage"] = {"include": True}

        return kwargs
