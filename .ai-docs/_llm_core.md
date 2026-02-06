# MODULE: _llm_core
# CLASSES: ModelOptions, Citation, ModelResponse, Role, TextContent, ImageContent, PDFContent, CoreMessage, TokenUsage
# DEPENDS: BaseModel, Generic, StrEnum
# SIZE: ~26KB

# === DEPENDENCIES (Resolved) ===

class BaseModel:
    """Pydantic base model. Fields are typed class attributes."""
    ...

class Generic:
    """Python generic base class for parameterized types."""
    ...

class StrEnum:
    """String enumeration base class."""
    ...

# === PUBLIC API ===

class ModelOptions(BaseModel):
    """Configuration options for LLM generation requests.

ModelOptions encapsulates all configuration parameters for model
generation, including model behavior settings, retry logic, and
advanced features. All fields are optional with sensible defaults.
Extra fields are forbidden to catch typos and incorrect usage.

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

    retry_delay_seconds: Seconds to wait between retries (default: 20).

    timeout: Maximum seconds to wait for response (default: 600).

    cache_ttl: Cache TTL for context messages (default: "300s").
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

    metadata: Custom metadata tags for tracking and observability.
             Dictionary of string key-value pairs for tagging requests.
             Useful for tracking experiments, versions, or custom attributes.
             Maximum of 16 key-value pairs, each key/value max 64 characters.
             Passed through to LMNR tracing and API provider metadata.

    extra_body: Additional provider-specific parameters to pass in request body.
               Dictionary of custom parameters not covered by standard options.
               Merged with usage_tracking if both are set.
               Useful for beta features or provider-specific capabilities.

Not all options apply to all models. search_context_size only works with search models,
reasoning_effort only works with models that support explicit reasoning, and
response_format is set internally by generate_structured(). cache_ttl accepts formats
like "120s", "5m", "1h" or None (default: "300s"). Stop sequences are limited to 4 by
most providers."""
    model_config = {'extra': 'forbid'}
    temperature: float | None = None
    system_prompt: str | None = None
    search_context_size: Literal['low', 'medium', 'high'] | None = None
    reasoning_effort: Literal['low', 'medium', 'high'] | None = None
    retries: int = 3
    retry_delay_seconds: int = 20
    timeout: int = 600
    cache_ttl: str | None = '300s'
    service_tier: Literal['auto', 'default', 'flex', 'scale', 'priority'] | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    response_format: type[BaseModel] | None = None
    verbosity: Literal['low', 'medium', 'high'] | None = None
    stream: bool = False
    usage_tracking: bool = True
    user: str | None = None
    metadata: dict[str, str] | None = None
    extra_body: dict[str, Any] | None = None

    def to_openai_completion_kwargs(self) -> dict[str, Any]:  # noqa: C901
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
            - metadata -> metadata (for tracking/observability)
            - extra_body -> extra_body (merged with usage tracking)

        Web Search Structure:
            When search_context_size is set, creates:
            {"web_search_options": {"search_context_size": "low|medium|high"}}
            Non-search models silently ignore this parameter.

        system_prompt is handled separately in _process_messages().
        retries and retry_delay_seconds are used by retry logic.
        extra_body always includes usage tracking for cost monitoring.
        """
        kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            "extra_body": {},
        }

        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        if self.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_completion_tokens

        if self.stop is not None:
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

        if self.metadata:
            kwargs["metadata"] = self.metadata

        if self.usage_tracking:
            kwargs["extra_body"]["usage"] = {"include": True}
            kwargs["stream_options"] = {"include_usage": True}

        return kwargs


@dataclass(frozen=True, slots=True)
class Citation:
    """A URL citation returned by search-enabled models.

The start_index and end_index fields indicate character positions in the response content
where the citation applies. Note that index behavior varies by model."""
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
for structured responses (use MyModel.model_validate(response.parsed) to reconstruct)."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    content: str
    parsed: T
    reasoning_content: str = ''
    citations: tuple[Citation, ...] = ()
    usage: TokenUsage
    cost: float | None = None
    model: str
    response_id: str = ''
    metadata: Mapping[str, Any] = Field(default_factory=dict)
    thinking_blocks: tuple[dict[str, Any], ...] | None = None
    provider_specific_fields: dict[str, Any] | None = None

    def get_laminar_metadata(self) -> dict[str, str | int | float]:
        """Extract metadata for LMNR (Laminar) observability.

        Returns dictionary with token usage, cost, and timing information.
        """
        result: dict[str, str | int | float] = {
            "gen_ai.response.id": self.response_id,
            "gen_ai.system": "litellm",
            "gen_ai.usage.prompt_tokens": self.usage.prompt_tokens,
            "gen_ai.usage.completion_tokens": self.usage.completion_tokens,
            "gen_ai.usage.total_tokens": self.usage.total_tokens,
        }

        if self.usage.cached_tokens:
            result["gen_ai.usage.cached_tokens"] = self.usage.cached_tokens

        if self.usage.reasoning_tokens:
            result["gen_ai.usage.reasoning_tokens"] = self.usage.reasoning_tokens

        if self.cost is not None:
            result["gen_ai.usage.output_cost"] = self.cost
            result["gen_ai.usage.cost"] = self.cost
            result["gen_ai.cost"] = self.cost

        # Include timing metadata
        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float)):
                result[key] = value

        return result

    @field_serializer("citations", when_used="always")
    def serialize_citations(self, value: tuple[Citation, ...]) -> list[dict[str, Any]]:
        """Serialize citations dataclass to dict."""
        return [{"title": c.title, "url": c.url, "start_index": c.start_index, "end_index": c.end_index} for c in value]

    @field_serializer("parsed", when_used="always")
    def serialize_parsed(self, value: T) -> Any:
        """Serialize parsed value - convert BaseModel to dict."""
        if isinstance(value, BaseModel):
            return value.model_dump()
        return value


class Role(StrEnum):
    """Message role in conversation."""
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'


class TextContent(BaseModel):
    """Plain text content part."""
    model_config = ConfigDict(frozen=True)
    type: Literal['text'] = 'text'
    text: str


class ImageContent(BaseModel):
    """Image content with binary data.

Uses Base64Bytes for automatic base64 encoding/decoding in JSON serialization."""
    model_config = ConfigDict(frozen=True)
    type: Literal['image'] = 'image'
    data: Base64Bytes
    mime_type: Literal['image/jpeg', 'image/png', 'image/gif', 'image/webp']


class PDFContent(BaseModel):
    """PDF document content with binary data.

Uses Base64Bytes for automatic base64 encoding/decoding in JSON serialization."""
    model_config = ConfigDict(frozen=True)
    type: Literal['pdf'] = 'pdf'
    data: Base64Bytes


class CoreMessage(BaseModel):
    """A single message in a conversation.

Content can be:
- str: Plain text (converted to TextContent internally)
- ContentPart: Single content part (text, image, or PDF)
- tuple[ContentPart, ...]: Multiple content parts (multimodal message)"""
    model_config = ConfigDict(frozen=True)
    role: Role
    content: str | ContentPart | tuple[ContentPart, ...]


class TokenUsage(BaseModel):
    """Token usage statistics from an LLM call."""
    model_config = ConfigDict(frozen=True)
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int = 0
    reasoning_tokens: int = 0


# === FUNCTIONS ===

async def generate(
    messages: list[CoreMessage],
    *,
    model: str,
    model_options: ModelOptions | None = None,
    purpose: str | None = None,
    expected_cost: float | None = None,
    context_count: int = 0,
) -> ModelResponse[str]:
    """Primitive LLM generation - NO Document dependency.

    This is the Layer 1 function used by internal modules. For app code,
    use the llm.Conversation class instead.

    Args:
        messages: List of CoreMessage objects.
        model: Model identifier (e.g., "gpt-5.1", "gemini-3-flash").
        model_options: Optional configuration for the model.
        purpose: Optional semantic label for tracing span name.
        expected_cost: Optional expected cost for cost-tracking attributes.
        context_count: Number of messages from start to apply cache_control to.

    Returns:
        ModelResponse[str] with content, parsed (same as content), usage, cost, etc.

    Raises:
        ValueError: If messages is empty or model is not provided.
        LLMError: If generation fails after all retries.
    """
    if not messages:
        raise ValueError("messages must not be empty")
    if not model:
        raise ValueError("model must be provided")

    if model_options is None:
        model_options = ModelOptions()
    else:
        model_options = model_options.model_copy()

    # Inject system_prompt as first system message if provided
    effective_messages = list(messages)
    effective_context_count = context_count
    if model_options.system_prompt:
        system_msg = CoreMessage(role=Role.SYSTEM, content=model_options.system_prompt)
        effective_messages = [system_msg] + effective_messages
        # Include system prompt in cache prefix
        effective_context_count = context_count + 1 if context_count > 0 else 1

    api_messages = _messages_to_api(effective_messages)

    if "openrouter" in settings.openai_base_url.lower():
        model = _model_name_to_openrouter_model(model)

    # Apply caching
    cache_ttl = model_options.cache_ttl
    if cache_ttl and effective_context_count > 0:
        # Gemini requires minimum ~10k tokens for caching to be effective
        if "gemini" in model.lower() and _estimate_token_count(api_messages[:effective_context_count]) < 10000:
            cache_ttl = None
            logger.debug("Disabling cache for Gemini with <10k context tokens")

        if cache_ttl:
            _apply_cache_control(api_messages, cache_ttl, effective_context_count)

    completion_kwargs: dict[str, Any] = {
        **model_options.to_openai_completion_kwargs(),
    }

    # Add cache key if caching
    if cache_ttl and effective_context_count > 0:
        completion_kwargs["prompt_cache_key"] = _compute_cache_key(api_messages[:effective_context_count], model_options.system_prompt)

    for attempt in range(model_options.retries):
        try:
            async with AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            ) as client:
                with Laminar.start_as_current_span(purpose or model, span_type="LLM", input=api_messages) as span:
                    # Use streaming or non-streaming based on options
                    if model_options.stream:
                        response, metadata, stream_usage = await _generate_streaming(client, model, api_messages, completion_kwargs)
                    else:
                        response, metadata, stream_usage = await _generate_non_streaming(client, model, api_messages, completion_kwargs)

                    # Normalize response to fix provider bugs
                    for choice in response.choices:
                        # Fix role duplication bug (some providers return "assistantassistant")
                        if hasattr(choice.message, "role") and choice.message.role != "assistant":
                            object.__setattr__(choice.message, "role", "assistant")
                        # Fix invalid finish_reason
                        if choice.finish_reason not in _VALID_FINISH_REASONS:
                            object.__setattr__(choice, "finish_reason", "stop")

                    content = response.choices[0].message.content or ""
                    # Strip thinking tags if present
                    if "</think>" in content:
                        content = content.split("</think>")[-1].strip()

                    reasoning_content = ""
                    msg = response.choices[0].message
                    if rc := getattr(msg, "reasoning_content", None):
                        reasoning_content = rc
                    elif "</think>" in (msg.content or ""):
                        reasoning_content = (msg.content or "").split("</think>")[0].strip()

                    # Extract provider-specific fields for multi-turn reasoning
                    thinking_blocks: tuple[dict[str, Any], ...] | None = None
                    provider_specific_fields: dict[str, Any] | None = None

                    if hasattr(msg, "thinking_blocks") and msg.thinking_blocks:
                        thinking_blocks = tuple(tb if isinstance(tb, dict) else tb.__dict__ for tb in msg.thinking_blocks)

                    if hasattr(msg, "provider_specific_fields") and msg.provider_specific_fields:
                        provider_specific_fields = dict(msg.provider_specific_fields)

                    usage = _extract_usage(response)
                    if stream_usage:
                        # Override with streaming usage if available
                        usage = TokenUsage(
                            prompt_tokens=stream_usage.prompt_tokens,
                            completion_tokens=stream_usage.completion_tokens,
                            total_tokens=stream_usage.total_tokens,
                            cached_tokens=usage.cached_tokens,
                            reasoning_tokens=usage.reasoning_tokens,
                        )
                    cost = _extract_cost(response)

                    # Set span attributes
                    span_attrs: dict[str, Any] = {
                        "time_taken": metadata["time_taken"],
                        "gen_ai.usage.prompt_tokens": usage.prompt_tokens,
                        "gen_ai.usage.completion_tokens": usage.completion_tokens,
                        "gen_ai.usage.total_tokens": usage.total_tokens,
                    }
                    if "first_token_time" in metadata:
                        span_attrs["first_token_time"] = metadata["first_token_time"]
                    if usage.cached_tokens:
                        span_attrs["gen_ai.usage.cached_tokens"] = usage.cached_tokens
                    if cost:
                        span_attrs["gen_ai.cost"] = cost
                    if expected_cost is not None:
                        span_attrs["expected_cost"] = expected_cost
                    if purpose:
                        span_attrs["purpose"] = purpose

                    span.set_attributes(span_attrs)

                    # Output both reasoning and content if available
                    outputs = [o for o in (reasoning_content, content) if o]
                    Laminar.set_span_output(outputs if len(outputs) > 1 else content)

                    if not content:
                        raise ValueError("Empty response content")

                    # Extract citations (filter by type to exclude non-URL annotations)
                    citations: tuple[Citation, ...] = ()
                    if annotations := response.choices[0].message.annotations:
                        url_citations = [a for a in annotations if getattr(a, "type", None) == "url_citation" and a.url_citation]
                        citations = tuple(
                            Citation(
                                title=a.url_citation.title,
                                url=a.url_citation.url,
                                start_index=a.url_citation.start_index,
                                end_index=a.url_citation.end_index,
                            )
                            for a in url_citations
                        )

                    # Build ModelResponse[str] for unstructured output
                    return ModelResponse[str](
                        content=content,
                        parsed=content,  # For unstructured, parsed = content
                        reasoning_content=reasoning_content,
                        citations=citations,
                        usage=usage,
                        cost=cost,
                        model=model,
                        response_id=response.id or "",
                        metadata=metadata,
                        thinking_blocks=thinking_blocks,
                        provider_specific_fields=provider_specific_fields,
                    )

        except TimeoutError:
            logger.warning(f"LLM generation timeout (attempt {attempt + 1}/{model_options.retries})")
            if attempt == model_options.retries - 1:
                raise LLMError("Exhausted all retry attempts for LLM generation.") from None
        except Exception as e:
            logger.warning(f"LLM generation failed (attempt {attempt + 1}/{model_options.retries}): {e}")

            # On non-timeout errors, disable caching to avoid cache-related failures
            completion_kwargs.setdefault("extra_body", {})
            completion_kwargs["extra_body"]["cache"] = {"no-cache": True}
            completion_kwargs.pop("prompt_cache_key", None)
            api_messages = _remove_cache_control(api_messages)

            if attempt == model_options.retries - 1:
                raise LLMError("Exhausted all retry attempts for LLM generation.") from e

        await asyncio.sleep(model_options.retry_delay_seconds)

    raise LLMError("Unknown error occurred during LLM generation.")

async def generate_structured(
    messages: list[CoreMessage],
    response_format: type[T],
    *,
    model: str,
    model_options: ModelOptions | None = None,
    purpose: str | None = None,
    expected_cost: float | None = None,
    context_count: int = 0,
) -> ModelResponse[T]:
    """Primitive structured LLM generation - NO Document dependency.

    This is the Layer 1 function used by internal modules. For app code,
    use the llm.Conversation class instead.

    Args:
        messages: List of CoreMessage objects.
        response_format: Pydantic model class for structured output.
        model: Model identifier.
        model_options: Optional configuration for the model.
        purpose: Optional semantic label for tracing span name.
        expected_cost: Optional expected cost for cost-tracking attributes.
        context_count: Number of messages from start to apply cache_control to.

    Returns:
        ModelResponse[T] with .parsed returning the typed model instance.

    Raises:
        ValueError: If messages is empty or model is not provided.
        LLMError: If generation and parsing fail after all retries.
    """
    if model_options is None:
        model_options = ModelOptions()
    else:
        model_options = model_options.model_copy()

    model_options.response_format = response_format

    last_error: Exception | None = None

    for attempt in range(model_options.retries):
        try:
            # Call generate to get the raw response
            result = await generate(
                messages,
                model=model,
                model_options=model_options,
                purpose=purpose,
                expected_cost=expected_cost,
                context_count=context_count,
            )

            # Parse the content into the response format
            parsed = response_format.model_validate_json(result.content)

            # Return ModelResponse[T] with the parsed model
            return ModelResponse[T](
                content=result.content,
                parsed=parsed,
                reasoning_content=result.reasoning_content,
                citations=result.citations,
                usage=result.usage,
                cost=result.cost,
                model=result.model,
                response_id=result.response_id,
                metadata=result.metadata,
                thinking_blocks=result.thinking_blocks,
                provider_specific_fields=result.provider_specific_fields,
            )
        except ValidationError as e:
            last_error = e
            logger.warning(f"Structured output validation failed (attempt {attempt + 1}/{model_options.retries}): {e}")
            if attempt < model_options.retries - 1:
                # Disable cache for retry
                if model_options.extra_body is None:
                    model_options.extra_body = {}
                model_options.extra_body["cache"] = {"no-cache": True}
                await asyncio.sleep(model_options.retry_delay_seconds)

    raise LLMError(f"Structured output validation failed after {model_options.retries} attempts") from last_error

# === EXAMPLES ===
# No test examples available.
