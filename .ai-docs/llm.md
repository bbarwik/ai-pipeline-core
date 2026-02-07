# MODULE: llm
# CLASSES: ModelOptions, Citation, ModelResponse, Role, TextContent, ImageContent, PDFContent, CoreMessage, TokenUsage, Conversation
# DEPENDS: BaseModel, Generic, StrEnum
# SIZE: ~29KB

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


class Conversation(BaseModel, Generic[T]):
    """Immutable conversation state for LLM interactions.

After calling send() or send_structured(), the returned Conversation has
response properties accessible directly (content, reasoning_content, usage,
cost, parsed, citations).

Generic parameter T represents the type of `.parsed` from the last
send_structured() call. For conversations created with send(), T is None.

Attributes:
    model: The model identifier (e.g., "gpt-5.1", "gemini-3-flash").
    context: Cacheable prefix documents.
    messages: Conversation history (Documents and ModelResponses).
    model_options: Optional model configuration.
    enable_substitutor: Whether to enable URL/address shortening (default True)."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    model: str
    context: tuple[Document, ...] = ()
    messages: tuple[MessageType | _UserMessage, ...] = ()
    model_options: ModelOptions | None = None
    enable_substitutor: bool = True
    substitutor: URLSubstitutor | None = Field(default=None, exclude=True, repr=False)

    @property
    def approximate_tokens_count(self) -> int:
        """Approximate token count for all context and messages."""
        total = 0
        for item in self.context + self.messages:
            if isinstance(item, ModelResponse):
                total += len(item.content) // 4
                if reasoning := item.reasoning_content:
                    total += len(reasoning) // 4
            elif isinstance(item, Document):
                if item.is_text:
                    total += len(item.content) // 4
                elif item.is_image or item.is_pdf:
                    total += _TOKENS_PER_IMAGE
                for att in item.attachments:
                    if att.is_text:
                        total += len(att.content) // 4
                    elif att.is_image or att.is_pdf:
                        total += _TOKENS_PER_IMAGE
        return total

    @property
    def citations(self) -> tuple[Citation, ...]:
        """Citations from last send() call (for search-enabled models)."""
        if r := self._last_response:
            return tuple(r.citations)
        return ()

    @property
    def content(self) -> str:
        """Response text from last send() call."""
        if r := self._last_response:
            return r.content
        return ""

    @property
    def cost(self) -> float | None:
        """Cost from last send() call (if available)."""
        if r := self._last_response:
            return r.cost
        return None

    @property
    def parsed(self) -> T | None:
        """Parsed Pydantic model from last send_structured() call."""
        if r := self._last_response:
            # For ModelResponse[str], parsed is the content string
            # For ModelResponse[SomeModel], parsed is the model instance (or dict after deser)
            if isinstance(r.parsed, str):
                return None  # Unstructured response, no typed parsed
            return r.parsed  # type: ignore[return-value]
        return None

    @property
    def reasoning_content(self) -> str:
        """Reasoning content from last send() call (if model supports it)."""
        if r := self._last_response:
            return r.reasoning_content
        return ""

    @property
    def usage(self) -> TokenUsage:
        """Token usage from last send() call."""
        if r := self._last_response:
            return r.usage
        return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    @field_validator("context", mode="before")
    @classmethod
    def convert_context_to_tuple(cls, v: list[Document] | tuple[Document, ...] | None) -> tuple[Document, ...]:
        """Coerce context list or None to immutable tuple."""
        if v is None:
            return ()
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("messages", mode="before")
    @classmethod
    def convert_messages_to_tuple(cls, v: list[MessageType] | tuple[MessageType, ...] | None) -> tuple[MessageType, ...]:
        """Coerce messages list or None to immutable tuple."""
        if v is None:
            return ()
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("model")
    @classmethod
    def validate_model_not_empty(cls, v: str) -> str:
        """Reject empty model name."""
        if not v:
            raise ValueError("model must be non-empty")
        return v

    def restore_content(self, text: str) -> str:
        """Restore shortened URLs/addresses in text to originals.

        Use this if you need to restore content that was shortened by the substitutor.
        """
        if self.substitutor:
            return self.substitutor.restore(text)
        return text

    async def send(
        self,
        content: ConversationContent,
        *,
        purpose: str | None = None,
        expected_cost: float | None = None,
    ) -> "Conversation[None]":
        """Send message, returns NEW Conversation with response.

        Args:
            content: Message content - str, Document, or list of Documents.
            purpose: Optional semantic label for tracing.
            expected_cost: Optional expected cost for tracking.

        Returns:
            New Conversation[None] with response accessible via .content, .reasoning_content, etc.
        """
        new_messages, response = await self._execute_send(content, None, purpose, expected_cost)
        return Conversation[None](
            model=self.model,
            context=self.context,
            messages=new_messages + (response,),
            model_options=self.model_options,
            enable_substitutor=self.enable_substitutor,
            substitutor=self.substitutor,
        )

    async def send_structured(
        self,
        content: ConversationContent,
        response_format: type[U],
        *,
        purpose: str | None = None,
        expected_cost: float | None = None,
    ) -> "Conversation[U]":
        """Send message expecting structured response.

        Args:
            content: Message content - str, Document, or list of Documents.
            response_format: Pydantic model class for structured output.
            purpose: Optional semantic label for tracing.
            expected_cost: Optional expected cost for tracking.

        Returns:
            New Conversation[U] with .parsed returning U instance.
        """
        new_messages, response = await self._execute_send(content, response_format, purpose, expected_cost)
        return Conversation[U](
            model=self.model,
            context=self.context,
            messages=new_messages + (response,),
            model_options=self.model_options,
            enable_substitutor=self.enable_substitutor,
            substitutor=self.substitutor,
        )

    def to_json(self) -> str:
        """Serialize conversation to JSON string for debugging.

        Note: Deserialization is not supported. Conversation is designed for
        transient use within a single task, not for persistence/restore.
        """
        return self.model_dump_json()

    def with_context(self, *docs: Document) -> "Conversation[T]":
        """Return NEW Conversation with documents added to context."""
        return Conversation[T](
            model=self.model,
            context=self.context + docs,
            messages=self.messages,
            model_options=self.model_options,
            enable_substitutor=self.enable_substitutor,
            substitutor=self.substitutor,
        )

    def with_document(self, doc: Document) -> "Conversation[T]":
        """Return NEW Conversation with document appended to messages."""
        return Conversation[T](
            model=self.model,
            context=self.context,
            messages=self.messages + (doc,),
            model_options=self.model_options,
            enable_substitutor=self.enable_substitutor,
            substitutor=self.substitutor,
        )

    def with_model_options(self, options: ModelOptions) -> "Conversation[T]":
        """Return NEW Conversation with updated model options."""
        return Conversation[T](
            model=self.model,
            context=self.context,
            messages=self.messages,
            model_options=options,
            enable_substitutor=self.enable_substitutor,
            substitutor=self.substitutor,
        )

    def with_substitutor(self, enabled: bool = True) -> "Conversation[T]":
        """Return NEW Conversation with substitutor enabled/disabled."""
        return Conversation[T](
            model=self.model,
            context=self.context,
            messages=self.messages,
            model_options=self.model_options,
            enable_substitutor=enabled,
            substitutor=self.substitutor if enabled else None,
        )


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
    options = model_options.model_copy() if model_options else ModelOptions()
    return await _generate_impl(
        messages,
        model=model,
        model_options=options,
        purpose=purpose,
        expected_cost=expected_cost,
        context_count=context_count,
    )

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
    options = model_options.model_copy() if model_options else ModelOptions()
    options.response_format = response_format
    return await _generate_impl(
        messages,
        model=model,
        model_options=options,
        purpose=purpose,
        expected_cost=expected_cost,
        context_count=context_count,
    )

# === EXAMPLES (from tests/) ===

# Example: Citation has slots
# Source: tests/llm/test_verified_issues.py:287
def test_citation_has_slots(self):
    """Citation dataclass should have __slots__."""
    assert hasattr(Citation, "__slots__"), "Citation should have slots=True"

# Example: Citations empty by default
# Source: tests/llm/test_model_response.py:156
def test_citations_empty_by_default(self):
    """Test citations returns empty tuple by default."""
    response = create_test_model_response(content="No citations")
    assert response.citations == ()

# Example: Citations preserved
# Source: tests/llm/test_model_response.py:161
def test_citations_preserved(self):
    """Test citations are preserved when set."""
    citations = (
        Citation(title="Page 1", url="https://example.com", start_index=0, end_index=10),
        Citation(title="Page 2", url="https://other.com", start_index=20, end_index=30),
    )
    response = ModelResponse[str](
        content="Test content",
        parsed="Test content",
        reasoning_content="",
        citations=citations,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        cost=None,
        model="test",
        response_id="test-id",
        metadata={},
    )

    assert len(response.citations) == 2
    assert response.citations[0].title == "Page 1"
    assert response.citations[1].url == "https://other.com"

# Example: Citations serialization
# Source: tests/llm/test_model_response.py:183
def test_citations_serialization(self):
    """Test citations serialize correctly."""
    citations = (Citation(title="Test", url="https://test.com", start_index=0, end_index=5),)
    response = ModelResponse[str](
        content="Test",
        parsed="Test",
        citations=citations,
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model="test",
        response_id="id",
    )

    json_str = response.model_dump_json()
    restored = ModelResponse.model_validate_json(json_str)

    # Citations are serialized as dicts
    assert len(restored.citations) == 1
    # After deserialization, citations are dicts (not Citation dataclass)
    citation = restored.citations[0]
    if isinstance(citation, dict):
        assert citation["title"] == "Test"
        assert citation["url"] == "https://test.com"
    else:
        assert citation.title == "Test"
        assert citation.url == "https://test.com"

# Example: Immutability
# Source: tests/llm/test_model_options.py:112
def test_immutability(self):
    """Test that ModelOptions is mutable (default Pydantic behavior)."""
    options = ModelOptions()
    options.timeout = 500
    assert options.timeout == 500

    options.response_format = None
    assert options.response_format is None

# Example: Small image document
# Source: tests/llm/test_text_doc_image_attachments.py:261
def test_small_image_document(self):
    """Small image (no splitting needed) must produce ImageContent."""
    doc = ConcreteDocument.create(name="photo.jpg", content=_make_image_bytes(100, 100))

    parts = _document_to_content_parts(doc, "gemini-3-pro")

    image_parts = [p for p in parts if isinstance(p, ImageContent)]
    assert len(image_parts) == 1, f"Expected 1 ImageContent, got {len(image_parts)}. Types: {[type(p).__name__ for p in parts]}"

# Example: Small png document
# Source: tests/llm/test_text_doc_image_attachments.py:281
def test_small_png_document(self):
    """PNG image must produce ImageContent with correct mime_type."""
    doc = ConcreteDocument.create(name="icon.png", content=_make_image_bytes(100, 100, fmt="PNG"))

    parts = _document_to_content_parts(doc, "gemini-3-pro")

    image_parts = [p for p in parts if isinstance(p, ImageContent)]
    assert len(image_parts) == 1
    assert image_parts[0].mime_type == "image/png"

# === ERROR EXAMPLES (What NOT to Do) ===

# Error: Frozen model
# Source: tests/llm/test_model_response.py:98
def test_frozen_model(self):
    """Test that ModelResponse is immutable."""
    response = create_test_model_response(content="test")

    with pytest.raises(ValidationError):
        response.content = "new content"  # type: ignore[misc]
