# MODULE: llm
# CLASSES: Conversation
# DEPENDS: BaseModel, Generic
# SIZE: ~12KB

# === DEPENDENCIES (Resolved) ===

class BaseModel:
    """Pydantic base model. Fields are typed class attributes."""
    ...

class Generic:
    """Python generic base class for parameterized types."""
    ...

# === PUBLIC API ===

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
    ) -> "Conversation[None]":  # type: ignore[type-var]
        """Send message, returns NEW Conversation with response.

        Args:
            content: Message content - str, Document, or list of Documents.
            purpose: Optional semantic label for tracing.
            expected_cost: Optional expected cost for tracking.

        Returns:
            New Conversation[None] with response accessible via .content, .reasoning_content, etc.
        """
        new_messages, response = await self._execute_send(content, None, purpose, expected_cost)
        return Conversation[None](  # type: ignore[type-var]
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
        response_format: type[T],
        *,
        purpose: str | None = None,
        expected_cost: float | None = None,
    ) -> "Conversation[T]":
        """Send message expecting structured response.

        Args:
            content: Message content - str, Document, or list of Documents.
            response_format: Pydantic model class for structured output.
            purpose: Optional semantic label for tracing.
            expected_cost: Optional expected cost for tracking.

        Returns:
            New Conversation[T] with .parsed returning T instance.
        """
        new_messages, response = await self._execute_send(content, response_format, purpose, expected_cost)
        return Conversation[T](
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


# === EXAMPLES (from tests/) ===

# Example: Simple url
# Source: tests/llm/test_substitutor_patterns.py:15
def test_simple_url(self):
    text = "Check https://example.com for details"
    matches = list(_URL_PATTERN.finditer(text))
    assert len(matches) == 1
    assert matches[0].group() == "https://example.com"

# Example: Structured response creation
# Source: tests/llm/test_model_response_features.py:137
def test_structured_response_creation(self):
    """Test creating a structured response."""
    parsed = self.TestModel(field="test", value=42)
    response = create_test_structured_model_response(parsed=parsed)

    assert response.parsed.field == "test"
    assert response.parsed.value == 42

# Example: Substitutor is regular field
# Source: tests/llm/test_verified_issues.py:127
def test_substitutor_is_regular_field(self):
    """Substitutor should be a regular Field, not PrivateAttr."""
    # Check that 'substitutor' is in model_fields (regular field)
    # and not in __private_attributes__ (PrivateAttr)
    assert "substitutor" in Conversation.model_fields, "substitutor should be a model field"

# Example: Three turn conversation
# Source: tests/llm/test_eager_restore.py:199
@pytest.mark.asyncio
async def test_three_turn_conversation(self, monkeypatch):
    """Verify 3-turn conversation: content always restored, history always shortened."""
    conv = Conversation(model="test-model")
    shorts = _prepare_and_get_short(conv.substitutor, LONG_URL)
    short = shorts[LONG_URL]

    call_count = 0

    async def fake_generate(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        return create_test_model_response(content=f"Turn {call_count}: {short}")

    monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

    conv = await conv.send("msg 1")
    assert conv.content == f"Turn 1: {LONG_URL}"

    conv = await conv.send("msg 2")
    assert conv.content == f"Turn 2: {LONG_URL}"

    conv = await conv.send("msg 3")
    assert conv.content == f"Turn 3: {LONG_URL}"
    assert call_count == 3

# Example: All punct segment falls back to domain only
# Source: tests/llm/test_url_substitutor.py:718
def test_all_punct_segment_falls_back_to_domain_only(self):
    """Path segment that's all punctuation after truncation should use domain-only format."""
    sub = URLSubstitutor()
    url = "https://example.com/----------/page/resource/detail"
    sub.prepare([url])
    shortened = sub.get_mappings()[url]
    assert shortened.startswith("https://example.com~"), f"Expected domain-only format: {shortened}"

# Example: Asyncio imported
# Source: tests/llm/test_verified_issues.py:271
def test_asyncio_imported(self):
    """asyncio module should be imported."""
    import ai_pipeline_core.llm.conversation as conv_module

    assert hasattr(conv_module, "asyncio") or "import asyncio" in inspect.getsource(conv_module)

# Example: Base64 with slash no plus not shortened
# Source: tests/llm/test_url_substitutor.py:793
def test_base64_with_slash_no_plus_not_shortened(self):
    """Base64 with / but no + or = is intentionally not shortened to prevent URL path false positives."""
    sub = URLSubstitutor()
    b64_like = "AAAAAAAABBBBBBBB/CCCCCCCCDDDDDDDDEEEE"
    text = f"Token: {b64_like}"
    sub.prepare([text])
    result = sub.substitute(text)
    # Intentional false negative — accepted to prevent URL path false positives
    assert result == text

# === ERROR EXAMPLES (What NOT to Do) ===

# Error: Frozen model
# Source: tests/llm/test_model_response.py:98
def test_frozen_model(self):
    """Test that ModelResponse is immutable."""
    response = create_test_model_response(content="test")

    with pytest.raises(ValidationError):
        response.content = "new content"  # type: ignore[misc]
