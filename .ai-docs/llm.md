# MODULE: llm
# CLASSES: ModelOptions, Citation, Conversation
# DEPENDS: BaseModel, Generic
# PURPOSE: Large Language Model integration via LiteLLM proxy.
# VERSION: 0.10.0
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import Citation, Conversation, ModelName, ModelOptions
```

## Rules

1. Never discard the return value — the original Conversation is unchanged.

## Types & Constants

```python
type ModelName = (
    Literal[
        # Core models
        "gemini-3-pro",
        "gpt-5.1",
        # Small models
        "gemini-3-flash",
        "gpt-5-mini",
        "grok-4.1-fast",
        # Search models
        "gemini-3-flash-search",
        "gpt-5-mini-search",
        "grok-4.1-fast-search",
        "sonar-pro-search",
    ]
    | str
)

SYSTEM_PROMPT_DOCUMENT_NAME = "system_prompt"

CHARS_PER_TOKEN = 4

```

## Public API

```python
class ModelOptions(BaseModel):
    """Configuration options for LLM generation requests.

All fields are optional with sensible defaults. Extra fields are forbidden.

Non-obvious behaviors:
- cache_ttl: string format ("60s", "5m", "1h") or None to disable. Default "300s".
- service_tier: only OpenAI models honor this; other providers silently ignore it.
- stop: accepts a single string or list; coerced to tuple. Max 4 by most providers.
- usage_tracking: when True (default), injects {"usage": {"include": True}} into extra_body.
- extra_body: merged with usage_tracking dict if both are set."""
    model_config = ConfigDict(frozen=True, extra='forbid')
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
    stop: tuple[str, ...] | None = None
    verbosity: Literal['low', 'medium', 'high'] | None = None
    stream: bool = False
    usage_tracking: bool = True
    user: str | None = None
    metadata: Mapping[str, str] | None = None
    extra_body: Mapping[str, Any] | None = None

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


@dataclass(frozen=True, slots=True)
class Citation:
    """A URL citation returned by search-enabled models.

The start_index and end_index fields indicate character positions in the response content
where the citation applies. Note that index behavior varies by model."""
    title: str
    url: str
    start_index: int
    end_index: int


class Conversation(BaseModel, Generic[T]):
    """Immutable conversation state for LLM interactions.

Every send()/send_structured() call returns a NEW Conversation instance.
Never discard the return value — the original Conversation is unchanged.

Content protection (URLs, addresses, high-entropy strings) is enabled by default,
auto-disabled for `-search` suffix models. Both `.content` and `.parsed` are
eagerly restored after each send.

Attachment rendering in LLM context:
- Text attachments: wrapped in <attachment name="..." description="..."> tags
- Binary attachments (images, PDFs): inserted as separate content parts"""
    model_config = ConfigDict(frozen=True)
    model: str
    context: tuple[Document, ...] = ()
    messages: tuple[_AnyMessage, ...] = ()
    model_options: ModelOptions | None = None
    enable_substitutor: bool = True
    extract_result_tags: bool = False

    @property
    def approximate_tokens_count(self) -> int:
        """Approximate token count for all context and messages."""
        total = 0
        for item in chain(self.context, self.messages):
            if isinstance(item, ModelResponse):
                total += len(item.content) // CHARS_PER_TOKEN
                if reasoning := item.reasoning_content:
                    total += len(reasoning) // CHARS_PER_TOKEN
            elif isinstance(item, (_UserMessage, _AssistantMessage)):
                total += len(item.text) // CHARS_PER_TOKEN
            elif isinstance(item, Document):  # pyright: ignore[reportUnnecessaryIsInstance]
                if item.is_text:
                    total += len(item.content) // CHARS_PER_TOKEN
                elif item.is_image or item.is_pdf:
                    total += TOKENS_PER_IMAGE
                for att in item.attachments:
                    if att.is_text:
                        total += len(att.content) // CHARS_PER_TOKEN
                    elif att.is_image or att.is_pdf:
                        total += TOKENS_PER_IMAGE
        return total

    @property
    def citations(self) -> tuple[Citation, ...]:
        """Citations from last send() call (for search-enabled models)."""
        return tuple(r.citations) if (r := self._last_response) else ()

    @property
    def content(self) -> str:
        """Response text from last send() call.

        When extract_result_tags is True, strips <result>...</result> tags
        from the raw response (used by send_spec with output_structure).
        """
        if r := self._last_response:
            return _extract_result(r.content) if self.extract_result_tags else r.content
        return ""

    @property
    def cost(self) -> float | None:
        """Cost from last send() call (if available)."""
        return r.cost if (r := self._last_response) else None

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
        return r.reasoning_content if (r := self._last_response) else ""

    @property
    def usage(self) -> TokenUsage:
        """Token usage from last send() call."""
        return r.usage if (r := self._last_response) else TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    @field_validator("model")
    @classmethod
    def validate_model_not_empty(cls, v: str) -> str:
        """Reject empty model name."""
        if not v:
            raise ValueError("model must be non-empty")
        return v

    async def send(
        self,
        content: ConversationContent,
        *,
        purpose: str | None = None,
        expected_cost: float | None = None,
    ) -> "Conversation[None]":
        """Send message, returns NEW Conversation with response."""
        new_messages, response = await self._execute_send(content, None, purpose, expected_cost)
        return self.model_copy(update={"messages": new_messages + (response,)})  # type: ignore[return-value]

    @overload
    async def send_spec(
        self,
        spec: PromptSpec[str],
        *,
        documents: list[Document] | None = None,
        include_input_documents: bool = True,
        purpose: str | None = None,
        expected_cost: float | None = None,
    ) -> "Conversation[None]": ...

    async def send_structured(
        self,
        content: ConversationContent,
        response_format: type[U],
        *,
        purpose: str | None = None,
        expected_cost: float | None = None,
    ) -> "Conversation[U]":
        """Send message expecting structured response, returns NEW Conversation[U] with .parsed."""
        new_messages, response = await self._execute_send(content, response_format, purpose, expected_cost)
        return self.model_copy(update={"messages": new_messages + (response,)})  # type: ignore[return-value]

    def with_assistant_message(self, content: str) -> "Conversation[T]":
        """Return NEW Conversation with an injected assistant turn in messages."""
        return self.model_copy(update={"messages": self.messages + (_AssistantMessage(content),)})

    def with_context(self, *docs: Document) -> "Conversation[T]":
        """Return NEW Conversation with documents added to the cacheable context prefix.

        All structured data for LLM context must be wrapped in a Document.
        Use Document.create() to wrap dicts, lists, or BaseModel instances.
        Never construct XML manually for LLM context.
        """
        return self.model_copy(update={"context": self.context + docs})

    def with_document(self, doc: Document) -> "Conversation[T]":
        """Return NEW Conversation with document appended to messages (not cached).

        All structured data for LLM context must be wrapped in a Document.
        Use Document.create() to wrap dicts, lists, or BaseModel instances.
        Never construct XML manually for LLM context.
        """
        return self.model_copy(update={"messages": self.messages + (doc,)})

    def with_documents(self, docs: Sequence[Document]) -> "Conversation[T]":
        """Return NEW Conversation with multiple documents appended to messages (not cached)."""
        return self.model_copy(update={"messages": self.messages + tuple(docs)})

    def with_model(self, model: str) -> "Conversation[T]":
        """Return NEW Conversation with a different model, preserving all state."""
        if not model:
            raise ValueError("model must be non-empty")
        return self.model_copy(update={"model": model})

    def with_model_options(self, options: ModelOptions) -> "Conversation[T]":
        """Return NEW Conversation with updated model options."""
        return self.model_copy(update={"model_options": options})

    def with_substitutor(self, enabled: bool = True) -> "Conversation[T]":
        """Return NEW Conversation with substitutor enabled/disabled."""
        return self.model_copy(update={"enable_substitutor": enabled})


```

## Examples

**Citation has slots** (`tests/llm/test_verified_issues.py:294`)

```python
def test_citation_has_slots(self):
    """Citation dataclass should have __slots__."""
    assert hasattr(Citation, "__slots__"), "Citation should have slots=True"
```

**Citations empty by default** (`tests/llm/test_model_response.py:157`)

```python
def test_citations_empty_by_default(self):
    """Test citations returns empty tuple by default."""
    response = create_test_model_response(content="No citations")
    assert response.citations == ()
```

**Citations preserved** (`tests/llm/test_model_response.py:162`)

```python
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
```

**Citations serialization** (`tests/llm/test_model_response.py:184`)

```python
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
```

**Cross conversation transfer** (`tests/llm/test_conversation_with_assistant_message.py:127`)

```python
@pytest.mark.asyncio
async def test_cross_conversation_transfer(self, monkeypatch):
    """Transfer content from conv_a to conv_b via with_assistant_message."""
    call_count = 0

    async def fake_generate(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return create_test_model_response(content="Analysis: the document discusses topic Z")
        return create_test_model_response(content="Based on the analysis, Z is important")

    monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

    with patch("ai_pipeline_core.llm.conversation.Laminar", _mock_laminar()):
        # Conv A does analysis
        conv_a = Conversation(model="test-model", enable_substitutor=False)
        conv_a = await conv_a.send("Analyze this")

        # Conv B receives the analysis
        conv_b = Conversation(model="test-model", enable_substitutor=False)
        conv_b = conv_b.with_assistant_message(conv_a.content)
        conv_b = await conv_b.send("What did you find?")

    assert conv_b.content == "Based on the analysis, Z is important"
```


## Error Examples

**Immutability** (`tests/llm/test_model_options.py:96`)

```python
def test_immutability(self):
    """Test that ModelOptions is frozen (immutable)."""
    options = ModelOptions()
    with pytest.raises(ValidationError):
        options.timeout = 500  # type: ignore[misc]
```

**With model validates empty** (`tests/llm/test_conversation_with_model.py:33`)

```python
def test_with_model_validates_empty(self):
    """with_model() rejects empty model name via field validator."""
    import pytest

    conv = Conversation(model="gemini-3-flash")
    with pytest.raises(Exception):
        conv.with_model("")
```

**Can be caught as exception** (`tests/llm/test_degeneration.py:497`)

```python
def test_can_be_caught_as_exception(self):
    with pytest.raises(Exception):
        raise OutputDegenerationError("test degeneration")
```

**Can be caught as llm error** (`tests/llm/test_degeneration.py:493`)

```python
def test_can_be_caught_as_llm_error(self):
    with pytest.raises(LLMError):
        raise OutputDegenerationError("test degeneration")
```

**Frozen model** (`tests/llm/test_model_response.py:99`)

```python
def test_frozen_model(self):
    """Test that ModelResponse is immutable."""
    response = create_test_model_response(content="test")

    with pytest.raises(ValidationError):
        response.content = "new content"  # type: ignore[misc]
```
