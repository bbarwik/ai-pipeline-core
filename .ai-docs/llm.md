# MODULE: llm
# CLASSES: AIMessages, ModelOptions, Citation, ModelResponse, StructuredModelResponse
# DEPENDS: BaseModel, ChatCompletion, Generic, list
# SIZE: ~54KB

# === DEPENDENCIES (Resolved) ===

class BaseModel:
    """Pydantic base model. Fields are typed class attributes."""
    ...

class ChatCompletion:
    """External base class (not fully documented)."""
    ...

class Generic:
    """Python generic base class for parameterized types."""
    ...

class list:
    """Python built-in list."""
    ...

# === PUBLIC API ===

class AIMessages(list[AIMessageType]):
    """Container for AI conversation messages supporting mixed types.

This class extends list to manage conversation messages between user
and AI, supporting text, Document objects, and ModelResponse instances.
Messages are converted to OpenAI-compatible format for LLM interactions.

Conversion Rules:
    - str: Becomes {"role": "user", "content": text}
    - Document: Becomes {"role": "user", "content": document_content}
      (automatically handles text, images, PDFs based on MIME type; attachments
      are rendered as <attachment> XML blocks)
    - ModelResponse: Becomes {"role": "assistant", "content": response.content}

Note: Document conversion is automatic. Text content becomes user text messages.

VISION/PDF MODEL COMPATIBILITY WARNING:
Images require vision-capable models (e.g., gpt-5.1, gemini-3-flash, gemini-3-pro).
Non-vision models will raise ValueError when encountering image documents.
PDFs require models with document processing support - check your model's capabilities
before including PDF documents in messages. Unsupported models may fall back to
text extraction or raise errors depending on provider configuration.
LiteLLM proxy handles the specific encoding requirements for each provider.

IMPORTANT: Although AIMessages can contain Document entries, the LLM client functions
expect `messages` to be `AIMessages` or `str`. If you start from a Document or a list
of Documents, build AIMessages first (e.g., `AIMessages([doc])` or `AIMessages(docs)`).

CAUTION: AIMessages is a list subclass. Always use list construction (e.g.,
`AIMessages(["text"])`) or empty constructor with append (e.g.,
`AIMessages(); messages.append("text")`). Never pass raw strings directly to the
constructor (`AIMessages("text")`) as this will raise a TypeError to prevent
accidental character iteration."""
    def __init__(self, iterable: Iterable[AIMessageType] | None = None, *, frozen: bool = False):
        """Initialize AIMessages with optional iterable.

        Args:
            iterable: Optional iterable of messages (list, tuple, etc.).
                     Must not be a string.
            frozen: If True, list is immutable from creation.

        Raises:
            TypeError: If a string is passed directly to the constructor.
        """
        if isinstance(iterable, str):
            raise TypeError(
                "AIMessages cannot be constructed from a string directly. "
                "Use AIMessages(['text']) for a single message or "
                "AIMessages() and then append('text')."
            )
        self._frozen = False  # Initialize as unfrozen to allow initial population
        if iterable is None:
            super().__init__()
        else:
            super().__init__(iterable)
        self._frozen = frozen  # Set frozen state after initial population

    @property
    def approximate_tokens_count(self) -> int:
        """Approximate tokens count for the messages.

        Uses tiktoken with gpt-4 encoding to estimate total token count
        across all messages in the conversation.

        Returns:
            Approximate tokens count for all messages.

        Raises:
            ValueError: If message contains unsupported type.

        """
        count = 0
        enc = get_tiktoken_encoding()
        for message in self:
            if isinstance(message, str):
                count += len(enc.encode(message))
            elif isinstance(message, Document):
                count += message.approximate_tokens_count
            elif isinstance(message, ModelResponse):  # type: ignore
                count += len(enc.encode(message.content))
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")
        return count

    def __delitem__(self, index: SupportsIndex | slice) -> None:
        """Delete item or slice from list."""
        self._check_frozen()
        super().__delitem__(index)

    def __iadd__(self, other: Iterable[AIMessageType]) -> "AIMessages":
        """In-place addition (+=).

        Returns:
            This AIMessages instance after modification.
        """
        self._check_frozen()
        return super().__iadd__(other)

    def __setitem__(
        self,
        index: SupportsIndex | slice,
        value: AIMessageType | Iterable[AIMessageType],
    ) -> None:
        """Set item or slice."""
        self._check_frozen()
        super().__setitem__(index, value)  # type: ignore[arg-type]

    def append(self, message: AIMessageType) -> None:
        """Add a message to the end of the list."""
        self._check_frozen()
        super().append(message)

    def clear(self) -> None:
        """Remove all items from list."""
        self._check_frozen()
        super().clear()

    def copy(self) -> "AIMessages":
        """Create an unfrozen deep copy of the list.

        Returns:
            New unfrozen AIMessages with deep-copied messages.
        """
        copied_messages = deepcopy(list(self))
        return AIMessages(copied_messages, frozen=False)

    @staticmethod
    def document_to_prompt(document: Document) -> list[ChatCompletionContentPartParam]:  # noqa: C901, PLR0912, PLR0914, PLR0915
        """Convert a document to prompt format for LLM consumption.

        Renders the document as XML with text/image/PDF content, followed by any
        attachments as separate <attachment> XML blocks with name and description attributes.

        Args:
            document: The document to convert.

        Returns:
            List of chat completion content parts for the prompt.
        """
        prompt: list[ChatCompletionContentPartParam] = []

        # Build the text header
        description = f"<description>{document.description}</description>\n" if document.description else ""
        header_text = f"<document>\n<id>{document.id}</id>\n<name>{document.name}</name>\n{description}"

        # Check if "PDF" is actually text (misnamed file from URL ending in .pdf)
        # Real PDFs start with %PDF- magic bytes; if missing and content is UTF-8, it's text
        is_text = document.is_text
        if not is_text and document.is_pdf and _looks_like_text(document.content) and not _has_pdf_signature(document.content):
            is_text = True
            logger.debug(f"Document '{document.name}' has PDF extension but contains text content - sending as text")

        # Handle text documents
        if is_text:
            text_content = document.content.decode("utf-8")
            content_text = f"{header_text}<content>\n{text_content}\n</content>\n"
            prompt.append({"type": "text", "text": content_text})

        # Handle binary documents (image/PDF)
        elif document.is_image or document.is_pdf:
            prompt.append({"type": "text", "text": f"{header_text}<content>\n"})

            if document.is_image:
                content_bytes, mime_type = _ensure_llm_compatible_image(document.content, document.mime_type)
            else:
                content_bytes, mime_type = document.content, document.mime_type
            base64_content = base64.b64encode(content_bytes).decode("utf-8")
            data_uri = f"data:{mime_type};base64,{base64_content}"

            if document.is_pdf:
                prompt.append({
                    "type": "file",
                    "file": {"file_data": data_uri},
                })
            else:
                prompt.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri, "detail": "high"},
                })

            prompt.append({"type": "text", "text": "</content>\n"})

        else:
            logger.error(f"Document is not a text, image or PDF: {document.name} - {document.mime_type}")
            return []

        # Render attachments
        for att in document.attachments:
            desc_attr = f' description="{att.description}"' if att.description else ""
            att_open = f'<attachment name="{att.name}"{desc_attr}>\n'

            # Check if "PDF" attachment is actually text (same logic as document)
            att_is_text = att.is_text
            if not att_is_text and att.is_pdf and _looks_like_text(att.content) and not _has_pdf_signature(att.content):
                att_is_text = True
                logger.debug(f"Attachment '{att.name}' has PDF extension but contains text content - sending as text")

            if att_is_text:
                # Use content.decode() directly - att.text property raises ValueError if is_text is False
                att_text = att.content.decode("utf-8")
                prompt.append({"type": "text", "text": f"{att_open}{att_text}\n</attachment>\n"})
            elif att.is_image or att.is_pdf:
                prompt.append({"type": "text", "text": att_open})

                if att.is_image:
                    att_bytes, att_mime = _ensure_llm_compatible_image(att.content, att.mime_type)
                else:
                    att_bytes, att_mime = att.content, att.mime_type
                att_b64 = base64.b64encode(att_bytes).decode("utf-8")
                att_uri = f"data:{att_mime};base64,{att_b64}"

                if att.is_pdf:
                    prompt.append({
                        "type": "file",
                        "file": {"file_data": att_uri},
                    })
                else:
                    prompt.append({
                        "type": "image_url",
                        "image_url": {"url": att_uri, "detail": "high"},
                    })

                prompt.append({"type": "text", "text": "</attachment>\n"})
            else:
                logger.warning(f"Skipping unsupported attachment type: {att.name} - {att.mime_type}")

        # Close document — merge into last text part to preserve JSON structure (and cache key)
        last = prompt[-1]
        if last["type"] == "text":
            prompt[-1] = {"type": "text", "text": last["text"] + "</document>\n"}
        else:
            prompt.append({"type": "text", "text": "</document>\n"})

        return prompt

    def extend(self, messages: Iterable[AIMessageType]) -> None:
        """Add multiple messages to the list."""
        self._check_frozen()
        super().extend(messages)

    def freeze(self) -> None:
        """Permanently freeze the list, preventing modifications.

        Once frozen, the list cannot be unfrozen.
        """
        self._frozen = True

    def get_last_message(self) -> AIMessageType:
        """Get the last message in the conversation.

        Returns:
            The last message in the conversation, which can be a string,
            Document, or ModelResponse.
        """
        return self[-1]

    def get_last_message_as_str(self) -> str:
        """Get the last message as a string, raising if not a string.

        Returns:
            The last message as a string.

        Raises:
            ValueError: If the last message is not a string.

        Safer Pattern:
            Instead of catching ValueError, check type first:
            >>> messages = AIMessages([user_msg, response, followup])
            >>> last = messages.get_last_message()
            >>> if isinstance(last, str):
            ...     text = last
            >>> elif isinstance(last, ModelResponse):
            ...     text = last.content
            >>> elif isinstance(last, Document):
            ...     text = last.text if last.is_text else "<binary>"
        """
        last_message = self.get_last_message()
        if isinstance(last_message, str):
            return last_message
        raise ValueError(f"Wrong message type: {type(last_message)}")

    def get_prompt_cache_key(self, system_prompt: str | None = None) -> str:
        """Generate cache key for message set.

        Args:
            system_prompt: Optional system prompt to include in cache key.

        Returns:
            SHA256 hash as hex string for cache key.
        """
        if not system_prompt:
            system_prompt = ""
        return hashlib.sha256((system_prompt + json.dumps(self.to_prompt())).encode()).hexdigest()

    def insert(self, index: SupportsIndex, message: AIMessageType) -> None:
        """Insert a message at the specified position."""
        self._check_frozen()
        super().insert(index, message)

    def pop(self, index: SupportsIndex = -1) -> AIMessageType:
        """Remove and return item at index.

        Returns:
            AIMessageType removed from the list.
        """
        self._check_frozen()
        return super().pop(index)

    def remove(self, message: AIMessageType) -> None:
        """Remove first occurrence of message."""
        self._check_frozen()
        super().remove(message)

    def reverse(self) -> None:
        """Reverse list in place."""
        self._check_frozen()
        super().reverse()

    def sort(self, *, key: Callable[[AIMessageType], Any] | None = None, reverse: bool = False) -> None:
        """Sort list in place."""
        self._check_frozen()
        if key is None:
            super().sort(reverse=reverse)  # type: ignore[call-arg]
        else:
            super().sort(key=key, reverse=reverse)

    def to_prompt(self) -> list[ChatCompletionMessageParam]:
        """Convert AIMessages to OpenAI-compatible format.

        Transforms the message list into the format expected by OpenAI API.
        Each message type is converted according to its role and content.
        Documents are rendered as XML with any attachments included as nested
        <attachment> blocks.

        Returns:
            List of ChatCompletionMessageParam dicts (from openai.types.chat)
            with 'role' and 'content' keys. Ready to be passed to generate()
            or OpenAI API directly.

        Raises:
            ValueError: If message type is not supported.

        """
        messages: list[ChatCompletionMessageParam] = []

        for message in self:
            if isinstance(message, str):
                messages.append({"role": "user", "content": [{"type": "text", "text": message}]})
            elif isinstance(message, Document):
                messages.append({"role": "user", "content": AIMessages.document_to_prompt(message)})
            elif isinstance(message, ModelResponse):  # type: ignore
                # Build base assistant message
                assistant_message: ChatCompletionMessageParam = {
                    "role": "assistant",
                    "content": [{"type": "text", "text": message.content}],
                }

                # Preserve reasoning_content (Gemini Flash 3+, O1, O3, GPT-5)
                if reasoning_content := message.reasoning_content:
                    assistant_message["reasoning_content"] = reasoning_content  # type: ignore[typeddict-item]

                # Preserve thinking_blocks (structured thinking)
                if hasattr(message.choices[0].message, "thinking_blocks"):
                    thinking_blocks = getattr(message.choices[0].message, "thinking_blocks", None)
                    if thinking_blocks:
                        assistant_message["thinking_blocks"] = thinking_blocks  # type: ignore[typeddict-item]

                # Preserve provider_specific_fields (thought_signatures for Gemini multi-turn)
                if hasattr(message.choices[0].message, "provider_specific_fields"):
                    provider_fields = getattr(message.choices[0].message, "provider_specific_fields", None)
                    if provider_fields:
                        assistant_message["provider_specific_fields"] = provider_fields  # type: ignore[typeddict-item]

                messages.append(assistant_message)
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        return messages

    def to_tracing_log(self) -> list[str]:
        """Convert AIMessages to a list of strings for tracing.

        Returns:
            List of string representations for tracing logs.
        """
        messages: list[str] = []
        for message in self:
            if isinstance(message, Document):
                serialized_document = message.serialize_model()
                filtered_doc = {k: v for k, v in serialized_document.items() if k != "content"}
                messages.append(json.dumps(filtered_doc, indent=2))
            elif isinstance(message, ModelResponse):
                messages.append(message.content)
            else:
                assert isinstance(message, str)
                messages.append(message)
        return messages


class ModelOptions(BaseModel):
    """Configuration options for LLM generation requests.

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

        if self.metadata:
            kwargs["metadata"] = self.metadata

        if self.usage_tracking:
            kwargs["extra_body"]["usage"] = {"include": True}
            kwargs["stream_options"] = {"include_usage": True}

        return kwargs


@dataclass(frozen=True)
class Citation:
    """A URL citation returned by search-enabled models (e.g. sonar-pro-search, gemini-3-flash-search)."""
    title: str
    url: str


class ModelResponse(ChatCompletion):
    """Response wrapper for LLM text generation.

Primary usage is adding to AIMessages for multi-turn conversations:

    >>> response = await llm.generate("gpt-5.1", messages=messages)
    >>> messages.append(response)  # Add assistant response to conversation
    >>> print(response.content)  # Access generated text

The two main interactions with ModelResponse:
1. Adding to AIMessages for conversation flow
2. Accessing .content property for the generated text

Almost all use cases are covered by these two patterns. Advanced features
like token usage and cost tracking are available but rarely needed.

Inherits from OpenAI's ChatCompletion for compatibility.
Other properties (usage, model, id) should only be accessed
when absolutely necessary."""
    def __init__(
        self,
        chat_completion: ChatCompletion,
        model_options: dict[str, Any],
        metadata: dict[str, Any],
        usage: CompletionUsage | None = None,
    ) -> None:
        """Initialize ModelResponse from ChatCompletion.

        Wraps an OpenAI ChatCompletion object with additional metadata
        and model options for tracking and observability.

        Args:
            chat_completion: ChatCompletion object from the API.
            model_options: Model configuration options used for the request.
                          Stored for metadata extraction and tracing.
            metadata: Custom metadata for tracking (time_taken, first_token_time, etc.).
                     Includes timing information and custom tags.
            usage: Optional usage information from streaming response.

        """
        data = chat_completion.model_dump()

        # fixes issue where the role is "assistantassistant" instead of "assistant"
        valid_finish_reasons = {"stop", "length", "tool_calls", "content_filter", "function_call"}
        for i in range(len(data["choices"])):
            data["choices"][i]["message"]["role"] = "assistant"
            # Only update finish_reason if it's not already a valid value
            current_finish_reason = data["choices"][i].get("finish_reason")
            if current_finish_reason not in valid_finish_reasons:
                data["choices"][i]["finish_reason"] = "stop"
            # Strip annotations with unsupported types (e.g. Grok returns type="file" for PDFs,
            # but OpenAI's ChatCompletion only accepts type="url_citation")
            if annotations := data["choices"][i]["message"].get("annotations"):
                data["choices"][i]["message"]["annotations"] = [a for a in annotations if a.get("type") == "url_citation"]

        super().__init__(**data)

        self._model_options = model_options
        self._metadata = metadata
        if usage:
            self.usage = usage

    @property
    def citations(self) -> list[Citation]:
        """Get URL citations from search-enabled models.

        Returns:
            List of Citation objects with title and url. Empty list for non-search models.
        """
        annotations = self.choices[0].message.annotations
        if not annotations:
            return []
        return [Citation(title=a.url_citation.title, url=a.url_citation.url) for a in annotations if a.url_citation]

    @property
    def content(self) -> str:
        """Get the generated text content.

        Primary property for accessing the LLM's response text.
        This is the main property you'll use with ModelResponse.

        Returns:
            Generated text from the model, or empty string if none.

        """
        content = self.choices[0].message.content or ""
        return content.split("</think>")[-1].strip()

    @property
    def reasoning_content(self) -> str:
        """Get the reasoning content.

        Returns:
            The reasoning content from the model, or empty string if none.
        """
        message = self.choices[0].message
        if reasoning_content := getattr(message, "reasoning_content", None):
            return reasoning_content
        if not message.content or "</think>" not in message.content:
            return ""
        return message.content.split("</think>")[0].strip()

    def get_laminar_metadata(self) -> dict[str, str | int | float]:  # noqa: C901
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

            Cost is stored in three fields for observability tool consumption:
            - gen_ai.usage.output_cost (OpenTelemetry GenAI semantic convention)
            - gen_ai.usage.cost (aggregated cost)
            - gen_ai.cost (short-form)

        Cost availability depends on LiteLLM proxy configuration. Not all providers
        return cost information. Cached tokens reduce actual cost but may not be reflected.
        Used internally by tracing but accessible for cost analysis.
        """
        metadata: dict[str, str | int | float] = deepcopy(self._metadata)

        # Add base metadata
        # NOTE: gen_ai.response.model is intentionally omitted — Laminar's UI uses it
        # to override the span display name in the tree view, hiding the actual span name
        # (set via `purpose` parameter). Tracked upstream: Laminar's getSpanDisplayName()
        # in frontend/components/traces/trace-view/utils.ts prefers model over span name
        # for LLM spans. Restore once Laminar shows both or prefers span name.
        metadata.update({
            "gen_ai.response.id": self.id,
            "gen_ai.system": "litellm",
        })

        # Add usage metadata if available
        cost = None
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
            if (completion_details := self.usage.completion_tokens_details) and (reasoning_tokens := completion_details.reasoning_tokens):
                metadata["gen_ai.usage.reasoning_tokens"] = reasoning_tokens

            # Add cached tokens if available
            if (prompt_details := self.usage.prompt_tokens_details) and (cached_tokens := prompt_details.cached_tokens):
                metadata["gen_ai.usage.cached_tokens"] = cached_tokens

        # Add cost metadata if available
        if cost and cost > 0:
            metadata.update({
                "gen_ai.usage.output_cost": cost,
                "gen_ai.usage.cost": cost,
                "gen_ai.cost": cost,
            })

        for key, value in self._model_options.items():
            if "messages" in key:
                continue
            metadata[f"model_options.{key}"] = str(value)

        other_fields = self.__dict__
        for key, value in other_fields.items():
            if key in {"_model_options", "_metadata", "choices"}:
                continue
            try:
                metadata[f"response.raw.{key}"] = json.dumps(value, indent=2, default=str)
            except Exception:
                metadata[f"response.raw.{key}"] = str(value)

        message = self.choices[0].message
        for key, value in message.__dict__.items():
            if key in {"content"}:
                continue
            metadata[f"response.raw.message.{key}"] = json.dumps(value, indent=2, default=str)

        return metadata

    def validate_output(self) -> None:
        """Validate response output content and format.

        Checks that response has non-empty content and validates against
        response_format if structured output was requested.

        Raises:
            ValueError: If response content is empty.
            ValidationError: If content doesn't match response_format schema.
        """
        if not self.content:
            raise ValueError("Empty response content")

        if (response_format := self._model_options.get("response_format")) and isinstance(response_format, BaseModel):
            response_format.model_validate_json(self.content)


class StructuredModelResponse(ModelResponse, Generic[T]):
    """Response wrapper for structured/typed LLM output.

Primary usage is accessing the .parsed property for the structured data."""
    # [Inherited from ModelResponse]
    # __init__, citations, content, get_laminar_metadata, reasoning_content, validate_output

    @property
    def parsed(self) -> T:
        """Get the parsed structured output.

        Lazily parses the JSON content into the specified Pydantic model.
        Result is cached after first access.

        Returns:
            Parsed Pydantic model instance.

        Raises:
            ValidationError: If content doesn't match the response_format schema.
        """
        if not hasattr(self, "_parsed_value"):
            response_format = self._model_options.get("response_format")
            self._parsed_value: T = response_format.model_validate_json(self.content)  # type: ignore[return-value]
        return self._parsed_value

    @classmethod
    def from_model_response(cls, model_response: ModelResponse) -> "StructuredModelResponse[T]":
        """Convert a ModelResponse to StructuredModelResponse.

        Takes an existing ModelResponse and converts it to a StructuredModelResponse
        for accessing parsed structured output. Used internally by generate_structured().

        Args:
            model_response: The ModelResponse to convert.

        Returns:
            StructuredModelResponse with lazy parsing support.
        """
        model_response.__class__ = cls
        return model_response  # type: ignore[return-value]


# === FUNCTIONS ===

async def generate(
    model: ModelName,
    *,
    context: AIMessages | None = None,
    messages: AIMessages | str,
    options: ModelOptions | None = None,
    purpose: str | None = None,
    expected_cost: float | None = None,
) -> ModelResponse:
    """Generate text response from a language model.

    Main entry point for LLM text generation with smart context caching.
    The context/messages split enables efficient token usage by caching
    expensive static content separately from dynamic queries.

    Best Practices:
        1. OPTIONS: DO NOT use the options parameter - omit it entirely for production use
        2. MESSAGES: Use AIMessages or str - wrap Documents in AIMessages
        3. CONTEXT vs MESSAGES: Use context for static/cacheable, messages for dynamic
        4. CONFIGURATION: Configure model behavior via LiteLLM proxy or environment variables

    Args:
        model: Model to use (e.g., "gpt-5.1", "gemini-3-pro", "grok-4.1-fast").
               Accepts predefined models or any string for custom models.
        context: Static context to cache (documents, examples, instructions).
                Defaults to None (empty context). Cached for 5 minutes by default.
        messages: Dynamic messages/queries. AIMessages or str ONLY.
                 Do not pass Document or list[Document] directly.
                 If string, converted to AIMessages internally.
        options: Internal framework parameter. Framework defaults are production-optimized
                (3 retries, 20s delay, 600s timeout). Configure model behavior centrally via
                LiteLLM proxy settings or environment variables, not per API call.
                Provider-specific settings should be configured at the proxy level.
        purpose: Optional semantic label used as the tracing span name
                instead of model name. Stored as a span attribute.
        expected_cost: Optional expected cost stored as a span attribute
                      for cost-tracking and comparison with actual cost.

    Returns:
        ModelResponse containing:
        - Generated text content
        - Usage statistics
        - Cost information (if available)
        - Model metadata

    Raises:
        ValueError: If model is empty or messages are invalid.
        LLMError: If generation fails after all retries.

    Document Handling:
        Wrap Documents in AIMessages - DO NOT pass directly or convert to .text:

        # CORRECT - wrap Document in AIMessages
        response = await llm.generate("gpt-5.1", messages=AIMessages([my_document]))

        # WRONG - don't pass Document directly
        response = await llm.generate("gpt-5.1", messages=my_document)  # NO!

        # WRONG - don't convert to string yourself
        response = await llm.generate("gpt-5.1", messages=my_document.text)  # NO!

    VISION/PDF MODEL COMPATIBILITY:
        When using Documents containing images or PDFs, ensure your model supports these formats:
        - Images require vision-capable models (gpt-5.1, gemini-3-flash, gemini-3-pro)
        - PDFs require document processing support (varies by provider)
        - Non-compatible models will raise ValueError or fall back to text extraction
        - Check model capabilities before including visual/PDF content

    Context vs Messages Strategy:
        context: Static, reusable content for caching efficiency
            - Large documents, instructions, examples
            - Remains constant across multiple calls
            - Cached when supported by provider/proxy configuration

        messages: Dynamic, per-call specific content
            - User questions, current conversation turn
            - Changes with each API call
            - Never cached, always processed fresh

    Performance:
        - Context caching saves ~50-90% tokens on repeated calls
        - First call: full token cost
        - Subsequent calls (within cache TTL): only messages tokens
        - Default cache TTL is 300s/5 minutes (production-optimized)
        - Default retry logic: 3 attempts with 20s delay (production-optimized)

    Caching:
        When enabled in your LiteLLM proxy and supported by the upstream provider,
        context messages may be cached to reduce token usage on repeated calls.
        Default TTL is 5m (optimized for production workloads). Configure caching
        behavior centrally via your LiteLLM proxy settings, not per API call.
        Savings depend on provider and payload; treat this as an optimization, not a guarantee.

    Configuration:
        All model behavior should be configured at the LiteLLM proxy level:
        - Temperature, max_tokens: Set in litellm_config.yaml model_list
        - Retry logic: Configure in proxy general_settings
        - Timeouts: Set via proxy configuration
        - Caching: Enable/configure in proxy cache settings

        This centralizes configuration and ensures consistency across all API calls.

    All models are accessed via LiteLLM proxy with automatic retry and
    cost tracking via response headers.
    """
    if isinstance(messages, str):
        messages = AIMessages([messages])

    if context is None:
        context = AIMessages()
    if options is None:
        options = ModelOptions()
    else:
        # Create a copy to avoid mutating the caller's options object
        options = options.model_copy()

    with contextlib.suppress(Exception):
        track_llm_documents(context, messages)

    try:
        return await _generate_with_retry(
            model,
            context,
            messages,
            options,
            purpose=purpose,
            expected_cost=expected_cost,
        )
    except (ValueError, LLMError):
        raise  # Explicitly re-raise to satisfy DOC502

async def generate_structured(  # noqa: UP047
    model: ModelName,
    response_format: type[T],
    *,
    context: AIMessages | None = None,
    messages: AIMessages | str,
    options: ModelOptions | None = None,
    purpose: str | None = None,
    expected_cost: float | None = None,
) -> StructuredModelResponse[T]:
    """Generate structured output conforming to a Pydantic model.

    Type-safe generation that returns validated Pydantic model instances.
    Uses OpenAI's structured output feature for guaranteed schema compliance.

    IMPORTANT: Search models (models with '-search' suffix) do not support
    structured output. Use generate() instead for search models.

    Best Practices:
        1. OPTIONS: DO NOT use the options parameter - omit it entirely for production use
        2. MESSAGES: Use AIMessages or str - wrap Documents in AIMessages
        3. CONFIGURATION: Configure model behavior via LiteLLM proxy or environment variables
        4. See generate() documentation for more details

    Context vs Messages Strategy:
        context: Static, reusable content for caching efficiency
            - Schemas, examples, instructions
            - Remains constant across multiple calls
            - Cached when supported by provider/proxy configuration

        messages: Dynamic, per-call specific content
            - Data to be structured, user queries
            - Changes with each API call
            - Never cached, always processed fresh

    Complex Task Pattern:
        For complex tasks like research or deep analysis, it's recommended to use
        a two-step approach:
        1. First use generate() with a capable model to perform the analysis
        2. Then use generate_structured() with a smaller model to convert the
           response into structured output

        This pattern is more reliable than trying to force complex reasoning
        directly into structured format:

        >>> # Step 1: Research/analysis with generate() - no options parameter
        >>> research = await llm.generate(
        ...     "gpt-5.1",
        ...     messages="Research and analyze this complex topic..."
        ... )
        >>>
        >>> # Step 2: Structure the results with generate_structured()
        >>> structured = await llm.generate_structured(
        ...     "gpt-5-mini",  # Smaller model is fine for structuring
        ...     response_format=ResearchSummary,
        ...     messages=f"Extract key information: {research.content}"
        ... )

    Args:
        model: Model to use (must support structured output).
               Search models (models with '-search' suffix) do not support structured output.
        response_format: Pydantic model class defining the output schema.
                        The model will generate JSON matching this schema.
        context: Static context to cache (documents, schemas, examples).
                Defaults to None (empty AIMessages).
        messages: Dynamic prompts/queries. AIMessages or str ONLY.
                 Do not pass Document or list[Document] directly.
        options: Optional ModelOptions for configuring temperature, retries, etc.
                If provided, it will NOT be mutated (a copy is created internally).
                The response_format field is set automatically from the response_format parameter.
                In most cases, leave as None to use framework defaults.
                Configure model behavior centrally via LiteLLM proxy settings when possible.
        purpose: Optional semantic label used as the tracing span name
                instead of model name. Stored as a span attribute.
        expected_cost: Optional expected cost stored as a span attribute
                      for cost-tracking and comparison with actual cost.

    Vision/PDF model compatibility: Images require vision-capable models that also support
    structured output. PDFs require models with both document processing AND structured output
    support. Consider two-step approach: generate() for analysis, then generate_structured()
    for formatting.

    Returns:
        StructuredModelResponse[T] containing:
        - parsed: Validated instance of response_format class
        - All fields from regular ModelResponse (content, usage, etc.)

    Raises:
        TypeError: If response_format is not a Pydantic model class.
        ValueError: If model doesn't support structured output or no parsed content returned.
                   Structured output support varies by provider and model.
        LLMError: If generation fails after retries.
        ValidationError: If response cannot be parsed into response_format.

    Supported models:
        Structured output support varies by provider and model. Generally includes:
        - OpenAI: GPT-4 and newer models
        - Anthropic: Claude 3+ models
        - Google: Gemini Pro models

        Search models (models with '-search' suffix) do not support structured output.
        Check provider documentation for specific support.

    Performance:
        - Structured output may use more tokens than free text
        - Complex schemas increase generation time
        - Validation overhead is minimal (Pydantic is fast)

    Pydantic model is converted to JSON Schema for the API. Validation happens
    automatically via Pydantic. Search models (models with '-search' suffix) do
    not support structured output.
    """
    if context is None:
        context = AIMessages()
    if options is None:
        options = ModelOptions()
    else:
        # Create a copy to avoid mutating the caller's options object
        options = options.model_copy()

    options.response_format = response_format

    if isinstance(messages, str):
        messages = AIMessages([messages])

    assert isinstance(messages, AIMessages)

    with contextlib.suppress(Exception):
        track_llm_documents(context, messages)

    # Call the internal generate function with structured output enabled
    try:
        response = await _generate_with_retry(
            model,
            context,
            messages,
            options,
            purpose=purpose,
            expected_cost=expected_cost,
        )
    except (ValueError, LLMError):
        raise  # Explicitly re-raise to satisfy DOC502

    return StructuredModelResponse[T].from_model_response(response)

# === EXAMPLES (from tests/) ===

# Example: Citation model returns citations
# Source: tests/llm/test_search_citations.py:23
@pytest.mark.asyncio
@pytest.mark.parametrize("model", CITATION_MODELS)
async def test_citation_model_returns_citations(self, model: str):
    """Models with structured citations should return non-empty Citation list."""
    # Unique query to avoid proxy cache returning responses without annotations
    query = f"Who is the current pope? (ref:{uuid.uuid4().hex[:8]})"
    response = await generate(model, messages=query, purpose=f"{model}-citation-test")

    assert response.content, f"{model} returned empty content"

    # Verify search was actually performed — Pope Leo XIV was elected in 2025
    content_lower = response.content.lower()
    assert "leo" in content_lower or "xiv" in content_lower, f"{model} did not return current search results about the pope"

    assert len(response.citations) > 0, f"{model} returned no citations"
    for citation in response.citations:
        assert isinstance(citation, Citation)
        assert citation.url.startswith("http"), f"{model} citation URL invalid: {citation.url}"
        assert citation.title, f"{model} citation has empty title"

# Example: Citations from annotations
# Source: tests/llm/test_model_response.py:273
def test_citations_from_annotations(self):
    """Test citations are extracted from url_citation annotations."""
    response = create_test_model_response(
        id="test",
        object="chat.completion",
        created=1234567890,
        model="sonar-pro-search",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Some search result",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "end_index": 0,
                                "start_index": 0,
                                "title": "Example Page",
                                "url": "https://example.com",
                            },
                        },
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "end_index": 0,
                                "start_index": 0,
                                "title": "Another Page",
                                "url": "https://another.com",
                            },
                        },
                    ],
                },
                "finish_reason": "stop",
            }
        ],
    )
    citations = response.citations
    assert len(citations) == 2
    assert citations[0] == Citation(title="Example Page", url="https://example.com")
    assert citations[1] == Citation(title="Another Page", url="https://another.com")

# Example: Citations empty when no annotations
# Source: tests/llm/test_model_response.py:256
def test_citations_empty_when_no_annotations(self):
    """Test citations returns empty list when no annotations present."""
    response = create_test_model_response(
        id="test",
        object="chat.completion",
        created=1234567890,
        model="gpt-5.1",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "No citations here"},
                "finish_reason": "stop",
            }
        ],
    )
    assert response.citations == []

# Example: Empty messages
# Source: tests/llm/test_client_process_messages.py:13
def test_empty_messages(self):
    """Test processing empty messages."""
    result = _process_messages(context=AIMessages(), messages=AIMessages(), system_prompt=None)
    assert result == []

# === ERROR EXAMPLES (What NOT to Do) ===

# Error: Empty frozen list
# Source: tests/llm/test_ai_messages_freeze.py:241
def test_empty_frozen_list(self) -> None:
    """Test empty frozen list."""
    messages = AIMessages(frozen=True)

    assert len(messages) == 0
    assert list(messages) == []

    with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
        messages.append("Test")

# Error: Frozen append
# Source: tests/llm/test_ai_messages_freeze.py:105
def test_frozen_append(self) -> None:
    """Test that append raises RuntimeError when frozen."""
    messages = AIMessages(frozen=True)
    with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
        messages.append("Test")

# Error: Frozen clear
# Source: tests/llm/test_ai_messages_freeze.py:171
def test_frozen_clear(self) -> None:
    """Test that clear raises RuntimeError when frozen."""
    messages = AIMessages(["Hello"], frozen=True)

    with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
        messages.clear()

# Error: Frozen extend
# Source: tests/llm/test_ai_messages_freeze.py:111
def test_frozen_extend(self) -> None:
    """Test that extend raises RuntimeError when frozen."""
    messages = AIMessages(frozen=True)
    with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
        messages.extend(["Test1", "Test2"])
