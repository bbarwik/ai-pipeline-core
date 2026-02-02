"""LLM client implementation for AI model interactions.

This module provides the core functionality for interacting with language models
through a unified interface. It handles retries, caching, structured outputs,
and integration with various LLM providers via LiteLLM.

Automatic image auto-tiling splits oversized images in attachments to meet
model-specific constraints (e.g., 3000x3000 for Gemini, 1000x1000 default).
Context caching separates static content from dynamic messages for 50-90% token savings.
Optional purpose and expected_cost parameters enable tracing and cost-tracking.
"""

import asyncio
import contextlib
import time
from io import BytesIO
from typing import Any, TypeVar

from lmnr import Laminar
from openai import AsyncOpenAI
from openai.lib.streaming.chat import ChunkEvent, ContentDeltaEvent, ContentDoneEvent
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
)
from PIL import Image
from pydantic import BaseModel, ValidationError

from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents.attachment import Attachment
from ai_pipeline_core.exceptions import LLMError
from ai_pipeline_core.images import ImageProcessingConfig, process_image, process_image_to_documents
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._document_tracking import track_llm_documents
from ai_pipeline_core.settings import settings

from .ai_messages import AIMessages, AIMessageType
from .model_options import ModelOptions
from .model_response import ModelResponse, StructuredModelResponse
from .model_types import ModelName

logger = get_pipeline_logger(__name__)

# Image splitting configs for automatic large-image handling at the LLM boundary.
# Gemini supports up to 3000x3000; all other models use a conservative 1000x1000 default.
_GEMINI_IMAGE_CONFIG = ImageProcessingConfig(max_dimension=3000, max_pixels=9_000_000, jpeg_quality=75)
_DEFAULT_IMAGE_CONFIG = ImageProcessingConfig(max_dimension=1000, max_pixels=1_000_000, jpeg_quality=75)


def _get_image_config(model: str) -> ImageProcessingConfig:
    """Return the image splitting config for a model."""
    if "gemini" in model.lower():
        return _GEMINI_IMAGE_CONFIG
    return _DEFAULT_IMAGE_CONFIG


def _prepare_images_for_model(messages: AIMessages, model: str) -> AIMessages:  # noqa: C901, PLR0912, PLR0915, PLR0914
    """Split image documents and image attachments that exceed model constraints.

    Returns a new AIMessages with oversized images replaced by tiles.
    Returns the original instance unchanged if no splitting is needed.
    """
    if not any(isinstance(m, Document) and (m.is_image or any(att.is_image for att in m.attachments)) for m in messages):
        return messages

    config = _get_image_config(model)
    result: list[AIMessageType] = []
    changed = False

    for msg in messages:
        if not isinstance(msg, Document):
            result.append(msg)
            continue

        # 1. Handle top-level image Documents (existing logic)
        if msg.is_image:
            try:
                with Image.open(BytesIO(msg.content)) as img:
                    w, h = img.size
            except Exception:
                result.append(msg)
                continue

            within_limits = w <= config.max_dimension and h <= config.max_dimension and w * h <= config.max_pixels
            if within_limits:
                pass  # Falls through to attachment handling
            else:
                name_prefix = msg.name.rsplit(".", 1)[0] if "." in msg.name else msg.name
                tiles = process_image_to_documents(msg, config=config, name_prefix=name_prefix)
                if msg.attachments and tiles:
                    tiles[0] = tiles[0].model_copy(update={"attachments": msg.attachments})
                result.extend(tiles)
                changed = True
                continue

        # 2. Handle image attachments
        if msg.attachments:
            new_attachments: list[Attachment] = []
            attachments_changed = False

            for att in msg.attachments:
                if not att.is_image:
                    new_attachments.append(att)
                    continue

                try:
                    with Image.open(BytesIO(att.content)) as img:
                        w, h = img.size
                except Exception:
                    new_attachments.append(att)
                    continue

                att_within_limits = w <= config.max_dimension and h <= config.max_dimension and w * h <= config.max_pixels
                if att_within_limits:
                    new_attachments.append(att)
                    continue

                # Tile the oversized attachment image
                processed = process_image(att.content, config=config)
                att_prefix = att.name.rsplit(".", 1)[0] if "." in att.name else att.name

                for part in processed.parts:
                    if part.total == 1:
                        tile_name = f"{att_prefix}.jpg"
                        tile_desc = att.description
                    else:
                        tile_name = f"{att_prefix}_{part.index + 1:02d}_of_{part.total:02d}.jpg"
                        tile_desc = f"{att.description} ({part.label})" if att.description else part.label

                    new_attachments.append(
                        Attachment(
                            name=tile_name,
                            content=part.data,
                            description=tile_desc,
                        )
                    )
                attachments_changed = True

            if attachments_changed:
                msg = msg.model_copy(update={"attachments": tuple(new_attachments)})  # noqa: PLW2901
                changed = True

        result.append(msg)

    if not changed:
        return messages
    return AIMessages(result)


def _process_messages(
    context: AIMessages,
    messages: AIMessages,
    system_prompt: str | None = None,
    cache_ttl: str | None = "300s",
) -> list[ChatCompletionMessageParam]:
    """Process and format messages for LLM API consumption.

    Internal function that combines context and messages into a single
    list of API-compatible messages. Applies caching directives to
    system prompt and context messages for efficiency.

    Args:
        context: Messages to be cached (typically expensive/static content).
        messages: Regular messages without caching (dynamic queries).
        system_prompt: Optional system instructions for the model.
        cache_ttl: Cache TTL for system and context messages (e.g. "120s", "300s", "1h").
                   Set to None or empty string to disable caching.

    Returns:
        List of formatted messages ready for API calls, with:
        - System prompt at the beginning with cache_control (if provided and cache_ttl set)
        - Context messages with cache_control on all messages (if cache_ttl set)
        - Regular messages without caching

    System Prompt Location:
        The system prompt parameter is always injected as the FIRST message
        with role="system". It is cached along with context when cache_ttl is set.

    Cache behavior:
        All system and context messages get ephemeral caching with specified TTL
        to reduce token usage on repeated calls with same context.
        If cache_ttl is None or empty string (falsy), no caching is applied.
        All system and context messages receive cache_control to maximize cache efficiency.

    This is an internal function used by _generate_with_retry().
    The context/messages split enables efficient token usage.
    """
    processed_messages: list[ChatCompletionMessageParam] = []

    # Add system prompt if provided
    if system_prompt:
        processed_messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        })

    # Process context messages with caching if provided
    if context:
        # Use AIMessages.to_prompt() for context
        context_messages = context.to_prompt()
        processed_messages.extend(context_messages)

    if cache_ttl:
        for message in processed_messages:
            message["cache_control"] = {  # type: ignore
                "type": "ephemeral",
                "ttl": cache_ttl,
            }
            if isinstance(message["content"], list):  # type: ignore
                message["content"][-1]["cache_control"] = {  # type: ignore
                    "type": "ephemeral",
                    "ttl": cache_ttl,
                }

    # Process regular messages without caching
    if messages:
        regular_messages = messages.to_prompt()
        processed_messages.extend(regular_messages)

    return processed_messages


def _remove_cache_control(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    """Remove cache control directives from messages.

    Internal utility that strips cache_control fields from both message-level
    and content-level entries. Used in retry logic when cache-related errors
    occur during LLM API calls.

    Args:
        messages: List of messages that may contain cache_control directives.

    Returns:
        The same message list (modified in-place) with all cache_control
        fields removed from both messages and their content items.

    Modifies the input list in-place but also returns it for convenience.
    Handles both list-based content (multipart) and string content (simple messages).
    """
    for message in messages:
        if (content := message.get("content")) and isinstance(content, list):
            for item in content:
                if "cache_control" in item:
                    del item["cache_control"]
        if "cache_control" in message:
            del message["cache_control"]
    return messages


def _model_name_to_openrouter_model(model: ModelName) -> str:
    """Convert a model name to an OpenRouter model name.

    Args:
        model: Model name to convert.

    Returns:
        OpenRouter model name.
    """
    if model == "gemini-3-flash-search":
        return "google/gemini-3-flash:online"
    if model == "sonar-pro-search":
        return "perplexity/sonar-pro-search"
    if model.startswith("gemini"):
        return f"google/{model}"
    elif model.startswith("gpt"):
        return f"openai/{model}"
    elif model.startswith("grok"):
        return f"x-ai/{model}"
    elif model.startswith("claude"):
        return f"anthropic/{model}"
    elif model.startswith("qwen3"):
        return f"qwen/{model}"
    elif model.startswith("deepseek-"):
        return f"deepseek/{model}"
    elif model.startswith("glm-"):
        return f"z-ai/{model}"
    elif model.startswith("kimi-"):
        return f"moonshotai/{model}"
    return model


async def _generate_streaming(client: AsyncOpenAI, model: str, messages: list[ChatCompletionMessageParam], completion_kwargs: dict[str, Any]) -> ModelResponse:
    """Execute a streaming LLM API call."""
    start_time = time.time()
    first_token_time = None
    usage = None
    async with client.chat.completions.stream(
        model=model,
        messages=messages,
        **completion_kwargs,
    ) as s:
        async for event in s:
            if isinstance(event, ContentDeltaEvent):
                if not first_token_time:
                    first_token_time = time.time()
            elif isinstance(event, ContentDoneEvent):
                pass
            elif isinstance(event, ChunkEvent) and event.chunk.usage:
                usage = event.chunk.usage
        if not first_token_time:
            first_token_time = time.time()
        raw_response = await s.get_final_completion()

    metadata = {
        "time_taken": round(time.time() - start_time, 2),
        "first_token_time": round(first_token_time - start_time, 2),
    }
    return ModelResponse(raw_response, model_options=completion_kwargs, metadata=metadata, usage=usage)


async def _generate_non_streaming(
    client: AsyncOpenAI, model: str, messages: list[ChatCompletionMessageParam], completion_kwargs: dict[str, Any]
) -> ModelResponse:
    """Execute a non-streaming LLM API call.

    Avoids OpenAI SDK delta accumulation — some providers (e.g. Grok) send
    streaming annotation deltas that crash the SDK's accumulate_delta().
    """
    start_time = time.time()
    kwargs = {k: v for k, v in completion_kwargs.items() if k != "stream_options"}
    response_format = kwargs.get("response_format")
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        raw_response: ChatCompletion = await client.chat.completions.parse(
            model=model,
            messages=messages,
            **kwargs,
        )
    else:
        raw_response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            **kwargs,
        )
    elapsed = round(time.time() - start_time, 2)
    metadata = {"time_taken": elapsed, "first_token_time": elapsed}
    return ModelResponse(raw_response, model_options=completion_kwargs, metadata=metadata)


async def _generate(model: str, messages: list[ChatCompletionMessageParam], completion_kwargs: dict[str, Any], *, stream: bool = True) -> ModelResponse:
    """Execute a single LLM API call.

    Args:
        model: Model identifier (e.g., "gpt-5.1", "gemini-3-pro").
        messages: Formatted messages for the API.
        completion_kwargs: Additional parameters for the completion API.
        stream: Whether to use streaming mode (default True). Non-streaming
                avoids OpenAI SDK delta accumulation issues with some providers.

    Returns:
        ModelResponse with generated content and metadata.
    """
    if "openrouter" in settings.openai_base_url.lower():
        model = _model_name_to_openrouter_model(model)

    async with AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    ) as client:
        if stream:
            return await _generate_streaming(client, model, messages, completion_kwargs)
        return await _generate_non_streaming(client, model, messages, completion_kwargs)


async def _generate_with_retry(  # noqa: PLR0917
    model: str,
    context: AIMessages,
    messages: AIMessages,
    options: ModelOptions,
    purpose: str | None = None,
    expected_cost: float | None = None,
) -> ModelResponse:
    """Core LLM generation with automatic retry logic.

    Internal function that orchestrates the complete generation process
    including message processing, retries, caching, and tracing.

    Args:
        model: Model identifier string.
        context: Cached context messages (can be empty).
        messages: Dynamic query messages.
        options: Configuration including retries, timeout, temperature.
        purpose: Optional semantic label for the LLM span name.
        expected_cost: Optional expected cost for cost-tracking attributes.

    Returns:
        ModelResponse with generated content.

    Raises:
        ValueError: If model is not provided or both context and messages are empty.
        LLMError: If all retry attempts are exhausted.

    Empty responses trigger a retry as they indicate API issues.
    """
    if not model:
        raise ValueError("Model must be provided")
    if not context and not messages:
        raise ValueError("Either context or messages must be provided")

    # Auto-split large images based on model-specific constraints
    context = _prepare_images_for_model(context, model)
    messages = _prepare_images_for_model(messages, model)

    if "gemini" in model.lower() and context.approximate_tokens_count < 10000:
        # Bug fix for minimum explicit context size for Gemini models
        options.cache_ttl = None

    processed_messages = _process_messages(context, messages, options.system_prompt, options.cache_ttl)
    completion_kwargs: dict[str, Any] = {
        **options.to_openai_completion_kwargs(),
    }

    if context and options.cache_ttl:
        completion_kwargs["prompt_cache_key"] = context.get_prompt_cache_key(options.system_prompt)

    for attempt in range(options.retries):
        try:
            with Laminar.start_as_current_span(purpose or model, span_type="LLM", input=processed_messages) as span:
                response = await _generate(model, processed_messages, completion_kwargs, stream=options.stream)
                laminar_metadata = response.get_laminar_metadata()
                if purpose:
                    laminar_metadata["purpose"] = purpose
                if expected_cost is not None:
                    laminar_metadata["expected_cost"] = expected_cost
                span.set_attributes(laminar_metadata)  # pyright: ignore[reportArgumentType]
                Laminar.set_span_output([r for r in (response.reasoning_content, response.content) if r])
                response.validate_output()
                return response
        except (TimeoutError, ValueError, ValidationError, Exception) as e:
            if not isinstance(e, asyncio.TimeoutError):
                # disable cache if it's not a timeout because it may cause an error
                completion_kwargs["extra_body"]["cache"] = {"no-cache": True}
                # sometimes there are issues with cache so cache is removed in case of failure
                processed_messages = _remove_cache_control(processed_messages)

            logger.warning(
                f"LLM generation failed (attempt {attempt + 1}/{options.retries}): {e}",
            )
            if attempt == options.retries - 1:
                raise LLMError("Exhausted all retry attempts for LLM generation.") from e

        await asyncio.sleep(options.retry_delay_seconds)

    raise LLMError("Unknown error occurred during LLM generation.")


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


T = TypeVar("T", bound=BaseModel)
"""Type variable for Pydantic model types in structured generation."""


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
