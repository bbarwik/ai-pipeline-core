"""LLM client implementation for AI model interactions.

@public

This module provides the core functionality for interacting with language models
through a unified interface. It handles retries, caching, structured outputs,
and integration with various LLM providers via LiteLLM.

Key functions:
- generate(): Text generation with optional context caching
- generate_structured(): Type-safe structured output generation
"""

import asyncio
from typing import Any, TypeVar

from lmnr import Laminar
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
)
from prefect.logging import get_logger
from pydantic import BaseModel

from ai_pipeline_core.exceptions import LLMError
from ai_pipeline_core.settings import settings
from ai_pipeline_core.tracing import trace

from .ai_messages import AIMessages
from .model_options import ModelOptions
from .model_response import ModelResponse, StructuredModelResponse
from .model_types import ModelName

logger = get_logger()


def _process_messages(
    context: AIMessages,
    messages: AIMessages,
    system_prompt: str | None = None,
) -> list[ChatCompletionMessageParam]:
    """Process and format messages for LLM API consumption.

    Internal function that combines context and messages into a single
    list of API-compatible messages. Applies caching directives to
    context messages for efficiency.

    Args:
        context: Messages to be cached (typically expensive/static content).
        messages: Regular messages without caching (dynamic queries).
        system_prompt: Optional system instructions for the model.

    Returns:
        List of formatted messages ready for API calls, with:
        - System prompt at the beginning (if provided)
        - Context messages with cache_control on the last one
        - Regular messages without caching

    System Prompt Location:
        The system prompt from ModelOptions.system_prompt is always injected
        as the FIRST message with role="system". It is NOT cached with context,
        allowing dynamic system prompts without breaking cache efficiency.

    Cache behavior:
        The last context message gets ephemeral caching (120s TTL)
        to reduce token usage on repeated calls with same context.

    Note:
        This is an internal function used by _generate_with_retry().
        The context/messages split enables efficient token usage.
    """
    processed_messages: list[ChatCompletionMessageParam] = []

    # Add system prompt if provided
    if system_prompt:
        processed_messages.append({"role": "system", "content": system_prompt})

    # Process context messages with caching if provided
    if context:
        # Use AIMessages.to_prompt() for context
        context_messages = context.to_prompt()

        # Apply caching to last context message
        context_messages[-1]["cache_control"] = {  # type: ignore
            "type": "ephemeral",
            "ttl": "120s",  # Cache for 2m
        }

        processed_messages.extend(context_messages)

    # Process regular messages without caching
    if messages:
        regular_messages = messages.to_prompt()
        processed_messages.extend(regular_messages)

    return processed_messages


async def _generate(
    model: str, messages: list[ChatCompletionMessageParam], completion_kwargs: dict[str, Any]
) -> ModelResponse:
    """Execute a single LLM API call.

    Internal function that makes the actual API request to the LLM provider.
    Handles both regular and structured output generation.

    Args:
        model: Model identifier (e.g., "gpt-5", "gemini-2.5-pro").
        messages: Formatted messages for the API.
        completion_kwargs: Additional parameters for the completion API.

    Returns:
        ModelResponse with generated content and metadata.

    API selection:
        - Uses client.chat.completions.parse() for structured output
        - Uses client.chat.completions.create() for regular text

    Note:
        - Uses AsyncOpenAI client configured via settings
        - Captures response headers for cost tracking
        - Response includes model options for debugging
    """
    async with AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    ) as client:
        # Use parse for structured output, create for regular
        if completion_kwargs.get("response_format"):
            raw_response = await client.chat.completions.with_raw_response.parse(  # type: ignore[var-annotated]
                **completion_kwargs,
            )
        else:
            raw_response = await client.chat.completions.with_raw_response.create(  # type: ignore[var-annotated]
                **completion_kwargs
            )

        response = ModelResponse(raw_response.parse())  # type: ignore[arg-type]
        response.set_model_options(completion_kwargs)
        response.set_headers(dict(raw_response.headers.items()))  # type: ignore[arg-type]
        return response


async def _generate_with_retry(
    model: str,
    context: AIMessages,
    messages: AIMessages,
    options: ModelOptions,
) -> ModelResponse:
    """Core LLM generation with automatic retry logic.

    Internal function that orchestrates the complete generation process
    including message processing, retries, caching, and tracing.

    Args:
        model: Model identifier string.
        context: Cached context messages (can be empty).
        messages: Dynamic query messages.
        options: Configuration including retries, timeout, temperature.

    Returns:
        ModelResponse with generated content.

    Raises:
        ValueError: If model is not provided or both context and messages are empty.
        LLMError: If all retry attempts are exhausted.

    Note:
        Empty responses trigger a retry as they indicate API issues.
    """
    if not model:
        raise ValueError("Model must be provided")
    if not context and not messages:
        raise ValueError("Either context or messages must be provided")

    processed_messages = _process_messages(context, messages, options.system_prompt)
    completion_kwargs: dict[str, Any] = {
        "model": model,
        "messages": processed_messages,
        **options.to_openai_completion_kwargs(),
    }

    if context:
        completion_kwargs["prompt_cache_key"] = context.get_prompt_cache_key(options.system_prompt)

    for attempt in range(options.retries):
        try:
            with Laminar.start_as_current_span(
                model, span_type="LLM", input=processed_messages
            ) as span:
                response = await _generate(model, processed_messages, completion_kwargs)
                span.set_attributes(response.get_laminar_metadata())
                Laminar.set_span_output(response.content)
                if not response.content:
                    raise ValueError(f"Model {model} returned an empty response.")
                return response
        except (asyncio.TimeoutError, ValueError, Exception) as e:
            if not isinstance(e, asyncio.TimeoutError):
                # disable cache if it's not a timeout because it may cause an error
                completion_kwargs["extra_body"]["cache"] = {"no-cache": True}

            logger.warning(
                "LLM generation failed (attempt %d/%d): %s",
                attempt + 1,
                options.retries,
                e,
            )
            if attempt == options.retries - 1:
                raise LLMError("Exhausted all retry attempts for LLM generation.") from e

        await asyncio.sleep(options.retry_delay_seconds)

    raise LLMError("Unknown error occurred during LLM generation.")


@trace(ignore_inputs=["context"])
async def generate(
    model: ModelName | str,
    *,
    context: AIMessages | None = None,
    messages: AIMessages | str,
    options: ModelOptions | None = None,
) -> ModelResponse:
    """Generate text response from a language model.

    @public

    Main entry point for LLM text generation with smart context caching.
    The context/messages split enables efficient token usage by caching
    expensive static content separately from dynamic queries.

    Args:
        model: Model to use (e.g., "gpt-5", "gemini-2.5-pro", "grok-4").
               Can be ModelName literal or any string for custom models.
        context: Static context to cache (documents, examples, instructions).
                Defaults to None (empty context). Cached for 120 seconds.
        messages: Dynamic messages/queries. Can be a single string or AIMessages.
                 Converted to AIMessages if string.
        options: Model configuration (temperature, retries, timeout, etc.).
                Defaults to None (uses ModelOptions() with standard settings).

    Returns:
        ModelResponse containing:
        - Generated text content
        - Usage statistics
        - Cost information (if available)
        - Model metadata

    Raises:
        ValueError: If model is empty or messages are invalid.
        LLMError: If generation fails after all retries.

    Example:
        >>> # Simple generation
        >>> response = await generate(
        ...     model="gpt-5",
        ...     messages="Explain quantum computing"
        ... )
        >>> print(response.content)

        >>> # With context caching
        >>> context = AIMessages([large_document])
        >>> response = await generate(
        ...     model="gemini-2.5-pro",
        ...     context=context,  # Cached for efficiency
        ...     messages="Summarize the key points",
        ...     options=ModelOptions(temperature=0.7)
        ... )

        >>> # Multi-turn conversation
        >>> messages = AIMessages([
        ...     "What is Python?",
        ...     previous_response,
        ...     "Can you give an example?"
        ... ])
        >>> response = await generate("gpt-5", messages=messages)

    Performance:
        - Context caching saves ~50-90% tokens on repeated calls
        - First call: full token cost
        - Subsequent calls (within 120s): only messages tokens
        - Default retry delay is 10s (configurable via ModelOptions.retry_delay_seconds)

    Caching:
        Context is cached by the LLM provider with a fixed 120-second TTL.
        The cache is based on the content hash and reduces token usage significantly.

    Note:
        - Context argument is ignored by the tracer to avoid recording large data
        - All models are accessed via LiteLLM proxy
        - Automatic retry with configurable delay between attempts
        - Cost tracking via response headers

    See Also:
        - generate_structured: For typed/structured output
        - AIMessages: Message container with document support
        - ModelOptions: Configuration options
    """
    if isinstance(messages, str):
        messages = AIMessages([messages])

    if context is None:
        context = AIMessages()
    if options is None:
        options = ModelOptions()

    try:
        return await _generate_with_retry(model, context, messages, options)
    except (ValueError, LLMError):
        raise  # Explicitly re-raise to satisfy DOC502


T = TypeVar("T", bound=BaseModel)
"""Type variable for Pydantic model types in structured generation."""


@trace(ignore_inputs=["context"])
async def generate_structured(
    model: ModelName | str,
    response_format: type[T],
    *,
    context: AIMessages | None = None,
    messages: AIMessages | str,
    options: ModelOptions | None = None,
) -> StructuredModelResponse[T]:
    """Generate structured output conforming to a Pydantic model.

    @public

    Type-safe generation that returns validated Pydantic model instances.
    Uses OpenAI's structured output feature for guaranteed schema compliance.

    Args:
        model: Model to use (must support structured output).
        response_format: Pydantic model class defining the output schema.
                        The model will generate JSON matching this schema.
        context: Static context to cache (documents, schemas, examples).
                Defaults to None (empty AIMessages).
        messages: Dynamic prompts/queries. String or AIMessages.
        options: Model configuration. response_format is set automatically.

    Returns:
        StructuredModelResponse[T] containing:
        - parsed: Validated instance of response_format class
        - All fields from regular ModelResponse (content, usage, etc.)

    Raises:
        TypeError: If response_format is not a Pydantic model class.
        ValueError: If model doesn't support structured output or no parsed content returned.
        LLMError: If generation fails after retries.
        ValidationError: If response cannot be parsed into response_format.

    Example:
        >>> from pydantic import BaseModel, Field
        >>>
        >>> class Analysis(BaseModel):
        ...     summary: str = Field(description="Brief summary")
        ...     sentiment: float = Field(ge=-1, le=1)
        ...     key_points: list[str] = Field(max_length=5)
        >>>
        >>> response = await generate_structured(
        ...     model="gpt-5",
        ...     response_format=Analysis,
        ...     messages="Analyze this product review: ..."
        ... )
        >>>
        >>> analysis = response.parsed  # Type: Analysis
        >>> print(f"Sentiment: {analysis.sentiment}")
        >>> for point in analysis.key_points:
        ...     print(f"- {point}")

    Supported models:
        Support varies by provider and model. Generally includes:
        - OpenAI: GPT-4 and newer models
        - Anthropic: Claude 3+ models
        - Google: Gemini Pro models
        Check provider documentation for specific model support.

    Performance:
        - Structured output may use more tokens than free text
        - Complex schemas increase generation time
        - Validation overhead is minimal (Pydantic is fast)

    Note:
        - Pydantic model is converted to JSON Schema for the API
        - The model generates JSON matching the schema
        - Validation happens automatically via Pydantic
        - Use Field() descriptions to guide generation

    See Also:
        - generate: For unstructured text generation
        - ModelOptions: Configuration including response_format
        - StructuredModelResponse: Response wrapper with .parsed property
    """
    if context is None:
        context = AIMessages()
    if options is None:
        options = ModelOptions()

    options.response_format = response_format

    if isinstance(messages, str):
        messages = AIMessages([messages])

    # Call the internal generate function with structured output enabled
    try:
        response = await _generate_with_retry(model, context, messages, options)
    except (ValueError, LLMError):
        raise  # Explicitly re-raise to satisfy DOC502

    # Extract the parsed value from the response
    parsed_value: T | None = None

    # Check if response has choices and parsed content
    if response.choices and hasattr(response.choices[0].message, "parsed"):
        parsed: Any = response.choices[0].message.parsed  # type: ignore[attr-defined]

        # If parsed is a dict, instantiate it as the response format class
        if isinstance(parsed, dict):
            parsed_value = response_format(**parsed)
        # If it's already the right type, use it
        elif isinstance(parsed, response_format):
            parsed_value = parsed
        else:
            # Otherwise try to convert it
            raise TypeError(
                f"Unable to convert parsed response to {response_format.__name__}: "
                f"got type {type(parsed).__name__}"  # type: ignore[reportUnknownArgumentType]
            )

    if parsed_value is None:
        raise ValueError("No parsed content available from the model response")

    # Create a StructuredModelResponse with the parsed value
    return StructuredModelResponse[T](chat_completion=response, parsed_value=parsed_value)
