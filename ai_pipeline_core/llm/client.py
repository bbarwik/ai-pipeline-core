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
    """Convert context and messages to OpenAI-compatible format.

    Args:
        context: Messages to be cached (optional)
        messages: Regular messages that won't be cached
        system_prompt: Optional system prompt

    Returns:
        List of formatted messages for OpenAI API
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
    """Core generation logic with exponential backoff retry."""
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
                    # disable cache in case of empty response
                    completion_kwargs["extra_body"]["cache"] = {"no-cache": True}
                    raise ValueError(f"Model {model} returned an empty response.")
                return response
        except (asyncio.TimeoutError, ValueError, Exception) as e:
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
    context: AIMessages = AIMessages(),
    messages: AIMessages | str,
    options: ModelOptions = ModelOptions(),
) -> ModelResponse:
    """Generate response using a large or small model.

    Args:
        model: The model to use for generation
        context: Messages to be cached (optional) - keyword only
        messages: Regular messages that won't be cached - keyword only
        options: Model options - keyword only

    Returns:
        Model response
    """
    if isinstance(messages, str):
        messages = AIMessages([messages])

    return await _generate_with_retry(model, context, messages, options)


T = TypeVar("T", bound=BaseModel)


@trace(ignore_inputs=["context"])
async def generate_structured(
    model: ModelName,
    response_format: type[T],
    *,
    context: AIMessages = AIMessages(),
    messages: AIMessages | str,
    options: ModelOptions = ModelOptions(),
) -> StructuredModelResponse[T]:
    """Generate structured response using Pydantic models.

    Args:
        model: The model to use for generation
        response_format: A Pydantic model class
        context: Messages to be cached (optional) - keyword only
        messages: Regular messages that won't be cached - keyword only
        options: Model options - keyword only

    Returns:
        A StructuredModelResponse containing the parsed Pydantic model instance
    """
    options.response_format = response_format

    if isinstance(messages, str):
        messages = AIMessages([messages])

    # Call the internal generate function with structured output enabled
    response = await _generate_with_retry(model, context, messages, options)

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
