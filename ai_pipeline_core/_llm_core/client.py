"""Primitive LLM client for internal modules.

This module provides low-level generate() and generate_structured() functions.
Expects pre-processed content (images already split/compressed by llm layer).

For app code, use the llm module's Conversation class instead.
"""

import asyncio
import base64
import hashlib
import json
import time
from io import BytesIO
from typing import Any, TypeVar

from lmnr import Laminar
from openai import AsyncOpenAI
from openai.lib.streaming.chat import ChunkEvent, ContentDeltaEvent, ContentDoneEvent
from openai.types.chat import ChatCompletionMessageParam
from PIL import Image
from pydantic import BaseModel, ValidationError

from ai_pipeline_core.exceptions import LLMError
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.settings import settings

from .model_options import ModelOptions
from .model_response import Citation, ModelResponse
from .types import (
    ContentPart,
    CoreMessage,
    ImageContent,
    PDFContent,
    Role,
    TextContent,
    TokenUsage,
)

logger = get_pipeline_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# Valid finish reasons accepted by downstream consumers
_VALID_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter", "function_call"})


def _validate_image(data: bytes, name: str = "image") -> str | None:
    """Validate image content. Returns error message or None if valid."""
    if not data:
        return f"empty image content in '{name}'"
    try:
        with Image.open(BytesIO(data)) as img:
            img.verify()
        return None
    except (OSError, ValueError, Image.DecompressionBombError) as e:
        return f"invalid image in '{name}': {e}"


def _validate_pdf(data: bytes, name: str = "pdf") -> str | None:
    """Validate PDF content. Returns error message or None if valid."""
    if not data:
        return f"empty PDF content in '{name}'"
    if not data.lstrip().startswith(b"%PDF-"):
        return f"invalid PDF header in '{name}' (missing %PDF- signature)"
    return None


def _looks_like_text(content: bytes) -> bool:
    """Check if content is valid UTF-8 text (not binary)."""
    if not content:
        return True
    if b"\x00" in content:
        return False
    try:
        content.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _has_pdf_signature(content: bytes) -> bool:
    """Check if content starts with PDF magic bytes (%PDF-)."""
    return content.lstrip().startswith(b"%PDF-")


def _content_to_api_parts(content: str | ContentPart | tuple[ContentPart, ...]) -> list[dict[str, Any]]:
    """Convert content to OpenAI API format.

    Expects pre-processed images (already split/compressed by llm layer).
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if isinstance(content, TextContent):
        return [{"type": "text", "text": content.text}]

    if isinstance(content, ImageContent):
        # Validate image
        if err := _validate_image(content.data):
            logger.warning(f"Skipping invalid image: {err}")
            return []

        # Encode pre-processed image (splitting done by llm layer)
        b64 = base64.b64encode(content.data).decode("utf-8")
        return [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{content.mime_type};base64,{b64}", "detail": "high"},
            }
        ]

    if isinstance(content, PDFContent):
        # Check if "PDF" is actually text (misnamed file)
        if _looks_like_text(content.data) and not _has_pdf_signature(content.data):
            logger.debug("PDF content is actually text - sending as text")
            return [{"type": "text", "text": content.data.decode("utf-8")}]

        # Validate PDF
        if err := _validate_pdf(content.data):
            logger.warning(f"Skipping invalid PDF: {err}")
            return []

        b64 = base64.b64encode(content.data).decode("utf-8")
        return [{"type": "file", "file": {"file_data": f"data:application/pdf;base64,{b64}"}}]

    # Tuple of parts
    result = []
    for part in content:
        result.extend(_content_to_api_parts(part))
    return result


def _messages_to_api(messages: list[CoreMessage]) -> list[ChatCompletionMessageParam]:
    """Convert CoreMessages to OpenAI API format."""
    result: list[ChatCompletionMessageParam] = []
    for msg in messages:
        parts = _content_to_api_parts(msg.content)
        if parts:  # Skip messages with no valid content
            result.append({"role": msg.role.value, "content": parts})  # type: ignore[arg-type]
    return result


def _apply_cache_control(messages: list[ChatCompletionMessageParam], cache_ttl: str, context_count: int) -> None:
    """Apply cache_control to context messages (first context_count messages)."""
    for message in messages[:context_count]:
        message["cache_control"] = {"type": "ephemeral", "ttl": cache_ttl}  # type: ignore[typeddict-unknown-key]
        if isinstance(message.get("content"), list):
            # Also apply to last content item for better cache hits
            message["content"][-1]["cache_control"] = {"type": "ephemeral", "ttl": cache_ttl}  # type: ignore[typeddict-unknown-key]


def _remove_cache_control(messages: list[ChatCompletionMessageParam]) -> list[ChatCompletionMessageParam]:
    """Remove cache_control directives from messages (for retry after cache errors)."""
    for message in messages:
        if (content := message.get("content")) and isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "cache_control" in item:
                    del item["cache_control"]
        if "cache_control" in message:
            del message["cache_control"]
    return messages


def _compute_cache_key(messages: list[ChatCompletionMessageParam], system_prompt: str | None) -> str:
    """Compute SHA256 cache key for messages."""
    key_data = (system_prompt or "") + json.dumps(messages, sort_keys=True, default=str)
    return hashlib.sha256(key_data.encode()).hexdigest()


def _estimate_token_count(messages: list[ChatCompletionMessageParam]) -> int:
    """Rough estimate of token count for Gemini cache threshold."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content) // 4
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(part.get("text", "")) // 4
                elif isinstance(part, dict) and part.get("type") in {"image_url", "file"}:
                    total += 1000  # Images/PDFs are ~1000 tokens
    return total


def _extract_usage(response: Any) -> TokenUsage:
    """Extract token usage from API response."""
    usage = response.usage
    if not usage:
        return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    cached = 0
    reasoning = 0

    if prompt_details := getattr(usage, "prompt_tokens_details", None):
        cached = getattr(prompt_details, "cached_tokens", 0) or 0

    if completion_details := getattr(usage, "completion_tokens_details", None):
        reasoning = getattr(completion_details, "reasoning_tokens", 0) or 0

    return TokenUsage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        cached_tokens=cached,
        reasoning_tokens=reasoning,
    )


def _extract_cost(response: Any) -> float | None:
    """Extract cost from API response if available."""
    if (usage := response.usage) and hasattr(usage, "cost"):
        return float(usage.cost)  # type: ignore[attr-defined]
    return None


def _model_name_to_openrouter_model(model: str) -> str:
    """Convert model name to OpenRouter format if needed."""
    if model == "sonar-pro-search":
        return "perplexity/sonar-pro-search"
    if model.endswith("-search"):
        model = model.replace("-search", ":online")
    if model.startswith("gemini"):
        return f"google/{model}"
    if model.startswith("gpt"):
        return f"openai/{model}"
    if model.startswith("grok"):
        return f"x-ai/{model}"
    if model.startswith("claude"):
        return f"anthropic/{model}"
    if model.startswith("qwen3"):
        return f"qwen/{model}"
    if model.startswith("deepseek-"):
        return f"deepseek/{model}"
    if model.startswith("glm-"):
        return f"z-ai/{model}"
    if model.startswith("kimi-"):
        return f"moonshotai/{model}"
    return model


async def _generate_streaming(
    client: AsyncOpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    completion_kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any], Any]:
    """Execute streaming LLM API call. Returns (response, metadata)."""
    start_time = time.time()
    first_token_time = None
    usage = None

    async with client.chat.completions.stream(
        model=model,
        messages=messages,
        **completion_kwargs,
    ) as stream:
        async for event in stream:
            if isinstance(event, ContentDeltaEvent):
                if not first_token_time:
                    first_token_time = time.time()
            elif isinstance(event, ContentDoneEvent):
                pass
            elif isinstance(event, ChunkEvent) and event.chunk.usage:
                usage = event.chunk.usage

        if not first_token_time:
            first_token_time = time.time()
        response = await stream.get_final_completion()

    metadata = {
        "time_taken": round(time.time() - start_time, 2),
        "first_token_time": round(first_token_time - start_time, 2),
    }
    return response, metadata, usage


async def _generate_non_streaming(
    client: AsyncOpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    completion_kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any], Any]:
    """Execute non-streaming LLM API call. Returns (response, metadata).

    Uses non-streaming to avoid OpenAI SDK delta accumulation issues with
    some providers (e.g., Grok annotation deltas crash the SDK).
    """
    start_time = time.time()
    kwargs = {k: v for k, v in completion_kwargs.items() if k != "stream_options"}

    response_format = kwargs.get("response_format")
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        response = await client.chat.completions.parse(
            model=model,
            messages=messages,
            **kwargs,
        )
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            **kwargs,
        )

    elapsed = round(time.time() - start_time, 2)
    metadata = {"time_taken": elapsed, "first_token_time": elapsed}
    return response, metadata, None


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
