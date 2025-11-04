"""Integration tests for prompt caching functionality."""

import uuid

import pytest

from ai_pipeline_core.llm import AIMessages, ModelOptions, generate
from ai_pipeline_core.llm.model_types import ModelName
from ai_pipeline_core.settings import settings

from .model_categories import ALL_MODELS

# Check if API keys are configured
HAS_API_KEYS = bool(settings.openai_api_key and settings.openai_base_url)

# Filter models that support caching, only Gemini 2.5 for now due to explicit caching support
CACHE_SUPPORTED_MODELS: tuple[ModelName, ...] = tuple(
    model for model in ALL_MODELS if model in ["gemini-2.5-flash", "gemini-2.5-pro"]
)

# Skip all tests if API keys not configured
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not HAS_API_KEYS,
        reason="OpenAI API keys not configured in settings or .env file",
    ),
]


def create_large_context(size_kb: int = 20) -> str:
    """Create a large context message with repeated words and random suffix.

    Args:
        size_kb: Approximate size of the message in kilobytes.

    Returns:
        Large string with repeated words and unique suffix.
    """
    # Add random prefix to avoid Redis caching
    unique_prefix = f"Unique ID: {uuid.uuid4()}\n\n"

    # Create a repeated word pattern
    word = "T3st M3ss4g3 "
    # Calculate repetitions needed (roughly 7 chars per word + space)
    repetitions = (size_kb * 1024) // (len(word) + 1)
    large_text = f"{word} " * repetitions

    return unique_prefix + large_text


@pytest.mark.parametrize("model", CACHE_SUPPORTED_MODELS)
@pytest.mark.asyncio
async def test_prompt_caching(model: ModelName):
    """Test that prompt caching reduces token costs on subsequent calls.

    This test verifies that:
    1. Large context messages are properly cached
    2. Subsequent calls with the same context have lower costs
    3. The caching mechanism works across different user messages
    """
    # Create large context message (10kb)
    system_prompt = "You are a helpful assistant."
    small_context = create_large_context(1)
    large_context = create_large_context(20)
    context = AIMessages([small_context, large_context, small_context])
    assert context.approximate_tokens_count > 10000 and context.approximate_tokens_count < 20000

    # First request with large context
    messages = AIMessages(["don't think, answer as quickly as possible 'working'"])

    response_1 = await generate(
        model=model,
        context=context,
        messages=messages,
        options=ModelOptions(reasoning_effort="low", retries=1, system_prompt=system_prompt),
    )

    # Verify first response
    assert response_1.content is not None
    assert "working" in response_1.content.lower()
    assert response_1.usage is not None
    assert response_1.usage.prompt_tokens > 0

    # Get cost of first request
    cost_1 = getattr(response_1.usage, "cost", 0)
    assert cost_1 >= 0

    # Second request with same context (should use cache)
    # Append the previous response to messages
    messages.append(response_1)
    messages.append("don't think, answer as quickly as possible 'hello'")

    response_2 = await generate(
        model=model,
        context=context,  # Same context should be cached
        messages=messages,
        options=ModelOptions(reasoning_effort="low", retries=1, system_prompt=system_prompt),
    )

    # Verify second response
    assert response_2.content is not None
    assert "hello" in response_2.content.lower()
    assert response_2.usage is not None
    assert response_2.usage.prompt_tokens > 0

    # Get cost of second request
    cost_2 = getattr(response_2.usage, "cost", 0)
    assert cost_2 >= 0

    # Get cached tokens count from second response
    cached_tokens = 0
    if hasattr(response_2.usage, "prompt_tokens_details"):
        prompt_details = response_2.usage.prompt_tokens_details
        if prompt_details and hasattr(prompt_details, "cached_tokens"):
            cached_tokens = prompt_details.cached_tokens or 0

    # Calculate cost difference percentage
    cost_reduction_pct = 0
    if cost_1 > 0:
        cost_reduction_pct = ((cost_1 - cost_2) / cost_1) * 100

    # Print caching statistics
    print(f"\n{model} caching stats:")
    print(f"  Cached tokens: {cached_tokens:,}")
    print(f"  Cost reduction: {cost_reduction_pct:.1f}% (${cost_1:.6f} -> ${cost_2:.6f})")

    # If significant caching (>1000 tokens), we can be more lenient with cost check
    if cached_tokens > 1000:
        # Just verify tokens were cached, cost calculation might vary
        assert cached_tokens > 0, f"Model {model} should have cached tokens"
    else:
        # Verify caching worked: second call should be cheaper
        assert cost_2 < cost_1, (
            f"Model {model} did not show cost reduction with caching: "
            f"First call cost: {cost_1}, Second call cost: {cost_2}"
        )

    # Also verify that prompt tokens are reported
    # (cached tokens are still counted but charged differently)
    assert response_2.usage.prompt_tokens > response_1.usage.prompt_tokens, (
        f"Model {model} second call should have more prompt tokens "
        f"(includes previous response): "
        f"{response_2.usage.prompt_tokens} vs {response_1.usage.prompt_tokens}"
    )
