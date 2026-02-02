"""Integration tests for usage tracking across all models."""

import uuid

import pytest

from ai_pipeline_core.llm import AIMessages, ModelOptions, generate
from ai_pipeline_core.llm.model_types import ModelName
from ai_pipeline_core.settings import settings

from .model_categories import ALL_MODELS

# Check if API keys are configured
HAS_API_KEYS = bool(settings.openai_api_key and settings.openai_base_url)

# Skip all tests if API keys not configured
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not HAS_API_KEYS,
        reason="OpenAI API keys not configured in settings or .env file",
    ),
]


@pytest.mark.parametrize("model", ALL_MODELS)
@pytest.mark.asyncio
async def test_usage_tracking_for_all_models(model: ModelName):
    """Test that usage information is correctly returned for all models.

    Verifies that each model returns proper usage statistics including:
    - prompt_tokens
    - completion_tokens
    - total_tokens
    - cost

    This test ensures models like grok-4.1-fast properly report usage data.
    """
    # Use UUID to make message unique and avoid caching
    unique_id = str(uuid.uuid4())
    messages = AIMessages([
        f"Solve this problem step by step (ID: {unique_id}): If a train travels at 60 mph for 2.5 hours, how far does it travel? Explain your reasoning."
    ])

    response = await generate(
        model=model,
        messages=messages,
        options=ModelOptions(max_completion_tokens=2048, retries=1),
    )

    # Verify response content exists
    assert response.content is not None
    assert len(response.content) > 0

    # Verify usage information exists
    assert response.usage is not None, f"Model {model} did not return usage information"

    # Verify token counts are present and valid
    assert response.usage.prompt_tokens > 0, f"Model {model} returned invalid prompt_tokens: {response.usage.prompt_tokens}"
    assert response.usage.completion_tokens > 0, f"Model {model} returned invalid completion_tokens: {response.usage.completion_tokens}"
    assert response.usage.total_tokens > 0, f"Model {model} returned invalid total_tokens: {response.usage.total_tokens}"

    # Verify total equals sum of prompt and completion
    expected_total = response.usage.prompt_tokens + response.usage.completion_tokens
    assert response.usage.total_tokens == expected_total, (
        f"Model {model} token counts don't add up: {response.usage.prompt_tokens} + {response.usage.completion_tokens} != {response.usage.total_tokens}"
    )

    # Verify cost attribute if present (added by LiteLLM, not all providers return it)
    cost = getattr(response.usage, "cost", None)
    if cost is not None:
        assert isinstance(cost, (int, float)), f"Model {model} cost is not numeric: {type(cost)}"
        assert cost >= 0, f"Model {model} returned negative cost: {cost}"
