"""Integration tests for core models (non-search models)."""

import pytest
from pydantic import BaseModel

from ai_pipeline_core.llm import AIMessages, ModelOptions, generate, generate_structured
from ai_pipeline_core.llm.model_types import ModelName
from ai_pipeline_core.settings import settings

from .model_categories import CORE_MODELS

# Check if API keys are configured in settings (respects .env file)
HAS_API_KEYS = bool(settings.openai_api_key and settings.openai_base_url)

# Skip all tests if API keys not configured
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not HAS_API_KEYS,
        reason="OpenAI API keys not configured in settings or .env file",
    ),
]


class SimpleResponse(BaseModel):
    """Simple structured response for testing."""

    answer: str
    confidence: float


class TestCoreModelsIntegration:
    """Test core (non-search) models with real API calls."""

    @pytest.mark.parametrize("model", CORE_MODELS)
    @pytest.mark.asyncio
    async def test_model_basic_generation(self, model: ModelName):
        """Test basic text generation for all models."""
        messages = AIMessages(["What is 2+2? Reply with just the number."])
        options = ModelOptions(max_completion_tokens=1000)

        response = await generate(model=model, messages=messages, options=options)

        # Should get some response
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0

        # The response should contain "4" somewhere
        assert "4" in response.content or "four" in response.content.lower()

    @pytest.mark.parametrize("model", CORE_MODELS)
    @pytest.mark.asyncio
    async def test_model_with_context(self, model: ModelName):
        """Test models with context and messages."""
        context = AIMessages(["You are a math tutor. Always respond concisely."])
        messages = AIMessages(["What is the square root of 16?"])
        options = ModelOptions(max_completion_tokens=1000)

        response = await generate(model=model, context=context, messages=messages, options=options)

        assert response is not None
        assert response.content is not None
        # Should mention 4 in the response
        assert "4" in response.content or "four" in response.content.lower()

    @pytest.mark.parametrize("model", CORE_MODELS)
    @pytest.mark.asyncio
    async def test_model_structured_generation(self, model: ModelName):
        """Test structured generation for models that support it."""
        messages = AIMessages(["Answer this math question with a simple response. What is 5 times 3? Provide your answer and confidence level from 0 to 1."])
        options = ModelOptions(max_completion_tokens=4000)

        response = await generate_structured(model=model, response_format=SimpleResponse, messages=messages, options=options)

        assert response is not None
        assert response.parsed is not None
        assert isinstance(response.parsed, SimpleResponse)

        # Check the structured response
        assert "15" in response.parsed.answer or "fifteen" in response.parsed.answer.lower()
        assert 0 <= response.parsed.confidence <= 1

    @pytest.mark.asyncio
    async def test_conversation_flow(self):
        """Test a multi-turn conversation with one model."""
        # Pick a reliable model for conversation test
        model = "gpt-5-mini"  # Small, fast model

        # First message
        messages1 = AIMessages(["My name is Alice. What's 2+2?"])
        response1 = await generate(
            model=model,
            messages=messages1,
            options=ModelOptions(max_completion_tokens=1000),
        )

        assert "4" in response1.content or "four" in response1.content.lower()

        # Follow-up referencing context
        messages2 = AIMessages(["My name is Alice. What's 2+2?", response1, "What's my name?"])
        response2 = await generate(
            model=model,
            messages=messages2,
            options=ModelOptions(max_completion_tokens=1000),
        )

        # Should remember the name
        assert "alice" in response2.content.lower()

    @pytest.mark.asyncio
    async def test_retry_mechanism(self):
        """Test that retry mechanism works with a reliable model."""
        model = "gpt-5-mini"
        messages = AIMessages(["Hi"])

        # Configure aggressive retry settings
        options = ModelOptions(retries=3, retry_delay_seconds=1, timeout=30, max_completion_tokens=1000)

        response = await generate(model=model, messages=messages, options=options)

        # Should eventually succeed
        assert response is not None
        assert len(response.content) > 0
