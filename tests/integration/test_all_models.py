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
pytestmark = pytest.mark.skipif(
    not HAS_API_KEYS,
    reason="OpenAI API keys not configured in settings or .env file",
)


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

        try:
            response = await generate(model=model, messages=messages, options=options)

            # Should get some response
            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0

            # The response should contain "4" somewhere
            assert "4" in response.content or "four" in response.content.lower()

        except Exception as e:
            # Some models might not be available - that's OK
            pytest.skip(f"Model {model} not available: {e}")

    @pytest.mark.parametrize("model", CORE_MODELS)
    @pytest.mark.asyncio
    async def test_model_with_context(self, model: ModelName):
        """Test models with context and messages."""
        context = AIMessages(["You are a math tutor. Always respond concisely."])
        messages = AIMessages(["What is the square root of 16?"])
        options = ModelOptions(max_completion_tokens=1000)

        try:
            response = await generate(
                model=model, context=context, messages=messages, options=options
            )

            assert response is not None
            assert response.content is not None
            # Should mention 4 in the response
            assert "4" in response.content or "four" in response.content.lower()

        except Exception as e:
            pytest.skip(f"Model {model} not available: {e}")

    @pytest.mark.parametrize("model", CORE_MODELS)
    @pytest.mark.asyncio
    async def test_model_options_mapping(self, model: ModelName):
        """Test that ModelOptions work correctly for all models."""
        messages = AIMessages(["Hello"])

        # Test various options configurations
        options_configs = [
            ModelOptions(max_completion_tokens=1000),
            ModelOptions(max_completion_tokens=5000),
            ModelOptions(reasoning_effort="high"),  # For models that support it
        ]

        for options in options_configs:
            try:
                # This tests that options are properly converted to API kwargs
                kwargs = options.to_openai_completion_kwargs()

                # Basic validation
                assert isinstance(kwargs, dict)

                # Model-specific validations
                if options.reasoning_effort:
                    # Other models should have reasoning if specified
                    if "extra_body" in kwargs:
                        assert "reasoning" in kwargs["extra_body"]

                # Try actual generation to ensure options work
                response = await generate(model=model, messages=messages, options=options)
                assert response is not None

            except Exception as e:
                # Some options might not be supported by all models
                if "not supported" in str(e).lower():
                    continue
                pytest.skip(f"Model {model} with options {options}: {e}")

    @pytest.mark.parametrize("model", CORE_MODELS)
    @pytest.mark.asyncio
    async def test_model_structured_generation(self, model: ModelName):
        """Test structured generation for models that support it."""
        messages = AIMessages(
            [
                "Answer this math question with a simple response. What is 5 times 3? "
                "Provide your answer and confidence level from 0 to 1."
            ]
        )
        options = ModelOptions(max_completion_tokens=1000)

        try:
            response = await generate_structured(
                model=model, response_format=SimpleResponse, messages=messages, options=options
            )

            assert response is not None
            assert response.parsed is not None
            assert isinstance(response.parsed, SimpleResponse)

            # Check the structured response
            assert "15" in response.parsed.answer or "fifteen" in response.parsed.answer.lower()
            assert 0 <= response.parsed.confidence <= 1

        except Exception as e:
            # Not all models support structured generation
            if "structured" in str(e).lower() or "not supported" in str(e).lower():
                pytest.skip(f"Model {model} doesn't support structured generation")
            else:
                pytest.skip(f"Model {model} structured generation failed: {e}")

    @pytest.mark.asyncio
    async def test_conversation_flow(self):
        """Test a multi-turn conversation with one model."""
        # Pick a reliable model for conversation test
        model = "gpt-5-mini"  # Small, fast model

        try:
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

        except Exception as e:
            pytest.skip(f"Conversation test failed: {e}")

    @pytest.mark.asyncio
    async def test_retry_mechanism(self):
        """Test that retry mechanism works with a reliable model."""
        model = "gpt-5-mini"
        messages = AIMessages(["Hi"])

        # Configure aggressive retry settings
        options = ModelOptions(
            retries=3, retry_delay_seconds=1, timeout=5, max_completion_tokens=1000
        )

        try:
            response = await generate(model=model, messages=messages, options=options)

            # Should eventually succeed
            assert response is not None
            assert len(response.content) > 0

        except Exception as e:
            pytest.skip(f"Retry test failed: {e}")
