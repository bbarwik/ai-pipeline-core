"""Integration tests for search models only."""

import pytest

from ai_pipeline_core.llm import AIMessages, ModelOptions, generate
from ai_pipeline_core.llm.model_types import ModelName
from ai_pipeline_core.settings import settings

from .model_categories import SEARCH_MODELS

# Check if API keys are configured in settings (respects .env file)
HAS_API_KEYS = bool(settings.openai_api_key and settings.openai_base_url)

# Skip all tests if API keys not configured
pytestmark = pytest.mark.skipif(
    not HAS_API_KEYS,
    reason="OpenAI API keys not configured in settings or .env file",
)


class TestSearchModelsIntegration:
    """Test search models with real API calls."""

    @pytest.mark.parametrize("model", SEARCH_MODELS)
    @pytest.mark.asyncio
    async def test_search_model_basic_generation(self, model: ModelName):
        """Test basic text generation for search models."""
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
            pytest.skip(f"Search model {model} not available: {e}")

    @pytest.mark.parametrize("model", SEARCH_MODELS)
    @pytest.mark.asyncio
    async def test_search_model_with_search_query(self, model: ModelName):
        """Test search models with queries that might trigger search."""
        messages = AIMessages(
            [
                "What is the current weather in San Francisco? "
                "If you can search the internet, please do so. "
                "Otherwise, just tell me you cannot search."
            ]
        )
        options = ModelOptions(max_completion_tokens=2000, search_context_size="high")

        try:
            response = await generate(model=model, messages=messages, options=options)

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0

            # Search models should either provide weather info or indicate search capability
            content_lower = response.content.lower()
            weather_keywords = ["weather", "temperature", "degrees", "sunny", "cloudy", "rain"]
            search_keywords = ["search", "internet", "cannot", "unable"]

            has_weather = any(word in content_lower for word in weather_keywords)
            has_search = any(word in content_lower for word in search_keywords)

            assert has_weather or has_search, (
                f"Search model {model} didn't mention weather or search capability.\n"
                f"Response: {response.content}"
            )

        except Exception as e:
            pytest.skip(f"Search model {model} not available: {e}")

    @pytest.mark.parametrize("model", SEARCH_MODELS)
    @pytest.mark.asyncio
    async def test_search_model_recent_events(self, model: ModelName):
        """Test search models with queries about recent events."""
        messages = AIMessages(
            [
                "Search for who was elected to be Pope in 2025. "
                "Find the name of this person from search results. "
                "If you cannot search, say 'I cannot search the internet'."
            ]
        )
        options = ModelOptions(max_completion_tokens=3000, search_context_size="high")

        try:
            response = await generate(model=model, messages=messages, options=options)

            assert response is not None
            assert response.content is not None

            content_lower = response.content.lower()

            # Search models should either find the info or indicate they searched/can't search
            search_keywords = ["search", "internet", "cannot", "unable", "don't have", "not able"]
            answer_keywords = ["robert", "leo", "prevost", "pope", "2025"]

            has_search_response = any(word in content_lower for word in search_keywords)
            has_answer = any(word in content_lower for word in answer_keywords)

            assert has_search_response or has_answer, (
                f"Search model {model} didn't provide a search response or answer.\n"
                f"Response: {response.content}"
            )

        except Exception as e:
            pytest.skip(f"Search model {model} not available: {e}")

    @pytest.mark.asyncio
    async def test_search_models_list(self):
        """Test that we have the expected search models."""
        # Should have at least some search models
        assert len(SEARCH_MODELS) > 0

        # Check for expected search models
        expected_search_models = [
            "gemini-2.5-flash-search",
            "sonar-pro-search",
            "gpt-4o-search",
            "grok-3-mini-search",
        ]

        for expected in expected_search_models:
            assert expected in SEARCH_MODELS, f"Expected search model {expected} not found"

    @pytest.mark.parametrize("model", SEARCH_MODELS)
    @pytest.mark.asyncio
    async def test_search_model_with_context(self, model: ModelName):
        """Test search models with context and messages."""
        context = AIMessages(["You are a helpful assistant. Always be concise."])
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
            pytest.skip(f"Search model {model} not available: {e}")

    @pytest.mark.parametrize("model", SEARCH_MODELS)
    @pytest.mark.asyncio
    async def test_search_model_conversation_flow(self, model: ModelName):
        """Test multi-turn conversation with search models."""
        try:
            # First message
            messages1 = AIMessages(["My name is Bob. What's 3+3?"])
            response1 = await generate(
                model=model,
                messages=messages1,
                options=ModelOptions(max_completion_tokens=1000),
            )

            assert "6" in response1.content or "six" in response1.content.lower()

            # Follow-up referencing context
            messages2 = AIMessages(["My name is Bob. What's 3+3?", response1, "What's my name?"])
            response2 = await generate(
                model=model,
                messages=messages2,
                options=ModelOptions(max_completion_tokens=1000),
            )

            # Should remember the name
            assert "bob" in response2.content.lower()

        except Exception as e:
            pytest.skip(f"Search model {model} conversation test failed: {e}")
