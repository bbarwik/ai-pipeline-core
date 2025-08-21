"""Integration tests for LLM functionality (requires API keys)."""

import os

import pytest
from pydantic import BaseModel

from ai_pipeline_core.llm import AIMessages, ModelOptions, generate, generate_structured
from ai_pipeline_core.settings import settings
from tests.test_helpers import ConcreteFlowDocument

# Skip all tests in this file if API key not available
pytestmark = pytest.mark.integration


# Check if API keys are configured in settings (respects .env file)
HAS_API_KEYS = bool(settings.openai_api_key and settings.openai_base_url)


@pytest.mark.skipif(
    not HAS_API_KEYS, reason="OpenAI API keys not configured in settings or .env file"
)
class TestLLMIntegration:
    """Integration tests that make real LLM calls."""

    @pytest.mark.asyncio
    async def test_simple_generation(self):
        """Test basic text generation."""
        messages = AIMessages(["Say 'Hello, World!' and nothing else."])

        response = await generate(
            model="gemini-2.5-flash",
            messages=messages,
            options=ModelOptions(max_completion_tokens=1000),
        )

        assert response.content
        assert "Hello" in response.content or "hello" in response.content
        assert response.model
        assert response.id

    @pytest.mark.asyncio
    async def test_structured_generation(self):
        """Test structured output generation."""

        class SimpleResponse(BaseModel):
            greeting: str
            number: int

        messages = AIMessages(["Return a JSON with greeting='Hello' and number=42"])

        response = await generate_structured(
            model="gemini-2.5-flash",
            response_format=SimpleResponse,
            messages=messages,
            options=ModelOptions(max_completion_tokens=1000),
        )

        assert response.parsed.greeting == "Hello"
        assert response.parsed.number == 42

    @pytest.mark.asyncio
    async def test_document_in_context(self):
        """Test using a document as context."""
        doc = ConcreteFlowDocument(
            name="info.txt",
            content=b"The capital of France is Paris.",
            description="Geographic information",
        )

        context = AIMessages([doc])
        messages = AIMessages(["What is the capital of France? Answer in one word."])

        response = await generate(
            model="gemini-2.5-flash",
            context=context,
            messages=messages,
            options=ModelOptions(max_completion_tokens=1000),
        )

        assert "Paris" in response.content

    @pytest.mark.asyncio
    async def test_conversation_with_history(self):
        """Test conversation with message history."""
        # First exchange
        messages1 = AIMessages(["My name is Alice. Remember it."])
        response1 = await generate(
            model="gemini-2.5-flash",
            messages=messages1,
            options=ModelOptions(max_completion_tokens=1000),
        )

        # Second exchange with history
        messages2 = AIMessages(["My name is Alice. Remember it.", response1, "What is my name?"])

        response2 = await generate(
            model="gemini-2.5-flash",
            messages=messages2,
            options=ModelOptions(max_completion_tokens=1000),
        )

        assert "Alice" in response2.content

    @pytest.mark.asyncio
    async def test_system_prompt(self):
        """Test using system prompt."""
        messages = AIMessages(["What are you?"])

        response = await generate(
            model="gemini-2.5-flash",
            messages=messages,
            options=ModelOptions(
                system_prompt="You are a pirate. Always respond like a pirate.",
                max_completion_tokens=1000,
            ),
        )

        # Should have pirate-like language
        content_lower = response.content.lower()
        pirate_words = ["arr", "ahoy", "matey", "ye", "aye", "sailor", "pirate"]
        assert any(word in content_lower for word in pirate_words)

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test that retry logic works (if we can trigger it)."""
        # This is hard to test reliably without mocking
        # Just ensure the retry parameters are accepted
        messages = AIMessages(["Hello"])

        response = await generate(
            model="gemini-2.5-flash",
            messages=messages,
            options=ModelOptions(
                retries=2, retry_delay_seconds=1, timeout=10, max_completion_tokens=1000
            ),
        )

        assert response.content

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("LMNR_PROJECT_API_KEY"), reason="LMNR_PROJECT_API_KEY not set"
    )
    async def test_with_tracing(self):
        """Test that tracing doesn't break generation."""
        from ai_pipeline_core.tracing import trace

        @trace(level="debug", tags=["test"])  # Mark as test to avoid polluting prod metrics
        async def traced_generation():
            messages = AIMessages(["Say 'traced'"])
            return await generate(
                model="gemini-2.5-flash",
                messages=messages,
                options=ModelOptions(max_completion_tokens=1000),
            )

        response = await traced_generation()
        assert response.content
