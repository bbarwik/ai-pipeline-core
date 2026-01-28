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
            model="gemini-3-flash",
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
            model="gemini-3-flash",
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
            model="gemini-3-flash",
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
            model="gemini-3-flash",
            messages=messages1,
            options=ModelOptions(max_completion_tokens=1000),
        )

        # Second exchange with history
        messages2 = AIMessages(["My name is Alice. Remember it.", response1, "What is my name?"])

        response2 = await generate(
            model="gemini-3-flash",
            messages=messages2,
            options=ModelOptions(max_completion_tokens=1000),
        )

        assert "Alice" in response2.content

    @pytest.mark.asyncio
    async def test_system_prompt(self):
        """Test using system prompt."""
        messages = AIMessages(["What are you?"])

        response = await generate(
            model="gemini-3-flash",
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
            model="gemini-3-flash",
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
                model="gemini-3-flash",
                messages=messages,
                options=ModelOptions(max_completion_tokens=1000),
            )

        response = await traced_generation()
        assert response.content

    @pytest.mark.asyncio
    async def test_approximate_tokens_count(self):
        """Test approximate tokens count with real conversation."""
        doc = ConcreteFlowDocument(
            name="context.txt",
            content=b"This is important context information for the AI to use.",
        )

        messages = AIMessages([
            "Hello, I have a question",
            doc,
            "Can you help me understand the context?",
        ])

        # Get token count
        token_count = messages.approximate_tokens_count

        # Should have reasonable token count
        assert token_count > 0
        assert isinstance(token_count, int)
        # Should be less than a few hundred for this small conversation
        assert token_count < 500

    @pytest.mark.asyncio
    async def test_response_metadata(self):
        """Test that response includes timing metadata."""
        messages = AIMessages(["Say 'metadata test' quickly"])

        response = await generate(
            model="gemini-3-flash",
            messages=messages,
            options=ModelOptions(max_completion_tokens=100),
        )

        # Check that response has content
        assert response.content

        # Check laminar metadata includes timing info
        metadata = response.get_laminar_metadata()
        assert "time_taken" in metadata
        assert "first_token_time" in metadata
        assert isinstance(metadata["time_taken"], (int, float))
        assert isinstance(metadata["first_token_time"], (int, float))
        # First token should be before total time
        assert metadata["first_token_time"] <= metadata["time_taken"]

    @pytest.mark.asyncio
    async def test_reasoning_content(self):
        """Test reasoning content extraction (if model supports it)."""
        messages = AIMessages(["Solve: 2 + 2 = ?"])

        response = await generate(
            model="gemini-3-flash",
            messages=messages,
            options=ModelOptions(max_completion_tokens=1000),
        )

        # Most models won't have reasoning content in think tags
        # But we should be able to call the property without error
        reasoning = response.reasoning_content
        assert isinstance(reasoning, str)

        # Content should be present
        assert response.content
        assert "4" in response.content

    @pytest.mark.asyncio
    async def test_structured_response_lazy_parsing(self):
        """Test that structured response parsing is lazy."""

        class MathResult(BaseModel):
            equation: str
            answer: int

        messages = AIMessages(["What is 10 + 15? Return JSON with equation and answer."])

        response = await generate_structured(
            model="gemini-3-flash",
            response_format=MathResult,
            messages=messages,
            options=ModelOptions(max_completion_tokens=1000),
        )

        # Response should be StructuredModelResponse
        from ai_pipeline_core.llm import StructuredModelResponse

        assert isinstance(response, StructuredModelResponse)

        # Access parsed property
        parsed = response.parsed
        assert parsed.answer == 25
        assert "10" in parsed.equation or "15" in parsed.equation

        # Accessing again should return cached value
        parsed2 = response.parsed
        assert parsed is parsed2  # Same object

    @pytest.mark.asyncio
    async def test_usage_tracking_enabled(self):
        """Test that usage tracking works when enabled."""
        messages = AIMessages(["Count to 3"])

        response = await generate(
            model="gemini-3-flash",
            messages=messages,
            options=ModelOptions(
                max_completion_tokens=100,
                usage_tracking=True,
            ),
        )

        # Should have usage information
        assert response.usage is not None
        assert response.usage.total_tokens > 0
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

        # Metadata should include usage
        metadata = response.get_laminar_metadata()
        total_tokens = metadata["gen_ai.usage.total_tokens"]
        assert isinstance(total_tokens, (int, float))
        assert total_tokens > 0

    @pytest.mark.asyncio
    async def test_conversation_with_tokens_tracking(self):
        """Test tracking tokens across conversation turns."""
        # First turn - simple question
        messages1 = AIMessages(["Count to 3"])
        count1 = messages1.approximate_tokens_count

        response1 = await generate(
            model="gemini-3-flash",
            messages=messages1,
            options=ModelOptions(max_completion_tokens=50),
        )

        # Second turn with history
        messages2 = AIMessages([
            "Count to 3",
            response1,
            "Now count to 5",
        ])
        count2 = messages2.approximate_tokens_count

        # Second turn should have more tokens (includes response from first turn)
        assert count2 > count1

        response2 = await generate(
            model="gemini-3-flash",
            messages=messages2,
            options=ModelOptions(max_completion_tokens=50),
        )

        assert response2.content
        assert "5" in response2.content

    @pytest.mark.asyncio
    async def test_metadata_in_options(self):
        """Test that metadata can be passed through ModelOptions."""
        metadata = {
            "experiment": "integration-test",
            "version": "1.0",
            "feature": "metadata-tracking",
        }

        messages = AIMessages(["Say 'metadata test'"])

        response = await generate(
            model="gemini-3-flash",
            messages=messages,
            options=ModelOptions(
                metadata=metadata,
                max_completion_tokens=100,
            ),
        )

        # Response should be successful
        assert response.content
        assert "metadata" in response.content.lower() or "test" in response.content.lower()

        # Metadata should be in the model options used for the call
        # (We can't directly verify it was sent to API, but we can verify it's in options)
        options = ModelOptions(metadata=metadata)
        kwargs = options.to_openai_completion_kwargs()
        assert kwargs["metadata"] == metadata
