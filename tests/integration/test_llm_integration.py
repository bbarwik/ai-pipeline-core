"""Integration tests for LLM functionality (requires API keys)."""

import pytest
from pydantic import BaseModel

from ai_pipeline_core.llm import Conversation, ModelOptions
from ai_pipeline_core.settings import settings
from tests.support.helpers import ConcreteDocument

# Skip all tests in this file if API key not available
pytestmark = pytest.mark.integration


# Check if API keys are configured in settings (respects .env file)
HAS_API_KEYS = bool(settings.openai_api_key and settings.openai_base_url)


@pytest.mark.skipif(not HAS_API_KEYS, reason="OpenAI API keys not configured in settings or .env file")
class TestLLMIntegration:
    """Integration tests that make real LLM calls using Conversation."""

    @pytest.mark.asyncio
    async def test_simple_generation(self):
        """Test basic text generation."""
        conv = Conversation(
            model="gemini-3-flash",
            model_options=ModelOptions(max_completion_tokens=1000),
        )

        conv = await conv.send("Say 'Hello, World!' and nothing else.")

        assert conv.content
        assert "Hello" in conv.content or "hello" in conv.content
        assert conv.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_structured_generation(self):
        """Test structured output generation."""

        class SimpleResponse(BaseModel):
            greeting: str
            number: int

        conv = Conversation(
            model="gemini-3-flash",
            model_options=ModelOptions(max_completion_tokens=1000),
        )

        conv = await conv.send_structured(
            "Return a JSON with greeting='Hello' and number=42",
            response_format=SimpleResponse,
        )

        assert conv.parsed is not None
        assert conv.parsed.greeting == "Hello"
        assert conv.parsed.number == 42

    @pytest.mark.asyncio
    async def test_document_in_context(self):
        """Test using a document as context."""
        doc = ConcreteDocument(
            name="info.txt",
            content=b"The capital of France is Paris.",
            description="Geographic information",
        )

        conv = Conversation(
            model="gemini-3-flash",
            context=[doc],
            model_options=ModelOptions(max_completion_tokens=1000),
        )

        conv = await conv.send("What is the capital of France? Answer in one word.")

        assert "Paris" in conv.content

    @pytest.mark.asyncio
    async def test_conversation_with_history(self):
        """Test conversation with message history."""
        conv = Conversation(
            model="gemini-3-flash",
            model_options=ModelOptions(max_completion_tokens=1000),
        )

        # First exchange
        conv = await conv.send("My name is Alice. Remember it.")

        # Second exchange continues conversation
        conv = await conv.send("What is my name?")

        assert "Alice" in conv.content

    @pytest.mark.asyncio
    async def test_system_prompt(self):
        """Test using system prompt."""
        system_doc = ConcreteDocument(
            name="system_prompt",
            content=b"You are a pirate. Always respond like a pirate.",
        )

        conv = Conversation(
            model="gemini-3-flash",
            context=[system_doc],
            model_options=ModelOptions(max_completion_tokens=1000),
        )

        conv = await conv.send("What are you?")

        # Should have pirate-like language
        content_lower = conv.content.lower()
        pirate_words = ["arr", "ahoy", "matey", "ye", "aye", "sailor", "pirate"]
        assert any(word in content_lower for word in pirate_words)

    @pytest.mark.asyncio
    async def test_retry_options(self):
        """Test that retry parameters are accepted."""
        conv = Conversation(
            model="gemini-3-flash",
            model_options=ModelOptions(retries=2, retry_delay_seconds=1, timeout=10, max_completion_tokens=1000),
        )

        conv = await conv.send("Hello")

        assert conv.content

    @pytest.mark.asyncio
    async def test_reasoning_content(self):
        """Test reasoning content extraction (if model supports it)."""
        conv = Conversation(
            model="gemini-3-flash",
            model_options=ModelOptions(max_completion_tokens=1000),
        )

        conv = await conv.send("Solve: 2 + 2 = ?")

        # Most models won't have reasoning content in think tags
        # But we should be able to access the property without error
        reasoning = conv.reasoning_content
        assert isinstance(reasoning, str)

        # Content should be present
        assert conv.content
        assert "4" in conv.content

    @pytest.mark.asyncio
    async def test_usage_tracking(self):
        """Test that usage tracking works."""
        conv = Conversation(
            model="gemini-3-flash",
            model_options=ModelOptions(max_completion_tokens=100, usage_tracking=True),
        )

        conv = await conv.send("Count to 3")

        # Should have usage information
        assert conv.usage.total_tokens > 0
        assert conv.usage.prompt_tokens > 0
        assert conv.usage.completion_tokens > 0

    @pytest.mark.asyncio
    async def test_conversation_immutability(self):
        """Test that Conversation is immutable - send returns new instance."""
        conv = Conversation(
            model="gemini-3-flash",
            model_options=ModelOptions(max_completion_tokens=100),
        )

        result = await conv.send("Hello")

        # Original conversation should be unchanged
        assert len(conv.messages) == 0
        # Result conversation should have the exchange
        assert len(result.messages) > 0

    @pytest.mark.asyncio
    async def test_fork_conversation(self):
        """Test forking a conversation for parallel calls.

        Conversation is immutable — 'forking' is reusing the same instance
        for multiple independent sends, each returning a new Conversation.
        """
        conv = Conversation(
            model="gemini-3-flash",
            model_options=ModelOptions(max_completion_tokens=100),
        )

        # First message
        conv = await conv.send("Remember: the secret code is ALPHA")

        # Send two different follow-ups from the same conversation state
        result2a = await conv.send("What is the secret code?")
        result2b = await conv.send("Tell me the code you remember")

        # Both should remember ALPHA
        assert "ALPHA" in result2a.content or "alpha" in result2a.content.lower()
        assert "ALPHA" in result2b.content or "alpha" in result2b.content.lower()

    @pytest.mark.asyncio
    async def test_with_document(self):
        """Test adding document to conversation."""
        conv = Conversation(
            model="gemini-3-flash",
            model_options=ModelOptions(max_completion_tokens=100),
        )

        doc = ConcreteDocument(
            name="data.txt",
            content=b"The answer is 42.",
        )

        conv_with_doc = conv.with_document(doc)
        conv_with_doc = await conv_with_doc.send("What is the answer?")

        assert "42" in conv_with_doc.content

    def test_serialization(self):
        """Test conversation JSON serialization produces valid JSON."""
        import json

        conv = Conversation(
            model="gemini-3-flash",
            model_options=ModelOptions(max_completion_tokens=100),
        )

        json_str = conv.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["model"] == "gemini-3-flash"
        assert data["context"] == []
        assert data["messages"] == []

    @pytest.mark.asyncio
    async def test_substitutor_roundtrip_with_real_llm(self):
        """End-to-end: substitutor shortens URLs/addresses for LLM, restores them in output."""
        urls = [
            "https://etherscan.io/address/0xdac17f958d2ee523a2206206994597c13d831ec7",
            "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG1GvW3nY37WMAKEhLfIK3tYPcvi96LKvsRVZEhz5tW7J0wwWaD9l3YuBXL6D4B0vSwgH6NpUB9stPrmV3mE",
            "https://github.com/aptos-labs/aptos-core/blob/main/documentation/specifications/network/messaging-v1.md",
            "https://explorer.solana.com/address/Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB?cluster=mainnet-beta",
            "https://docs.uniswap.org/contracts/v3/reference/periphery/interfaces/ISwapRouter",
            "https://polygonscan.com/token/0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
            "https://bscscan.com/token/0x55d398326f99059fF775485246999027B3197955#balances",
        ]
        addresses = [
            "0xdac17f958d2ee523a2206206994597c13d831ec7",
            "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
            "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
        ]
        all_items = urls + addresses

        content = "Items to sort:\n" + "\n".join(f"- {item}" for item in all_items)
        doc = ConcreteDocument(
            name="items_list.txt",
            content=content.encode(),
            description="URLs and crypto addresses to sort alphabetically",
        )

        conv = Conversation(
            model="gemini-3-flash",
            context=[doc],
            enable_substitutor=True,
            model_options=ModelOptions(max_completion_tokens=4000),
        )

        conv = await conv.send(
            "Sort all URLs and crypto addresses from the document alphabetically. "
            "Output each item on its own line, exactly as written in the document. "
            "No headers, numbering, or commentary."
        )

        # Substitutor should have created patterns for URLs and addresses
        assert conv.substitutor is not None
        assert conv.substitutor.pattern_count > 0
        mappings = conv.substitutor.get_mappings()
        assert any(url in mappings for url in urls), "At least one URL should be shortened"
        assert any(addr in mappings for addr in addresses), "At least one address should be shortened"

        # LLM response should exist and contain restored (original) URLs
        assert conv.content

        # Content is eagerly restored — original URLs should appear directly
        found = [item for item in all_items if item in conv.content]
        assert len(found) >= len(all_items) - 2, (
            f"Expected at least {len(all_items) - 2} items in restored content, found {len(found)}/{len(all_items)}. Missing: {set(all_items) - set(found)}"
        )

        # restore_content() on already-restored content should be a no-op
        assert conv.restore_content(conv.content) == conv.content

    @pytest.mark.asyncio
    async def test_substitutor_structured_output_roundtrip(self):
        """Substitutor should work correctly with structured output (send_structured)."""

        class ItemList(BaseModel):
            items: list[str]

        urls = [
            "https://etherscan.io/address/0xdac17f958d2ee523a2206206994597c13d831ec7",
            "https://github.com/aptos-labs/aptos-core/blob/main/documentation/specifications/network/messaging-v1.md",
            "https://polygonscan.com/token/0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
        ]
        addresses = [
            "0xdac17f958d2ee523a2206206994597c13d831ec7",
            "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
        ]
        all_items = urls + addresses

        content = "Blockchain items:\n" + "\n".join(f"- {item}" for item in all_items)
        doc = ConcreteDocument(
            name="blockchain_items.txt",
            content=content.encode(),
            description="Blockchain URLs and addresses",
        )

        conv = Conversation(
            model="gemini-3-flash",
            context=[doc],
            enable_substitutor=True,
            model_options=ModelOptions(max_completion_tokens=4000),
        )

        conv = await conv.send_structured(
            "Return all URLs and addresses from the document as a list, sorted alphabetically.",
            response_format=ItemList,
        )

        assert conv.substitutor is not None
        assert conv.substitutor.pattern_count > 0
        assert conv.parsed is not None
        assert len(conv.parsed.items) >= 3, f"Expected at least 3 items, got {len(conv.parsed.items)}"

        # Parsed items are eagerly restored — original URLs should appear directly
        found = [item for item in all_items if any(item in pi for pi in conv.parsed.items)]
        assert len(found) >= len(all_items) - 1, (
            f"Expected at least {len(all_items) - 1} items in restored parsed output, "
            f"found {len(found)}/{len(all_items)}. Missing: {set(all_items) - set(found)}"
        )
