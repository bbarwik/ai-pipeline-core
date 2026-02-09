"""Tests verifying that generate() sends correct Laminar span attributes.

Regression: v0.6.0 refactor built span attributes manually instead of using
ModelResponse.get_laminar_metadata(), dropping cost fields and other metadata.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from ai_pipeline_core._llm_core.client import generate
from ai_pipeline_core._llm_core.types import CoreMessage, Role


def _make_chat_completion(
    content: str = "Hello",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    cost: float | None = 0.003,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
) -> ChatCompletion:
    """Build a fake ChatCompletion with optional cost on usage."""
    usage = CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    if cost is not None:
        usage.cost = cost  # type: ignore[attr-defined]  # LiteLLM extension
    if reasoning_tokens:
        usage.completion_tokens_details = MagicMock(reasoning_tokens=reasoning_tokens)
    else:
        usage.completion_tokens_details = None
    if cached_tokens:
        usage.prompt_tokens_details = MagicMock(cached_tokens=cached_tokens)
    else:
        usage.prompt_tokens_details = None

    return ChatCompletion(
        id="chatcmpl-test",
        model="test-model",
        object="chat.completion",
        created=1234567890,
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=content, annotations=None),
            )
        ],
        usage=usage,
    )


@pytest.fixture
def mock_openai_and_laminar():
    """Mock OpenAI client and Laminar to capture span attributes."""
    captured_attrs: dict[str, object] = {}
    completion = _make_chat_completion(content="Test response", cost=0.003)

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=completion)

    mock_span = MagicMock()
    mock_span.set_attributes = MagicMock(side_effect=lambda attrs: captured_attrs.update(attrs))
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=None)

    with (
        patch("ai_pipeline_core._llm_core.client.AsyncOpenAI") as mock_openai_cls,
        patch("ai_pipeline_core._llm_core.client.Laminar") as mock_laminar,
    ):
        mock_openai_instance = AsyncMock()
        mock_openai_instance.chat.completions.create = AsyncMock(return_value=completion)
        mock_openai_instance.__aenter__ = AsyncMock(return_value=mock_openai_instance)
        mock_openai_instance.__aexit__ = AsyncMock(return_value=None)
        mock_openai_cls.return_value = mock_openai_instance

        mock_laminar.start_as_current_span.return_value = mock_span
        mock_laminar.set_span_output = MagicMock()

        yield captured_attrs, mock_openai_instance


class TestGenerateLaminarMetadata:
    """Verify that generate() sends complete metadata to Laminar spans."""

    async def test_cost_fields_sent_to_span(self, mock_openai_and_laminar):
        """Cost must use Laminar-recognized attribute names."""
        captured_attrs, _ = mock_openai_and_laminar

        messages = [CoreMessage(role=Role.USER, content="Hello")]
        await generate(messages, model="test-model")

        # gen_ai.usage.cost is the total cost recognized by Laminar backend
        assert captured_attrs["gen_ai.usage.cost"] == 0.003
        assert captured_attrs["gen_ai.usage.output_cost"] == 0.003
        # gen_ai.cost (without .usage.) is NOT recognized by Laminar — must not be set
        assert "gen_ai.cost" not in captured_attrs

    async def test_system_identifier_sent_to_span(self, mock_openai_and_laminar):
        """gen_ai.system and gen_ai.response.id must be present."""
        captured_attrs, _ = mock_openai_and_laminar

        messages = [CoreMessage(role=Role.USER, content="Hello")]
        await generate(messages, model="test-model")

        assert captured_attrs["gen_ai.system"] == "litellm"
        assert captured_attrs["gen_ai.response.id"] == "chatcmpl-test"

    async def test_usage_tokens_sent_to_span(self, mock_openai_and_laminar):
        """Token usage must use Laminar-preferred attribute names (input/output, not prompt/completion)."""
        captured_attrs, _ = mock_openai_and_laminar

        messages = [CoreMessage(role=Role.USER, content="Hello")]
        await generate(messages, model="test-model")

        assert captured_attrs["gen_ai.usage.input_tokens"] == 100
        assert captured_attrs["gen_ai.usage.output_tokens"] == 50
        assert captured_attrs["gen_ai.usage.total_tokens"] == 150
        # Legacy names must not be set — use canonical names only
        assert "gen_ai.usage.prompt_tokens" not in captured_attrs
        assert "gen_ai.usage.completion_tokens" not in captured_attrs

    async def test_purpose_and_expected_cost_sent_to_span(self, mock_openai_and_laminar):
        """Purpose and expected_cost are appended to metadata."""
        captured_attrs, _ = mock_openai_and_laminar

        messages = [CoreMessage(role=Role.USER, content="Hello")]
        await generate(messages, model="test-model", purpose="test_purpose", expected_cost=0.01)

        assert captured_attrs["purpose"] == "test_purpose"
        assert captured_attrs["expected_cost"] == 0.01

    async def test_no_cost_fields_when_cost_is_none(self, mock_openai_and_laminar):
        """Cost fields must be absent when no cost is available."""
        captured_attrs, mock_client = mock_openai_and_laminar

        # Override with no-cost completion
        no_cost_completion = _make_chat_completion(content="Test", cost=None)
        mock_client.chat.completions.create = AsyncMock(return_value=no_cost_completion)

        messages = [CoreMessage(role=Role.USER, content="Hello")]
        await generate(messages, model="test-model")

        assert "gen_ai.usage.cost" not in captured_attrs
        assert "gen_ai.usage.output_cost" not in captured_attrs
        assert "gen_ai.usage.input_cost" not in captured_attrs

    async def test_cached_tokens_use_laminar_attribute_name(self, mock_openai_and_laminar):
        """Cached tokens must use gen_ai.usage.cache_read_input_tokens (Laminar's attribute name)."""
        captured_attrs, mock_client = mock_openai_and_laminar

        completion = _make_chat_completion(content="Test", cached_tokens=80)
        mock_client.chat.completions.create = AsyncMock(return_value=completion)

        messages = [CoreMessage(role=Role.USER, content="Hello")]
        await generate(messages, model="test-model")

        assert captured_attrs["gen_ai.usage.cache_read_input_tokens"] == 80
        # gen_ai.usage.cached_tokens is NOT recognized by Laminar
        assert "gen_ai.usage.cached_tokens" not in captured_attrs

    async def test_request_model_sent_to_span(self, mock_openai_and_laminar):
        """gen_ai.request_model must be set for Laminar model identification."""
        captured_attrs, _ = mock_openai_and_laminar

        messages = [CoreMessage(role=Role.USER, content="Hello")]
        await generate(messages, model="gpt-5.1")

        assert captured_attrs["gen_ai.request_model"] == "gpt-5.1"

    async def test_metadata_matches_get_laminar_metadata(self, mock_openai_and_laminar):
        """Span attributes must be a superset of what get_laminar_metadata() returns."""
        captured_attrs, _ = mock_openai_and_laminar

        messages = [CoreMessage(role=Role.USER, content="Hello")]
        result = await generate(messages, model="test-model")

        laminar_metadata = result.get_laminar_metadata()
        for key, value in laminar_metadata.items():
            assert key in captured_attrs, f"Missing key in span attrs: {key}"
            assert captured_attrs[key] == value, f"Mismatch for {key}: span={captured_attrs[key]}, metadata={value}"
