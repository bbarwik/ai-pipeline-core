"""Tests verifying Conversation.send() creates a Laminar span with pre-substitution input."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from ai_pipeline_core._llm_core.types import CoreMessage
from ai_pipeline_core.llm.conversation import Conversation
from tests.support.helpers import create_test_model_response, create_test_structured_model_response

LONG_URL = "https://etherscan.io/address/0xdac17f958d2ee523a2206206994597c13d831ec7"


class ItemList(BaseModel):
    items: list[str]


def _mock_laminar():
    """Create a mock Laminar with span capturing start_as_current_span / set_span_output calls."""
    mock_span = MagicMock()
    captured: dict[str, object] = {"input": None, "output": None, "attrs": {}}

    mock_span.set_attributes = MagicMock(side_effect=lambda a: captured["attrs"].update(a))  # type: ignore[union-attr]
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=None)

    mock_laminar = MagicMock()
    mock_laminar.start_as_current_span.return_value = mock_span
    mock_laminar.set_span_output = MagicMock(side_effect=lambda v: captured.__setitem__("output", v))

    # Capture input from start_as_current_span(name, input=...)
    original_start = mock_laminar.start_as_current_span

    def _capture_start(*args, **kwargs):
        captured["input"] = kwargs.get("input")
        return original_start(*args, **kwargs)

    mock_laminar.start_as_current_span = MagicMock(side_effect=_capture_start)
    mock_laminar.start_as_current_span.return_value = mock_span

    return mock_laminar, captured


def _text_from_core_messages(messages: list[CoreMessage]) -> str:
    """Extract all text content from CoreMessages into a single string."""
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg.content, str):
            parts.append(msg.content)
        elif isinstance(msg.content, tuple):
            parts.extend(part.text for part in msg.content if hasattr(part, "text"))
    return " ".join(parts)


class TestConversationTracing:
    """Verify Conversation._execute_send() creates a Laminar span with pre-substitution input."""

    @pytest.mark.asyncio
    async def test_span_input_captures_pre_substitution_content(self, monkeypatch):
        """The span input must contain original URLs, not shortened forms."""
        mock_laminar, captured = _mock_laminar()

        async def fake_generate(messages, **kwargs):
            return create_test_model_response(content="response")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        with patch("ai_pipeline_core.llm.conversation.Laminar", mock_laminar):
            conv = Conversation(model="test-model")
            await conv.send(f"Check {LONG_URL}")

        # Span input should be a list of CoreMessages with pre-substitution content
        span_input = captured["input"]
        assert isinstance(span_input, list)
        assert all(isinstance(m, CoreMessage) for m in span_input)

        full_text = _text_from_core_messages(span_input)
        assert LONG_URL in full_text

    @pytest.mark.asyncio
    async def test_span_input_not_shortened(self, monkeypatch):
        """Verify the shortened form does NOT appear in span input."""
        mock_laminar, captured = _mock_laminar()

        async def fake_generate(messages, **kwargs):
            return create_test_model_response(content="response")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        with patch("ai_pipeline_core.llm.conversation.Laminar", mock_laminar):
            conv = Conversation(model="test-model")
            # Prepare substitutor so it has patterns
            assert conv.substitutor is not None
            conv.substitutor.prepare([LONG_URL])
            short = conv.substitutor.get_mappings()[LONG_URL]
            assert "..." in short  # sanity: substitutor shortened it

            await conv.send(f"Check {LONG_URL}")

        full_text = _text_from_core_messages(captured["input"])
        # Must contain original, must NOT contain shortened form
        assert LONG_URL in full_text
        assert short not in full_text

    @pytest.mark.asyncio
    async def test_span_output_set_to_response_content(self, monkeypatch):
        mock_laminar, captured = _mock_laminar()

        async def fake_generate(messages, **kwargs):
            return create_test_model_response(content="LLM says hello")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        with patch("ai_pipeline_core.llm.conversation.Laminar", mock_laminar):
            conv = Conversation(model="test-model")
            await conv.send("test")

        assert captured["output"] == "LLM says hello"

    @pytest.mark.asyncio
    async def test_span_attributes_include_model_metadata(self, monkeypatch):
        mock_laminar, captured = _mock_laminar()

        async def fake_generate(messages, **kwargs):
            return create_test_model_response(content="response", prompt_tokens=100, completion_tokens=50)

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        with patch("ai_pipeline_core.llm.conversation.Laminar", mock_laminar):
            conv = Conversation(model="test-model")
            await conv.send("test")

        attrs = captured["attrs"]
        assert isinstance(attrs, dict)
        assert attrs["gen_ai.usage.prompt_tokens"] == 100
        assert attrs["gen_ai.usage.completion_tokens"] == 50
        assert attrs["gen_ai.system"] == "litellm"

    @pytest.mark.asyncio
    async def test_span_name_uses_purpose(self, monkeypatch):
        mock_laminar, _ = _mock_laminar()

        async def fake_generate(messages, **kwargs):
            return create_test_model_response(content="response")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        with patch("ai_pipeline_core.llm.conversation.Laminar", mock_laminar):
            conv = Conversation(model="test-model")
            await conv.send("test", purpose="verify_source")

        call_args = mock_laminar.start_as_current_span.call_args
        assert call_args[0][0] == "verify_source"

    @pytest.mark.asyncio
    async def test_span_name_defaults_to_send(self, monkeypatch):
        mock_laminar, _ = _mock_laminar()

        async def fake_generate(messages, **kwargs):
            return create_test_model_response(content="response")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        with patch("ai_pipeline_core.llm.conversation.Laminar", mock_laminar):
            conv = Conversation(model="test-model")
            await conv.send("test")

        call_args = mock_laminar.start_as_current_span.call_args
        assert call_args[0][0] == "conversation.send"

    @pytest.mark.asyncio
    async def test_span_name_defaults_to_send_structured(self, monkeypatch):
        mock_laminar, _ = _mock_laminar()

        parsed = ItemList(items=["a"])

        async def fake_generate_structured(messages, response_format, **kwargs):
            return create_test_structured_model_response(parsed=parsed)

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate_structured", fake_generate_structured)

        with patch("ai_pipeline_core.llm.conversation.Laminar", mock_laminar):
            conv = Conversation(model="test-model")
            await conv.send_structured("test", ItemList)

        call_args = mock_laminar.start_as_current_span.call_args
        assert call_args[0][0] == "conversation.send_structured"
