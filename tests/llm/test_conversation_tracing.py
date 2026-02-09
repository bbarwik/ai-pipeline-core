"""Tests verifying Conversation.send() creates a Laminar span with correct input format."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from ai_pipeline_core.llm.conversation import Conversation
from tests.support.helpers import ConcreteDocument, create_test_model_response, create_test_structured_model_response

LONG_URL = "https://etherscan.io/address/0xdac17f958d2ee523a2206206994597c13d831ec7"


class ItemList(BaseModel):
    items: list[str]


def _mock_laminar():
    """Create a mock Laminar with span capturing start_as_current_span / set_span_output calls."""
    mock_span = MagicMock()
    captured: dict[str, Any] = {"input": None, "output": None, "attrs": {}}

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


def _text_from_span_input(span_input: list[dict[str, Any]]) -> str:
    """Extract all text content from span input messages into a single string."""
    parts: list[str] = []
    for msg in span_input:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
    return " ".join(parts)


class TestConversationTracing:
    """Verify Conversation._execute_send() creates a Laminar span with correct chat-format input."""

    @pytest.mark.asyncio
    async def test_span_input_is_list_of_role_content_dicts(self, monkeypatch):
        """Span input must be a list of dicts with 'role' and 'content' keys for Laminar chat display."""
        mock_laminar, captured = _mock_laminar()

        async def fake_generate(messages, **kwargs):
            return create_test_model_response(content="response")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        with patch("ai_pipeline_core.llm.conversation.Laminar", mock_laminar):
            conv = Conversation(model="test-model")
            await conv.send("Hello")

        span_input = captured["input"]
        assert isinstance(span_input, list)
        for msg in span_input:
            assert isinstance(msg, dict), f"Expected dict, got {type(msg)}"
            assert "role" in msg, f"Missing 'role' key in {msg}"
            assert "content" in msg, f"Missing 'content' key in {msg}"
            assert msg["role"] in ("user", "assistant", "system"), f"Invalid role: {msg['role']}"

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

        span_input = captured["input"]
        assert isinstance(span_input, list)

        full_text = _text_from_span_input(span_input)
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
            assert conv.substitutor is not None
            conv.substitutor.prepare([LONG_URL])
            short = conv.substitutor.get_mappings()[LONG_URL]
            assert "..." in short

            await conv.send(f"Check {LONG_URL}")

        full_text = _text_from_span_input(captured["input"])
        assert LONG_URL in full_text
        assert short not in full_text

    @pytest.mark.asyncio
    async def test_span_input_user_message_has_correct_role(self, monkeypatch):
        """User string messages must have role='user'."""
        mock_laminar, captured = _mock_laminar()

        async def fake_generate(messages, **kwargs):
            return create_test_model_response(content="response")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        with patch("ai_pipeline_core.llm.conversation.Laminar", mock_laminar):
            conv = Conversation(model="test-model")
            await conv.send("Hello world")

        span_input = captured["input"]
        user_msgs = [m for m in span_input if m["role"] == "user"]
        assert len(user_msgs) >= 1
        assert "Hello world" in _text_from_span_input(user_msgs)

    @pytest.mark.asyncio
    async def test_span_input_conversation_history_preserves_roles(self, monkeypatch):
        """Multi-turn conversation must show alternating user/assistant roles."""
        mock_laminar, captured = _mock_laminar()

        call_count = 0

        async def fake_generate(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            return create_test_model_response(content=f"response {call_count}")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        with patch("ai_pipeline_core.llm.conversation.Laminar", mock_laminar):
            conv = Conversation(model="test-model")
            conv = await conv.send("First message")
            conv = await conv.send("Second message")

        span_input = captured["input"]
        roles = [m["role"] for m in span_input]
        assert roles == ["user", "assistant", "user"]

    @pytest.mark.asyncio
    async def test_span_input_context_documents_as_user_messages(self, monkeypatch):
        """Context documents must appear as user messages with their content."""
        mock_laminar, captured = _mock_laminar()

        async def fake_generate(messages, **kwargs):
            return create_test_model_response(content="response")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        doc = ConcreteDocument.create(name="context.md", content="Important context data")

        with patch("ai_pipeline_core.llm.conversation.Laminar", mock_laminar):
            conv = Conversation(model="test-model", context=(doc,))
            await conv.send("Analyze this")

        span_input = captured["input"]
        full_text = _text_from_span_input(span_input)
        assert "Important context data" in full_text

    @pytest.mark.asyncio
    async def test_span_input_image_content_uses_placeholder(self, monkeypatch):
        """Image content in span input should use a placeholder, not raw base64 data."""
        mock_laminar, captured = _mock_laminar()

        async def fake_generate(messages, **kwargs):
            return create_test_model_response(content="response")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        # Create a minimal valid PNG image
        import struct
        import zlib

        png_header = b"\x89PNG\r\n\x1a\n"
        ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
        ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
        raw_data = zlib.compress(b"\x00\x00\x00\x00")
        idat_crc = zlib.crc32(b"IDAT" + raw_data)
        idat = struct.pack(">I", len(raw_data)) + b"IDAT" + raw_data + struct.pack(">I", idat_crc)
        iend_crc = zlib.crc32(b"IEND")
        iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
        png_bytes = png_header + ihdr + idat + iend

        img_doc = ConcreteDocument.create(name="screenshot.png", content=png_bytes)

        with patch("ai_pipeline_core.llm.conversation.Laminar", mock_laminar):
            conv = Conversation(model="test-model")
            await conv.send(img_doc)

        span_input = captured["input"]
        serialized = str(span_input)
        # Should NOT contain raw base64 image data in the span input
        assert len(serialized) < 5000, "Span input should not contain large base64 image data"

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
        assert attrs["gen_ai.usage.input_tokens"] == 100
        assert attrs["gen_ai.usage.output_tokens"] == 50
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
