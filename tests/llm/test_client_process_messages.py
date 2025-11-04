"""Tests for LLM client message processing."""

from typing import Any, cast

from ai_pipeline_core.llm import AIMessages
from ai_pipeline_core.llm.client import _process_messages  # pyright: ignore[reportPrivateUsage]
from tests.test_helpers import ConcreteFlowDocument, create_test_model_response


class TestProcessMessages:
    """Test message processing for LLM client."""

    def test_empty_messages(self):
        """Test processing empty messages."""
        result = _process_messages(context=AIMessages(), messages=AIMessages(), system_prompt=None)
        assert result == []

    def test_system_prompt_only(self):
        """Test processing with only system prompt."""
        result = _process_messages(
            context=AIMessages(), messages=AIMessages(), system_prompt="You are a helpful assistant"
        )

        assert len(result) == 1
        assert result[0]["role"] == "system"
        content = cast(list[dict[str, Any]], result[0]["content"])
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "You are a helpful assistant"
        # System prompt gets cache_control by default (300s)
        result_0 = cast(dict[str, Any], result[0])
        assert result_0["cache_control"]["ttl"] == "300s"

    def test_messages_only(self):
        """Test processing regular messages without context."""
        messages = AIMessages(["Hello", "How are you?"])
        result = _process_messages(context=AIMessages(), messages=messages, system_prompt=None)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Hello"
        assert result[1]["role"] == "user"
        assert isinstance(result[1]["content"], list)
        assert result[1]["content"][0]["type"] == "text"
        assert result[1]["content"][0]["text"] == "How are you?"

        # No cache control on regular messages
        assert "cache_control" not in result[0]
        assert "cache_control" not in result[1]

    def test_context_with_cache_control(self):
        """Test that all context messages get cache control."""
        context = AIMessages(["Context message 1", "Context message 2"])
        messages = AIMessages(["Regular message"])

        result = _process_messages(context=context, messages=messages, system_prompt=None)

        assert len(result) == 3

        # Both context messages should have cache control
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Context message 1"
        assert "cache_control" in result[0]
        assert result[0]["cache_control"]["type"] == "ephemeral"
        assert result[0]["cache_control"]["ttl"] == "300s"

        assert result[1]["role"] == "user"
        assert isinstance(result[1]["content"], list)
        assert result[1]["content"][0]["type"] == "text"
        assert result[1]["content"][0]["text"] == "Context message 2"
        assert "cache_control" in result[1]
        assert result[1]["cache_control"]["type"] == "ephemeral"
        assert result[1]["cache_control"]["ttl"] == "300s"
        # Also check that the last content part has cache_control
        assert isinstance(result[1]["content"], list)
        assert "cache_control" in result[1]["content"][0]
        assert result[1]["content"][0]["cache_control"]["type"] == "ephemeral"
        assert result[1]["content"][0]["cache_control"]["ttl"] == "300s"

        # Regular message should not have cache control
        assert result[2]["role"] == "user"
        assert isinstance(result[2]["content"], list)
        assert result[2]["content"][0]["type"] == "text"
        assert result[2]["content"][0]["text"] == "Regular message"
        assert "cache_control" not in result[2]

    def test_full_message_ordering(self):
        """Test complete message ordering with all components."""
        # Create mixed context
        doc = ConcreteFlowDocument(name="context.txt", content=b"Document content")
        context_response = create_test_model_response(
            id="ctx-resp",
            object="chat.completion",
            created=1234567890,
            model="test",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Context response"},
                    "finish_reason": "stop",
                }
            ],
        )
        context = AIMessages(["Context string", doc, context_response])

        # Regular messages
        messages = AIMessages(["User question", "Follow-up"])

        result = _process_messages(
            context=context, messages=messages, system_prompt="System instructions"
        )

        # Check ordering: system -> context (all cached) -> messages (no cache)
        assert result[0]["role"] == "system"
        content_0 = cast(list[dict[str, Any]], result[0]["content"])
        assert content_0[0]["type"] == "text"
        assert content_0[0]["text"] == "System instructions"
        assert "cache_control" in result[0]
        result_0 = cast(dict[str, Any], result[0])
        assert result_0["cache_control"]["ttl"] == "300s"

        assert result[1]["role"] == "user"
        assert isinstance(result[1]["content"], list)
        assert result[1]["content"][0]["type"] == "text"
        assert result[1]["content"][0]["text"] == "Context string"
        assert "cache_control" in result[1]

        assert result[2]["role"] == "user"
        # Document content will be a list of parts
        assert isinstance(result[2]["content"], list)
        assert "cache_control" in result[2]

        assert result[3]["role"] == "assistant"
        assert "content" in result[3]
        assert isinstance(result[3]["content"], list)
        assert result[3]["content"][0]["type"] == "text"
        assert result[3]["content"][0]["text"] == "Context response"
        # All context messages get cache control
        assert "cache_control" in result[3]
        assert result[3]["cache_control"]["type"] == "ephemeral"
        assert result[3]["cache_control"]["ttl"] == "300s"
        # Also check that the content part has cache_control
        assert "cache_control" in result[3]["content"][0]
        assert result[3]["content"][0]["cache_control"]["ttl"] == "300s"

        assert result[4]["role"] == "user"
        assert isinstance(result[4]["content"], list)
        assert result[4]["content"][0]["type"] == "text"
        assert result[4]["content"][0]["text"] == "User question"
        assert "cache_control" not in result[4]

        assert result[5]["role"] == "user"
        assert isinstance(result[5]["content"], list)
        assert result[5]["content"][0]["type"] == "text"
        assert result[5]["content"][0]["text"] == "Follow-up"
        assert "cache_control" not in result[5]

    def test_cache_control_only_on_last_context(self):
        """Test that all context messages get cache control."""
        # Mix of user and assistant messages in context
        response = create_test_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Assistant msg"},
                    "finish_reason": "stop",
                }
            ],
        )

        context = AIMessages(["User context", response, "Another user msg"])

        result = _process_messages(context=context, messages=AIMessages(), system_prompt=None)

        # All context messages get cache control
        assert result[0]["role"] == "user"
        assert "cache_control" in result[0]
        assert result[0]["cache_control"]["ttl"] == "300s"

        assert result[1]["role"] == "assistant"
        assert "cache_control" in result[1]
        assert result[1]["cache_control"]["ttl"] == "300s"

        assert result[2]["role"] == "user"
        assert "cache_control" in result[2]
        assert result[2]["cache_control"]["type"] == "ephemeral"
        assert result[2]["cache_control"]["ttl"] == "300s"

    def test_no_system_prompt_when_none(self):
        """Test that no system message is added when prompt is None."""
        messages = AIMessages(["Test"])
        result = _process_messages(context=AIMessages(), messages=messages, system_prompt=None)

        # Should not have system message
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_empty_string_system_prompt(self):
        """Test handling of empty string system prompt."""
        messages = AIMessages(["Test"])
        result = _process_messages(context=AIMessages(), messages=messages, system_prompt="")

        # Empty string should NOT add system message (falsy check)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Test"

    def test_single_context_message_gets_cache(self):
        """Test that a single context message gets cache control."""
        context = AIMessages(["Single context message"])
        messages = AIMessages(["Regular message"])

        result = _process_messages(context=context, messages=messages, system_prompt=None)

        assert len(result) == 2

        # Single context message should get cache control
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Single context message"
        assert "cache_control" in result[0]
        assert result[0]["cache_control"]["type"] == "ephemeral"
        assert result[0]["cache_control"]["ttl"] == "300s"
        # Also check that the content part has cache_control
        assert "cache_control" in result[0]["content"][0]
        assert result[0]["content"][0]["cache_control"]["type"] == "ephemeral"
        assert result[0]["content"][0]["cache_control"]["ttl"] == "300s"

        # Regular message should not have cache control
        assert result[1]["role"] == "user"
        assert isinstance(result[1]["content"], list)
        assert result[1]["content"][0]["type"] == "text"
        assert result[1]["content"][0]["text"] == "Regular message"
        assert "cache_control" not in result[1]

    def test_assistant_as_last_context_gets_cache(self):
        """Test that all context messages get cache control including assistant."""
        response = create_test_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Last assistant msg"},
                    "finish_reason": "stop",
                }
            ],
        )

        context = AIMessages(["User msg", response])
        messages = AIMessages(["New question"])

        result = _process_messages(context=context, messages=messages, system_prompt=None)

        assert len(result) == 3

        # First context message gets cache
        assert result[0]["role"] == "user"
        assert "cache_control" in result[0]
        assert result[0]["cache_control"]["ttl"] == "300s"

        # Second context message (assistant) also gets cache
        assert result[1]["role"] == "assistant"
        assert "cache_control" in result[1]
        assert result[1]["cache_control"]["type"] == "ephemeral"
        assert result[1]["cache_control"]["ttl"] == "300s"

        # Regular message no cache
        assert result[2]["role"] == "user"
        assert "cache_control" not in result[2]
