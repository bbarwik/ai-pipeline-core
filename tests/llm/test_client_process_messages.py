"""Tests for LLM client message processing."""

from ai_pipeline_core.documents import FlowDocument
from ai_pipeline_core.llm import AIMessages, ModelResponse
from ai_pipeline_core.llm.client import _process_messages  # type: ignore[reportPrivateUsage]


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
        assert result[0]["content"] == "You are a helpful assistant"

    def test_messages_only(self):
        """Test processing regular messages without context."""
        messages = AIMessages(["Hello", "How are you?"])
        result = _process_messages(context=AIMessages(), messages=messages, system_prompt=None)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "How are you?"

        # No cache control on regular messages
        assert "cache_control" not in result[0]
        assert "cache_control" not in result[1]

    def test_context_with_cache_control(self):
        """Test that context messages get cache control."""
        context = AIMessages(["Context message 1", "Context message 2"])
        messages = AIMessages(["Regular message"])

        result = _process_messages(context=context, messages=messages, system_prompt=None)

        assert len(result) == 3

        # Context messages should have cache control
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Context message 1"
        assert "cache_control" in result[0]
        assert result[0]["cache_control"]["type"] == "ephemeral"
        assert result[0]["cache_control"]["ttl"] == "120s"

        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Context message 2"
        assert "cache_control" in result[1]

        # Regular message should not have cache control
        assert result[2]["role"] == "user"
        assert result[2]["content"] == "Regular message"
        assert "cache_control" not in result[2]

    def test_full_message_ordering(self):
        """Test complete message ordering with all components."""
        # Create mixed context
        doc = FlowDocument(name="context.txt", content=b"Document content")
        context_response = ModelResponse(
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

        # Check ordering: system -> context (with cache) -> messages (no cache)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "System instructions"

        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Context string"
        assert "cache_control" in result[1]

        assert result[2]["role"] == "user"
        # Document content will be a list of parts
        assert isinstance(result[2]["content"], list)
        assert "cache_control" in result[2]

        assert result[3]["role"] == "assistant"
        assert result[3].get("content") == "Context response"  # type: ignore[attr-defined]
        # Assistant messages don't get cache control
        assert "cache_control" not in result[3]

        assert result[4]["role"] == "user"
        assert result[4]["content"] == "User question"
        assert "cache_control" not in result[4]

        assert result[5]["role"] == "user"
        assert result[5]["content"] == "Follow-up"
        assert "cache_control" not in result[5]

    def test_cache_control_only_on_user_context(self):
        """Test that only user messages in context get cache control."""
        # Mix of user and assistant messages in context
        response = ModelResponse(
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

        # First user message gets cache control
        assert result[0]["role"] == "user"
        assert "cache_control" in result[0]

        # Assistant message does not
        assert result[1]["role"] == "assistant"
        assert "cache_control" not in result[1]

        # Second user message gets cache control
        assert result[2]["role"] == "user"
        assert "cache_control" in result[2]

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
        assert result[0]["content"] == "Test"
