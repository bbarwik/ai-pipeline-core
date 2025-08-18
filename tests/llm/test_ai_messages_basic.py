"""Tests for AIMessages basic functionality."""

import pytest

from ai_pipeline_core.llm import AIMessages, ModelResponse
from tests.test_helpers import ConcreteFlowDocument


class TestAIMessagesBasic:
    """Test basic AIMessages functionality."""

    def test_get_last_message_as_str(self):
        """Test getting last message as string."""
        # String message should return the string
        messages = AIMessages(["first", "second", "last"])
        assert messages.get_last_message_as_str() == "last"

        # Single string
        single = AIMessages(["only"])
        assert single.get_last_message_as_str() == "only"

        # Document as last should raise
        doc = ConcreteFlowDocument(name="test.txt", content=b"content")
        messages_with_doc = AIMessages(["first", doc])
        with pytest.raises(ValueError) as exc_info:
            messages_with_doc.get_last_message_as_str()
        assert "Wrong message type" in str(exc_info.value)

        # ModelResponse as last should raise
        response = ModelResponse(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "response"},
                    "finish_reason": "stop",
                }
            ],
        )
        messages_with_response = AIMessages(["first", response])
        with pytest.raises(ValueError) as exc_info:
            messages_with_response.get_last_message_as_str()
        assert "Wrong message type" in str(exc_info.value)

    def test_get_last_message(self):
        """Test getting last message of any type."""
        # String
        messages = AIMessages(["first", "last"])
        assert messages.get_last_message() == "last"

        # Document
        doc = ConcreteFlowDocument(name="test.txt", content=b"content")
        messages_doc = AIMessages(["first", doc])
        assert messages_doc.get_last_message() == doc

        # ModelResponse
        response = ModelResponse(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "response"},
                    "finish_reason": "stop",
                }
            ],
        )
        messages_response = AIMessages(["first", response])
        assert messages_response.get_last_message() == response

    def test_to_prompt_string_messages(self):
        """Test converting string messages to prompt format."""
        messages = AIMessages(["Hello", "How are you?"])
        prompt = messages.to_prompt()

        assert len(prompt) == 2
        assert prompt[0]["role"] == "user"
        assert prompt[0]["content"] == "Hello"
        assert prompt[1]["role"] == "user"
        assert prompt[1]["content"] == "How are you?"

    def test_to_prompt_model_response(self):
        """Test converting ModelResponse to assistant message."""
        response = ModelResponse(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "I am an AI assistant"},
                    "finish_reason": "stop",
                }
            ],
        )

        messages = AIMessages(["Hello", response, "Thanks"])
        prompt = messages.to_prompt()

        assert len(prompt) == 3
        assert prompt[0]["role"] == "user"
        assert prompt[0]["content"] == "Hello"
        assert prompt[1]["role"] == "assistant"
        assert prompt[1].get("content") == "I am an AI assistant"  # type: ignore[attr-defined]
        assert prompt[2]["role"] == "user"
        assert prompt[2]["content"] == "Thanks"

    def test_to_prompt_mixed_types(self):
        """Test converting mixed message types."""
        doc = ConcreteFlowDocument(name="test.txt", content=b"Document content")
        response = ModelResponse(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }
            ],
        )

        messages = AIMessages(["Start", doc, response, "End"])
        prompt = messages.to_prompt()

        assert len(prompt) == 4
        assert prompt[0]["role"] == "user"
        assert prompt[0]["content"] == "Start"
        assert prompt[1]["role"] == "user"
        # Document content will be a list of parts
        assert isinstance(prompt[1]["content"], list)
        assert prompt[2]["role"] == "assistant"
        assert prompt[2].get("content") == "Response"  # type: ignore[attr-defined]
        assert prompt[3]["role"] == "user"
        assert prompt[3]["content"] == "End"

    def test_to_prompt_empty(self):
        """Test converting empty messages."""
        messages = AIMessages([])
        prompt = messages.to_prompt()
        assert prompt == []

    def test_unsupported_message_type(self):
        """Test that unsupported message types raise an error."""
        messages = AIMessages([123])  # type: ignore
        with pytest.raises(ValueError) as exc_info:
            messages.to_prompt()
        assert "Unsupported message type" in str(exc_info.value)
