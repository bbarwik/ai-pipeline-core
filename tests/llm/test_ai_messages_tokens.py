"""Comprehensive tests for AIMessages approximate_tokens_count functionality."""

import pytest

from ai_pipeline_core.llm import AIMessages
from tests.test_helpers import ConcreteFlowDocument, create_test_model_response


class TestAIMessagesApproximateTokensCount:
    """Test approximate tokens count with comprehensive coverage."""

    def test_empty_messages(self):
        """Test token count for empty messages."""
        messages = AIMessages([])
        count = messages.approximate_tokens_count
        assert count == 0

    def test_single_short_string(self):
        """Test token count for single short string."""
        messages = AIMessages(["Hi"])
        count = messages.approximate_tokens_count
        assert count > 0
        assert count < 10  # Should be very few tokens

    def test_single_long_string(self):
        """Test token count for single long string."""
        long_text = "This is a much longer message that should have significantly more tokens " * 10
        messages = AIMessages([long_text])
        count = messages.approximate_tokens_count
        assert count > 100  # Should have many tokens

    def test_multiple_strings(self):
        """Test token count accumulates across multiple strings."""
        messages = AIMessages(["First message", "Second message", "Third message"])
        count = messages.approximate_tokens_count

        # Count should be sum of all messages
        single_counts = [
            AIMessages(["First message"]).approximate_tokens_count,
            AIMessages(["Second message"]).approximate_tokens_count,
            AIMessages(["Third message"]).approximate_tokens_count,
        ]
        assert count == sum(single_counts)

    def test_unicode_string(self):
        """Test token count with unicode characters."""
        messages = AIMessages(["Hello 世界 🌍"])
        count = messages.approximate_tokens_count
        assert count > 0
        assert isinstance(count, int)

    def test_multiline_string(self):
        """Test token count with multiline strings."""
        multiline = """This is a
        multiline message
        that spans multiple
        lines"""
        messages = AIMessages([multiline])
        count = messages.approximate_tokens_count
        assert count > 0

    def test_text_document(self):
        """Test token count for text document."""
        doc = ConcreteFlowDocument(name="test.txt", content=b"This is document content")
        messages = AIMessages([doc])
        count = messages.approximate_tokens_count
        assert count > 0

    def test_non_text_document(self):
        """Test token count for non-text document (uses fixed estimate)."""
        # Create a document with binary content (will be detected as non-text)
        doc = ConcreteFlowDocument(name="test.bin", content=b"\x89PNG\r\n\x1a\n")
        messages = AIMessages([doc])
        count = messages.approximate_tokens_count
        assert count == 1024  # Fixed estimate for non-text

    def test_model_response_content(self):
        """Test token count for model response."""
        response = create_test_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a response from the model",
                    },
                    "finish_reason": "stop",
                }
            ],
        )
        messages = AIMessages([response])
        count = messages.approximate_tokens_count
        assert count > 0

    def test_model_response_with_think_tags(self):
        """Test that token count includes think tags content."""
        response = create_test_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "<think>Internal reasoning</think> Visible content",
                    },
                    "finish_reason": "stop",
                }
            ],
        )
        messages = AIMessages([response])
        count = messages.approximate_tokens_count

        # Should count the visible content, not thinking content
        # (content property strips think tags)
        assert count > 0
        # Count should be based on "Visible content" only
        assert count < 10

    def test_mixed_message_types(self):
        """Test token count with all message types mixed."""
        doc = ConcreteFlowDocument(name="test.txt", content=b"Document content here")
        response = create_test_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response content"},
                    "finish_reason": "stop",
                }
            ],
        )

        messages = AIMessages(["User message", doc, response, "Follow-up question"])
        count = messages.approximate_tokens_count

        # Should be sum of all parts
        assert count >= 10  # At least some tokens
        assert isinstance(count, int)

    def test_empty_string_in_messages(self):
        """Test token count with empty string message."""
        messages = AIMessages([""])
        count = messages.approximate_tokens_count
        assert count == 0

    def test_multiple_empty_strings(self):
        """Test token count with multiple empty strings."""
        messages = AIMessages(["", "", ""])
        count = messages.approximate_tokens_count
        assert count == 0

    def test_mixed_empty_and_content(self):
        """Test token count with mix of empty and content."""
        messages = AIMessages(["", "Hello", "", "World", ""])
        count = messages.approximate_tokens_count
        assert count > 0

    def test_very_long_document(self):
        """Test token count for very long document."""
        long_content = b"This is a very long document. " * 1000
        doc = ConcreteFlowDocument(name="long.txt", content=long_content)
        messages = AIMessages([doc])
        count = messages.approximate_tokens_count
        assert count > 1000  # Should have many tokens

    def test_document_with_description(self):
        """Test token count includes document description in the encoding."""
        doc = ConcreteFlowDocument(
            name="test.txt",
            content=b"Content",
            description="This is a description",
        )
        messages = AIMessages([doc])
        count = messages.approximate_tokens_count
        assert count > 0

    def test_consecutive_responses(self):
        """Test token count with multiple consecutive responses."""
        response1 = create_test_model_response(
            id="test1",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "First response"},
                    "finish_reason": "stop",
                }
            ],
        )
        response2 = create_test_model_response(
            id="test2",
            object="chat.completion",
            created=1234567891,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Second response"},
                    "finish_reason": "stop",
                }
            ],
        )

        messages = AIMessages([response1, response2])
        count = messages.approximate_tokens_count

        # Should be sum of both responses
        count1 = AIMessages([response1]).approximate_tokens_count
        count2 = AIMessages([response2]).approximate_tokens_count
        assert count == count1 + count2

    def test_special_characters(self):
        """Test token count with special characters."""
        messages = AIMessages(["Hello! @#$% ^&*() <>?"])
        count = messages.approximate_tokens_count
        assert count > 0

    def test_numbers_and_text(self):
        """Test token count with numbers."""
        messages = AIMessages(["The year 2024 has 365 days"])
        count = messages.approximate_tokens_count
        assert count > 0

    def test_code_content(self):
        """Test token count with code content."""
        code = """
        def hello_world():
            print("Hello, World!")
            return True
        """
        messages = AIMessages([code])
        count = messages.approximate_tokens_count
        assert count > 0

    def test_json_content(self):
        """Test token count with JSON content."""
        json_content = '{"name": "test", "value": 123, "nested": {"key": "value"}}'
        messages = AIMessages([json_content])
        count = messages.approximate_tokens_count
        assert count > 0

    def test_consistency(self):
        """Test that token count is consistent for same input."""
        messages = AIMessages(["Consistent message"])
        count1 = messages.approximate_tokens_count
        count2 = messages.approximate_tokens_count
        count3 = messages.approximate_tokens_count
        assert count1 == count2 == count3

    def test_unsupported_message_type_error(self):
        """Test that unsupported message types raise ValueError."""
        messages = AIMessages([123])  # type: ignore
        with pytest.raises(ValueError, match="Unsupported message type"):
            messages.approximate_tokens_count

    def test_realistic_conversation(self):
        """Test token count for realistic conversation."""
        doc = ConcreteFlowDocument(
            name="context.txt",
            content=b"This is important context information for the conversation.",
        )
        response1 = create_test_model_response(
            id="resp1",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I understand. Based on the context, I can help you.",
                    },
                    "finish_reason": "stop",
                }
            ],
        )

        messages = AIMessages([
            "Hello, I need help with this document",
            doc,
            response1,
            "Can you explain more about the context?",
        ])

        count = messages.approximate_tokens_count
        assert count > 20  # Should have reasonable number of tokens
        assert isinstance(count, int)
