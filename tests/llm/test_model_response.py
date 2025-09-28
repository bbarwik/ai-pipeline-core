"""Tests for ModelResponse and StructuredModelResponse."""

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel

from ai_pipeline_core.llm import ModelResponse, StructuredModelResponse


class TestModelResponse:
    """Test ModelResponse class."""

    def test_construct_from_kwargs(self):
        """Test constructing ModelResponse from kwargs."""
        response = ModelResponse(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Test response"},
                    "finish_reason": "stop",
                }
            ],
        )

        assert response.id == "test-id"
        assert response.model == "test-model"
        assert response.headers == {}  # Default empty headers
        assert response.content == "Test response"

    def test_construct_from_chat_completion(self):
        """Test constructing ModelResponse from ChatCompletion."""
        # Create a ChatCompletion object
        completion = ChatCompletion(
            id="chat-id",
            object="chat.completion",
            created=1234567890,
            model="gpt-4",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Hello from GPT-4"),
                    finish_reason="stop",
                )
            ],
        )

        response = ModelResponse(completion)
        assert response.id == "chat-id"
        assert response.model == "gpt-4"
        assert response.content == "Hello from GPT-4"
        assert response.headers == {}

    def test_content_property(self):
        """Test content property accessor."""
        response = ModelResponse(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Content here"},
                    "finish_reason": "stop",
                }
            ],
        )

        assert response.content == "Content here"

    def test_content_strips_think_tags(self):
        """Test that content property removes think tags."""
        response = ModelResponse(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test",
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

        assert response.content == "Visible content"

    def test_content_strips_think_tags_multiline(self):
        """Test stripping think tags with multiline content."""
        response = ModelResponse(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "<think>\nMultiline\nthinking\n</think>\nActual response",
                    },
                    "finish_reason": "stop",
                }
            ],
        )

        assert response.content == "Actual response"

    def test_set_headers(self):
        """Test setting response headers."""
        response = ModelResponse(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "test"},
                    "finish_reason": "stop",
                }
            ],
        )

        headers = {"x-litellm-call-id": "call-123", "x-litellm-response-cost": "0.002"}
        response.set_headers(headers)
        assert response.headers == headers

    def test_get_laminar_metadata_basic(self):
        """Test basic Laminar metadata extraction."""
        response = ModelResponse(
            id="resp-id",
            object="chat.completion",
            created=1234567890,
            model="gpt-4",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "test"},
                    "finish_reason": "stop",
                }
            ],
        )

        metadata = response.get_laminar_metadata()

        assert metadata["gen_ai.response.id"] == "resp-id"
        assert metadata["gen_ai.response.model"] == "gpt-4"
        assert metadata["get_ai.system"] == "litellm"

    def test_get_laminar_metadata_with_headers(self):
        """Test Laminar metadata with LiteLLM headers."""
        response = ModelResponse(
            id="resp-id",
            object="chat.completion",
            created=1234567890,
            model="gpt-4",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "test"},
                    "finish_reason": "stop",
                }
            ],
        )

        response.set_headers({
            "x-litellm-call-id": "litellm-123",
            "x-litellm-response-cost": "0.003",
            "x-litellm-model-id": "model-456",
            "x-litellm-custom": "value",
        })

        metadata = response.get_laminar_metadata()

        # Should use litellm ID when available
        assert metadata["gen_ai.response.id"] == "litellm-123"
        assert metadata["litellm.call-id"] == "litellm-123"
        assert metadata["litellm.response-cost"] == "0.003"
        assert metadata["litellm.model-id"] == "model-456"
        assert metadata["litellm.custom"] == "value"

        # Cost metadata
        assert metadata["gen_ai.usage.output_cost"] == 0.003
        assert metadata["gen_ai.usage.cost"] == 0.003
        assert metadata["get_ai.cost"] == 0.003

    def test_get_laminar_metadata_with_usage(self):
        """Test Laminar metadata with usage information."""
        response = ModelResponse(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "test"},
                    "finish_reason": "stop",
                }
            ],
        )

        metadata = response.get_laminar_metadata()

        assert metadata["gen_ai.usage.prompt_tokens"] == 100
        assert metadata["gen_ai.usage.completion_tokens"] == 50
        assert metadata["gen_ai.usage.total_tokens"] == 150

    def test_get_laminar_metadata_with_reasoning_tokens(self):
        """Test metadata with reasoning tokens (for o1 models)."""
        response = ModelResponse(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="o1-preview",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "completion_tokens_details": {"reasoning_tokens": 30},
            },
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "test"},
                    "finish_reason": "stop",
                }
            ],
        )

        metadata = response.get_laminar_metadata()
        assert metadata["gen_ai.usage.reasoning_tokens"] == 30

    def test_get_laminar_metadata_with_cached_tokens(self):
        """Test metadata with cached tokens."""
        response = ModelResponse(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="gpt-4",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 80},
            },
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "test"},
                    "finish_reason": "stop",
                }
            ],
        )

        metadata = response.get_laminar_metadata()
        assert metadata["gen_ai.usage.cached_tokens"] == 80


class TestStructuredModelResponse:
    """Test StructuredModelResponse class."""

    class ExampleModel(BaseModel):
        """Example Pydantic model for testing."""

        field1: str
        field2: int

    def test_construct_with_parsed_value(self):
        """Test constructing with explicit parsed value."""
        parsed = self.ExampleModel(field1="test", field2=42)

        response = StructuredModelResponse(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "JSON output"},
                    "finish_reason": "stop",
                }
            ],
            parsed_value=parsed,
        )

        assert response.parsed == parsed
        assert response.parsed.field1 == "test"
        assert response.parsed.field2 == 42

    def test_parsed_property_error_when_none(self):
        """Test that accessing parsed raises when no value available."""
        response = StructuredModelResponse[self.ExampleModel](
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "test"},
                    "finish_reason": "stop",
                }
            ],
            parsed_value=None,
        )

        with pytest.raises(ValueError) as exc_info:
            _ = response.parsed
        assert "No parsed content available" in str(exc_info.value)

    def test_inherits_from_model_response(self):
        """Test that StructuredModelResponse inherits ModelResponse functionality."""
        parsed = self.ExampleModel(field1="value", field2=123)

        response = StructuredModelResponse(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="gpt-4",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Structured output"},
                    "finish_reason": "stop",
                }
            ],
            parsed_value=parsed,
        )

        # Should have ModelResponse properties
        assert response.content == "Structured output"
        assert response.model == "gpt-4"

        # Should have parsed property
        assert response.parsed == parsed

        # Should support headers
        response.set_headers({"x-test": "value"})
        assert response.headers["x-test"] == "value"

        # Should support metadata extraction
        metadata = response.get_laminar_metadata()
        assert metadata["gen_ai.response.model"] == "gpt-4"
