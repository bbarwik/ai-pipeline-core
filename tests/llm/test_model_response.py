"""Tests for ModelResponse and StructuredModelResponse."""

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel

from ai_pipeline_core.llm import Citation, ModelResponse
from tests.support.helpers import create_test_model_response, create_test_structured_model_response


class TestModelResponse:
    """Test ModelResponse class."""

    def test_construct_from_kwargs(self):
        """Test constructing ModelResponse from kwargs."""
        response = create_test_model_response(
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
        assert response.content == "Test response"

    def test_construct_from_chat_completion(self):
        """Test constructing ModelResponse from ChatCompletion."""
        # Create a ChatCompletion object
        completion = ChatCompletion(
            id="chat-id",
            object="chat.completion",
            created=1234567890,
            model="gpt-5.1",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Hello from GPT-4"),
                    finish_reason="stop",
                )
            ],
        )

        response = ModelResponse(
            chat_completion=completion,
            model_options={},
            metadata={},
        )
        assert response.id == "chat-id"
        assert response.model == "gpt-5.1"
        assert response.content == "Hello from GPT-4"

    def test_content_property(self):
        """Test content property accessor."""
        response = create_test_model_response(
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
        response = create_test_model_response(
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
        response = create_test_model_response(
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

    def test_get_laminar_metadata_basic(self):
        """Test basic Laminar metadata extraction."""
        response = create_test_model_response(
            id="resp-id",
            object="chat.completion",
            created=1234567890,
            model="gpt-5.1",
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
        assert metadata["gen_ai.system"] == "litellm"

    def test_get_laminar_metadata_with_usage(self):
        """Test Laminar metadata with usage information."""
        response = create_test_model_response(
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
        response = create_test_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="gpt-5.1",
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
        response = create_test_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="gpt-5.1",
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

    def test_reasoning_content_empty(self):
        """Test reasoning_content when there is no reasoning."""
        response = create_test_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Simple response"},
                    "finish_reason": "stop",
                }
            ],
        )
        assert response.reasoning_content == ""

    def test_reasoning_content_with_think_tags(self):
        """Test reasoning_content extraction from think tags."""
        response = create_test_model_response(
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
        assert response.reasoning_content == "<think>Internal reasoning"
        assert response.content == "Visible content"


class TestCitations:
    """Test citations property."""

    def test_citations_empty_when_no_annotations(self):
        """Test citations returns empty list when no annotations present."""
        response = create_test_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="gpt-5.1",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "No citations here"},
                    "finish_reason": "stop",
                }
            ],
        )
        assert response.citations == []

    def test_citations_from_annotations(self):
        """Test citations are extracted from url_citation annotations."""
        response = create_test_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="sonar-pro-search",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Some search result",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "end_index": 0,
                                    "start_index": 0,
                                    "title": "Example Page",
                                    "url": "https://example.com",
                                },
                            },
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "end_index": 0,
                                    "start_index": 0,
                                    "title": "Another Page",
                                    "url": "https://another.com",
                                },
                            },
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
        )
        citations = response.citations
        assert len(citations) == 2
        assert citations[0] == Citation(title="Example Page", url="https://example.com")
        assert citations[1] == Citation(title="Another Page", url="https://another.com")

    def test_citations_filters_unsupported_annotation_types(self):
        """Test that non-url_citation annotations are filtered out in __init__."""
        response = create_test_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="grok-4.1-fast-search",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Result",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "end_index": 0,
                                    "start_index": 0,
                                    "title": "Valid",
                                    "url": "https://valid.com",
                                },
                            },
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
        )
        assert len(response.citations) == 1
        assert response.citations[0].url == "https://valid.com"


class TestStructuredModelResponse:
    """Test StructuredModelResponse class."""

    class ExampleModel(BaseModel):
        """Example Pydantic model for testing."""

        field1: str
        field2: int

    def test_parsed_property(self):
        """Test that parsed property works with lazy parsing."""
        import json

        response = create_test_structured_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"field1": "test", "field2": 42}),
                    },
                    "finish_reason": "stop",
                }
            ],
        )
        # Set response_format in model_options
        response._model_options["response_format"] = self.ExampleModel  # type: ignore[reportPrivateUsage]

        parsed = response.parsed
        assert parsed.field1 == "test"
        assert parsed.field2 == 42

    def test_inherits_from_model_response(self):
        """Test that StructuredModelResponse inherits ModelResponse functionality."""
        import json

        response = create_test_structured_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="gpt-5.1",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"field1": "value", "field2": 123}),
                    },
                    "finish_reason": "stop",
                }
            ],
        )
        # Set response_format in model_options
        response._model_options["response_format"] = self.ExampleModel  # type: ignore[reportPrivateUsage]

        # Should have ModelResponse properties
        assert response.model == "gpt-5.1"

        # Should have parsed property
        parsed = response.parsed
        assert parsed.field1 == "value"
        assert parsed.field2 == 123

        # Should support metadata extraction
        metadata = response.get_laminar_metadata()
        assert metadata["gen_ai.response.id"] == "test"
