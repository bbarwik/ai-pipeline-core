"""Comprehensive tests for new ModelResponse features."""

import json

import pytest
from pydantic import BaseModel

from ai_pipeline_core.llm import ModelResponse, StructuredModelResponse
from tests.support.helpers import create_test_model_response


class TestModelResponseMetadata:
    """Test metadata tracking in ModelResponse."""

    def test_metadata_stored_in_response(self):
        """Test that metadata is stored in the response."""
        metadata = {"time_taken": 1.5, "first_token_time": 0.3}
        response = ModelResponse(
            chat_completion=create_test_model_response(
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
            ),
            model_options={"model": "test"},
            metadata=metadata,
        )

        laminar_metadata = response.get_laminar_metadata()
        assert laminar_metadata["time_taken"] == 1.5
        assert laminar_metadata["first_token_time"] == 0.3

    def test_model_options_stored(self):
        """Test that model options are stored and retrievable."""
        model_options = {
            "model": "gpt-5.1",
            "temperature": 0.7,
            "max_tokens": 1000,
        }
        response = ModelResponse(
            chat_completion=create_test_model_response(
                id="test",
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
            ),
            model_options=model_options,
            metadata={},
        )

        laminar_metadata = response.get_laminar_metadata()
        assert "model_options.model" in laminar_metadata
        assert laminar_metadata["model_options.model"] == "gpt-5.1"
        assert "model_options.temperature" in laminar_metadata
        assert laminar_metadata["model_options.temperature"] == "0.7"

    def test_metadata_excludes_messages(self):
        """Test that messages are excluded from metadata."""
        model_options = {
            "model": "gpt-5.1",
            "messages": [{"role": "user", "content": "long message"}],
            "temperature": 0.7,
        }
        response = ModelResponse(
            chat_completion=create_test_model_response(
                id="test",
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
            ),
            model_options=model_options,
            metadata={},
        )

        laminar_metadata = response.get_laminar_metadata()
        # messages should not be in metadata
        assert not any("messages" in key for key in laminar_metadata.keys())


class TestModelResponseReasoningContent:
    """Test reasoning_content property comprehensively."""

    def test_no_reasoning_content(self):
        """Test when there is no reasoning content."""
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
        assert response.content == "Simple response"

    def test_reasoning_with_think_tags(self):
        """Test reasoning content extraction from think tags."""
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
                        "content": "<think>My internal reasoning here</think> Final answer",
                    },
                    "finish_reason": "stop",
                }
            ],
        )
        assert response.reasoning_content == "<think>My internal reasoning here"
        assert response.content == "Final answer"

    def test_reasoning_with_multiline_think(self):
        """Test reasoning extraction with multiline think content."""
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
                        "content": """<think>
Step 1: Analyze the problem
Step 2: Consider options
Step 3: Choose best solution
</think>
Here is the final answer""",
                    },
                    "finish_reason": "stop",
                }
            ],
        )
        assert "<think>" in response.reasoning_content
        assert "Step 1" in response.reasoning_content
        assert "Step 3" in response.reasoning_content
        assert response.content == "Here is the final answer"

    def test_reasoning_with_empty_think_tags(self):
        """Test reasoning with empty think tags."""
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
                        "content": "<think></think> Content",
                    },
                    "finish_reason": "stop",
                }
            ],
        )
        assert response.reasoning_content == "<think>"
        assert response.content == "Content"

    def test_reasoning_with_only_think_tags(self):
        """Test when message is only think tags."""
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
                        "content": "<think>Only reasoning</think>",
                    },
                    "finish_reason": "stop",
                }
            ],
        )
        assert "<think>Only reasoning" in response.reasoning_content
        assert response.content == ""  # Empty after stripping

    def test_content_property_strips_think_tags(self):
        """Test that content property shows text after the LAST closing think tag."""
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
                        "content": "<think>Hidden</think>Visible<think>More hidden</think>Final",
                    },
                    "finish_reason": "stop",
                }
            ],
        )
        # Content should only show text after the LAST </think>
        assert "Hidden" not in response.content
        assert "Visible" not in response.content
        assert "More hidden" not in response.content
        assert response.content == "Final"


class TestModelResponseValidateOutput:
    """Test validate_output method comprehensively."""

    def test_validate_non_empty_content(self):
        """Test validation passes for non-empty content."""
        response = create_test_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Valid content"},
                    "finish_reason": "stop",
                }
            ],
        )
        response._model_options = {}  # type: ignore[reportPrivateUsage]
        # Should not raise
        response.validate_output()

    def test_validate_empty_content_raises(self):
        """Test validation raises for empty content."""
        response = create_test_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ],
        )
        response._model_options = {}  # type: ignore[reportPrivateUsage]

        with pytest.raises(ValueError, match="Empty response content"):
            response.validate_output()

    def test_validate_none_content_raises(self):
        """Test validation raises for None content."""
        response = create_test_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "stop",
                }
            ],
        )
        response._model_options = {}  # type: ignore[reportPrivateUsage]

        with pytest.raises(ValueError, match="Empty response content"):
            response.validate_output()

    def test_validate_structured_output_valid_json(self):
        """Test validation with valid structured JSON output."""

        class OutputModel(BaseModel):
            name: str
            value: int

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
                        "content": json.dumps({"name": "test", "value": 42}),
                    },
                    "finish_reason": "stop",
                }
            ],
        )
        response._model_options = {"response_format": OutputModel}  # type: ignore[reportPrivateUsage]

        # Should not raise
        response.validate_output()

    def test_validate_without_response_format(self):
        """Test validation without response_format just checks for empty content."""

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
                        "content": "Any content works",
                    },
                    "finish_reason": "stop",
                }
            ],
        )
        response._model_options = {}  # type: ignore[reportPrivateUsage]

        # Should not raise for non-empty content
        response.validate_output()

    def test_validate_output_with_response_format_type(self):
        """Test that response_format as type doesn't trigger validation."""

        class OutputModel(BaseModel):
            name: str
            value: int

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
                        "content": json.dumps({"name": "test", "wrong_field": "invalid"}),
                    },
                    "finish_reason": "stop",
                }
            ],
        )
        # response_format is a class/type, not instance, so isinstance check fails
        response._model_options = {"response_format": OutputModel}  # type: ignore[reportPrivateUsage]

        # Should not raise because isinstance(OutputModel, BaseModel) is False
        response.validate_output()


class TestStructuredModelResponseFromModelResponse:
    """Test StructuredModelResponse.from_model_response class method."""

    def test_from_model_response_conversion(self):
        """Test converting ModelResponse to StructuredModelResponse."""

        class TestModel(BaseModel):
            field: str

        model_response = create_test_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"field": "value"}),
                    },
                    "finish_reason": "stop",
                }
            ],
        )
        model_response._model_options = {"response_format": TestModel}  # type: ignore[reportPrivateUsage]

        structured = StructuredModelResponse.from_model_response(model_response)

        assert isinstance(structured, StructuredModelResponse)
        assert structured.id == "test"
        assert structured.parsed.field == "value"

    def test_from_model_response_preserves_metadata(self):
        """Test that conversion preserves all metadata."""

        class TestModel(BaseModel):
            value: int

        model_response = ModelResponse(
            chat_completion=create_test_model_response(
                id="test",
                object="chat.completion",
                created=1234567890,
                model="test",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"value": 42}),
                        },
                        "finish_reason": "stop",
                    }
                ],
            ),
            model_options={"response_format": TestModel, "temperature": 0.7},
            metadata={"time_taken": 1.5, "first_token_time": 0.3},
        )

        structured = StructuredModelResponse.from_model_response(model_response)

        # Check metadata is preserved
        assert structured.usage.total_tokens == 15  # type: ignore
        laminar_metadata = structured.get_laminar_metadata()
        assert laminar_metadata["time_taken"] == 1.5
        assert laminar_metadata["first_token_time"] == 0.3

    def test_from_model_response_lazy_parsing(self):
        """Test that parsing is lazy (only happens when accessed)."""

        class TestModel(BaseModel):
            field: str

        model_response = create_test_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"field": "test"}),
                    },
                    "finish_reason": "stop",
                }
            ],
        )
        model_response._model_options = {"response_format": TestModel}  # type: ignore[reportPrivateUsage]

        structured = StructuredModelResponse.from_model_response(model_response)

        # Check that _parsed_value doesn't exist yet
        assert not hasattr(structured, "_parsed_value")

        # Access parsed property triggers parsing
        parsed = structured.parsed
        assert parsed.field == "test"

        # Now _parsed_value should exist
        assert hasattr(structured, "_parsed_value")

    def test_from_model_response_caches_parsed(self):
        """Test that parsed value is cached after first access."""

        class TestModel(BaseModel):
            field: str

        model_response = create_test_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"field": "cached"}),
                    },
                    "finish_reason": "stop",
                }
            ],
        )
        model_response._model_options = {"response_format": TestModel}  # type: ignore[reportPrivateUsage]

        structured = StructuredModelResponse.from_model_response(model_response)

        # Access twice
        parsed1 = structured.parsed
        parsed2 = structured.parsed

        # Should be the same object (cached)
        assert parsed1 is parsed2

    def test_from_model_response_with_complex_model(self):
        """Test with complex nested Pydantic model."""

        class NestedModel(BaseModel):
            inner: str

        class ComplexModel(BaseModel):
            name: str
            count: int
            nested: NestedModel
            items: list[str]

        model_response = create_test_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "name": "test",
                            "count": 5,
                            "nested": {"inner": "value"},
                            "items": ["a", "b", "c"],
                        }),
                    },
                    "finish_reason": "stop",
                }
            ],
        )
        model_response._model_options = {"response_format": ComplexModel}  # type: ignore[reportPrivateUsage]

        structured = StructuredModelResponse.from_model_response(model_response)
        parsed = structured.parsed

        assert parsed.name == "test"
        assert parsed.count == 5
        assert parsed.nested.inner == "value"
        assert parsed.items == ["a", "b", "c"]


class TestModelResponseLaminarMetadata:
    """Test comprehensive Laminar metadata extraction."""

    def test_metadata_includes_all_fields(self):
        """Test that get_laminar_metadata includes all expected fields."""
        response = ModelResponse(
            chat_completion=create_test_model_response(
                id="test-id",
                object="chat.completion",
                created=1234567890,
                model="gpt-5.1",
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "test response"},
                        "finish_reason": "stop",
                    }
                ],
            ),
            model_options={"model": "gpt-5.1", "temperature": 0.7},
            metadata={"time_taken": 2.5, "first_token_time": 0.5},
        )

        metadata = response.get_laminar_metadata()

        # Basic fields
        assert "gen_ai.response.id" in metadata
        assert "gen_ai.system" in metadata

        # Usage fields
        assert "gen_ai.usage.prompt_tokens" in metadata
        assert "gen_ai.usage.completion_tokens" in metadata
        assert "gen_ai.usage.total_tokens" in metadata

        # Timing metadata
        assert "time_taken" in metadata
        assert "first_token_time" in metadata

        # Model options
        assert "model_options.model" in metadata
        assert "model_options.temperature" in metadata

    def test_metadata_json_serialization(self):
        """Test that metadata is JSON serializable."""
        response = ModelResponse(
            chat_completion=create_test_model_response(
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
            ),
            model_options={"model": "test"},
            metadata={"time_taken": 1.5},
        )

        metadata = response.get_laminar_metadata()

        # Should be JSON serializable
        json_str = json.dumps(metadata, default=str)
        assert json_str is not None
        assert len(json_str) > 0
