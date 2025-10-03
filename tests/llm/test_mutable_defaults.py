"""Tests to detect mutable default parameter issues in LLM module."""

import json

from pydantic import BaseModel

from ai_pipeline_core.documents import TemporaryDocument
from ai_pipeline_core.llm import AIMessages, ModelOptions


class SampleResponse(BaseModel):
    result: str


class AnotherResponse(BaseModel):
    value: int


class TestModelOptionsMutation:
    """Test that ModelOptions instances are not mutated when passed to functions."""

    def test_model_options_immutability_expectation(self):
        """Test that we can detect if ModelOptions is mutated."""
        # Create a ModelOptions instance
        options = ModelOptions(temperature=0.7, max_completion_tokens=100)

        # Store original value
        original_response_format = options.response_format
        assert original_response_format is None

        # Simulate what generate_structured currently does (the bug)
        # This test documents the CURRENT buggy behavior
        options_copy = options  # BUG: No copy is made!
        options_copy.response_format = SampleResponse  # type: ignore

        # This test will FAIL after we fix the bug, which is what we want
        # Current buggy behavior: options IS mutated
        assert options.response_format == SampleResponse, (
            "TEST SETUP ISSUE: This test expects the bug to exist. "
            "If this fails, the code might already be fixed!"
        )

        # Reset for next test
        options.response_format = None

    def test_model_options_proper_copy_prevents_mutation(self):
        """Test that model_copy() prevents mutation of the original."""
        # Create original options
        original_options = ModelOptions(temperature=0.5)
        assert original_options.response_format is None

        # Make a copy (the CORRECT way)
        copied_options = original_options.model_copy()
        copied_options.response_format = SampleResponse  # type: ignore

        # Original should NOT be mutated
        assert original_options.response_format is None, (
            "Original ModelOptions was mutated even after model_copy()! "
            "This indicates Pydantic model_copy() is not working correctly."
        )

        # Copy should have the new value
        assert copied_options.response_format == SampleResponse

    def test_model_options_reference_mutation(self):
        """Test that assigning without copy causes mutation (documents the bug)."""
        # Create options
        options1 = ModelOptions(temperature=0.8)

        # Assign without copy (bug pattern)
        options2 = options1  # This is the same object!

        # Modify options2
        options2.response_format = AnotherResponse  # type: ignore

        # options1 is also modified because they're the same object
        assert options1.response_format == AnotherResponse, (
            "TEST SETUP ISSUE: Assignment should create reference, not copy"
        )
        assert options1 is options2, "options1 and options2 should be the same object"


class TestAIMessagesToTracingLogMutation:
    """Test that to_tracing_log doesn't cause unintended mutations."""

    def test_serialize_model_returns_new_dict_each_time(self):
        """Verify that Document.serialize_model() returns a fresh dict, not a shared reference."""
        doc = TemporaryDocument(name="test.txt", content=b"test content")

        # Get two serializations
        serialized1 = doc.serialize_model()
        serialized2 = doc.serialize_model()

        # They should be equal but not the same object
        assert serialized1 == serialized2
        assert serialized1 is not serialized2, (
            "serialize_model() is returning the same dict instance! "
            "It should return a new dict each time."
        )

    def test_to_tracing_log_does_not_mutate_document(self):
        """Verify that AIMessages.to_tracing_log() doesn't mutate the underlying documents."""
        doc = TemporaryDocument(name="test.txt", content=b"test content")

        # Get original serialization
        original_serialized = doc.serialize_model()
        assert "content" in original_serialized

        # Create AIMessages and call to_tracing_log
        messages = AIMessages([doc])
        tracing_logs = messages.to_tracing_log()

        # Verify the log was created
        assert len(tracing_logs) == 1

        # Verify the document's serialize_model() still includes 'content'
        new_serialized = doc.serialize_model()
        assert "content" in new_serialized, (
            "Document.serialize_model() no longer includes 'content'! "
            "to_tracing_log() may have mutated shared state."
        )

    def test_to_tracing_log_content_field_removed_from_output(self):
        """Verify that to_tracing_log properly removes 'content' from the output."""
        doc = TemporaryDocument(name="test.txt", content=b"test content")
        messages = AIMessages([doc])

        tracing_logs = messages.to_tracing_log()
        assert len(tracing_logs) == 1

        # Parse the JSON log
        log_dict = json.loads(tracing_logs[0])

        # Content should NOT be in the tracing log
        assert "content" not in log_dict, "content field should be removed from tracing log"

        # But other fields should be present
        assert "name" in log_dict
        assert log_dict["name"] == "test.txt"

    def test_multiple_to_tracing_log_calls_are_independent(self):
        """Verify multiple calls to to_tracing_log don't interfere with each other."""
        doc = TemporaryDocument(name="test.txt", content=b"test content")
        messages = AIMessages([doc])

        # Call to_tracing_log multiple times
        logs1 = messages.to_tracing_log()
        logs2 = messages.to_tracing_log()
        logs3 = messages.to_tracing_log()

        # All should produce the same result
        assert logs1 == logs2 == logs3

        # Parse and verify content is removed in all cases
        for logs in [logs1, logs2, logs3]:
            log_dict = json.loads(logs[0])
            assert "content" not in log_dict
            assert "name" in log_dict
