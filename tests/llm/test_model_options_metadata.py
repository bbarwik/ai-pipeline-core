"""Tests for ModelOptions metadata field."""

from ai_pipeline_core.llm import ModelOptions


class TestModelOptionsMetadata:
    """Test ModelOptions metadata field."""

    def test_metadata_none_by_default(self):
        """Test that metadata is None by default."""
        options = ModelOptions()
        assert options.metadata is None

    def test_metadata_set_in_constructor(self):
        """Test setting metadata in constructor."""
        metadata = {"experiment": "v1", "version": "2.0"}
        options = ModelOptions(metadata=metadata)
        assert options.metadata == metadata

    def test_metadata_in_completion_kwargs(self):
        """Test that metadata is included in completion kwargs."""
        metadata = {"experiment": "test", "feature": "search"}
        options = ModelOptions(metadata=metadata)
        kwargs = options.to_openai_completion_kwargs()
        assert "metadata" in kwargs
        assert kwargs["metadata"] == metadata

    def test_metadata_not_in_kwargs_when_none(self):
        """Test that metadata is not in kwargs when None."""
        options = ModelOptions(metadata=None)
        kwargs = options.to_openai_completion_kwargs()
        assert "metadata" not in kwargs

    def test_metadata_with_single_tag(self):
        """Test metadata with single tag."""
        metadata = {"version": "1.0"}
        options = ModelOptions(metadata=metadata)
        kwargs = options.to_openai_completion_kwargs()
        assert kwargs["metadata"] == {"version": "1.0"}

    def test_metadata_with_multiple_tags(self):
        """Test metadata with multiple tags."""
        metadata = {
            "experiment": "exp-123",
            "version": "2.0",
            "feature": "caching",
            "environment": "production",
        }
        options = ModelOptions(metadata=metadata)
        kwargs = options.to_openai_completion_kwargs()
        assert kwargs["metadata"] == metadata

    def test_metadata_with_empty_dict(self):
        """Test metadata with empty dictionary."""
        options = ModelOptions(metadata={})
        kwargs = options.to_openai_completion_kwargs()
        # Empty dict is falsy, so not included
        assert "metadata" not in kwargs

    def test_metadata_values_are_strings(self):
        """Test that metadata values are strings."""
        metadata = {"count": "42", "enabled": "true", "name": "test"}
        options = ModelOptions(metadata=metadata)
        assert options.metadata == metadata

    def test_metadata_with_special_characters(self):
        """Test metadata with special characters in values."""
        metadata = {
            "experiment": "test-123",
            "feature": "cache_optimization",
            "version": "v2.0.1",
        }
        options = ModelOptions(metadata=metadata)
        kwargs = options.to_openai_completion_kwargs()
        assert kwargs["metadata"] == metadata

    def test_metadata_combined_with_other_options(self):
        """Test metadata combined with other options."""
        metadata = {"experiment": "v1"}
        options = ModelOptions(
            metadata=metadata,
            temperature=0.7,
            max_completion_tokens=1000,
            user="user_123",
        )
        kwargs = options.to_openai_completion_kwargs()
        assert kwargs["metadata"] == metadata
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_completion_tokens"] == 1000
        assert kwargs["user"] == "user_123"

    def test_metadata_independent_of_extra_body(self):
        """Test that metadata is separate from extra_body."""
        metadata = {"experiment": "test"}
        extra_body = {"custom_param": "value"}
        options = ModelOptions(metadata=metadata, extra_body=extra_body, usage_tracking=True)
        kwargs = options.to_openai_completion_kwargs()
        assert kwargs["metadata"] == metadata
        # extra_body gets merged with usage tracking
        assert kwargs["extra_body"]["custom_param"] == "value"
        assert kwargs["extra_body"]["usage"] == {"include": True}
        assert "metadata" not in kwargs["extra_body"]

    def test_metadata_with_usage_tracking(self):
        """Test metadata with usage tracking enabled."""
        metadata = {"experiment": "test"}
        options = ModelOptions(metadata=metadata, usage_tracking=True)
        kwargs = options.to_openai_completion_kwargs()
        assert kwargs["metadata"] == metadata
        assert kwargs["extra_body"]["usage"] == {"include": True}

    def test_metadata_preserved_across_conversions(self):
        """Test that metadata is preserved when converting to kwargs."""
        original_metadata = {
            "exp": "123",
            "ver": "2.0",
            "feat": "test",
        }
        options = ModelOptions(metadata=original_metadata)
        kwargs = options.to_openai_completion_kwargs()
        # Should be the exact same dict
        assert kwargs["metadata"] is not original_metadata  # New dict in kwargs
        assert kwargs["metadata"] == original_metadata

    def test_metadata_realistic_use_case(self):
        """Test realistic metadata use case."""
        metadata = {
            "experiment_id": "exp-2024-01",
            "model_version": "v3.5",
            "feature_flag": "enable_cache",
            "environment": "staging",
            "user_cohort": "beta",
        }
        options = ModelOptions(
            metadata=metadata,
            temperature=0.8,
            max_completion_tokens=2000,
            cache_ttl="10m",
        )
        kwargs = options.to_openai_completion_kwargs()
        assert kwargs["metadata"] == metadata
        assert kwargs["temperature"] == 0.8
        assert kwargs["max_completion_tokens"] == 2000

    def test_metadata_keys_are_strings(self):
        """Test that metadata keys are strings."""
        metadata = {"key1": "value1", "key2": "value2"}
        options = ModelOptions(metadata=metadata)
        assert options.metadata is not None
        for key in options.metadata.keys():
            assert isinstance(key, str)

    def test_metadata_values_are_strings_in_type(self):
        """Test that metadata values are typed as strings."""
        metadata = {"version": "1.0", "name": "test"}
        options = ModelOptions(metadata=metadata)
        assert options.metadata is not None
        for value in options.metadata.values():
            assert isinstance(value, str)
