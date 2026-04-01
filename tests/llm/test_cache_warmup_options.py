"""Tests for ModelOptions cache warmup fields (cache_warmup_max_wait, cache_warmup_max_qps)."""

import pytest

from ai_pipeline_core.llm import ModelOptions


class TestCacheWarmupDefaults:
    """Verify default values and basic construction."""

    def test_cache_warmup_max_wait_default(self):
        options = ModelOptions()
        assert options.cache_warmup_max_wait is None

    def test_cache_warmup_max_qps_default(self):
        options = ModelOptions()
        assert options.cache_warmup_max_qps is None

    def test_cache_warmup_disabled(self):
        options = ModelOptions(cache_warmup_max_wait=None)
        assert options.cache_warmup_max_wait is None

    def test_cache_warmup_custom_values(self):
        options = ModelOptions(cache_warmup_max_wait=300.0, cache_warmup_max_qps=15)
        assert options.cache_warmup_max_wait == 300.0
        assert options.cache_warmup_max_qps == 15

    def test_cache_warmup_zero_wait(self):
        options = ModelOptions(cache_warmup_max_wait=0)
        assert options.cache_warmup_max_wait == 0.0

    def test_cache_warmup_frozen(self):
        options = ModelOptions(cache_warmup_max_wait=300.0)
        with pytest.raises(Exception):
            options.cache_warmup_max_wait = 500.0  # type: ignore[misc]


class TestCacheWarmupInKwargs:
    """Verify cache warmup fields are injected into metadata, not as top-level API params."""

    def test_max_wait_injected_into_metadata(self):
        options = ModelOptions(cache_warmup_max_wait=600.0)
        kwargs = options.to_openai_completion_kwargs()
        assert "cache_warmup_max_wait" not in kwargs
        assert kwargs["metadata"]["cache_warmup_max_wait"] == "600.0"

    def test_max_qps_injected_into_metadata(self):
        options = ModelOptions(cache_warmup_max_qps=15)
        kwargs = options.to_openai_completion_kwargs()
        assert "cache_warmup_max_qps" not in kwargs
        assert kwargs["metadata"]["cache_warmup_max_qps"] == "15"

    def test_both_fields_injected(self):
        options = ModelOptions(cache_warmup_max_wait=300.0, cache_warmup_max_qps=10)
        kwargs = options.to_openai_completion_kwargs()
        assert kwargs["metadata"]["cache_warmup_max_wait"] == "300.0"
        assert kwargs["metadata"]["cache_warmup_max_qps"] == "10"

    def test_disabled_not_in_metadata(self):
        options = ModelOptions(cache_warmup_max_wait=None, cache_warmup_max_qps=None)
        kwargs = options.to_openai_completion_kwargs()
        assert "metadata" not in kwargs

    def test_max_wait_none_max_qps_set(self):
        """QPS-only mode: no warmup waiting but throttle requests."""
        options = ModelOptions(cache_warmup_max_wait=None, cache_warmup_max_qps=15)
        kwargs = options.to_openai_completion_kwargs()
        assert "cache_warmup_max_wait" not in kwargs.get("metadata", {})
        assert kwargs["metadata"]["cache_warmup_max_qps"] == "15"

    def test_values_serialized_as_strings(self):
        """LiteLLM metadata flows through JSON — values must be strings."""
        options = ModelOptions(cache_warmup_max_wait=123.456, cache_warmup_max_qps=42)
        kwargs = options.to_openai_completion_kwargs()
        assert isinstance(kwargs["metadata"]["cache_warmup_max_wait"], str)
        assert isinstance(kwargs["metadata"]["cache_warmup_max_qps"], str)


class TestCacheWarmupWithExistingMetadata:
    """Verify warmup fields coexist with user-supplied metadata."""

    def test_warmup_merged_with_user_metadata(self):
        options = ModelOptions(
            metadata={"experiment": "v1", "user_id": "abc"},
            cache_warmup_max_wait=300.0,
            cache_warmup_max_qps=15,
        )
        kwargs = options.to_openai_completion_kwargs()
        assert kwargs["metadata"]["experiment"] == "v1"
        assert kwargs["metadata"]["user_id"] == "abc"
        assert kwargs["metadata"]["cache_warmup_max_wait"] == "300.0"
        assert kwargs["metadata"]["cache_warmup_max_qps"] == "15"

    def test_warmup_does_not_overwrite_user_metadata(self):
        """User metadata set first, warmup appended after."""
        options = ModelOptions(
            metadata={"existing_key": "existing_value"},
            cache_warmup_max_wait=600.0,
        )
        kwargs = options.to_openai_completion_kwargs()
        assert kwargs["metadata"]["existing_key"] == "existing_value"
        assert kwargs["metadata"]["cache_warmup_max_wait"] == "600.0"

    def test_user_metadata_none_warmup_creates_metadata(self):
        """When user metadata is None, warmup fields create the metadata dict."""
        options = ModelOptions(metadata=None, cache_warmup_max_wait=600.0)
        kwargs = options.to_openai_completion_kwargs()
        assert "metadata" in kwargs
        assert kwargs["metadata"]["cache_warmup_max_wait"] == "600.0"

    def test_user_metadata_empty_warmup_creates_metadata(self):
        """When user metadata is empty, warmup fields still get injected."""
        options = ModelOptions(metadata={}, cache_warmup_max_wait=600.0)
        kwargs = options.to_openai_completion_kwargs()
        assert kwargs["metadata"]["cache_warmup_max_wait"] == "600.0"


class TestCacheWarmupExcludedFromApiParams:
    """Verify warmup fields never appear as top-level API kwargs."""

    def test_not_in_top_level_kwargs(self):
        options = ModelOptions(cache_warmup_max_wait=600.0, cache_warmup_max_qps=15)
        kwargs = options.to_openai_completion_kwargs()
        assert "cache_warmup_max_wait" not in kwargs
        assert "cache_warmup_max_qps" not in kwargs

    def test_not_in_extra_body(self):
        options = ModelOptions(
            cache_warmup_max_wait=600.0,
            extra_body={"custom": "value"},
        )
        kwargs = options.to_openai_completion_kwargs()
        assert "cache_warmup_max_wait" not in kwargs["extra_body"]
        assert "cache_warmup_max_qps" not in kwargs["extra_body"]
