"""Tests for ModelOptions."""

from pydantic import BaseModel

from ai_pipeline_core.llm import ModelOptions


class TestModelOptions:
    """Test ModelOptions configuration."""

    def test_default_values(self):
        """Test default ModelOptions values."""
        options = ModelOptions()
        assert options.system_prompt is None
        assert options.search_context_size is None
        assert options.reasoning_effort is None
        assert options.retries == 3
        assert options.retry_delay_seconds == 10
        assert options.timeout == 600
        assert options.cache_ttl == "5m"
        assert options.max_completion_tokens is None
        assert options.response_format is None
        assert options.verbosity is None
        assert options.usage_tracking is True

    def test_to_openai_kwargs_defaults(self):
        """Test conversion to OpenAI kwargs with defaults."""
        options = ModelOptions()
        kwargs = options.to_openai_completion_kwargs()

        assert kwargs["timeout"] == 600
        assert "extra_body" in kwargs
        assert kwargs["extra_body"] == {"usage": {"include": True}}

        # These should not be in kwargs when None
        assert "max_completion_tokens" not in kwargs
        assert "response_format" not in kwargs
        assert "verbosity" not in kwargs

    def test_max_completion_tokens(self):
        """Test max_completion_tokens pass-through."""
        options = ModelOptions(max_completion_tokens=10000)
        kwargs = options.to_openai_completion_kwargs()
        assert kwargs["max_completion_tokens"] == 10000

    def test_search_context_size(self):
        """Test search_context_size values."""
        for size in ["low", "medium", "high"]:
            options = ModelOptions(search_context_size=size)  # type: ignore
            kwargs = options.to_openai_completion_kwargs()
            assert kwargs["web_search_options"]["search_context_size"] == size

    def test_response_format_with_pydantic_model(self):
        """Test response_format with Pydantic model."""

        class TestModel(BaseModel):
            field: str

        options = ModelOptions(response_format=TestModel)
        kwargs = options.to_openai_completion_kwargs()
        assert kwargs["response_format"] == TestModel

    def test_all_options_combined(self):
        """Test all options combined."""

        class ResponseModel(BaseModel):
            result: str

        options = ModelOptions(
            system_prompt="You are a helpful assistant",
            search_context_size="high",
            reasoning_effort="medium",
            retries=5,
            retry_delay_seconds=15,
            timeout=600,
            max_completion_tokens=20000,
            response_format=ResponseModel,
        )

        # Test with non-grok model
        kwargs = options.to_openai_completion_kwargs()

        assert kwargs["timeout"] == 600
        assert kwargs["web_search_options"]["search_context_size"] == "high"
        assert kwargs["reasoning_effort"] == "medium"
        assert kwargs["max_completion_tokens"] == 20000
        assert kwargs["response_format"] == ResponseModel
        assert "extra_body" in kwargs

        # System prompt is not in kwargs (handled separately)
        assert "system_prompt" not in kwargs
        # Retry settings are not in OpenAI kwargs
        assert "retries" not in kwargs
        assert "retry_delay_seconds" not in kwargs

    def test_reasoning_effort_always_included_when_set(self):
        """Test that reasoning_effort is included when set, regardless of model check."""
        options = ModelOptions(reasoning_effort="low")

        # Without model parameter
        kwargs = options.to_openai_completion_kwargs()
        assert kwargs["reasoning_effort"] == "low"

        # With non-grok model
        kwargs = options.to_openai_completion_kwargs()
        assert kwargs["reasoning_effort"] == "low"

    def test_immutability(self):
        """Test that ModelOptions is mutable (default Pydantic behavior)."""
        options = ModelOptions()
        options.timeout = 500
        assert options.timeout == 500

        options.response_format = None
        assert options.response_format is None

    def test_verbosity_option(self):
        """Test verbosity option."""
        for level in ["low", "medium", "high"]:
            options = ModelOptions(verbosity=level)  # type: ignore
            kwargs = options.to_openai_completion_kwargs()
            assert kwargs["verbosity"] == level

    def test_usage_tracking_disabled(self):
        """Test usage tracking can be disabled."""
        options = ModelOptions(usage_tracking=False)
        kwargs = options.to_openai_completion_kwargs()
        assert "usage" not in kwargs["extra_body"]

    def test_usage_tracking_enabled(self):
        """Test usage tracking is enabled by default."""
        options = ModelOptions(usage_tracking=True)
        kwargs = options.to_openai_completion_kwargs()
        assert kwargs["extra_body"]["usage"] == {"include": True}
