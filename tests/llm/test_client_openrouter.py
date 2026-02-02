"""Tests for OpenRouter model name conversion."""

# pyright: reportPrivateUsage=false

import pytest

from ai_pipeline_core.llm.client import _model_name_to_openrouter_model


class TestModelNameToOpenRouterModel:
    """Test _model_name_to_openrouter_model conversion."""

    def test_gemini_flash_search(self):
        """Test gemini-3-flash-search maps to online variant."""
        assert _model_name_to_openrouter_model("gemini-3-flash-search") == "google/gemini-3-flash:online"

    def test_sonar_pro_search(self):
        """Test sonar-pro-search maps to perplexity."""
        assert _model_name_to_openrouter_model("sonar-pro-search") == "perplexity/sonar-pro-search"

    def test_gemini_model(self):
        """Test gemini models get google/ prefix."""
        assert _model_name_to_openrouter_model("gemini-3-pro") == "google/gemini-3-pro"
        assert _model_name_to_openrouter_model("gemini-3-flash") == "google/gemini-3-flash"

    def test_gpt_model(self):
        """Test gpt models get openai/ prefix."""
        assert _model_name_to_openrouter_model("gpt-5.1") == "openai/gpt-5.1"
        assert _model_name_to_openrouter_model("gpt-5-mini") == "openai/gpt-5-mini"

    def test_grok_model(self):
        """Test grok models get x-ai/ prefix."""
        assert _model_name_to_openrouter_model("grok-4.1-fast") == "x-ai/grok-4.1-fast"

    def test_claude_model(self):
        """Test claude models get anthropic/ prefix."""
        assert _model_name_to_openrouter_model("claude-opus-4-5") == "anthropic/claude-opus-4-5"

    def test_qwen_model(self):
        """Test qwen3 models get qwen/ prefix."""
        assert _model_name_to_openrouter_model("qwen3-32b") == "qwen/qwen3-32b"

    def test_deepseek_model(self):
        """Test deepseek models get deepseek/ prefix."""
        assert _model_name_to_openrouter_model("deepseek-r1") == "deepseek/deepseek-r1"

    def test_glm_model(self):
        """Test glm models get z-ai/ prefix."""
        assert _model_name_to_openrouter_model("glm-4") == "z-ai/glm-4"

    def test_kimi_model(self):
        """Test kimi models get moonshotai/ prefix."""
        assert _model_name_to_openrouter_model("kimi-k2") == "moonshotai/kimi-k2"

    def test_unknown_model_passthrough(self):
        """Test unknown model names pass through unchanged."""
        assert _model_name_to_openrouter_model("custom-model") == "custom-model"

    @pytest.mark.parametrize(
        "model",
        [
            "openrouter/some-model",
            "some-provider/model-name",
        ],
    )
    def test_already_prefixed_passthrough(self, model: str):
        """Test already-prefixed models pass through."""
        assert _model_name_to_openrouter_model(model) == model
