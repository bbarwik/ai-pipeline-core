"""Tests for Conversation.with_model()."""

from ai_pipeline_core.llm import Conversation, ModelOptions


class TestWithModel:
    """Test Conversation.with_model() method."""

    def test_with_model_changes_model(self):
        """with_model() returns a new Conversation with the specified model."""
        conv = Conversation(model="gemini-3-flash")
        new_conv = conv.with_model("gemini-3-pro")

        assert new_conv.model == "gemini-3-pro"
        assert conv.model == "gemini-3-flash"  # original unchanged

    def test_with_model_preserves_options(self):
        """with_model() preserves model_options from the original."""
        opts = ModelOptions(system_prompt="test", reasoning_effort="high")
        conv = Conversation(model="gemini-3-flash", model_options=opts)
        new_conv = conv.with_model("gpt-5.1")

        assert new_conv.model_options is opts
        assert new_conv.model_options.system_prompt == "test"

    def test_with_model_preserves_substitutor_setting(self):
        """with_model() preserves enable_substitutor flag."""
        conv = Conversation(model="gemini-3-flash", enable_substitutor=False)
        new_conv = conv.with_model("gpt-5.1")

        assert new_conv.enable_substitutor is False

    def test_with_model_validates_empty(self):
        """with_model() rejects empty model name via field validator."""
        import pytest

        conv = Conversation(model="gemini-3-flash")
        with pytest.raises(Exception):
            conv.with_model("")
