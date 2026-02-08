"""Tests for auto-disabling URLSubstitutor on search models."""

from ai_pipeline_core.llm.conversation import Conversation


class TestSubstitutorSearchModels:
    """Verify substitutor is auto-disabled for search models."""

    def test_search_model_auto_disables_substitutor(self):
        """Search models should have substitutor disabled by default."""
        conv = Conversation(model="sonar-pro-search")
        assert conv.enable_substitutor is False
        assert conv.substitutor is None

    def test_search_model_grok_auto_disables(self):
        """Grok search model should have substitutor disabled."""
        conv = Conversation(model="grok-4.1-fast-search")
        assert conv.enable_substitutor is False
        assert conv.substitutor is None

    def test_search_model_gemini_auto_disables(self):
        """Gemini search model should have substitutor disabled."""
        conv = Conversation(model="gemini-3-flash-search")
        assert conv.enable_substitutor is False
        assert conv.substitutor is None

    def test_non_search_model_keeps_substitutor(self):
        """Non-search models should keep substitutor enabled."""
        conv = Conversation(model="gemini-3-flash")
        assert conv.enable_substitutor is True
        assert conv.substitutor is not None

    def test_non_search_model_grok(self):
        """Non-search grok model keeps substitutor."""
        conv = Conversation(model="grok-4.1-fast")
        assert conv.enable_substitutor is True
        assert conv.substitutor is not None

    def test_explicit_enable_overrides_auto_disable(self):
        """Explicitly enabling substitutor on search model should be respected."""
        conv = Conversation(model="sonar-pro-search", enable_substitutor=True)
        assert conv.enable_substitutor is True
        assert conv.substitutor is not None

    def test_explicit_disable_on_non_search(self):
        """Explicitly disabling substitutor on non-search model should work."""
        conv = Conversation(model="gemini-3-flash", enable_substitutor=False)
        assert conv.enable_substitutor is False
        assert conv.substitutor is None

    def test_model_name_not_ending_with_search(self):
        """Model name containing 'search' but not ending with it keeps substitutor."""
        conv = Conversation(model="search-enhanced-gpt")
        assert conv.enable_substitutor is True
        assert conv.substitutor is not None
