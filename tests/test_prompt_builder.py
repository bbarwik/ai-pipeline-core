"""Tests for prompt_builder module."""

# pyright: reportPrivateUsage=false

import pytest

from ai_pipeline_core.documents import TemporaryDocument
from ai_pipeline_core.llm import AIMessages
from ai_pipeline_core.prompt_builder import EnvironmentVariable, PromptBuilder


class TestEnvironmentVariable:
    """Test EnvironmentVariable model."""

    def test_creation(self):
        """Test basic creation."""
        var = EnvironmentVariable(name="test", value="hello")
        assert var.name == "test"
        assert var.value == "hello"


class TestPromptBuilder:
    """Test PromptBuilder model and methods."""

    def test_default_creation(self):
        """Test default builder creation."""
        builder = PromptBuilder()
        assert len(builder.core_documents) == 0
        assert len(builder.new_documents) == 0
        assert len(builder.environment) == 0
        assert builder.mode == "full"

    def test_add_variable_string(self):
        """Test adding a string variable."""
        builder = PromptBuilder()
        builder.add_variable("config", "value123")
        assert len(builder.environment) == 1
        assert builder.environment[0].name == "config"
        assert builder.environment[0].value == "value123"

    def test_add_variable_document(self):
        """Test adding a Document variable."""
        builder = PromptBuilder()
        doc = TemporaryDocument(name="test.txt", content=b"doc content")
        builder.add_variable("doc_var", doc)
        assert len(builder.environment) == 1
        assert "doc content" in builder.environment[0].value

    def test_add_variable_none_skips(self):
        """Test adding None value skips the variable."""
        builder = PromptBuilder()
        builder.add_variable("empty", None)
        assert len(builder.environment) == 0

    def test_add_variable_reserved_name_raises(self):
        """Test 'document' name is reserved."""
        builder = PromptBuilder()
        with pytest.raises(AssertionError, match="reserved"):
            builder.add_variable("document", "value")

    def test_add_variable_duplicate_raises(self):
        """Test duplicate variable name raises."""
        builder = PromptBuilder()
        builder.add_variable("x", "1")
        with pytest.raises(AssertionError, match="already exists"):
            builder.add_variable("x", "2")

    def test_remove_variable(self):
        """Test removing a variable."""
        builder = PromptBuilder()
        builder.add_variable("to_remove", "value")
        assert len(builder.environment) == 1
        builder.remove_variable("to_remove")
        assert len(builder.environment) == 0

    def test_remove_variable_missing_raises(self):
        """Test removing non-existent variable raises."""
        builder = PromptBuilder()
        with pytest.raises(AssertionError, match="not found"):
            builder.remove_variable("nonexistent")

    def test_add_new_core_document(self):
        """Test adding a new core document."""
        builder = PromptBuilder()
        doc = TemporaryDocument(name="new.txt", content=b"new content")
        builder.add_new_core_document(doc)
        assert len(builder.new_core_documents) == 1

    def test_get_system_prompt(self):
        """Test system prompt template rendering."""
        builder = PromptBuilder()
        prompt = builder._get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_documents_prompt(self):
        """Test documents prompt template rendering."""
        builder = PromptBuilder()
        prompt = builder._get_documents_prompt()
        assert isinstance(prompt, str)

    def test_get_new_core_documents_prompt(self):
        """Test new core documents prompt rendering."""
        builder = PromptBuilder()
        prompt = builder._get_new_core_documents_prompt()
        assert isinstance(prompt, str)

    def test_get_context(self):
        """Test context construction."""
        builder = PromptBuilder()
        context = builder._get_context()
        assert isinstance(context, AIMessages)
        assert len(context) > 0

    def test_get_messages_string(self):
        """Test messages with string prompt."""
        builder = PromptBuilder()
        messages = builder._get_messages("Hello world")
        assert isinstance(messages, AIMessages)
        assert any("Hello world" in str(m) for m in messages)

    def test_get_messages_with_ai_messages(self):
        """Test messages with AIMessages prompt."""
        builder = PromptBuilder()
        prompt = AIMessages(["part1", "part2"])
        messages = builder._get_messages(prompt)
        assert isinstance(messages, AIMessages)

    def test_get_messages_includes_variables(self):
        """Test messages include environment variables."""
        builder = PromptBuilder()
        builder.add_variable("context_info", "important data")
        messages = builder._get_messages("Analyze")
        msg_text = " ".join(str(m) for m in messages)
        assert "context_info" in msg_text

    def test_get_messages_includes_new_core_documents(self):
        """Test messages include new core documents when present."""
        builder = PromptBuilder()
        doc = TemporaryDocument(name="core.txt", content=b"core content")
        builder.add_new_core_document(doc)
        messages = builder._get_messages("Analyze")
        assert len(messages) > 1

    def test_approximate_tokens_count(self):
        """Test approximate token count calculation."""
        builder = PromptBuilder()
        count = builder.approximate_tokens_count
        assert isinstance(count, int)
        assert count > 0


class TestPromptBuilderGetOptions:
    """Test _get_options method for various models."""

    def test_default_options(self):
        """Test default options are returned when none provided."""
        builder = PromptBuilder()
        options, cache_lock = builder._get_options("gpt-5-mini")
        assert options.system_prompt is not None
        assert cache_lock is True

    def test_custom_options_preserved(self):
        """Test custom options are passed through."""
        from ai_pipeline_core.llm import ModelOptions

        builder = PromptBuilder()
        custom = ModelOptions(reasoning_effort="low", max_completion_tokens=1000)
        options, _ = builder._get_options("gpt-5-mini", custom)
        assert options.max_completion_tokens == 1000

    def test_qwen_model_disables_features(self):
        """Test qwen3 models disable cache and tracking."""
        builder = PromptBuilder()
        options, cache_lock = builder._get_options("qwen3-32b")
        assert cache_lock is False
        assert options.usage_tracking is False
        assert options.verbosity is None

    def test_test_mode_sets_low_reasoning(self):
        """Test 'test' mode sets low reasoning effort."""
        builder = PromptBuilder(mode="test")
        options, _ = builder._get_options("gpt-5-mini")
        assert options.reasoning_effort == "low"

    def test_o3_model_sets_medium_reasoning(self):
        """Test o3 model uses medium reasoning."""
        builder = PromptBuilder()
        options, _ = builder._get_options("o3")
        assert options.reasoning_effort == "medium"
        assert options.verbosity is None

    def test_gpt5_model_sets_flex_tier(self):
        """Test gpt-5 model uses flex service tier."""
        builder = PromptBuilder()
        options, _ = builder._get_options("gpt-5")
        assert options.service_tier == "flex"

    def test_grok4_fast_limits_tokens(self):
        """Test grok-4-fast limits max tokens."""
        builder = PromptBuilder()
        options, _ = builder._get_options("grok-4-fast")
        assert options.max_completion_tokens == 30000
