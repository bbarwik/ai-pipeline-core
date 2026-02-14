"""Tests for default substitutor system prompt injection in Conversation._execute_send().

When the substitutor is active with patterns and no user system prompt is provided,
Conversation should inject a default system prompt instructing the LLM to preserve ~ markers.
"""

import pytest

from ai_pipeline_core._llm_core import ModelOptions
from ai_pipeline_core.llm.conversation import Conversation, _DEFAULT_SUBSTITUTOR_PROMPT
from tests.support.helpers import create_test_model_response

LONG_URL = "https://example.com/docs/api/v2/reference/contracts/very/long/path/to/resource/page"


class TestPromptInjection:
    """Category 1: Default prompt injection conditions."""

    @pytest.mark.asyncio
    async def test_default_prompt_injected_when_no_system_prompt(self, monkeypatch):
        """Active substitutor with patterns → default prompt should be injected."""
        captured_options: list[ModelOptions | None] = []

        async def fake_generate(messages, **kwargs):
            captured_options.append(kwargs.get("model_options"))
            return create_test_model_response(content="OK")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        conv = Conversation(model="test-model")
        await conv.send(f"Check {LONG_URL}")

        assert len(captured_options) == 1
        opts = captured_options[0]
        assert opts is not None
        assert opts.system_prompt == _DEFAULT_SUBSTITUTOR_PROMPT

    @pytest.mark.asyncio
    async def test_default_prompt_not_injected_when_user_has_system_prompt(self, monkeypatch):
        """User-provided system prompt should not be overridden."""
        captured_options: list[ModelOptions | None] = []
        user_prompt = "You are a blockchain expert."

        async def fake_generate(messages, **kwargs):
            captured_options.append(kwargs.get("model_options"))
            return create_test_model_response(content="OK")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        conv = Conversation(model="test-model", model_options=ModelOptions(system_prompt=user_prompt))
        await conv.send(f"Check {LONG_URL}")

        assert len(captured_options) == 1
        opts = captured_options[0]
        assert opts is not None
        assert opts.system_prompt == user_prompt

    @pytest.mark.asyncio
    async def test_default_prompt_not_injected_when_substitutor_disabled(self, monkeypatch):
        """Substitutor disabled → no prompt injection."""
        captured_options: list[ModelOptions | None] = []

        async def fake_generate(messages, **kwargs):
            captured_options.append(kwargs.get("model_options"))
            return create_test_model_response(content="OK")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        conv = Conversation(model="test-model", enable_substitutor=False)
        await conv.send(f"Check {LONG_URL}")

        assert len(captured_options) == 1
        opts = captured_options[0]
        # No model_options at all (substitutor disabled, no user options)
        assert opts is None

    @pytest.mark.asyncio
    async def test_default_prompt_not_injected_when_no_patterns(self, monkeypatch):
        """Substitutor active but 0 patterns → no prompt injection."""
        captured_options: list[ModelOptions | None] = []

        async def fake_generate(messages, **kwargs):
            captured_options.append(kwargs.get("model_options"))
            return create_test_model_response(content="OK")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        conv = Conversation(model="test-model")
        await conv.send("Plain text with no URLs or addresses")

        assert len(captured_options) == 1
        opts = captured_options[0]
        # No patterns found → no injection → model_options stays None
        assert opts is None


class TestPromptIntegration:
    """Category 2: Integration tests for prompt injection."""

    @pytest.mark.asyncio
    async def test_default_prompt_with_model_options_none(self, monkeypatch):
        """model_options=None + patterns → creates new ModelOptions with default prompt."""
        captured_options: list[ModelOptions | None] = []

        async def fake_generate(messages, **kwargs):
            captured_options.append(kwargs.get("model_options"))
            return create_test_model_response(content="OK")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        conv = Conversation(model="test-model", model_options=None)
        await conv.send(f"Check {LONG_URL}")

        opts = captured_options[0]
        assert opts is not None
        assert opts.system_prompt == _DEFAULT_SUBSTITUTOR_PROMPT

    @pytest.mark.asyncio
    async def test_default_prompt_does_not_mutate_original_options(self, monkeypatch):
        """Original model_options should not be modified after send."""

        async def fake_generate(messages, **kwargs):
            return create_test_model_response(content="OK")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        original_opts = ModelOptions(reasoning_effort="high")
        conv = Conversation(model="test-model", model_options=original_opts)
        await conv.send(f"Check {LONG_URL}")

        # Original should be unchanged
        assert original_opts.system_prompt is None
        assert original_opts.reasoning_effort == "high"

    @pytest.mark.asyncio
    async def test_default_prompt_preserves_other_model_options(self, monkeypatch):
        """When injecting prompt, other model_options fields should be preserved."""
        captured_options: list[ModelOptions | None] = []

        async def fake_generate(messages, **kwargs):
            captured_options.append(kwargs.get("model_options"))
            return create_test_model_response(content="OK")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        conv = Conversation(model="test-model", model_options=ModelOptions(reasoning_effort="high", retries=5))
        await conv.send(f"Check {LONG_URL}")

        opts = captured_options[0]
        assert opts is not None
        assert opts.system_prompt == _DEFAULT_SUBSTITUTOR_PROMPT
        assert opts.reasoning_effort == "high"
        assert opts.retries == 5

    @pytest.mark.asyncio
    async def test_default_prompt_works_with_send_structured(self, monkeypatch):
        """Default prompt should also be injected for send_structured()."""
        from pydantic import BaseModel as PydanticBaseModel

        from tests.support.helpers import create_test_structured_model_response

        captured_options: list[ModelOptions | None] = []

        class Result(PydanticBaseModel):
            answer: str

        async def fake_generate_structured(messages, response_format, **kwargs):
            captured_options.append(kwargs.get("model_options"))
            return create_test_structured_model_response(parsed=Result(answer="OK"))

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate_structured", fake_generate_structured)

        conv = Conversation(model="test-model")
        await conv.send_structured(f"Check {LONG_URL}", Result)

        opts = captured_options[0]
        assert opts is not None
        assert opts.system_prompt == _DEFAULT_SUBSTITUTOR_PROMPT
