"""AI-doc examples for Conversation caching, forking, and document serialization."""

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ai_pipeline_core._llm_core.types import CoreMessage, Role
from ai_pipeline_core.llm.conversation import Conversation
from tests.support.helpers import ConcreteDocument, create_test_model_response


def _mock_laminar():
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=None)
    mock_laminar = MagicMock()
    mock_laminar.start_as_current_span.return_value = mock_span
    return mock_laminar


@pytest.mark.ai_docs
@pytest.mark.asyncio
async def test_warmup_then_parallel_forks(monkeypatch):
    """Warmup populates cache, forks share the warmed prefix including warmup response."""
    captured_calls: list[list[CoreMessage]] = []

    async def fake_generate(messages: list[CoreMessage], **kwargs: Any) -> Any:
        captured_calls.append(messages)
        for msg in reversed(messages):
            if msg.role == Role.USER and isinstance(msg.content, str):
                return create_test_model_response(content=f"Answer: {msg.content}")
        return create_test_model_response(content="ok")

    monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

    with patch("ai_pipeline_core.llm.conversation.Laminar", _mock_laminar()):
        shared_doc = ConcreteDocument(name="shared_context.md", content=b"Shared context for all forks.")
        base = Conversation(model="test-model", enable_substitutor=False).with_context(shared_doc)
        warmed = await base.send("Acknowledge the context.")
        fork_a, fork_b = await asyncio.gather(
            warmed.send("Question A"),
            warmed.send("Question B"),
        )

    assert warmed.content == "Answer: Acknowledge the context."
    assert fork_a.content == "Answer: Question A"
    assert fork_b.content == "Answer: Question B"

    # Fork calls include the warmup response as an assistant message in their history
    fork_calls = captured_calls[1:]
    assert len(fork_calls) == 2
    for call in fork_calls:
        assistant_msgs = [m for m in call if m.role == Role.ASSISTANT]
        assert any("Answer: Acknowledge the context." in m.content for m in assistant_msgs if isinstance(m.content, str))


@pytest.mark.ai_docs
@pytest.mark.asyncio
async def test_context_is_cached_prefix_messages_are_dynamic_suffix(monkeypatch):
    """with_context() adds to cacheable prefix (context_count); with_document() adds to dynamic suffix."""
    captured_kwargs: list[dict[str, Any]] = []

    async def fake_generate(messages: list[CoreMessage], **kwargs: Any) -> Any:
        captured_kwargs.append(kwargs)
        return create_test_model_response(content="ok")

    monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

    with patch("ai_pipeline_core.llm.conversation.Laminar", _mock_laminar()):
        context_doc = ConcreteDocument(name="cached.md", content=b"Cached context")
        message_doc = ConcreteDocument(name="dynamic.md", content=b"Dynamic data")
        conv = Conversation(model="test-model", enable_substitutor=False)
        conv = conv.with_context(context_doc)
        conv = conv.with_document(message_doc)
        await conv.send("Question")

    # context_count tells the provider how many leading messages form the cached prefix
    assert captured_kwargs[0]["context_count"] == 1


@pytest.mark.ai_docs
def test_document_wrapped_in_xml_tags():
    """Documents sent to the LLM are wrapped in <document> XML tags with id, name, description, content."""
    doc = ConcreteDocument(
        name="report.md",
        content=b"Final findings here.",
        description="Research report",
    )
    conv = Conversation(model="test-model", enable_substitutor=False)
    core_messages = conv._to_core_messages((doc,))

    assert len(core_messages) == 1
    assert core_messages[0].role == Role.USER
    content = core_messages[0].content
    assert isinstance(content, str)
    assert "<document>" in content
    assert f"<id>{doc.id}</id>" in content
    assert "<name>report.md</name>" in content
    assert "<description>Research report</description>" in content
    assert "<content>" in content
    assert "Final findings here." in content
    assert "</document>" in content
