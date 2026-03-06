"""Tests for replay payload execute() methods with mocked LLM.

Each replay type (ConversationReplay, TaskReplay, FlowReplay) is tested for
correct argument resolution, document reference expansion, and dispatch to
the underlying function or Conversation.
"""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from ai_pipeline_core.documents import Document
from ai_pipeline_core.llm.conversation import Conversation
from ai_pipeline_core.pipeline import PipelineFlow, PipelineTask
from ai_pipeline_core.replay import ConversationReplay, FlowReplay, HistoryEntry, TaskReplay
from tests.replay.conftest import (
    ReplayArgsModel,
    ReplayFlowOptions,
    ReplayResultDocument,
    ReplayTextDocument,
    doc_ref_dict,
)
from tests.support.helpers import create_test_model_response


# ---------------------------------------------------------------------------
# Helpers — simple async functions used as task/flow targets
# ---------------------------------------------------------------------------


async def _test_task_fn(source: ReplayTextDocument, *, label: str) -> ReplayResultDocument:
    """Minimal task function for replay testing."""
    return ReplayResultDocument(name="result.txt", content=source.content, description=label)


async def _test_task_with_model_arg(source: ReplayTextDocument, *, config: ReplayArgsModel) -> ReplayResultDocument:
    """Task that accepts a BaseModel keyword argument."""
    return ReplayResultDocument(name="result.txt", content=source.content, description=config.label)


class _ReplayPipelineTask(PipelineTask):
    """Class-based task used to verify replay of PipelineTask.run(...)."""

    @classmethod
    async def run(cls, source: ReplayTextDocument, label: str) -> list[ReplayResultDocument]:
        _ = cls
        return [ReplayResultDocument(name="result.txt", content=source.content, description=label)]


async def _test_flow_fn(
    run_id: str,
    documents: list[ReplayTextDocument],
    flow_options: ReplayFlowOptions,
) -> list[ReplayResultDocument]:
    """Minimal flow function for replay testing."""
    return [
        ReplayResultDocument(
            name="flow_result.txt",
            content=documents[0].content,
            description=f"{flow_options.replay_label}:{flow_options.replay_mode}",
        )
    ]


class _ReplayClassFlow(PipelineFlow):
    """Class-based PipelineFlow for replay testing."""

    async def run(self, run_id: str, documents: list[ReplayTextDocument], options: ReplayFlowOptions) -> list[ReplayResultDocument]:
        return [ReplayResultDocument(name="flow_out.txt", content=documents[0].content if documents else b"empty")]


class _ParameterizedFlow(PipelineFlow):
    """PipelineFlow with constructor params for replay testing."""

    model_name: str = "default-model"

    async def run(self, run_id: str, documents: list[ReplayTextDocument], options: ReplayFlowOptions) -> list[ReplayResultDocument]:
        return [ReplayResultDocument(name="parameterized.txt", content=self.model_name.encode(), description=self.model_name)]


# ---------------------------------------------------------------------------
# Structured output model for conversation tests
# ---------------------------------------------------------------------------


class _SummaryOutput(BaseModel):
    summary: str
    word_count: int


# ---------------------------------------------------------------------------
# ConversationReplay tests
# ---------------------------------------------------------------------------


class TestConversationReplayExecute:
    """ConversationReplay.execute() builds a Conversation and sends the prompt."""

    @pytest.mark.asyncio
    async def test_conversation_execute_sends_prompt(self, monkeypatch: pytest.MonkeyPatch, populated_store: Path) -> None:
        """The rendered prompt text reaches Conversation.send()."""
        captured_content: list[str] = []

        async def fake_generate(messages: Any, **kwargs: Any) -> Any:
            # Last message is the user prompt
            last = messages[-1]
            text = last.content if isinstance(last.content, str) else str(last.content)
            captured_content.append(text)
            return create_test_model_response(content="mocked reply")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        replay = ConversationReplay(
            payload_type="conversation",
            model="test-model",
            prompt="Summarize everything.",
            context=[],
            history=[],
        )
        with patch("ai_pipeline_core.llm.conversation.Laminar", MagicMock()):
            conv = await replay.execute(populated_store)

        assert len(captured_content) == 1
        assert "Summarize everything." in captured_content[0]
        assert isinstance(conv, Conversation)
        assert conv.content == "mocked reply"

    @pytest.mark.asyncio
    async def test_conversation_execute_resolves_context_docs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        populated_store: Path,
        sample_text_doc: ReplayTextDocument,
    ) -> None:
        """$doc_ref entries in context are resolved to full Document instances."""
        resolved_docs: list[Document] = []

        async def fake_generate(messages: Any, **kwargs: Any) -> Any:
            return create_test_model_response(content="ok")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        ref = doc_ref_dict(sample_text_doc)
        replay = ConversationReplay(
            payload_type="conversation",
            model="test-model",
            prompt="Describe the doc.",
            context=[ref],
            history=[],
        )

        original_send = Conversation.send

        async def capturing_send(self_conv: Conversation, content: Any, **kw: Any) -> Conversation:
            resolved_docs.extend(self_conv.context)
            return await original_send(self_conv, content, **kw)

        monkeypatch.setattr(Conversation, "send", capturing_send)

        with patch("ai_pipeline_core.llm.conversation.Laminar", MagicMock()):
            await replay.execute(populated_store)

        assert len(resolved_docs) >= 1
        names = [d.name for d in resolved_docs]
        assert sample_text_doc.name in names

    @pytest.mark.asyncio
    async def test_conversation_execute_with_history(self, monkeypatch: pytest.MonkeyPatch, populated_store: Path) -> None:
        """History entries are reconstructed as user/assistant message pairs."""
        seen_messages: list[Any] = []

        async def fake_generate(messages: Any, **kwargs: Any) -> Any:
            seen_messages.extend(messages)
            return create_test_model_response(content="final answer")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        replay = ConversationReplay(
            payload_type="conversation",
            model="test-model",
            prompt="Continue the conversation.",
            context=[],
            history=[
                HistoryEntry(type="user_text", text="What is AI?"),
                HistoryEntry(type="response", content="Artificial intelligence is..."),
            ],
        )

        with patch("ai_pipeline_core.llm.conversation.Laminar", MagicMock()):
            conv = await replay.execute(populated_store)

        assert conv.content == "final answer"
        # At least 3 messages: history user, history assistant, final prompt
        assert len(seen_messages) >= 3

    @pytest.mark.asyncio
    async def test_conversation_execute_structured_output(self, monkeypatch: pytest.MonkeyPatch, populated_store: Path) -> None:
        """When response_format is a valid importable class, send_structured() is used."""
        structured_called = False

        async def fake_generate_structured(messages: Any, response_format: Any, **kwargs: Any) -> Any:
            nonlocal structured_called
            structured_called = True
            from tests.support.helpers import create_test_structured_model_response

            return create_test_structured_model_response(
                parsed=_SummaryOutput(summary="AI summary", word_count=42),
            )

        monkeypatch.setattr(
            "ai_pipeline_core.llm.conversation.core_generate_structured",
            fake_generate_structured,
        )
        # Also need to mock regular generate in case fallback is tested
        monkeypatch.setattr(
            "ai_pipeline_core.llm.conversation.core_generate",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not be called")),
        )

        replay = ConversationReplay(
            payload_type="conversation",
            model="test-model",
            prompt="Summarize.",
            context=[],
            history=[],
            response_format=f"{_SummaryOutput.__module__}:{_SummaryOutput.__qualname__}",
        )

        with patch("ai_pipeline_core.llm.conversation.Laminar", MagicMock()):
            conv = await replay.execute(populated_store)

        assert structured_called
        assert conv.parsed is not None
        assert conv.parsed.summary == "AI summary"

    @pytest.mark.asyncio
    async def test_conversation_execute_unimportable_format_falls_back(
        self,
        monkeypatch: pytest.MonkeyPatch,
        populated_store: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """If response_format cannot be imported, fall back to send() with a warning."""

        async def fake_generate(messages: Any, **kwargs: Any) -> Any:
            return create_test_model_response(content="fallback text")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        replay = ConversationReplay(
            payload_type="conversation",
            model="test-model",
            prompt="Summarize.",
            context=[],
            history=[],
            response_format="nonexistent.module.BogusClass",
        )

        with patch("ai_pipeline_core.llm.conversation.Laminar", MagicMock()):
            with caplog.at_level(logging.WARNING):
                conv = await replay.execute(populated_store)

        assert conv.content == "fallback text"
        # A warning should have been logged about the import failure
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("nonexistent" in msg or "import" in msg.lower() for msg in warning_messages)


# ---------------------------------------------------------------------------
# TaskReplay tests
# ---------------------------------------------------------------------------


class TestTaskReplayExecute:
    """TaskReplay.execute() imports the function, resolves args, and calls it."""

    @pytest.mark.asyncio
    async def test_task_execute_calls_function(self, populated_store: Path) -> None:
        """function_path is resolved and the function is called with given arguments."""
        fn_path = f"{_test_task_fn.__module__}:{_test_task_fn.__qualname__}"
        text_doc = ReplayTextDocument(name="src.txt", content=b"hello world")

        replay = TaskReplay(
            payload_type="pipeline_task",
            function_path=fn_path,
            arguments={
                "source": doc_ref_dict(text_doc),
                "label": "test-label",
            },
        )

        # Save the document so it can be resolved
        from ai_pipeline_core.document_store._local import LocalDocumentStore
        from ai_pipeline_core.documents import RunScope

        store = LocalDocumentStore(base_path=populated_store)
        await store.save(text_doc, RunScope("replay/test"))

        result = await replay.execute(populated_store)

        assert isinstance(result, ReplayResultDocument)
        assert result.description == "test-label"

    @pytest.mark.asyncio
    async def test_task_execute_resolves_doc_refs(self, populated_store: Path, sample_text_doc: ReplayTextDocument) -> None:
        """$doc_ref values in arguments are resolved to Document instances from the store."""
        fn_path = f"{_test_task_fn.__module__}:{_test_task_fn.__qualname__}"

        replay = TaskReplay(
            payload_type="pipeline_task",
            function_path=fn_path,
            arguments={
                "source": doc_ref_dict(sample_text_doc),
                "label": "resolved",
            },
        )

        result = await replay.execute(populated_store)

        assert isinstance(result, ReplayResultDocument)
        assert result.description == "resolved"
        # Content should match the original document
        assert result.content == sample_text_doc.content

    @pytest.mark.asyncio
    async def test_task_execute_validates_basemodel_args(self, populated_store: Path, sample_text_doc: ReplayTextDocument) -> None:
        """Dict arguments matching BaseModel type hints are validated via model_validate."""
        fn_path = f"{_test_task_with_model_arg.__module__}:{_test_task_with_model_arg.__qualname__}"

        replay = TaskReplay(
            payload_type="pipeline_task",
            function_path=fn_path,
            arguments={
                "source": doc_ref_dict(sample_text_doc),
                "config": {"max_items": 10, "label": "validated"},
            },
        )

        result = await replay.execute(populated_store)

        assert isinstance(result, ReplayResultDocument)
        assert result.description == "validated"

    @pytest.mark.asyncio
    async def test_task_execute_pipeline_task_class(self, populated_store: Path, sample_text_doc: ReplayTextDocument) -> None:
        """TaskReplay executes PipelineTask subclasses through the wrapped run() entry point."""
        fn_path = f"{_ReplayPipelineTask.__module__}:{_ReplayPipelineTask.__qualname__}"

        replay = TaskReplay(
            payload_type="pipeline_task",
            function_path=fn_path,
            arguments={
                "source": doc_ref_dict(sample_text_doc),
                "label": "pipeline-task",
            },
        )

        result = await replay.execute(populated_store)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], ReplayResultDocument)
        assert result[0].description == "pipeline-task"


# ---------------------------------------------------------------------------
# FlowReplay tests
# ---------------------------------------------------------------------------


class TestFlowReplayExecute:
    """FlowReplay.execute() resolves documents and flow options, then calls the flow function."""

    @pytest.mark.asyncio
    async def test_flow_execute_resolves_docs_and_options(self, populated_store: Path, sample_text_doc: ReplayTextDocument) -> None:
        """Documents are resolved from $doc_ref, flow_options are reconstructed."""
        fn_path = f"{_test_flow_fn.__module__}:{_test_flow_fn.__qualname__}"

        replay = FlowReplay(
            payload_type="pipeline_flow",
            function_path=fn_path,
            run_id="run-99",
            documents=[doc_ref_dict(sample_text_doc)],
            flow_options={
                "replay_label": "prod",
                "replay_mode": "deep",
            },
        )

        result = await replay.execute(populated_store)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], ReplayResultDocument)
        assert result[0].description is not None and "prod:deep" in result[0].description

    @pytest.mark.asyncio
    async def test_flow_execute_handles_pipeline_flow_class(self, populated_store: Path, sample_text_doc: ReplayTextDocument) -> None:
        """execute_flow must instantiate PipelineFlow classes, not call them directly."""
        fn_path = f"{_ReplayClassFlow.__module__}:{_ReplayClassFlow.__qualname__}"

        replay = FlowReplay(
            payload_type="pipeline_flow",
            function_path=fn_path,
            run_id="test-run",
            documents=[doc_ref_dict(sample_text_doc)],
            flow_options={},
        )

        # Before fix: TypeError because PipelineFlow.__init__ doesn't accept positional args
        result = await replay.execute(populated_store)
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_flow_execute_preserves_constructor_params(self, populated_store: Path, sample_text_doc: ReplayTextDocument) -> None:
        """flow_params are passed to PipelineFlow constructor during replay."""
        fn_path = f"{_ParameterizedFlow.__module__}:{_ParameterizedFlow.__qualname__}"

        replay = FlowReplay(
            payload_type="pipeline_flow",
            function_path=fn_path,
            run_id="test-run",
            documents=[doc_ref_dict(sample_text_doc)],
            flow_options={},
            flow_params={"model_name": "gpt-5"},
        )

        result = await replay.execute(populated_store)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].description == "gpt-5"
