"""Tests for replay payload execute() methods with database-backed document resolution."""

import logging
from typing import Any

import pytest
from pydantic import BaseModel

from ai_pipeline_core.documents import Document
from ai_pipeline_core.llm.conversation import Conversation
from ai_pipeline_core.pipeline import PipelineFlow, PipelineTask, get_run_id
from ai_pipeline_core.replay import ConversationReplay, DocumentRef, FlowReplay, HistoryEntry, TaskReplay
from tests.replay.conftest import (
    ReplayArgsModel,
    ReplayFlowOptions,
    ReplayResultDocument,
    ReplayTextDocument,
    doc_ref_dict,
    store_document_in_database,
)
from tests.support.helpers import create_test_model_response


async def _test_task_fn(source: ReplayTextDocument, *, label: str) -> ReplayResultDocument:
    return ReplayResultDocument(name="result.txt", content=source.content, description=label)


async def _test_task_with_model_arg(source: ReplayTextDocument, *, config: ReplayArgsModel) -> ReplayResultDocument:
    return ReplayResultDocument(name="result.txt", content=source.content, description=config.label)


class _ReplayPipelineTask(PipelineTask):
    @classmethod
    async def run(cls, source: ReplayTextDocument, label: str) -> tuple[ReplayResultDocument, ...]:
        _ = cls
        return (ReplayResultDocument(name="result.txt", content=source.content, description=label),)


async def _test_flow_fn(
    documents: tuple[ReplayTextDocument, ...],
    flow_options: ReplayFlowOptions,
) -> tuple[ReplayResultDocument, ...]:
    return (
        ReplayResultDocument(
            name="flow_result.txt",
            content=documents[0].content,
            description=f"{get_run_id()}:{flow_options.replay_label}:{flow_options.replay_mode}",
        ),
    )


class _ReplayClassFlow(PipelineFlow):
    async def run(self, documents: tuple[ReplayTextDocument, ...], options: ReplayFlowOptions) -> tuple[ReplayResultDocument, ...]:
        _ = options
        return (
            ReplayResultDocument(
                name="flow_out.txt",
                content=documents[0].content if documents else b"empty",
                description=get_run_id(),
            ),
        )


class _ParameterizedFlow(PipelineFlow):
    model_name: str = "default-model"

    async def run(self, documents: tuple[ReplayTextDocument, ...], options: ReplayFlowOptions) -> tuple[ReplayResultDocument, ...]:
        _ = (documents, options)
        return (
            ReplayResultDocument(
                name="parameterized.txt",
                content=self.model_name.encode(),
                description=f"{self.model_name}:{get_run_id()}",
            ),
        )


class _SummaryOutput(BaseModel):
    summary: str
    word_count: int


class TestConversationReplayExecute:
    @pytest.mark.asyncio
    async def test_conversation_execute_sends_prompt(self, monkeypatch: pytest.MonkeyPatch, memory_database) -> None:
        captured_content: list[str] = []

        async def fake_generate(messages: Any, **kwargs: Any) -> Any:
            _ = kwargs
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
        conv = await replay.execute(memory_database)

        assert captured_content == ["Summarize everything."]
        assert isinstance(conv, Conversation)
        assert conv.content == "mocked reply"

    @pytest.mark.asyncio
    async def test_conversation_execute_resolves_context_docs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        memory_database,
        sample_text_doc: ReplayTextDocument,
    ) -> None:
        resolved_docs: list[Document] = []
        await store_document_in_database(memory_database, sample_text_doc)

        async def fake_generate(messages: Any, **kwargs: Any) -> Any:
            _ = (messages, kwargs)
            return create_test_model_response(content="ok")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        replay = ConversationReplay(
            payload_type="conversation",
            model="test-model",
            prompt="Describe the doc.",
            context=[doc_ref_dict(sample_text_doc)],
            history=[],
        )

        original_send = Conversation.send

        async def capturing_send(self_conv: Conversation, content: Any, **kwargs: Any) -> Conversation:
            _ = (content, kwargs)
            resolved_docs.extend(self_conv.context)
            return await original_send(self_conv, content)

        monkeypatch.setattr(Conversation, "send", capturing_send)

        await replay.execute(memory_database)

        assert [doc.name for doc in resolved_docs] == [sample_text_doc.name]

    @pytest.mark.asyncio
    async def test_conversation_execute_with_history(self, monkeypatch: pytest.MonkeyPatch, memory_database) -> None:
        seen_messages: list[Any] = []

        async def fake_generate(messages: Any, **kwargs: Any) -> Any:
            _ = kwargs
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

        conv = await replay.execute(memory_database)

        assert conv.content == "final answer"
        assert len(seen_messages) >= 3

    @pytest.mark.asyncio
    async def test_conversation_execute_structured_output(self, monkeypatch: pytest.MonkeyPatch, memory_database) -> None:
        structured_called = False

        async def fake_generate_structured(messages: Any, response_format: Any, **kwargs: Any) -> Any:
            _ = (messages, response_format, kwargs)
            nonlocal structured_called
            structured_called = True
            from tests.support.helpers import create_test_structured_model_response

            return create_test_structured_model_response(
                parsed=_SummaryOutput(summary="AI summary", word_count=42),
            )

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate_structured", fake_generate_structured)
        monkeypatch.setattr(
            "ai_pipeline_core.llm.conversation.core_generate",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not be called")),
        )

        replay = ConversationReplay(
            payload_type="conversation",
            model="test-model",
            prompt="Summarize.",
            context=[],
            history=[],
            response_format=f"{_SummaryOutput.__module__}:{_SummaryOutput.__qualname__}",
        )

        conv = await replay.execute(memory_database)

        assert structured_called
        assert conv.parsed is not None
        assert conv.parsed.summary == "AI summary"

    @pytest.mark.asyncio
    async def test_conversation_execute_resolves_single_prompt_document(
        self,
        monkeypatch: pytest.MonkeyPatch,
        memory_database,
        sample_text_doc: ReplayTextDocument,
    ) -> None:
        await store_document_in_database(memory_database, sample_text_doc)
        captured_content: list[Any] = []

        async def fake_send(self_conv: Conversation, content: Any, **kwargs: Any) -> Conversation:
            _ = kwargs
            captured_content.append(content)
            return self_conv.model_copy(update={"messages": (*self_conv.messages, create_test_model_response(content="ok"))})

        monkeypatch.setattr(Conversation, "send", fake_send)

        replay = ConversationReplay(
            payload_type="conversation",
            model="test-model",
            prompt_documents=[DocumentRef.model_validate(doc_ref_dict(sample_text_doc))],
            context=[],
            history=[],
        )

        await replay.execute(memory_database)

        assert len(captured_content) == 1
        assert isinstance(captured_content[0], ReplayTextDocument)
        assert captured_content[0].sha256 == sample_text_doc.sha256

    @pytest.mark.asyncio
    async def test_conversation_execute_resolves_prompt_document_list_and_frozen_date(
        self,
        monkeypatch: pytest.MonkeyPatch,
        memory_database,
        sample_text_doc: ReplayTextDocument,
    ) -> None:
        await store_document_in_database(memory_database, sample_text_doc)
        second_doc = ReplayTextDocument(name="second.txt", content=b"second replay document")
        await store_document_in_database(memory_database, second_doc)

        captured_content: list[Any] = []
        captured_dates: list[tuple[bool, str | None]] = []

        async def fake_send(self_conv: Conversation, content: Any, **kwargs: Any) -> Conversation:
            _ = kwargs
            captured_content.append(content)
            captured_dates.append((self_conv.include_date, self_conv.current_date))
            return self_conv.model_copy(update={"messages": (*self_conv.messages, create_test_model_response(content="ok"))})

        monkeypatch.setattr(Conversation, "send", fake_send)

        replay = ConversationReplay(
            payload_type="conversation",
            model="test-model",
            prompt_documents=[
                DocumentRef.model_validate(doc_ref_dict(sample_text_doc)),
                DocumentRef.model_validate(doc_ref_dict(second_doc)),
            ],
            context=[],
            history=[],
            include_date=True,
            current_date="2025-03-15",
        )

        await replay.execute(memory_database)

        assert len(captured_content) == 1
        assert isinstance(captured_content[0], list)
        assert [doc.name for doc in captured_content[0]] == [sample_text_doc.name, second_doc.name]
        assert captured_dates == [(True, "2025-03-15")]

    @pytest.mark.asyncio
    async def test_conversation_execute_unimportable_format_falls_back(
        self,
        monkeypatch: pytest.MonkeyPatch,
        memory_database,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        async def fake_generate(messages: Any, **kwargs: Any) -> Any:
            _ = (messages, kwargs)
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

        with caplog.at_level(logging.WARNING):
            conv = await replay.execute(memory_database)

        assert conv.content == "fallback text"
        assert any("import" in record.message.lower() or "nonexistent" in record.message for record in caplog.records)


class TestTaskReplayExecute:
    @pytest.mark.asyncio
    async def test_task_execute_calls_function(self, memory_database) -> None:
        fn_path = f"{_test_task_fn.__module__}:{_test_task_fn.__qualname__}"
        text_doc = ReplayTextDocument(name="src.txt", content=b"hello world")
        await store_document_in_database(memory_database, text_doc)

        replay = TaskReplay(
            payload_type="pipeline_task",
            function_path=fn_path,
            arguments={"source": doc_ref_dict(text_doc), "label": "test-label"},
        )

        result = await replay.execute(memory_database)

        assert isinstance(result, ReplayResultDocument)
        assert result.description == "test-label"

    @pytest.mark.asyncio
    async def test_task_execute_resolves_doc_refs(self, memory_database, sample_text_doc: ReplayTextDocument) -> None:
        fn_path = f"{_test_task_fn.__module__}:{_test_task_fn.__qualname__}"
        await store_document_in_database(memory_database, sample_text_doc)

        replay = TaskReplay(
            payload_type="pipeline_task",
            function_path=fn_path,
            arguments={"source": doc_ref_dict(sample_text_doc), "label": "resolved"},
        )

        result = await replay.execute(memory_database)

        assert isinstance(result, ReplayResultDocument)
        assert result.description == "resolved"
        assert result.content == sample_text_doc.content

    @pytest.mark.asyncio
    async def test_task_execute_validates_basemodel_args(self, memory_database, sample_text_doc: ReplayTextDocument) -> None:
        fn_path = f"{_test_task_with_model_arg.__module__}:{_test_task_with_model_arg.__qualname__}"
        await store_document_in_database(memory_database, sample_text_doc)

        replay = TaskReplay(
            payload_type="pipeline_task",
            function_path=fn_path,
            arguments={
                "source": doc_ref_dict(sample_text_doc),
                "config": {"max_items": 10, "label": "validated"},
            },
        )

        result = await replay.execute(memory_database)

        assert isinstance(result, ReplayResultDocument)
        assert result.description == "validated"

    @pytest.mark.asyncio
    async def test_task_execute_pipeline_task_class(self, memory_database, sample_text_doc: ReplayTextDocument) -> None:
        fn_path = f"{_ReplayPipelineTask.__module__}:{_ReplayPipelineTask.__qualname__}"
        await store_document_in_database(memory_database, sample_text_doc)

        replay = TaskReplay(
            payload_type="pipeline_task",
            function_path=fn_path,
            arguments={"source": doc_ref_dict(sample_text_doc), "label": "pipeline-task"},
        )

        result = await replay.execute(memory_database)

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], ReplayResultDocument)
        assert result[0].description == "pipeline-task"


class TestFlowReplayExecute:
    @pytest.mark.asyncio
    async def test_flow_execute_resolves_docs_and_options(self, memory_database, sample_text_doc: ReplayTextDocument) -> None:
        fn_path = f"{_test_flow_fn.__module__}:{_test_flow_fn.__qualname__}"
        await store_document_in_database(memory_database, sample_text_doc)

        replay = FlowReplay(
            payload_type="pipeline_flow",
            function_path=fn_path,
            run_id="run-99",
            documents=[doc_ref_dict(sample_text_doc)],
            flow_options={"replay_label": "prod", "replay_mode": "deep"},
        )

        result = await replay.execute(memory_database)

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], ReplayResultDocument)
        assert result[0].description == "run-99:prod:deep"

    @pytest.mark.asyncio
    async def test_flow_execute_handles_pipeline_flow_class(self, memory_database, sample_text_doc: ReplayTextDocument) -> None:
        fn_path = f"{_ReplayClassFlow.__module__}:{_ReplayClassFlow.__qualname__}"
        await store_document_in_database(memory_database, sample_text_doc)

        replay = FlowReplay(
            payload_type="pipeline_flow",
            function_path=fn_path,
            run_id="test-run",
            documents=[doc_ref_dict(sample_text_doc)],
            flow_options={},
        )

        result = await replay.execute(memory_database)
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0].description == "test-run"

    @pytest.mark.asyncio
    async def test_flow_execute_preserves_constructor_params(self, memory_database, sample_text_doc: ReplayTextDocument) -> None:
        fn_path = f"{_ParameterizedFlow.__module__}:{_ParameterizedFlow.__qualname__}"
        await store_document_in_database(memory_database, sample_text_doc)

        replay = FlowReplay(
            payload_type="pipeline_flow",
            function_path=fn_path,
            run_id="test-run",
            documents=[doc_ref_dict(sample_text_doc)],
            flow_options={},
            flow_params={"model_name": "gpt-5"},
        )

        result = await replay.execute(memory_database)
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0].description == "gpt-5:test-run"
