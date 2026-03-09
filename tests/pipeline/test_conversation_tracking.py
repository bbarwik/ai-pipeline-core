"""Tests for conversation turn tracking via contextvar.

Verifies that:
- ConversationTurnData is appended to the contextvar during Conversation._execute_send()
- Task wrapper creates conversation and conversation_turn nodes from captured data
- Turn data is reset after task completion
- No tracking when contextvar is not set (outside task)
"""

# pyright: reportPrivateUsage=false

import asyncio
from datetime import datetime, UTC
from types import MappingProxyType
from uuid import uuid4

import pytest

from ai_pipeline_core.database import MemoryDatabase, NodeKind, NodeStatus
from ai_pipeline_core.deployment._types import _NoopPublisher
from ai_pipeline_core.documents import Document
from ai_pipeline_core.llm.conversation import Conversation
from ai_pipeline_core.pipeline import PipelineTask
from ai_pipeline_core.pipeline._execution_context import (
    ConversationTurnData,
    ExecutionContext,
    get_conversation_turns,
    reset_conversation_turns,
    set_conversation_turns,
)
from ai_pipeline_core.pipeline._execution_context import reset_execution_context, set_execution_context
from ai_pipeline_core.pipeline._task import _create_conversation_turn_nodes
from ai_pipeline_core.pipeline.limits import _SharedStatus


# --- Helpers ---


def _make_turn(
    *,
    conversation_id: str = "default-conv-id",
    conversation_name: str = "test-conv",
    model: str = "gpt-4o",
    cost_usd: float = 0.01,
    tokens_input: int = 100,
    tokens_output: int = 50,
    tokens_cache_read: int = 0,
    tokens_reasoning: int = 0,
    response_id: str = "resp-123",
    citations_json: str = "",
    first_token_time: float = 0.1,
    time_taken: float = 0.5,
    replay_payload_json: str = '{"payload_type":"conversation"}',
    status: str = "completed",
    error_type: str = "",
    error_message: str = "",
) -> ConversationTurnData:
    now = datetime.now(tz=UTC)
    return ConversationTurnData(
        conversation_id=conversation_id,
        conversation_name=conversation_name,
        model=model,
        cost_usd=cost_usd,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        tokens_cache_read=tokens_cache_read,
        tokens_reasoning=tokens_reasoning,
        prompt_content="test prompt",
        response_content="test response",
        reasoning_content="",
        started_at=now,
        ended_at=now,
        time_taken=time_taken,
        first_token_time=first_token_time,
        context_document_shas=(),
        model_options_json="{}",
        response_format_class="",
        response_id=response_id,
        citations_json=citations_json,
        replay_payload_json=replay_payload_json,
        status=status,
        error_type=error_type,
        error_message=error_message,
    )


def _make_context_with_db(db: MemoryDatabase) -> ExecutionContext:
    dep_id = uuid4()
    return ExecutionContext(
        run_id="test-run",
        run_scope="test-run/scope",
        execution_id=None,
        publisher=_NoopPublisher(),
        limits=MappingProxyType({}),
        limits_status=_SharedStatus(),
        database=db,
        deployment_id=dep_id,
        root_deployment_id=dep_id,
        deployment_name="test-pipeline",
        current_node_id=uuid4(),
    )


class _ConversationTrackingOutput(Document):
    """Output document for real conversation tracking tests."""


class _FailingConversationTask(PipelineTask):
    @classmethod
    async def run(cls) -> list[_ConversationTrackingOutput]:
        _ = cls
        conv = Conversation(model="test-model", enable_substitutor=False)
        await conv.send("track this failure", purpose="tracked-failure")
        return []


# --- Tests ---


@pytest.fixture
def db() -> MemoryDatabase:
    return MemoryDatabase()


class TestConversationTurnContextvar:
    """Test the contextvar-based turn capture mechanism."""

    def test_set_and_get_turns(self) -> None:
        turns: list[ConversationTurnData] = []
        token = set_conversation_turns(turns)
        try:
            result = get_conversation_turns()
            assert result is turns
        finally:
            reset_conversation_turns(token)

    def test_default_is_none(self) -> None:
        assert get_conversation_turns() is None

    def test_append_to_turns_list(self) -> None:
        turns: list[ConversationTurnData] = []
        token = set_conversation_turns(turns)
        try:
            turn = _make_turn()
            turns.append(turn)
            assert len(get_conversation_turns() or []) == 1
        finally:
            reset_conversation_turns(token)

    def test_reset_restores_previous(self) -> None:
        outer: list[ConversationTurnData] = [_make_turn()]
        outer_token = set_conversation_turns(outer)
        try:
            inner: list[ConversationTurnData] = []
            inner_token = set_conversation_turns(inner)
            assert get_conversation_turns() is inner
            reset_conversation_turns(inner_token)
            assert get_conversation_turns() is outer
        finally:
            reset_conversation_turns(outer_token)


class TestConversationTurnDataFields:
    """Test ConversationTurnData dataclass construction."""

    def test_all_fields_set(self) -> None:
        turn = _make_turn(
            conversation_name="my-conv",
            model="gemini-3-flash",
            cost_usd=0.05,
            tokens_input=200,
            tokens_output=100,
            tokens_cache_read=50,
            tokens_reasoning=25,
            citations_json='[{"title":"Citation","url":"https://example.com"}]',
        )
        assert turn.conversation_name == "my-conv"
        assert turn.model == "gemini-3-flash"
        assert turn.cost_usd == 0.05
        assert turn.tokens_input == 200
        assert turn.tokens_output == 100
        assert turn.tokens_cache_read == 50
        assert turn.tokens_reasoning == 25
        assert turn.response_id == "resp-123"
        assert turn.citations_json == '[{"title":"Citation","url":"https://example.com"}]'
        assert turn.first_token_time == 0.1
        assert turn.time_taken == 0.5
        assert turn.replay_payload_json == '{"payload_type":"conversation"}'
        assert turn.status == "completed"
        assert turn.error_type == ""
        assert turn.error_message == ""

    def test_frozen(self) -> None:
        turn = _make_turn()
        with pytest.raises(AttributeError):
            turn.model = "other-model"  # type: ignore[misc]


class TestCreateConversationTurnNodes:
    """Test _create_conversation_turn_nodes helper function."""

    @pytest.mark.asyncio
    async def test_single_conversation_single_turn(self, db: MemoryDatabase) -> None:
        ctx = _make_context_with_db(db)
        task_node_id = uuid4()
        turns = [_make_turn(conversation_name="extract")]

        await _create_conversation_turn_nodes(turns, task_node_id, ctx)

        nodes = list(db._nodes.values())
        conv_nodes = [n for n in nodes if n.node_kind == NodeKind.CONVERSATION]
        turn_nodes = [n for n in nodes if n.node_kind == NodeKind.CONVERSATION_TURN]

        assert len(conv_nodes) == 1
        assert len(turn_nodes) == 1

        conv = conv_nodes[0]
        assert conv.name == "extract"
        assert conv.parent_node_id == task_node_id
        assert conv.status == NodeStatus.COMPLETED
        assert conv.turn_count == 1

        turn = turn_nodes[0]
        assert turn.parent_node_id == conv.node_id
        assert turn.model == "gpt-4o"
        assert turn.cost_usd == 0.01
        assert turn.tokens_input == 100
        assert turn.tokens_output == 50
        assert turn.tokens_reasoning == 0

    @pytest.mark.asyncio
    async def test_multiple_turns_same_conversation(self, db: MemoryDatabase) -> None:
        ctx = _make_context_with_db(db)
        task_node_id = uuid4()
        turns = [
            _make_turn(conversation_name="multi-turn", tokens_input=100),
            _make_turn(conversation_name="multi-turn", tokens_input=200),
            _make_turn(conversation_name="multi-turn", tokens_input=300),
        ]

        await _create_conversation_turn_nodes(turns, task_node_id, ctx)

        conv_nodes = [n for n in db._nodes.values() if n.node_kind == NodeKind.CONVERSATION]
        turn_nodes = [n for n in db._nodes.values() if n.node_kind == NodeKind.CONVERSATION_TURN]

        assert len(conv_nodes) == 1
        assert conv_nodes[0].turn_count == 3
        assert len(turn_nodes) == 3

    @pytest.mark.asyncio
    async def test_multiple_conversations(self, db: MemoryDatabase) -> None:
        ctx = _make_context_with_db(db)
        task_node_id = uuid4()
        turns = [
            _make_turn(conversation_id="id-a", conversation_name="conv-a"),
            _make_turn(conversation_id="id-b", conversation_name="conv-b"),
            _make_turn(conversation_id="id-a", conversation_name="conv-a"),
        ]

        await _create_conversation_turn_nodes(turns, task_node_id, ctx)

        conv_nodes = [n for n in db._nodes.values() if n.node_kind == NodeKind.CONVERSATION]
        assert len(conv_nodes) == 2

        conv_a = next(n for n in conv_nodes if n.name == "conv-a")
        conv_b = next(n for n in conv_nodes if n.name == "conv-b")
        assert conv_a.turn_count == 2
        assert conv_b.turn_count == 1

    @pytest.mark.asyncio
    async def test_turn_nodes_have_conversation_id(self, db: MemoryDatabase) -> None:
        ctx = _make_context_with_db(db)
        task_node_id = uuid4()
        turns = [_make_turn(conversation_name="linked")]

        await _create_conversation_turn_nodes(turns, task_node_id, ctx)

        conv = [n for n in db._nodes.values() if n.node_kind == NodeKind.CONVERSATION][0]
        turn = [n for n in db._nodes.values() if n.node_kind == NodeKind.CONVERSATION_TURN][0]
        assert turn.conversation_id == conv.node_id
        assert turn.task_id == task_node_id

    @pytest.mark.asyncio
    async def test_no_turns_no_nodes(self, db: MemoryDatabase) -> None:
        ctx = _make_context_with_db(db)
        task_node_id = uuid4()

        await _create_conversation_turn_nodes([], task_node_id, ctx)

        assert len(db._nodes) == 0

    @pytest.mark.asyncio
    async def test_no_database_is_noop(self) -> None:
        ctx = ExecutionContext(
            run_id="test-run",
            run_scope="test-run/scope",
            execution_id=None,
            publisher=_NoopPublisher(),
            limits=MappingProxyType({}),
            limits_status=_SharedStatus(),
            database=None,
            deployment_id=None,
        )
        # Should not raise
        await _create_conversation_turn_nodes([_make_turn()], uuid4(), ctx)

    @pytest.mark.asyncio
    async def test_turn_payload_has_prompt_and_response(self, db: MemoryDatabase) -> None:
        ctx = _make_context_with_db(db)
        task_node_id = uuid4()
        turns = [_make_turn()]

        await _create_conversation_turn_nodes(turns, task_node_id, ctx)

        turn_node = [n for n in db._nodes.values() if n.node_kind == NodeKind.CONVERSATION_TURN][0]
        assert "prompt_content" in turn_node.payload
        assert "response_content" in turn_node.payload
        assert "reasoning_content" in turn_node.payload
        assert "response_id" in turn_node.payload
        assert "citations_json" in turn_node.payload
        assert "first_token_time" in turn_node.payload
        assert "time_taken" in turn_node.payload
        assert "replay_payload" in turn_node.payload
        assert turn_node.payload["prompt_content"] == "test prompt"
        assert turn_node.payload["response_content"] == "test response"

    @pytest.mark.asyncio
    async def test_failed_turn_marks_conversation_and_turn_failed(self, db: MemoryDatabase) -> None:
        ctx = _make_context_with_db(db)
        task_node_id = uuid4()
        turns = [
            _make_turn(),
            _make_turn(
                status="failed",
                error_type="LLMError",
                error_message="provider error",
                response_id="",
                replay_payload_json='{"payload_type":"conversation","original":{"status":"failed"}}',
            ),
        ]

        await _create_conversation_turn_nodes(turns, task_node_id, ctx)

        conv = [n for n in db._nodes.values() if n.node_kind == NodeKind.CONVERSATION][0]
        failed_turn = [n for n in db._nodes.values() if n.node_kind == NodeKind.CONVERSATION_TURN and n.status == NodeStatus.FAILED][0]

        assert conv.status == NodeStatus.FAILED
        assert failed_turn.error_type == "LLMError"
        assert failed_turn.error_message == "provider error"

    @pytest.mark.asyncio
    async def test_conversation_inherits_deployment_ids(self, db: MemoryDatabase) -> None:
        ctx = _make_context_with_db(db)
        task_node_id = uuid4()
        turns = [_make_turn()]

        await _create_conversation_turn_nodes(turns, task_node_id, ctx)

        conv = [n for n in db._nodes.values() if n.node_kind == NodeKind.CONVERSATION][0]
        assert conv.deployment_id == ctx.deployment_id
        assert conv.root_deployment_id == ctx.root_deployment_id


class TestRealConversationTracking:
    @pytest.mark.asyncio
    async def test_failed_llm_call_is_persisted_as_failed_turn(
        self,
        db: MemoryDatabase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def fake_generate(*args: object, **kwargs: object) -> object:
            raise RuntimeError("llm boom")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        ctx = _make_context_with_db(db)
        token = set_execution_context(ctx)
        try:
            with pytest.raises(RuntimeError, match="llm boom"):
                await _FailingConversationTask.run()
        finally:
            reset_execution_context(token)

        task_node = [n for n in db._nodes.values() if n.node_kind == NodeKind.TASK][0]
        conversation_node = [n for n in db._nodes.values() if n.node_kind == NodeKind.CONVERSATION][0]
        turn_node = [n for n in db._nodes.values() if n.node_kind == NodeKind.CONVERSATION_TURN][0]

        assert task_node.status == NodeStatus.FAILED
        assert conversation_node.status == NodeStatus.FAILED
        assert turn_node.status == NodeStatus.FAILED
        assert turn_node.error_type == "RuntimeError"
        assert turn_node.error_message == "llm boom"
        assert turn_node.payload["replay_payload"]["original"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_failed_llm_call_survives_failed_replay_payload_capture(
        self,
        db: MemoryDatabase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def fake_generate(*args: object, **kwargs: object) -> object:
            raise RuntimeError("llm boom")

        def fake_failed_payload(*args: object, **kwargs: object) -> dict[str, object]:
            raise ValueError("payload boom")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)
        monkeypatch.setattr(Conversation, "_build_failed_replay_payload", fake_failed_payload)

        ctx = _make_context_with_db(db)
        token = set_execution_context(ctx)
        try:
            with pytest.raises(RuntimeError, match="llm boom"):
                await _FailingConversationTask.run()
        finally:
            reset_execution_context(token)

        turn_node = [node for node in db._nodes.values() if node.node_kind == NodeKind.CONVERSATION_TURN][0]
        assert turn_node.status == NodeStatus.FAILED
        assert turn_node.error_message == "llm boom"

    @pytest.mark.asyncio
    async def test_cancelled_llm_call_is_persisted_as_failed_turn(
        self,
        db: MemoryDatabase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def fake_generate(*args: object, **kwargs: object) -> object:
            raise asyncio.CancelledError("llm cancelled")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        ctx = _make_context_with_db(db)
        token = set_execution_context(ctx)
        try:
            with pytest.raises(asyncio.CancelledError):
                await _FailingConversationTask.run()
        finally:
            reset_execution_context(token)

        task_node = [n for n in db._nodes.values() if n.node_kind == NodeKind.TASK][0]
        conversation_node = [n for n in db._nodes.values() if n.node_kind == NodeKind.CONVERSATION][0]
        turn_node = [n for n in db._nodes.values() if n.node_kind == NodeKind.CONVERSATION_TURN][0]

        assert task_node.status == NodeStatus.FAILED
        assert conversation_node.status == NodeStatus.FAILED
        assert turn_node.status == NodeStatus.FAILED
        assert turn_node.error_type == "CancelledError"
        assert turn_node.error_message == "llm cancelled"
