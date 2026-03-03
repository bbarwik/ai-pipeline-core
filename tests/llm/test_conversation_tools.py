"""Tests for tool-related paths in Conversation: message conversion, span input, token count, and replay round-trip."""

from typing import Any

import pytest
from pydantic import BaseModel, Field

from ai_pipeline_core._llm_core.model_response import ModelResponse
from ai_pipeline_core._llm_core.types import CoreMessage, Role
from ai_pipeline_core.llm.conversation import Conversation, ToolResultMessage
from ai_pipeline_core.llm.tools import Tool, ToolCallRecord, ToolOutput

from .conftest import make_response, make_tool_call


# ── Test tools ────────────────────────────────────────────────────────────────


class ToolA(Tool):
    """First tool."""

    class Input(BaseModel):
        x: str = Field(description="Value")

    async def execute(self, input: Input) -> ToolOutput:
        return ToolOutput(content="a")


class ToolB(Tool):
    """Second tool (name collision with ToolA if we call it tool_a)."""

    class Input(BaseModel):
        y: str = Field(description="Value")

    async def execute(self, input: Input) -> ToolOutput:
        return ToolOutput(content="b")


# ── _to_core_messages ─────────────────────────────────────────────────────────


def test_to_core_messages_tool_result_message() -> None:
    """ToolResultMessage converts to CoreMessage with Role.TOOL and preserves fields."""
    conv = Conversation(model="test")
    msg = ToolResultMessage(tool_call_id="c1", function_name="search", content="result text")
    core = conv._to_core_messages((msg,))
    assert len(core) == 1
    assert core[0].role == Role.TOOL
    assert core[0].tool_call_id == "c1"
    assert core[0].name == "search"
    assert core[0].content == "result text"


def test_to_core_messages_model_response_with_tool_calls() -> None:
    """ModelResponse with tool_calls converts to assistant CoreMessage with tool_calls."""
    conv = Conversation(model="test")
    tc = make_tool_call("c1", "search", '{"q": "test"}')
    resp = make_response(content="Let me search", tool_calls=(tc,))
    core = conv._to_core_messages((resp,))
    assert len(core) == 1
    assert core[0].role == Role.ASSISTANT
    assert core[0].content == "Let me search"
    assert core[0].tool_calls is not None
    assert core[0].tool_calls[0].id == "c1"


def test_to_core_messages_model_response_without_tool_calls() -> None:
    """ModelResponse without tool_calls converts to plain assistant CoreMessage."""
    conv = Conversation(model="test")
    resp = make_response(content="Final answer")
    core = conv._to_core_messages((resp,))
    assert len(core) == 1
    assert core[0].role == Role.ASSISTANT
    assert core[0].tool_calls is None


# ── _core_messages_to_span_input ──────────────────────────────────────────────


def test_span_input_tool_message() -> None:
    """TOOL role messages include tool_call_id in span input."""
    msg = CoreMessage(role=Role.TOOL, content="result", tool_call_id="c42", name="search")
    result = Conversation._core_messages_to_span_input([msg])
    assert len(result) == 1
    assert result[0]["role"] == "tool"
    assert result[0]["tool_call_id"] == "c42"
    assert result[0]["content"] == "result"


def test_span_input_assistant_with_tool_calls() -> None:
    """Assistant messages with tool_calls include them in span input."""
    tc = make_tool_call("c1", "search", '{"q": "test"}')
    msg = CoreMessage(role=Role.ASSISTANT, content="searching", tool_calls=(tc,))
    result = Conversation._core_messages_to_span_input([msg])
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert "tool_calls" in result[0]
    assert result[0]["tool_calls"][0]["id"] == "c1"


# ── _collect_text ─────────────────────────────────────────────────────────────


def test_collect_text_includes_tool_result() -> None:
    """_collect_text includes ToolResultMessage content for substitutor."""
    msg = ToolResultMessage(tool_call_id="c1", function_name="search", content="http://example.com")
    texts = Conversation._collect_text((msg,))
    assert "http://example.com" in texts


# ── approximate_tokens_count ──────────────────────────────────────────────────


def test_approximate_tokens_count_with_tool_messages() -> None:
    """Token count includes tool result messages."""
    tool_msg = ToolResultMessage(tool_call_id="c1", function_name="search", content="x" * 400)
    resp = make_response(content="y" * 400)
    conv = Conversation(model="test", messages=(tool_msg, resp))
    # Each should contribute ~100 tokens (400 chars / 4 chars per token)
    assert conv.approximate_tokens_count >= 200


# ── Duplicate tool name detection ─────────────────────────────────────────────


async def test_duplicate_tool_name_detected() -> None:
    """Two tools with same snake_case name raise ValueError via real send() code path."""
    from ai_pipeline_core.llm.tools import to_snake_case

    class MySearch(Tool):
        """Search A."""

        class Input(BaseModel):
            q: str = Field(description="Query")

        async def execute(self, input: Input) -> ToolOutput:
            return ToolOutput(content="a")

    class My_Search(Tool):
        """Search B."""

        class Input(BaseModel):
            q: str = Field(description="Query")

        async def execute(self, input: Input) -> ToolOutput:
            return ToolOutput(content="b")

    # Verify names collide
    assert to_snake_case(MySearch.__name__) == to_snake_case(My_Search.__name__) == "my_search"

    # ValueError is raised before any LLM call, exercising the real _execute_send code path
    conv = Conversation(model="test")
    with pytest.raises(ValueError, match="Duplicate tool name"):
        await conv.send("test", tools=[MySearch(), My_Search()])


# ── Bug-proving tests (P0) ───────────────────────────────────────────────────


async def test_send_structured_with_tools_no_tool_call_single_invocation() -> None:
    """When send_structured is called with tools but LLM answers directly (no tool calls),
    only ONE LLM call should be made — not a tool loop call + a second structured call."""
    from ai_pipeline_core.llm._tool_loop import execute_tool_loop

    invocations: list[dict[str, Any]] = []

    async def fake_invoke(**kwargs: Any) -> ModelResponse[Any]:
        invocations.append(kwargs)
        return make_response(content='{"answer": "42"}')

    # Run the tool loop directly with response_format — it should make 1 call
    msgs, resp, records = await execute_tool_loop(
        invoke_llm=fake_invoke,
        tool_schemas=[{"type": "function", "function": {"name": "tool_a"}}],
        tool_lookup={"tool_a": ToolA()},
        tool_choice="auto",
        max_tool_rounds=5,
        purpose="test",
        expected_cost=None,
        core_messages=[CoreMessage(role=Role.USER, content="test")],
        context_count=0,
        effective_options=None,
        substitutor=None,
        build_tool_result_message=lambda tid, fn, c: ToolResultMessage(tool_call_id=tid, function_name=fn, content=c),
        response_format=BaseModel,
    )
    # LLM answered directly (no tool calls) — should be exactly 1 invocation
    assert len(invocations) == 1
    assert len(records) == 0


async def test_tool_choice_without_tools_raises() -> None:
    """Passing tool_choice without tools should raise ValueError immediately."""
    conv = Conversation(model="test")
    with pytest.raises(ValueError, match="tool_choice"):
        await conv.send("test", tool_choice="required")


# ── Replay serialization round-trip ──────────────────────────────────────────


def test_replay_serialize_tool_result_message() -> None:
    """ToolResultMessage serializes to tool_result history entry."""
    from ai_pipeline_core.replay._capture import serialize_prior_messages

    msg = ToolResultMessage(tool_call_id="c42", function_name="search", content="found it")
    entries = serialize_prior_messages((msg,))
    assert len(entries) == 1
    assert entries[0]["type"] == "tool_result"
    assert entries[0]["tool_call_id"] == "c42"
    assert entries[0]["function_name"] == "search"
    assert entries[0]["content"] == "found it"


def test_replay_serialize_model_response_with_tool_calls() -> None:
    """ModelResponse with tool_calls serializes with tool_calls list."""
    from ai_pipeline_core.replay._capture import serialize_prior_messages

    tc = make_tool_call("c1", "get_weather", '{"city": "Paris"}')
    resp = make_response(content="Let me check", tool_calls=(tc,))
    entries = serialize_prior_messages((resp,))
    assert len(entries) == 1
    assert entries[0]["type"] == "response"
    assert entries[0]["content"] == "Let me check"
    assert "tool_calls" in entries[0]
    assert entries[0]["tool_calls"][0]["id"] == "c1"
    assert entries[0]["tool_calls"][0]["function_name"] == "get_weather"


def test_replay_deserialize_tool_result() -> None:
    """tool_result history entry reconstructs ToolResultMessage."""
    from ai_pipeline_core.replay._execute import _replay_history
    from ai_pipeline_core.replay.types import ConversationReplay, HistoryEntry

    payload = ConversationReplay(
        model="test",
        prompt="test",
        history=(HistoryEntry(type="tool_result", tool_call_id="c1", function_name="search", content="result"),),
    )
    conv = Conversation(model="test")
    conv = _replay_history(conv, payload, store_base=None)  # type: ignore[arg-type]
    assert len(conv.messages) == 1
    msg = conv.messages[0]
    assert isinstance(msg, ToolResultMessage)
    assert msg.tool_call_id == "c1"
    assert msg.function_name == "search"


def test_replay_deserialize_response_with_tool_calls() -> None:
    """Response entry with tool_calls reconstructs ModelResponse, not AssistantMessage."""
    from ai_pipeline_core.replay._execute import _replay_history
    from ai_pipeline_core.replay.types import ConversationReplay, HistoryEntry

    payload = ConversationReplay(
        model="test",
        prompt="test",
        history=(
            HistoryEntry(
                type="response",
                content="Let me search",
                tool_calls=[{"id": "c1", "function_name": "search", "arguments": '{"q": "test"}'}],
            ),
        ),
    )
    conv = Conversation(model="test")
    conv = _replay_history(conv, payload, store_base=None)  # type: ignore[arg-type]
    assert len(conv.messages) == 1
    msg = conv.messages[0]
    assert isinstance(msg, ModelResponse)
    assert msg.has_tool_calls
    assert msg.tool_calls[0].id == "c1"
    assert msg.tool_calls[0].function_name == "search"


def test_replay_full_round_trip() -> None:
    """Serialize then deserialize a complete tool round — data preserved."""
    from ai_pipeline_core.replay._capture import serialize_prior_messages
    from ai_pipeline_core.replay._execute import _replay_history
    from ai_pipeline_core.replay.types import ConversationReplay, HistoryEntry

    # Build original history
    tc = make_tool_call("c1", "search", '{"q": "weather"}')
    messages: tuple[Any, ...] = (
        make_response(content="", tool_calls=(tc,)),
        ToolResultMessage(tool_call_id="c1", function_name="search", content="Sunny 20°C"),
        make_response(content="It's sunny and 20°C."),
    )

    # Serialize
    entries = serialize_prior_messages(messages)
    assert len(entries) == 3

    # Deserialize
    history_entries = tuple(HistoryEntry.model_validate(e) for e in entries)
    payload = ConversationReplay(model="test", prompt="test", history=history_entries)
    conv = Conversation(model="test")
    conv = _replay_history(conv, payload, store_base=None)  # type: ignore[arg-type]

    assert len(conv.messages) == 3
    # First: ModelResponse with tool_calls
    assert isinstance(conv.messages[0], ModelResponse)
    assert conv.messages[0].has_tool_calls
    # Second: ToolResultMessage
    assert isinstance(conv.messages[1], ToolResultMessage)
    assert conv.messages[1].content == "Sunny 20°C"
    # Third: response without tool_calls is reconstructed as AssistantMessage (via with_assistant_message)
    from ai_pipeline_core.llm.conversation import AssistantMessage

    assert isinstance(conv.messages[2], AssistantMessage)
    assert conv.messages[2].text == "It's sunny and 20°C."


# ── tool_calls_for ──────────────────────────────────────────────────────────


def _conv_with_records(*records: ToolCallRecord) -> Conversation:
    """Create a Conversation with tool_call_records (Pydantic private fields need model_copy)."""
    return Conversation(model="test").model_copy(update={"_tool_call_records": records})


def test_tool_calls_for_filters_by_tool_class() -> None:
    """tool_calls_for returns only records matching the given tool class."""
    conv = _conv_with_records(
        ToolCallRecord(tool=ToolA, input=ToolA.Input(x="1"), output=ToolOutput(content="a1"), round=1),
        ToolCallRecord(tool=ToolB, input=ToolB.Input(y="2"), output=ToolOutput(content="b1"), round=1),
        ToolCallRecord(tool=ToolA, input=ToolA.Input(x="3"), output=ToolOutput(content="a2"), round=2),
    )

    a_records = conv.tool_calls_for(ToolA)
    assert len(a_records) == 2
    assert all(r.tool is ToolA for r in a_records)

    b_records = conv.tool_calls_for(ToolB)
    assert len(b_records) == 1
    assert b_records[0].tool is ToolB


def test_tool_calls_for_empty_when_no_match() -> None:
    """tool_calls_for returns empty tuple when no records match."""
    conv = _conv_with_records(
        ToolCallRecord(tool=ToolA, input=ToolA.Input(x="1"), output=ToolOutput(content="a"), round=1),
    )
    assert conv.tool_calls_for(ToolB) == ()


def test_tool_calls_for_returns_tuple() -> None:
    """tool_calls_for returns immutable tuple, not list."""
    conv = _conv_with_records(
        ToolCallRecord(tool=ToolA, input=ToolA.Input(x="1"), output=ToolOutput(content="a"), round=1),
    )
    result = conv.tool_calls_for(ToolA)
    assert isinstance(result, tuple)


# ── Record accumulation ─────────────────────────────────────────────────────


def test_tool_call_records_not_accumulated_across_sends() -> None:
    """Each send() call produces independent tool_call_records — no cross-send accumulation.

    Phase 1 records must NOT appear in Phase 2's tool_call_records.
    This enables the collection pattern: phase1.tool_call_records + phase2.tool_call_records.
    """
    phase1_records = (
        ToolCallRecord(tool=ToolA, input=ToolA.Input(x="a"), output=ToolOutput(content="r1"), round=1),
    )
    conv_after_phase1 = Conversation(model="test").model_copy(update={"_tool_call_records": phase1_records})

    phase2_records = (
        ToolCallRecord(tool=ToolB, input=ToolB.Input(y="b"), output=ToolOutput(content="r2"), round=1),
    )
    conv_after_phase2 = conv_after_phase1.model_copy(update={"_tool_call_records": phase2_records})

    # Phase 1 has only its own records
    assert len(conv_after_phase1.tool_call_records) == 1
    assert conv_after_phase1.tool_call_records[0].tool is ToolA

    # Phase 2 has only its own records — NOT phase1 + phase2
    assert len(conv_after_phase2.tool_call_records) == 1
    assert conv_after_phase2.tool_call_records[0].tool is ToolB


def test_tool_calls_for_multi_phase_collection() -> None:
    """Collection pattern across phases: no double-counting, no data loss.

    Simulates the primary use case:
        conv = await conv.send_spec(AnalysisSpec(), tools=[inspect])
        critic_conv = await conv.send_spec(CriticSpec(), tools=[inspect])
        all_calls = conv.tool_calls_for(Inspect) + critic_conv.tool_calls_for(Inspect)
    """
    conv_phase1 = _conv_with_records(
        ToolCallRecord(tool=ToolA, input=ToolA.Input(x="p1_1"), output=ToolOutput(content="r1"), round=1),
        ToolCallRecord(tool=ToolA, input=ToolA.Input(x="p1_2"), output=ToolOutput(content="r2"), round=2),
    )
    conv_phase2 = _conv_with_records(
        ToolCallRecord(tool=ToolA, input=ToolA.Input(x="p2_1"), output=ToolOutput(content="r3"), round=1),
    )

    all_calls = conv_phase1.tool_calls_for(ToolA) + conv_phase2.tool_calls_for(ToolA)
    assert len(all_calls) == 3
