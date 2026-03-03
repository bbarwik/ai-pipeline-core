"""Tests for _llm_core primitive tool support: types, message mapping, response building."""

import json

import pytest

from ai_pipeline_core._llm_core.types import CoreMessage, RawToolCall, Role


# ── RawToolCall ──────────────────────────────────────────────────────────────


def test_raw_tool_call_string_arguments() -> None:
    tc = RawToolCall(id="call_1", function_name="search", arguments='{"q": "test"}')
    assert tc.arguments == '{"q": "test"}'


def test_raw_tool_call_dict_coercion() -> None:
    """Dict arguments are coerced to JSON string (Gemini/LiteLLM quirk)."""
    tc = RawToolCall(id="call_1", function_name="search", arguments={"q": "test"})  # type: ignore[arg-type]
    assert tc.arguments == json.dumps({"q": "test"})


def test_raw_tool_call_frozen() -> None:
    from pydantic import ValidationError

    tc = RawToolCall(id="call_1", function_name="search", arguments="{}")
    with pytest.raises(ValidationError):
        tc.id = "call_2"  # type: ignore[misc]


# ── CoreMessage tool field validation ────────────────────────────────────────


def test_core_message_assistant_with_tool_calls() -> None:
    tc = RawToolCall(id="call_1", function_name="get_weather", arguments='{"city": "Paris"}')
    msg = CoreMessage(role=Role.ASSISTANT, content="", tool_calls=(tc,))
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1


def test_core_message_tool_result() -> None:
    msg = CoreMessage(role=Role.TOOL, content='{"temp": 20}', tool_call_id="call_1", name="get_weather")
    assert msg.tool_call_id == "call_1"
    assert msg.name == "get_weather"


def test_core_message_tool_calls_only_on_assistant() -> None:
    tc = RawToolCall(id="call_1", function_name="search", arguments="{}")
    with pytest.raises(ValueError, match="tool_calls is only valid on ASSISTANT"):
        CoreMessage(role=Role.USER, content="hi", tool_calls=(tc,))


def test_core_message_tool_call_id_only_on_tool() -> None:
    with pytest.raises(ValueError, match="tool_call_id is only valid on TOOL"):
        CoreMessage(role=Role.ASSISTANT, content="hello", tool_call_id="call_1")


def test_core_message_tool_requires_tool_call_id() -> None:
    with pytest.raises(ValueError, match="TOOL messages require tool_call_id"):
        CoreMessage(role=Role.TOOL, content="result")


def test_core_message_tool_requires_str_content() -> None:
    """TOOL messages must have str content — catches non-string at creation time."""
    from ai_pipeline_core._llm_core.types import TextContent

    with pytest.raises(ValueError, match="TOOL messages must have str content"):
        CoreMessage(role=Role.TOOL, content=TextContent(text="wrong"), tool_call_id="c1")


# ── Role.TOOL ────────────────────────────────────────────────────────────────


def test_role_tool_value() -> None:
    assert Role.TOOL.value == "tool"


# ── _messages_to_api with tool messages ──────────────────────────────────────


def test_messages_to_api_tool_result() -> None:
    from ai_pipeline_core._llm_core.client import _messages_to_api

    msg = CoreMessage(role=Role.TOOL, content="result text", tool_call_id="call_42", name="search")
    api = _messages_to_api([msg])
    assert len(api) == 1
    assert api[0]["role"] == "tool"
    assert api[0]["tool_call_id"] == "call_42"  # type: ignore[typeddict-item]
    assert api[0]["content"] == "result text"


def test_messages_to_api_assistant_with_tool_calls() -> None:
    from ai_pipeline_core._llm_core.client import _messages_to_api

    tc = RawToolCall(id="call_1", function_name="get_weather", arguments='{"city": "Paris"}')
    msg = CoreMessage(role=Role.ASSISTANT, content="", tool_calls=(tc,))
    api = _messages_to_api([msg])
    assert len(api) == 1
    assert api[0]["role"] == "assistant"
    assert api[0].get("content") is None  # empty content becomes None
    assert len(api[0]["tool_calls"]) == 1  # type: ignore[typeddict-item]
    tc_api = api[0]["tool_calls"][0]  # type: ignore[typeddict-item]
    assert tc_api["id"] == "call_1"
    assert tc_api["function"]["name"] == "get_weather"


def test_messages_to_api_assistant_with_content_and_tool_calls() -> None:
    """Claude-style: assistant messages can have both text and tool calls."""
    from ai_pipeline_core._llm_core.client import _messages_to_api

    tc = RawToolCall(id="call_1", function_name="search", arguments='{"q": "test"}')
    msg = CoreMessage(role=Role.ASSISTANT, content="Let me search for that.", tool_calls=(tc,))
    api = _messages_to_api([msg])
    assert api[0].get("content") is not None  # text parts preserved


def test_apply_substitution_preserves_tool_fields() -> None:
    """Regression: _apply_substitution must preserve tool_call_id and name on TOOL messages."""
    from ai_pipeline_core.llm.conversation import Conversation
    from ai_pipeline_core.llm._substitutor import URLSubstitutor

    messages = [
        CoreMessage(role=Role.TOOL, content="result from tool", tool_call_id="call_42", name="my_tool"),
        CoreMessage(role=Role.ASSISTANT, content="thinking", tool_calls=(RawToolCall(id="call_1", function_name="search", arguments="{}"),)),
    ]
    sub = URLSubstitutor()
    sub.prepare(["result from tool"])
    result = Conversation._apply_substitution(messages, sub)
    assert result[0].tool_call_id == "call_42"
    assert result[0].name == "my_tool"
    assert result[1].tool_calls is not None
    assert result[1].tool_calls[0].id == "call_1"


def test_messages_to_api_assistant_tuple_content_with_tool_calls() -> None:
    """Multimodal (tuple) content on assistant messages with tool_calls must be preserved."""
    from ai_pipeline_core._llm_core.client import _messages_to_api
    from ai_pipeline_core._llm_core.types import TextContent

    tc = RawToolCall(id="call_1", function_name="search", arguments="{}")
    msg = CoreMessage(
        role=Role.ASSISTANT,
        content=(TextContent(text="Let me search for that"),),
        tool_calls=(tc,),
    )
    api = _messages_to_api([msg])
    content_parts = api[0].get("content")
    assert content_parts is not None
    assert isinstance(content_parts, list)
    first_part: dict[str, str] = content_parts[0]  # type: ignore[assignment]  # narrowing for TypedDict union
    assert first_part["text"] == "Let me search for that"


def test_messages_to_api_full_tool_round() -> None:
    """Complete tool interaction round: user → assistant (tool call) → tool result."""
    from ai_pipeline_core._llm_core.client import _messages_to_api

    tc = RawToolCall(id="call_1", function_name="search", arguments='{"q": "weather"}')
    messages = [
        CoreMessage(role=Role.USER, content="What's the weather?"),
        CoreMessage(role=Role.ASSISTANT, content="", tool_calls=(tc,)),
        CoreMessage(role=Role.TOOL, content="Sunny, 20°C", tool_call_id="call_1", name="search"),
    ]
    api = _messages_to_api(messages)
    assert len(api) == 3
    assert api[0]["role"] == "user"
    assert api[1]["role"] == "assistant"
    assert api[2]["role"] == "tool"
