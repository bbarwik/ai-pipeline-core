"""Schema injection into transport messages for ``supports_json_schema=False``.

When ``AIModel.supports_json_schema`` is ``False`` (the safe default), the
framework appends a schema prompt to the last USER message at the transport
boundary. Simplified legacy rendering is the default; full JSON Schema is an
explicit per-model mode. This unit suite proves:

- injection happens for plain-string USER content (string concatenation),
- injection happens for multimodal tuple content (TextContent appended),
- injection is suppressed when the flag is ``True``,
- injection is suppressed for forced tool calls and unstructured requests,
- ``req.call.messages`` stays frozen so retries cannot stack injections,
- the structured-output base SYSTEM prompt is still emitted alongside.
"""

import base64

import pytest
from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError
from pydantic.alias_generators import to_camel

from ai_pipeline_core._llm_core.client import _messages_for_transport
from ai_pipeline_core._llm_core.request import (
    AttemptRequest,
    LLMRequest,
    ResponseSpec,
    ToolSpec,
)
from ai_pipeline_core._llm_core.types import (
    AIModel,
    CoreMessage,
    ImageContent,
    Role,
    TextContent,
)
from tests.support.model_catalog import DEFAULT_TEST_MODEL, WEAK_SCHEMA_TEST_MODEL


class _Answer(BaseModel):
    value: int
    label: str


class _AliasedAnswer(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel)

    published_or_effective_at: str | None


def _attempt(req: LLMRequest, model: AIModel) -> AttemptRequest:
    return AttemptRequest(call=req, model=model, attempt_index=0, call_id="c1")


def _make_request(
    model: AIModel,
    content: str | TextContent | ImageContent | tuple[TextContent | ImageContent, ...],
) -> LLMRequest:
    user = CoreMessage(role=Role.USER, content=content)
    return LLMRequest(
        model=model,
        messages=(CoreMessage(role=Role.SYSTEM, content="You are helpful."), user),
        response=ResponseSpec(format=_Answer),
    )


def test_default_schema_prompt_preserves_simplified_legacy_rendering() -> None:
    req = _make_request(WEAK_SCHEMA_TEST_MODEL, "Pick a number.")
    out = _messages_for_transport(_attempt(req, WEAK_SCHEMA_TEST_MODEL))
    user = out[-1]
    assert user.role == Role.USER
    assert isinstance(user.content, str)
    assert "Pick a number." in user.content
    assert "## Response schema" in user.content
    assert "JSON Schema:" not in user.content
    assert "`_Answer` fields:" in user.content
    assert "`value` (integer, required)" in user.content


def test_schema_injection_uses_wire_aliases() -> None:
    full_schema_model = WEAK_SCHEMA_TEST_MODEL.model_copy(update={"schema_prompt_mode": "full_json_schema"})
    req = LLMRequest(
        model=full_schema_model,
        messages=(CoreMessage(role=Role.USER, content="Return the date."),),
        response=ResponseSpec(format=_AliasedAnswer),
    )

    out = _messages_for_transport(_attempt(req, full_schema_model))

    assert isinstance(out[-1].content, str)
    assert "JSON Schema:" in out[-1].content
    assert '"publishedOrEffectiveAt"' in out[-1].content
    assert '"published_or_effective_at"' not in out[-1].content


def test_schema_appended_as_textcontent_for_multimodal_user() -> None:
    """Multimodal USER (tuple of parts) receives schema text as a new TextContent suffix."""
    weak_model_with_images = WEAK_SCHEMA_TEST_MODEL.model_copy(update={"supports_images": True})
    image = ImageContent(data=base64.b64encode(b"fake-png-bytes"), mime_type="image/png")
    req = _make_request(weak_model_with_images, (TextContent(text="Look:"), image))
    out = _messages_for_transport(_attempt(req, weak_model_with_images))
    user = out[-1]
    assert isinstance(user.content, tuple)
    assert len(user.content) == 3
    assert isinstance(user.content[0], TextContent)
    assert user.content[0].text == "Look:"
    assert isinstance(user.content[1], ImageContent)
    assert isinstance(user.content[2], TextContent)
    assert "## Response schema" in user.content[2].text


def test_schema_not_injected_when_flag_true() -> None:
    native_model = DEFAULT_TEST_MODEL.model_copy(update={"schema_prompt_mode": "full_json_schema"})
    req = _make_request(native_model, "Pick a number.")
    out = _messages_for_transport(_attempt(req, native_model))
    user = out[-1]
    assert user.role == Role.USER
    assert isinstance(user.content, str)
    assert user.content == "Pick a number."


def test_schema_prompt_mode_serialization_is_backward_compatible() -> None:
    default_model = DEFAULT_TEST_MODEL
    full_model = DEFAULT_TEST_MODEL.model_copy(update={"schema_prompt_mode": "full_json_schema"})

    assert "schema_prompt_mode" not in default_model.model_dump()
    assert full_model.model_dump()["schema_prompt_mode"] == "full_json_schema"
    assert AIModel.model_validate(full_model.model_dump()).schema_prompt_mode == "full_json_schema"
    with pytest.raises(ValidationError):
        AIModel.model_validate({"name": "invalid", "schema_prompt_mode": "other"})


def test_schema_not_injected_without_response_format() -> None:
    """Unstructured requests never receive schema injection regardless of the flag."""
    req = LLMRequest(
        model=WEAK_SCHEMA_TEST_MODEL,
        messages=(CoreMessage(role=Role.USER, content="Hello"),),
    )
    out = _messages_for_transport(_attempt(req, WEAK_SCHEMA_TEST_MODEL))
    assert out == req.messages


def test_schema_not_injected_for_forced_tool_call() -> None:
    """Forced tool calls produce tool messages, not structured JSON; injection is suppressed."""
    req = LLMRequest(
        model=WEAK_SCHEMA_TEST_MODEL,
        messages=(CoreMessage(role=Role.USER, content="Use the tool"),),
        response=ResponseSpec(format=_Answer),
        tools=ToolSpec(
            schemas=({"type": "function", "function": {"name": "noop"}},),
            choice="required",
        ),
    )
    out = _messages_for_transport(_attempt(req, WEAK_SCHEMA_TEST_MODEL))
    assert out == req.messages


def test_injection_does_not_mutate_request_messages() -> None:
    """``req.call.messages`` must stay frozen so retries cannot stack injections."""
    req = _make_request(WEAK_SCHEMA_TEST_MODEL, "Pick a number.")
    snapshot = req.messages
    _ = _messages_for_transport(_attempt(req, WEAK_SCHEMA_TEST_MODEL))
    _ = _messages_for_transport(_attempt(req, WEAK_SCHEMA_TEST_MODEL))
    assert req.messages is snapshot
    assert req.messages[-1].content == "Pick a number."


def test_base_system_prompt_emitted_alongside_injection() -> None:
    """SYSTEM gets the base structured-output instruction even when schema is injected into USER."""
    req = _make_request(WEAK_SCHEMA_TEST_MODEL, "Pick a number.")
    out = _messages_for_transport(_attempt(req, WEAK_SCHEMA_TEST_MODEL))
    system = out[0]
    assert system.role == Role.SYSTEM
    assert isinstance(system.content, str)
    assert "Return valid JSON matching the response schema" in system.content


def test_schema_injected_into_last_user_when_assistant_is_final() -> None:
    """When the message tail is ASSISTANT (tool-loop turn), injection targets the last USER above it."""
    req = LLMRequest(
        model=WEAK_SCHEMA_TEST_MODEL,
        messages=(
            CoreMessage(role=Role.SYSTEM, content="You are helpful."),
            CoreMessage(role=Role.USER, content="Original question"),
            CoreMessage(role=Role.ASSISTANT, content="Partial answer"),
        ),
        response=ResponseSpec(format=_Answer),
    )
    out = _messages_for_transport(_attempt(req, WEAK_SCHEMA_TEST_MODEL))
    user = out[1]
    assert user.role == Role.USER
    assert isinstance(user.content, str)
    assert "Original question" in user.content
    assert "## Response schema" in user.content
    assert out[2].role == Role.ASSISTANT
    assert out[2].content == "Partial answer"
