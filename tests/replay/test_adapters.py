"""Tests for replay adapter resolution and invocation."""

from typing import Any

import pytest

from ai_pipeline_core.llm.conversation import Conversation
from ai_pipeline_core.replay._adapters import _invoke_callable, resolve_callable
from tests.support.helpers import create_test_model_response


async def adapter_function(*, value: str) -> str:
    """Return a simple function result."""
    return f"function:{value}"


class AdapterClassMethod:
    """Classmethod target used by replay adapter tests."""

    @classmethod
    async def run(cls, value: str) -> str:
        return f"{cls.__name__}:{value}"


class AdapterInstanceMethod:
    """Instance method target used by replay adapter tests."""

    def __init__(self, prefix: str, ignored: str = "") -> None:
        self.prefix = prefix
        self.ignored = ignored

    async def run(self, value: str, extra: str = "") -> str:
        return f"{self.prefix}:{value}:{extra}"


@pytest.mark.asyncio
async def test_function_adapter_resolves_and_invokes() -> None:
    callable_obj = resolve_callable(
        f"function:{__name__}:adapter_function",
        receiver=None,
    )
    result = await _invoke_callable(callable_obj, {"value": "ok"})

    assert result == "function:ok"


@pytest.mark.asyncio
async def test_classmethod_adapter_resolves_and_invokes() -> None:
    callable_obj = resolve_callable(
        f"classmethod:{__name__}:AdapterClassMethod.run",
        receiver=None,
    )
    result = await _invoke_callable(callable_obj, {"value": "ok"})

    assert result == "AdapterClassMethod:ok"


@pytest.mark.asyncio
async def test_instance_method_adapter_constructs_receiver_and_filters_kwargs() -> None:
    callable_obj = resolve_callable(
        f"instance_method:{__name__}:AdapterInstanceMethod.run",
        receiver={"mode": "constructor_args", "value": {"prefix": "instance", "ignored": "ignored"}},
    )
    result = await _invoke_callable(callable_obj, {"value": "ok", "extra": "x", "round_index": 2})

    assert result == "instance:ok:x"


@pytest.mark.asyncio
async def test_decoded_method_adapter_splits_method_name_from_class_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_generate(messages: Any, **kwargs: Any) -> Any:
        _ = (messages, kwargs)
        return create_test_model_response(content="adapter response")

    monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

    conversation = Conversation(model="test-model")
    callable_obj = resolve_callable(
        "decoded_method:ai_pipeline_core.llm.conversation:Conversation.send",
        receiver={"mode": "decoded_state", "value": conversation},
    )
    result = await _invoke_callable(
        callable_obj,
        {
            "content": "hello",
            "tools": None,
            "tool_choice": None,
            "max_tool_rounds": 1,
            "purpose": "adapter",
            "expected_cost": None,
            "response_format": None,
        },
    )

    assert result.content == "adapter response"


def test_resolve_callable_rejects_unknown_target_kind() -> None:
    with pytest.raises(ValueError, match="is not supported"):
        resolve_callable(f"unknown:{__name__}:adapter_function", receiver=None)


def test_resolve_callable_rejects_tool_call_target() -> None:
    """resolve_callable raises ValueError for tool_call:... targets."""
    with pytest.raises(ValueError, match="tool_call span"):
        resolve_callable("tool_call:some.module:SomeTool", receiver=None)
