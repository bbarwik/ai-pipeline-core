"""Conversation type detection helpers used by pipeline internals."""

from typing import Any

__all__ = [
    "is_conversation_instance",
    "is_conversation_type",
]

_CONVERSATION_MODULE = "ai_pipeline_core.llm.conversation"
_CONVERSATION_CLASS = "Conversation"


def is_conversation_type(value: Any) -> bool:
    """Return whether ``value`` is the Conversation class or a parameterized variant."""
    return isinstance(value, type) and value.__module__ == _CONVERSATION_MODULE and value.__name__ == _CONVERSATION_CLASS


def is_conversation_instance(value: Any) -> bool:
    """Return whether ``value`` looks like a Conversation instance without importing llm."""
    value_type = type(value)
    return (
        value_type.__module__ == _CONVERSATION_MODULE
        and value_type.__name__ == _CONVERSATION_CLASS
        and isinstance(getattr(value, "model", None), str)
        and hasattr(value, "context")
        and hasattr(value, "messages")
    )
