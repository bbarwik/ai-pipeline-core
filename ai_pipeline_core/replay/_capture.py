"""Serialization helpers for replay payload capture.

Converts runtime Python objects (Documents, BaseModels, Enums) into
JSON-serializable dicts suitable for YAML replay payloads.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel

from ai_pipeline_core._llm_core.model_response import ModelResponse
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.llm.conversation import _AssistantMessage, _UserMessage

__all__ = ["serialize_kwargs", "serialize_prior_messages"]


def _serialize_value(value: Any) -> Any:
    """Serialize a single value for replay payload.

    Recursively handles list, tuple, and dict containers so that nested
    Documents, BaseModels, and Enums are properly serialized.
    """
    if isinstance(value, Document):
        return {
            "$doc_ref": value.sha256,
            "class_name": value.__class__.__name__,
            "name": value.name,
        }
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(val) for key, val in value.items()}
    return value


def serialize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Serialize task/flow keyword arguments for replay payload.

    Document -> $doc_ref reference dict
    BaseModel -> model_dump(mode="json")
    Enum -> .value
    Primitives -> pass through
    """
    return {key: _serialize_value(value) for key, value in kwargs.items()}


def serialize_prior_messages(messages: tuple[Any, ...]) -> list[dict[str, Any]]:
    """Serialize conversation message history for replay payload.

    _UserMessage -> {"type": "user_text", "text": ...}
    _AssistantMessage -> {"type": "assistant_text", "text": ...}
    ModelResponse -> {"type": "response", "content": ...}
    Document -> {"type": "document", "$doc_ref": ..., "class_name": ..., "name": ...}
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, _UserMessage):
            result.append({"type": "user_text", "text": msg.text})
        elif isinstance(msg, _AssistantMessage):
            result.append({"type": "assistant_text", "text": msg.text})
        elif isinstance(msg, ModelResponse):
            result.append({"type": "response", "content": msg.content})
        elif isinstance(msg, Document):
            result.append({
                "type": "document",
                "$doc_ref": msg.sha256,
                "class_name": msg.__class__.__name__,
                "name": msg.name,
            })
    return result
