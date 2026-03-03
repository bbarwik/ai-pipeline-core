"""Serialization helpers for replay payload capture.

Converts runtime Python objects (Documents, BaseModels, Enums) into
JSON-serializable dicts suitable for YAML replay payloads.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel

from ai_pipeline_core._llm_core.model_response import ModelResponse
from ai_pipeline_core._llm_core.types import ModelOptions
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.llm.conversation import AssistantMessage, ConversationContent, ToolResultMessage, UserMessage
from ai_pipeline_core.replay.types import ToolCallEntry

__all__ = ["build_conversation_replay_payload", "serialize_kwargs", "serialize_prior_messages"]


def _serialize_value(value: Any) -> Any:
    """Serialize a single value for replay payload.

    Recursively handles list, tuple, and dict containers so that nested
    Documents, BaseModels, and Enums are properly serialized.
    """
    if isinstance(value, Document):
        return {
            "$doc_ref": value.sha256,
            "class_name": type(value).__name__,
            "name": value.name,
        }
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        items: list[Any] = list(value)
        return [_serialize_value(item) for item in items]
    if isinstance(value, dict):
        entries: dict[str, Any] = dict(value)
        return {key: _serialize_value(val) for key, val in entries.items()}
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

    UserMessage -> {"type": "user_text", "text": ...}
    AssistantMessage -> {"type": "assistant_text", "text": ...}
    ModelResponse -> {"type": "response", "content": ...} (with optional tool_calls)
    ToolResultMessage -> {"type": "tool_result", "tool_call_id": ..., "function_name": ..., "content": ...}
    Document -> {"type": "document", "$doc_ref": ..., "class_name": ..., "name": ...}
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, UserMessage):
            result.append({"type": "user_text", "text": msg.text})
        elif isinstance(msg, AssistantMessage):
            result.append({"type": "assistant_text", "text": msg.text})
        elif isinstance(msg, ToolResultMessage):
            result.append({
                "type": "tool_result",
                "tool_call_id": msg.tool_call_id,
                "function_name": msg.function_name,
                "content": msg.content,
            })
        elif isinstance(msg, ModelResponse):
            entry: dict[str, Any] = {"type": "response", "content": msg.content}
            if msg.has_tool_calls:
                entry["tool_calls"] = [ToolCallEntry(id=tc.id, function_name=tc.function_name, arguments=tc.arguments).model_dump() for tc in msg.tool_calls]
            result.append(entry)
        elif isinstance(msg, Document):
            result.append({
                "type": "document",
                "$doc_ref": msg.sha256,
                "class_name": msg.__class__.__name__,
                "name": msg.name,
            })
    return result


def build_conversation_replay_payload(
    *,
    content: ConversationContent,
    response_format: type[BaseModel] | None,
    purpose: str | None,
    response: ModelResponse[Any],
    context: tuple[Document, ...],
    model: str,
    model_options: ModelOptions | None,
    messages: tuple[Any, ...],
    enable_substitutor: bool,
    extract_result_tags: bool,
) -> dict[str, Any]:
    """Build a replay payload dict capturing the full conversation state."""
    # Serialize context as document references
    ctx_refs = [{"$doc_ref": d.sha256, "class_name": type(d).__name__, "name": d.name} for d in context]

    # Extract prompt text
    prompt: str
    if isinstance(content, str):
        prompt = content
    elif isinstance(content, Document):
        prompt = content.text if content.is_text else f"[Document: {content.name}]"
    else:
        prompt = "\n".join(doc.text if doc.is_text else f"[Document: {doc.name}]" for doc in content)

    # Serialize response format as importable path
    rf_path: str | None = None
    if response_format is not None:
        rf_path = f"{response_format.__module__}:{response_format.__qualname__}"

    # Build model options dict (use model_options, not effective_options,
    # because effective_options may include substitutor system_prompt that
    # would be re-applied during replay execution)
    options_dict: dict[str, Any] = {}
    if model_options is not None:
        options_dict = model_options.model_dump(exclude_defaults=True)

    payload: dict[str, Any] = {
        "version": 1,
        "payload_type": "conversation",
        "model": model,
        "model_options": options_dict,
        "prompt": prompt,
        "response_format": rf_path,
        "purpose": purpose,
        "context": ctx_refs,
        "history": serialize_prior_messages(messages),
        "enable_substitutor": enable_substitutor,
        "extract_result_tags": extract_result_tags,
        "original": {
            "cost": response.cost,
            "tokens": {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
                "cached": response.usage.cached_tokens,
                "reasoning": response.usage.reasoning_tokens,
            },
        },
    }
    return payload
