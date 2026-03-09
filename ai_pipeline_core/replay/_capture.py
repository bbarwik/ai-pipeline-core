"""Serialization helpers for replay payload capture.

Converts runtime Python objects (Documents, BaseModels, Enums) into
JSON-serializable dicts suitable for YAML replay payloads.
"""

from collections.abc import Sequence
from enum import Enum
from typing import Any, cast

from pydantic import BaseModel

from ai_pipeline_core._llm_core.model_response import ModelResponse
from ai_pipeline_core._llm_core.types import ModelOptions
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.llm.conversation import AssistantMessage, Conversation, ConversationContent, ToolResultMessage, UserMessage
from ai_pipeline_core.replay.types import ToolCallEntry

__all__ = ["build_conversation_replay_payload", "serialize_kwargs", "serialize_prior_messages"]


def _serialize_document_ref(doc: Document[Any]) -> dict[str, str]:
    """Serialize a Document into a replay reference dict."""
    return {
        "$doc_ref": doc.sha256,
        "class_name": type(doc).__name__,
        "name": doc.name,
    }


def _serialize_conversation(value: Conversation[Any]) -> dict[str, Any]:
    """Serialize a Conversation for use as a task argument."""
    model_options: dict[str, Any] = {}
    if value.model_options is not None:
        model_options = value.model_options.model_dump(exclude_defaults=True)

    return {
        "$conversation": {
            "model": value.model,
            "model_options": model_options,
            "context": [_serialize_document_ref(doc) for doc in value.context],
            "history": serialize_prior_messages(value.messages),
            "enable_substitutor": value.enable_substitutor,
            "extract_result_tags": value.extract_result_tags,
            "include_date": value.include_date,
            "current_date": value.current_date,
        }
    }


def _serialize_send_content(content: ConversationContent) -> tuple[str, tuple[dict[str, str], ...]]:
    """Serialize Conversation.send() content into replay-safe prompt fields."""
    if isinstance(content, str):
        return content, ()
    if isinstance(content, Document):
        return "", (_serialize_document_ref(content),)
    documents = cast("Sequence[Document[Any]]", content)
    return "", tuple(_serialize_document_ref(doc) for doc in documents)


def _serialize_value(value: Any) -> Any:
    """Serialize a single value for replay payload.

    Recursively handles list, tuple, and dict containers so that nested
    Documents, BaseModels, and Enums are properly serialized.
    """
    result: Any = value
    if isinstance(value, Conversation):
        result = _serialize_conversation(cast(Conversation[Any], value))
    elif isinstance(value, Document):
        result = _serialize_document_ref(cast(Document[Any], value))
    elif isinstance(value, BaseModel):
        result = value.model_dump(mode="json")
    elif isinstance(value, Enum):
        result = value.value
    elif isinstance(value, (list, tuple)):
        result = [_serialize_value(item) for item in cast("Sequence[Any]", value)]
    elif isinstance(value, dict):
        result = {key: _serialize_value(val) for key, val in cast(dict[str, Any], value).items()}
    return result


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
            result.append({"type": "document", **_serialize_document_ref(cast(Document[Any], msg))})
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
    include_date: bool,
    current_date: str | None,
) -> dict[str, Any]:
    """Build a replay payload dict capturing the full conversation state."""
    # Serialize context as document references
    ctx_refs = [_serialize_document_ref(document) for document in context]
    prompt, prompt_documents = _serialize_send_content(content)

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
        "prompt_documents": prompt_documents,
        "response_format": rf_path,
        "purpose": purpose,
        "context": ctx_refs,
        "history": serialize_prior_messages(messages),
        "enable_substitutor": enable_substitutor,
        "extract_result_tags": extract_result_tags,
        "include_date": include_date,
        "current_date": current_date,
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
