"""Two-pass deserialization for replay payloads.

Pass 1: Recursively walk parsed YAML, replacing sentinel dicts with runtime objects.
Pass 2: Validate task kwargs against the target callable's type hints.
"""

import importlib
import inspect
import types
import typing
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Union, cast, get_args, get_origin

from pydantic import BaseModel

from ai_pipeline_core._llm_core.model_response import ModelResponse
from ai_pipeline_core._llm_core.types import ModelOptions, RawToolCall, TokenUsage
from ai_pipeline_core.documents._context import _suppress_document_registration
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.llm.conversation import AssistantMessage, Conversation, ToolResultMessage, UserMessage
from ai_pipeline_core.pipeline._task import PipelineTask
from ai_pipeline_core.replay.types import DocumentRef, HistoryEntry

from ._resolve import _find_document_class, resolve_document_ref

__all__ = ["resolve_doc_refs", "resolve_task_kwargs"]

_UNHANDLED = object()


def _unwrap_annotated(tp: Any) -> Any:
    """Strip Annotated metadata from a type annotation."""
    while get_origin(tp) is Annotated:
        args = get_args(tp)
        if not args:
            break
        tp = args[0]
    return tp


def _import_by_path(path: str) -> Any:
    """Import an object from ``module:qualname`` format."""
    module_path, qualname = path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    obj: Any = module
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


def _create_inline_document(data: dict[str, Any]) -> Document:
    """Create an ephemeral Document from inline content (no ``$doc_ref``)."""
    class_name = data["class_name"]
    name = data.get("name", "inline.txt")
    content = data["content"]
    doc_cls = _find_document_class(class_name)

    content_bytes = content.encode("utf-8") if isinstance(content, str) else content
    with _suppress_document_registration():
        return doc_cls(
            name=name,
            content=content_bytes,
            description=data.get("description"),
        )


def _deserialize_history_entry(entry: HistoryEntry, store_base: Path) -> Any:
    """Deserialize one Conversation history entry."""
    if entry.type == "user_text":
        return UserMessage(entry.text or "")
    if entry.type == "assistant_text":
        return AssistantMessage(entry.text or "")
    if entry.type == "tool_result":
        return ToolResultMessage(
            tool_call_id=entry.tool_call_id or "",
            function_name=entry.function_name or "",
            content=entry.content or "",
        )
    if entry.type == "response":
        tool_calls = ()
        if entry.tool_calls:
            tool_calls = tuple(
                RawToolCall(id=tool_call.id, function_name=tool_call.function_name, arguments=tool_call.arguments) for tool_call in entry.tool_calls
            )
        return ModelResponse(
            content=entry.content or "",
            parsed=entry.content or "",
            model="",
            usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            tool_calls=tool_calls,
        )
    if entry.type == "document" and entry.doc_ref:
        ref = DocumentRef.model_validate({
            "$doc_ref": entry.doc_ref,
            "class_name": entry.class_name or "",
            "name": entry.name or "",
        })
        return resolve_document_ref(ref, store_base)
    raise TypeError(f"Unsupported conversation history entry: {entry!r}")


def _deserialize_conversation(data: dict[str, Any], store_base: Path) -> Conversation[Any]:
    """Deserialize a Conversation sentinel dict used in task replay arguments."""
    payload = cast(dict[str, Any], data["$conversation"])
    model_options = ModelOptions.model_validate(payload["model_options"]) if payload.get("model_options") else None
    context = tuple(resolve_document_ref(DocumentRef.model_validate(ref), store_base) for ref in payload.get("context", []))
    history_entries = [HistoryEntry.model_validate(entry) for entry in payload.get("history", [])]
    messages = tuple(_deserialize_history_entry(entry, store_base) for entry in history_entries)
    return Conversation(
        model=payload["model"],
        context=context,
        messages=messages,
        model_options=model_options,
        enable_substitutor=payload.get("enable_substitutor", True),
        extract_result_tags=payload.get("extract_result_tags", False),
        include_date=payload.get("include_date", True),
        current_date=payload.get("current_date"),
    )


def resolve_doc_refs(data: Any, store_base: Path) -> Any:
    """Recursively walk parsed YAML data, replacing replay sentinel dicts with runtime objects."""
    if isinstance(data, dict):
        mapping = cast(dict[str, Any], data)
        if "$doc_ref" in mapping:
            ref = DocumentRef.model_validate(mapping)
            return resolve_document_ref(ref, store_base)
        if "$conversation" in mapping:
            return _deserialize_conversation(mapping, store_base)
        if "class_name" in mapping and "content" in mapping and "$doc_ref" not in mapping:
            return _create_inline_document(mapping)
        return {key: resolve_doc_refs(value, store_base) for key, value in mapping.items()}
    if isinstance(data, list):
        return [resolve_doc_refs(item, store_base) for item in cast(list[Any], data)]
    return data


def _is_basemodel_subclass(hint: Any) -> bool:
    """Check if a type hint is a concrete BaseModel subclass (excluding Document)."""
    return isinstance(hint, type) and issubclass(hint, BaseModel) and not issubclass(hint, Document)


def _is_union_annotation(annotation: Any) -> bool:
    """Check whether an annotation is ``Union`` or ``X | Y``."""
    unwrapped = _unwrap_annotated(annotation)
    origin = get_origin(unwrapped)
    return origin is Union or isinstance(unwrapped, types.UnionType)


def _coerce_union_value(value: Any, annotation: Any) -> Any | object:
    if not _is_union_annotation(annotation):
        return _UNHANDLED

    for branch in get_args(annotation):
        try:
            return _coerce_value_for_annotation(value, branch)
        except (TypeError, ValueError):
            continue
    return value


def _coerce_list_value(value: Any, annotation: Any) -> Any | object:
    if get_origin(annotation) is not list:
        return _UNHANDLED
    if not isinstance(value, list):
        return value

    args = get_args(annotation)
    if len(args) != 1:
        return value

    item_annotation = args[0]
    return [_coerce_value_for_annotation(item, item_annotation) for item in cast(list[Any], value)]


def _coerce_tuple_value(value: Any, annotation: Any) -> Any | object:
    if get_origin(annotation) is not tuple:
        return _UNHANDLED
    if not isinstance(value, (list, tuple)):
        return value

    args = get_args(annotation)
    items = cast(list[Any] | tuple[Any, ...], value)
    if len(args) == 2 and args[1] is Ellipsis:
        return tuple(_coerce_value_for_annotation(item, args[0]) for item in items)
    if len(items) != len(args):
        raise ValueError(f"Fixed-length tuple expects {len(args)} items but replay data has {len(items)} for annotation {annotation!r}")
    return tuple(_coerce_value_for_annotation(item, item_annotation) for item, item_annotation in zip(items, args, strict=True))


def _coerce_dict_value(value: Any, annotation: Any) -> Any | object:
    if get_origin(annotation) is not dict:
        return _UNHANDLED
    if not isinstance(value, dict):
        return value

    args = get_args(annotation)
    if len(args) != 2:
        return value

    key_annotation, value_annotation = args
    typed_items = cast(dict[Any, Any], value)
    return {_coerce_value_for_annotation(key, key_annotation): _coerce_value_for_annotation(item, value_annotation) for key, item in typed_items.items()}


def _coerce_basemodel_value(value: Any, annotation: Any) -> Any | object:
    if not _is_basemodel_subclass(annotation):
        return _UNHANDLED
    if isinstance(value, annotation):
        return value
    if isinstance(value, dict):
        return annotation.model_validate(value)
    return value


def _coerce_enum_value(value: Any, annotation: Any) -> Any | object:
    if not (isinstance(annotation, type) and issubclass(annotation, Enum)):
        return _UNHANDLED
    if isinstance(value, annotation):
        return value
    return annotation(value)


def _coerce_value_for_annotation(value: Any, annotation: Any) -> Any:
    """Coerce replay-deserialized values into their annotated runtime types."""
    unwrapped = _unwrap_annotated(annotation)
    if unwrapped is type(None) or unwrapped is None:
        return None
    for coercer in (_coerce_union_value, _coerce_list_value, _coerce_tuple_value, _coerce_dict_value, _coerce_basemodel_value, _coerce_enum_value):
        coerced = coercer(value, unwrapped)
        if coerced is not _UNHANDLED:
            return coerced
    return value


def _task_target(function_path: str) -> Any:
    """Resolve the callable whose signature should be used for task replay kwargs."""
    fn = _import_by_path(function_path)
    if isinstance(fn, type) and issubclass(fn, PipelineTask):
        return fn._run_spec.user_run
    return cast(Any, getattr(fn, "fn", fn))


def resolve_task_kwargs(function_path: str, raw_kwargs: dict[str, Any], store_base: Path) -> dict[str, Any]:
    """Resolve task kwargs: replay sentinels -> runtime objects -> annotated model coercion."""
    target = _task_target(function_path)
    hints = typing.get_type_hints(target, include_extras=True)
    resolved = cast(dict[str, Any], resolve_doc_refs(raw_kwargs, store_base))

    signature = inspect.signature(target)
    parameter_names = [name for name in signature.parameters if name != "cls"]
    for key in parameter_names:
        if key not in resolved:
            continue
        hint = hints.get(key)
        if hint is None:
            continue
        resolved[key] = _coerce_value_for_annotation(resolved[key], hint)

    return resolved
