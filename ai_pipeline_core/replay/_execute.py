"""Replay execution engine.

Core execution logic for each payload type: conversation, task, and flow.
"""

import inspect
import typing
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ai_pipeline_core._llm_core.model_options import ModelOptions
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.llm.conversation import Conversation, _UserMessage
from ai_pipeline_core.logging import get_pipeline_logger

from ._deserialize import _import_by_path, resolve_task_kwargs
from ._resolve import resolve_document_ref
from .types import ConversationReplay, DocumentRef, FlowReplay, TaskReplay

__all__: list[str] = []

logger = get_pipeline_logger(__name__)


def _resolve_context_docs(payload: ConversationReplay, store_base: Path) -> list[Document]:
    """Resolve context document references from a ConversationReplay payload."""
    return [resolve_document_ref(ref, store_base) for ref in payload.context]


def _replay_history(conv: Conversation, payload: ConversationReplay, store_base: Path) -> Conversation:
    """Reconstruct conversation history from replay payload entries."""
    for entry in payload.history:
        if entry.type == "user_text":
            conv = conv.model_copy(update={"messages": (*conv.messages, _UserMessage(entry.text or ""))})
        elif entry.type in {"response", "assistant_text"}:
            text = entry.content if entry.type == "response" else entry.text
            conv = conv.with_assistant_message(text or "")
        elif entry.type == "document" and entry.doc_ref:
            ref = DocumentRef.model_validate({
                "$doc_ref": entry.doc_ref,
                "class_name": entry.class_name or "",
                "name": entry.name or "",
            })
            conv = conv.with_document(resolve_document_ref(ref, store_base))
    return conv


def _resolve_response_format(response_format_path: str | None) -> type | None:
    """Import response_format class, returning None with a warning on failure."""
    if not response_format_path:
        return None
    try:
        return _import_by_path(response_format_path)
    except (ImportError, AttributeError, ModuleNotFoundError, ValueError):
        logger.warning(
            "Could not import response_format '%s'. Falling back to unstructured send().",
            response_format_path,
        )
        return None


async def execute_conversation(payload: ConversationReplay, store_base: Path) -> Any:
    """Execute a ConversationReplay payload."""
    context_docs = _resolve_context_docs(payload, store_base)

    model_options = ModelOptions.model_validate(payload.model_options) if payload.model_options else None

    conv = Conversation(
        model=payload.model,
        model_options=model_options,
        enable_substitutor=payload.enable_substitutor,
        extract_result_tags=payload.extract_result_tags,
    )

    if context_docs:
        conv = conv.with_context(*context_docs)

    conv = _replay_history(conv, payload, store_base)

    response_format_cls = _resolve_response_format(payload.response_format)
    if response_format_cls is not None:
        return await conv.send_structured(payload.prompt, response_format=response_format_cls, purpose=payload.purpose)
    return await conv.send(payload.prompt, purpose=payload.purpose)


async def execute_task(payload: TaskReplay, store_base: Path) -> Any:
    """Execute a TaskReplay payload."""
    resolved_kwargs = resolve_task_kwargs(payload.function_path, dict(payload.arguments), store_base)
    fn = _import_by_path(payload.function_path)

    # Handle Prefect-wrapped functions: try .fn attribute first
    actual_fn = getattr(fn, "fn", fn)
    return await actual_fn(**resolved_kwargs)


async def execute_flow(payload: FlowReplay, store_base: Path) -> Any:
    """Execute a FlowReplay payload."""
    fn = _import_by_path(payload.function_path)
    actual_fn = getattr(fn, "fn", fn)

    # Resolve document references
    resolved_docs = [resolve_document_ref(ref, store_base) for ref in payload.documents]

    # Resolve flow_options via type hints
    flow_options: Any = dict(payload.flow_options)
    hints = typing.get_type_hints(actual_fn)

    # Find the flow_options parameter (third parameter)
    sig = inspect.signature(actual_fn)
    params = list(sig.parameters.keys())
    if len(params) >= 3:
        options_param = params[2]
        options_hint = hints.get(options_param)
        if options_hint is not None and isinstance(options_hint, type) and issubclass(options_hint, BaseModel):
            # Filter to known fields — deployment CLI adds extra fields (working_directory, start, end, etc.)
            # that aren't part of the user's FlowOptions subclass
            known_fields = set(options_hint.model_fields)
            filtered = {k: v for k, v in flow_options.items() if k in known_fields}
            flow_options = options_hint.model_validate(filtered)

    return await actual_fn(payload.run_id, resolved_docs, flow_options)
