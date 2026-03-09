"""Replay execution engine.

Core execution logic for each payload type: conversation, task, and flow.
"""

import inspect
import typing
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from ai_pipeline_core._llm_core.model_response import ModelResponse
from ai_pipeline_core._llm_core.types import ModelOptions, RawToolCall, TokenUsage
from ai_pipeline_core.database._protocol import DatabaseReader
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.llm.conversation import Conversation, ToolResultMessage, UserMessage
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.pipeline._execution_context import pipeline_test_context
from ai_pipeline_core.pipeline._flow import PipelineFlow
from ai_pipeline_core.pipeline._task import PipelineTask

from ._deserialize import _import_by_path, _unwrap_prefect_callable, resolve_task_kwargs
from ._resolve import resolve_document_ref
from .types import ConversationReplay, DocumentRef, FlowReplay, TaskReplay

__all__: list[str] = []

logger = get_pipeline_logger(__name__)


async def _execute_pipeline_task(
    task_cls: type[PipelineTask],
    *,
    resolved_kwargs: dict[str, Any],
) -> Any:
    """Execute a class-based pipeline task inside replay test context."""
    with pipeline_test_context():
        handle = task_cls.run(**resolved_kwargs)
        return await handle


async def _resolve_context_docs(payload: ConversationReplay, database: DatabaseReader) -> list[Document]:
    """Resolve context document references from a ConversationReplay payload."""
    return [await resolve_document_ref(ref, database) for ref in payload.context]


async def _resolve_prompt_content(payload: ConversationReplay, database: DatabaseReader) -> str | Document | list[Document]:
    """Resolve the original Conversation.send() content from a replay payload."""
    if not payload.prompt_documents:
        return payload.prompt
    prompt_documents = [await resolve_document_ref(ref, database) for ref in payload.prompt_documents]
    if len(prompt_documents) == 1:
        return prompt_documents[0]
    return prompt_documents


async def _replay_history(conv: Conversation, payload: ConversationReplay, database: DatabaseReader) -> Conversation:
    """Reconstruct conversation history from replay payload entries."""
    for entry in payload.history:
        if entry.type == "user_text":
            conv = conv.model_copy(update={"messages": (*conv.messages, UserMessage(entry.text or ""))})
        elif entry.type in {"response", "assistant_text"}:
            # Reconstruct ModelResponse with tool_calls when present (preserves tool round history)
            if entry.type == "response" and entry.tool_calls:
                resp = ModelResponse(
                    content=entry.content or "",
                    parsed=entry.content or "",
                    model="",
                    usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                    tool_calls=tuple(RawToolCall(id=tc.id, function_name=tc.function_name, arguments=tc.arguments) for tc in entry.tool_calls),
                )
                conv = conv.model_copy(update={"messages": (*conv.messages, resp)})
            else:
                text = entry.content if entry.type == "response" else entry.text
                conv = conv.with_assistant_message(text or "")
        elif entry.type == "tool_result":
            tool_msg = ToolResultMessage(
                tool_call_id=entry.tool_call_id or "",
                function_name=entry.function_name or "",
                content=entry.content or "",
            )
            conv = conv.model_copy(update={"messages": (*conv.messages, tool_msg)})
        elif entry.type == "document" and entry.doc_ref:
            ref = DocumentRef.model_validate({
                "$doc_ref": entry.doc_ref,
                "class_name": entry.class_name or "",
                "name": entry.name or "",
            })
            conv = conv.with_document(await resolve_document_ref(ref, database))
    return conv


def _resolve_response_format(response_format_path: str | None) -> type[BaseModel] | None:
    """Import response_format class, returning None with a warning on failure."""
    if not response_format_path:
        return None
    try:
        response_format = _import_by_path(response_format_path)
    except (ImportError, AttributeError, ModuleNotFoundError, ValueError):
        logger.warning(
            "Could not import response_format '%s'. Falling back to unstructured send().",
            response_format_path,
        )
        return None
    if not isinstance(response_format, type) or not issubclass(response_format, BaseModel):
        logger.warning(
            "Replay response_format '%s' is not a BaseModel subclass. Falling back to unstructured send().",
            response_format_path,
        )
        return None
    return response_format


async def execute_conversation(payload: ConversationReplay, database: DatabaseReader) -> Any:
    """Execute a ConversationReplay payload."""
    context_docs = await _resolve_context_docs(payload, database)

    model_options = ModelOptions.model_validate(payload.model_options) if payload.model_options else None

    conv = Conversation(
        model=payload.model,
        model_options=model_options,
        enable_substitutor=payload.enable_substitutor,
        extract_result_tags=payload.extract_result_tags,
        include_date=payload.include_date,
        current_date=payload.current_date,
    )

    if context_docs:
        conv = conv.with_context(*context_docs)

    conv = await _replay_history(conv, payload, database)
    prompt_content = await _resolve_prompt_content(payload, database)

    response_format_cls = _resolve_response_format(payload.response_format)
    if response_format_cls is not None:
        return await conv.send_structured(prompt_content, response_format=response_format_cls, purpose=payload.purpose)
    return await conv.send(prompt_content, purpose=payload.purpose)


async def execute_task(payload: TaskReplay, database: DatabaseReader) -> Any:
    """Execute a TaskReplay payload (class-based PipelineTask or function task)."""
    fn = _import_by_path(payload.function_path)
    resolved_kwargs = await resolve_task_kwargs(payload.function_path, dict(payload.arguments), database)

    # Class-based pipeline tasks use Task.run(...) and return awaitable TaskHandle objects.
    if isinstance(fn, type) and issubclass(fn, PipelineTask):
        return await _execute_pipeline_task(fn, resolved_kwargs=resolved_kwargs)

    # Handle Prefect-wrapped functions: try .fn attribute first
    actual_fn = _unwrap_prefect_callable(fn)
    return await actual_fn(**resolved_kwargs)


async def execute_flow(payload: FlowReplay, database: DatabaseReader) -> Any:
    """Execute a FlowReplay payload."""
    fn = _import_by_path(payload.function_path)

    # Resolve document references
    resolved_docs_list: list[Document[Any]] = [await resolve_document_ref(ref, database) for ref in payload.documents]
    resolved_docs = tuple(resolved_docs_list)

    # Class-based PipelineFlow: instantiate with captured constructor params, then call .run()
    actual_fn: Callable[..., Any]
    if isinstance(fn, type) and issubclass(fn, PipelineFlow):
        flow_instance = fn(**payload.flow_params)
        actual_fn = flow_instance.run
    else:
        # Replay supports direct callable flows and Prefect-wrapped callables via .fn.
        actual_fn = _unwrap_prefect_callable(fn)

    # Resolve flow_options via type hints
    flow_options: Any = dict(payload.flow_options)
    hints = typing.get_type_hints(actual_fn)

    sig = inspect.signature(actual_fn)
    params = list(sig.parameters.keys())
    if len(params) >= 2:
        options_param = params[1]
        options_hint = hints.get(options_param)
        if options_hint is not None and isinstance(options_hint, type) and issubclass(options_hint, BaseModel):
            known_fields = set(options_hint.model_fields)
            filtered = {k: v for k, v in flow_options.items() if k in known_fields}
            flow_options = options_hint.model_validate(filtered)

    with pipeline_test_context(run_id=payload.run_id):
        return await actual_fn(resolved_docs, flow_options)
