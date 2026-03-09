"""Class-based pipeline task runtime.

Rules enforced at class definition time for concrete tasks:
1. Subclasses must define ``@classmethod async def run(cls, ...)`` or inherit one from a parent task.
2. Every ``run()`` parameter after ``cls`` must use a supported annotation.
3. ``run()`` must return ``Document``, ``None``, ``list[Document]``, ``tuple[Document, ...]``, or unions of those shapes.
4. Bare ``Document`` is forbidden in both inputs and outputs; use concrete subclasses.
5. Class names must not start with ``Test``.
6. ``estimated_minutes`` must be >= 1, ``retries`` >= 0, and ``timeout_seconds`` positive when set.

Classes that explicitly declare ``_abstract_task = True`` skip definition-time validation.
Concrete subclasses of those abstract bases are validated normally.

Runtime behavior:
1. ``Task.run(...)`` returns an awaitable ``TaskHandle``.
2. ``await Task.run(...)`` executes the full lifecycle: retries, timeout, events, persistence, summaries, and replay capture.
3. Tasks must run inside an active pipeline execution context (or ``pipeline_test_context()`` in tests).
"""

import asyncio
import inspect
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import update_wrapper
from types import MappingProxyType
from typing import Any, ClassVar, cast
from uuid import UUID, uuid4

from pydantic import BaseModel

from ai_pipeline_core.database import NULL_PARENT, BlobRecord, DocumentRecord, ExecutionNode, NodeKind, NodeStatus
from ai_pipeline_core.deployment._types import DocumentRef, TaskCompletedEvent, TaskFailedEvent, TaskStartedEvent
from ai_pipeline_core.documents import Document, DocumentSha256, RunScope
from ai_pipeline_core.documents._context import TaskContext, _TaskDocumentContext, reset_task_context, set_task_context
from ai_pipeline_core.documents._hashing import compute_content_sha256
from ai_pipeline_core.documents.document import get_inline_summary, pop_inline_summary
from ai_pipeline_core.llm.conversation import Conversation
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.pipeline._execution_context import (
    ConversationTurnData,
    ExecutionContext,
    FlowFrame,
    TaskFrame,
    get_execution_context,
    record_lifecycle_event,
    reset_conversation_turns,
    reset_execution_context,
    set_conversation_turns,
    set_execution_context,
)
from ai_pipeline_core.pipeline._parallel import TaskHandle
from ai_pipeline_core.pipeline._type_validation import (
    resolve_type_hints,
    validate_task_argument_value,
    validate_task_input_annotation,
    validate_task_return_annotation,
)
from ai_pipeline_core.replay._capture import serialize_kwargs

logger = get_pipeline_logger(__name__)

MILLISECONDS_PER_SECOND = 1000

EVENT_PUBLISH_EXCEPTIONS = (OSError, RuntimeError, ValueError, TypeError)
TASK_EXECUTION_EXCEPTIONS = (Exception, asyncio.CancelledError)
REPLAY_CAPTURE_EXCEPTIONS = (Exception,)
RETRY_CAPTURE_EXCEPTIONS = (Exception,)

DATABASE_EXCEPTIONS = (Exception,)

__all__ = ["PipelineTask"]


async def _insert_db_node_safe(database: Any, node: ExecutionNode) -> None:
    """Insert an execution node, logging warnings on failure."""
    if database is None:
        return
    try:
        await database.insert_node(node)
    except DATABASE_EXCEPTIONS as exc:
        logger.warning("Database node insert failed for %s %s: %s", node.node_kind, node.node_id, exc)


async def _update_db_node_safe(database: Any, node_id: UUID, **updates: Any) -> None:
    """Update an execution node, logging warnings on failure."""
    if database is None:
        return
    try:
        await database.update_node(node_id, **updates)
    except DATABASE_EXCEPTIONS as exc:
        logger.warning("Database node update failed for %s: %s", node_id, exc)


async def _persist_documents_to_database(
    documents: Sequence[Document],
    database: Any,
    deployment_id: UUID | None,
    run_scope: RunScope,
    producing_node_id: UUID | None,
) -> None:
    """Persist documents and blobs to the new database backend (fire-and-forget)."""
    if database is None or deployment_id is None:
        return

    doc_records: list[DocumentRecord] = []
    blob_records: list[BlobRecord] = []
    seen_content_sha256s: set[str] = set()

    for doc in documents:
        content_sha = compute_content_sha256(doc.content)
        doc_records.append(
            DocumentRecord(
                document_sha256=doc.sha256,
                content_sha256=content_sha,
                deployment_id=deployment_id,
                producing_node_id=producing_node_id,
                document_type=type(doc).__name__,
                name=doc.name,
                run_scope=run_scope,
                description=doc.description or "",
                mime_type=doc.mime_type,
                size_bytes=doc.size,
                publicly_visible=getattr(type(doc), "publicly_visible", False),
                derived_from=doc.derived_from,
                triggered_by=doc.triggered_by,
                attachment_names=tuple(att.name for att in doc.attachments),
                attachment_descriptions=tuple(att.description or "" for att in doc.attachments),
                attachment_sha256s=tuple(compute_content_sha256(att.content) for att in doc.attachments),
                attachment_mime_types=tuple(att.mime_type for att in doc.attachments),
                attachment_sizes=tuple(att.size for att in doc.attachments),
            )
        )
        if content_sha not in seen_content_sha256s:
            seen_content_sha256s.add(content_sha)
            blob_records.append(
                BlobRecord(
                    content_sha256=content_sha,
                    content=doc.content,
                    size_bytes=len(doc.content),
                )
            )
        # Also persist attachment blobs
        for att in doc.attachments:
            att_sha = compute_content_sha256(att.content)
            if att_sha not in seen_content_sha256s:
                seen_content_sha256s.add(att_sha)
                blob_records.append(
                    BlobRecord(
                        content_sha256=att_sha,
                        content=att.content,
                        size_bytes=len(att.content),
                    )
                )

    try:
        if blob_records:
            await database.save_blob_batch(blob_records)
        if doc_records:
            await database.save_document_batch(doc_records)
    except DATABASE_EXCEPTIONS as exc:
        logger.warning("Database document persistence failed: %s", exc)


async def _create_conversation_turn_nodes(
    turns: list[ConversationTurnData],
    task_node_id: UUID,
    execution_ctx: ExecutionContext,
) -> None:
    """Create conversation and conversation_turn nodes from captured turn data."""
    database = execution_ctx.database
    if database is None or execution_ctx.deployment_id is None:
        return

    # Group turns by conversation_id (stable UUID) so two Conversation objects
    # with the same purpose produce separate nodes, and a single Conversation
    # that changes purpose stays unified.
    conversations: dict[str, list[tuple[int, ConversationTurnData]]] = {}
    for idx, turn in enumerate(turns):
        conversations.setdefault(turn.conversation_id, []).append((idx, turn))

    for conv_seq, conv_turns in enumerate(conversations.values(), start=1):
        conv_node_id = uuid4()
        conv_display_name = conv_turns[0][1].conversation_name
        conversation_status = NodeStatus.FAILED if any(turn.status == "failed" for _, turn in conv_turns) else NodeStatus.COMPLETED

        # Create conversation node
        conv_node = ExecutionNode(
            node_id=conv_node_id,
            node_kind=NodeKind.CONVERSATION,
            deployment_id=execution_ctx.deployment_id,
            root_deployment_id=execution_ctx.root_deployment_id or execution_ctx.deployment_id,
            parent_node_id=task_node_id,
            run_id=execution_ctx.run_id,
            run_scope=RunScope(str(execution_ctx.run_scope)),
            deployment_name=execution_ctx.deployment_name,
            name=conv_display_name,
            sequence_no=conv_seq,
            task_id=task_node_id,
            status=conversation_status,
            started_at=conv_turns[0][1].started_at,
            ended_at=conv_turns[-1][1].ended_at,
            turn_count=len(conv_turns),
            context_document_shas=conv_turns[0][1].context_document_shas,
            payload={
                "model_options": conv_turns[0][1].model_options_json,
                "response_format_class": conv_turns[0][1].response_format_class,
                "purpose": conv_display_name,
            },
        )
        await _insert_db_node_safe(database, conv_node)

        # Create turn nodes
        for turn_seq, (_, turn) in enumerate(conv_turns):
            turn_node_id = uuid4()
            turn_node = ExecutionNode(
                node_id=turn_node_id,
                node_kind=NodeKind.CONVERSATION_TURN,
                deployment_id=execution_ctx.deployment_id,
                root_deployment_id=execution_ctx.root_deployment_id or execution_ctx.deployment_id,
                parent_node_id=conv_node_id,
                run_id=execution_ctx.run_id,
                run_scope=RunScope(str(execution_ctx.run_scope)),
                deployment_name=execution_ctx.deployment_name,
                name=f"{conv_display_name}:turn-{turn_seq}",
                sequence_no=turn_seq,
                task_id=task_node_id,
                conversation_id=conv_node_id,
                model=turn.model,
                cost_usd=turn.cost_usd,
                tokens_input=turn.tokens_input,
                tokens_output=turn.tokens_output,
                tokens_cache_read=turn.tokens_cache_read,
                tokens_reasoning=turn.tokens_reasoning,
                status=NodeStatus.FAILED if turn.status == "failed" else NodeStatus.COMPLETED,
                started_at=turn.started_at,
                ended_at=turn.ended_at,
                error_type=turn.error_type,
                error_message=turn.error_message,
                payload={
                    "prompt_content": turn.prompt_content,
                    "response_content": turn.response_content,
                    "reasoning_content": turn.reasoning_content,
                    "response_format_class": turn.response_format_class,
                    "response_id": turn.response_id,
                    "citations_json": turn.citations_json,
                    "first_token_time": turn.first_token_time,
                    "time_taken": turn.time_taken,
                    "replay_payload": json.loads(turn.replay_payload_json) if turn.replay_payload_json else {},
                },
            )
            await _insert_db_node_safe(database, turn_node)


def _aggregate_conversation_turn_metrics(turns: Sequence[ConversationTurnData]) -> dict[str, int | float]:
    """Aggregate LLM usage captured during a task for task-level execution metrics."""
    return {
        "cost_usd": sum(turn.cost_usd for turn in turns),
        "tokens_input": sum(turn.tokens_input for turn in turns),
        "tokens_output": sum(turn.tokens_output for turn in turns),
        "tokens_cache_read": sum(turn.tokens_cache_read for turn in turns),
        "tokens_reasoning": sum(turn.tokens_reasoning for turn in turns),
        "turn_count": len(turns),
    }


def _consume_log_summary(execution_ctx: ExecutionContext, node_id: UUID) -> dict[str, int | str]:
    """Return and clear lightweight log counters for a terminal task node."""
    if execution_ctx.log_buffer is None:
        return {"total": 0, "warnings": 0, "errors": 0, "last_error": ""}
    return execution_ctx.log_buffer.consume_summary(node_id)


@dataclass(frozen=True, slots=True)
class _TaskRunSpec:
    """Validated task run metadata stored on the task class."""

    user_run: Callable[..., Awaitable[Any]]
    signature: inspect.Signature
    hints: Mapping[str, Any]
    input_document_types: tuple[type[Document], ...]
    output_document_types: tuple[type[Document], ...]


def _class_name(value: Any) -> str:
    return getattr(value, "__name__", str(value))


def _ordered_unique_document_types(document_types: list[type[Document]]) -> tuple[type[Document], ...]:
    ordered: dict[str, type[Document]] = {}
    for document_type in document_types:
        ordered.setdefault(document_type.__name__, document_type)
    return tuple(ordered.values())


async def _maybe_with_timeout[T](timeout_seconds: int | None, call: Callable[[], Awaitable[T]]) -> T:
    if timeout_seconds is None:
        return await call()
    async with asyncio.timeout(timeout_seconds):
        return await call()


def _collect_documents(value: Any, collected_docs: list[Document]) -> None:
    """Collect Documents nested in supported task input values."""
    if isinstance(value, Document):
        collected_docs.append(cast(Document[Any], value))
        return
    if isinstance(value, Conversation):
        for document in value.context:
            _collect_documents(document, collected_docs)
        for message in value.messages:
            _collect_documents(message, collected_docs)
        return
    if isinstance(value, (list, tuple)):
        for item in cast(Sequence[Any], value):
            _collect_documents(item, collected_docs)
        return
    if isinstance(value, dict):
        for item in cast(dict[Any, Any], value).values():
            _collect_documents(item, collected_docs)
        return
    if isinstance(value, BaseModel):
        for field_name in type(value).model_fields:
            _collect_documents(getattr(value, field_name), collected_docs)


def _input_documents(arguments: Mapping[str, Any]) -> tuple[Document, ...]:
    """Flatten Document inputs from task arguments while preserving order."""
    collected_docs: list[Document] = []
    for value in arguments.values():
        _collect_documents(value, collected_docs)
    deduped: dict[str, Document] = {}
    for document in collected_docs:
        deduped.setdefault(document.sha256, document)
    return tuple(deduped.values())


class PipelineTask:
    """Base class for pipeline tasks.

    Tasks are stateless units of work. Define ``run`` as a **@classmethod** because tasks
    carry no per-invocation instance state — all inputs arrive as arguments, all outputs
    are returned documents. The framework wraps ``run`` with retries, persistence,
    and event emission automatically.

    Set ``_abstract_task = True`` on an intermediate base class to skip ``run()``
    validation on that class. Concrete subclasses do not inherit that skip; they must
    define ``run()`` or inherit a validated implementation from a non-abstract parent.

    Minimal example::

        class SummarizeTask(PipelineTask):
            @classmethod
            async def run(cls, documents: tuple[ArticleDocument, ...]) -> tuple[SummaryDocument, ...]:
                conv = Conversation(model="gemini-3-flash").with_context(documents[0])
                conv = await conv.send("Summarize this article.")
                return (SummaryDocument.derive(from_documents=(documents[0],), name="summary.md", content=conv.content),)

    Calling ``await SummarizeTask.run((doc,))`` dispatches the full lifecycle. Calling without
    ``await`` returns a ``TaskHandle`` for parallel execution via ``collect_tasks``.
    """

    name: ClassVar[str]
    estimated_minutes: ClassVar[float] = 1.0
    retries: ClassVar[int] = 0
    retry_delay_seconds: ClassVar[int] = 20
    timeout_seconds: ClassVar[int | None] = None
    _abstract_task: ClassVar[bool] = False
    expected_cost: ClassVar[float | None] = None

    input_document_types: ClassVar[list[type[Document]]] = []
    output_document_types: ClassVar[list[type[Document]]] = []
    _run_spec: ClassVar[_TaskRunSpec]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is PipelineTask:
            return
        if cls.__dict__.get("_abstract_task", False) is True:
            return

        cls._validate_class_config()

        own_run = cls.__dict__.get("run")
        if own_run is None:
            inherited_spec = getattr(cls, "_run_spec", None)
            if inherited_spec is None:
                raise TypeError(f"PipelineTask '{cls.__name__}' must define @classmethod async def run(cls, ...) or inherit a validated run() implementation.")
            cls.input_document_types = list(inherited_spec.input_document_types)
            cls.output_document_types = list(inherited_spec.output_document_types)
            return

        spec = cls._validate_run_signature(own_run)
        cls._run_spec = spec
        cls.input_document_types = list(spec.input_document_types)
        cls.output_document_types = list(spec.output_document_types)
        cls.run = classmethod(cls._build_run_wrapper(spec))

    @classmethod
    def _validate_class_config(cls) -> None:
        if cls.__name__.startswith("Test"):
            raise TypeError(
                f"PipelineTask class name cannot start with 'Test': {cls.__name__}. Use a production-style class name; pytest classes reserve the Test* prefix."
            )
        if "name" not in cls.__dict__:
            cls.name = cls.__name__
        if cls.estimated_minutes < 1:
            raise TypeError(f"PipelineTask '{cls.__name__}' has estimated_minutes={cls.estimated_minutes}. Use a value >= 1.")
        if cls.retries < 0:
            raise TypeError(f"PipelineTask '{cls.__name__}' has retries={cls.retries}. Use a value >= 0.")
        if cls.timeout_seconds is not None and cls.timeout_seconds <= 0:
            raise TypeError(f"PipelineTask '{cls.__name__}' has timeout_seconds={cls.timeout_seconds}. Use a positive integer timeout or None.")

    @classmethod
    def _validate_run_signature(cls, run_descriptor: Any) -> _TaskRunSpec:
        if not isinstance(run_descriptor, classmethod):
            raise TypeError(f"PipelineTask '{cls.__name__}'.run must be declared with @classmethod.")

        descriptor = cast(object, run_descriptor)
        descriptor_func = getattr(descriptor, "__func__", None)
        if descriptor_func is None or not callable(descriptor_func):
            raise TypeError(f"PipelineTask '{cls.__name__}'.run descriptor is invalid. Declare it as @classmethod async def run(cls, ...).")

        user_run = cast(Callable[..., Awaitable[Any]], descriptor_func)
        if not inspect.iscoroutinefunction(user_run):
            raise TypeError(f"PipelineTask '{cls.__name__}'.run must be async def. Use async operations in task code and return Documents.")

        signature = inspect.signature(user_run)
        parameters = list(signature.parameters.values())
        if not parameters:
            raise TypeError(f"PipelineTask '{cls.__name__}'.run must accept 'cls' as the first parameter.")
        if parameters[0].name != "cls":
            raise TypeError(
                f"PipelineTask '{cls.__name__}'.run must use signature @classmethod async def run(cls, ...). Found first parameter '{parameters[0].name}'."
            )

        hints = resolve_type_hints(user_run)
        input_document_types: list[type[Document]] = []
        for parameter in parameters[1:]:
            annotation = hints.get(parameter.name)
            if annotation is None:
                raise TypeError(
                    f"PipelineTask '{cls.__name__}'.run parameter '{parameter.name}' is missing a type annotation. Annotate every task input explicitly."
                )
            input_document_types.extend(
                validate_task_input_annotation(
                    annotation,
                    task_name=cls.__name__,
                    parameter_name=parameter.name,
                )
            )

        return_annotation = hints.get("return")
        if return_annotation is None:
            raise TypeError(
                f"PipelineTask '{cls.__name__}'.run is missing a return annotation. "
                "Return Document, None, list[Document], tuple[Document, ...], or unions of those shapes."
            )
        output_document_types = validate_task_return_annotation(return_annotation, task_name=cls.__name__)

        return _TaskRunSpec(
            user_run=user_run,
            signature=signature,
            hints=MappingProxyType(hints),
            input_document_types=_ordered_unique_document_types(input_document_types),
            output_document_types=_ordered_unique_document_types(output_document_types),
        )

    @classmethod
    def _public_signature(cls) -> inspect.Signature:
        parameters = tuple(cls._run_spec.signature.parameters.values())[1:]
        return cls._run_spec.signature.replace(parameters=parameters)

    @classmethod
    def _bind_run_arguments(cls, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            bound = cls._run_spec.signature.bind(cls, *args, **kwargs)
        except TypeError as exc:
            raise TypeError(f"PipelineTask '{cls.__name__}.run' called with invalid arguments. Expected signature {cls._public_signature()}: {exc}") from exc

        bound.apply_defaults()
        arguments = {name: value for name, value in bound.arguments.items() if name != "cls"}
        for name, value in arguments.items():
            validate_task_argument_value(
                task_name=cls.__name__,
                parameter_name=name,
                value=value,
                annotation=cls._run_spec.hints[name],
            )
        return arguments

    @classmethod
    def _build_run_wrapper(cls, spec: _TaskRunSpec) -> Callable[..., TaskHandle[tuple[Document[Any], ...]]]:
        def wrapped(task_cls: type["PipelineTask"], *args: Any, **kwargs: Any) -> TaskHandle[tuple[Document[Any], ...]]:
            arguments = task_cls._bind_run_arguments(args, kwargs)
            try:
                asyncio.get_running_loop()
            except RuntimeError as exc:
                raise RuntimeError(
                    f"PipelineTask '{task_cls.__name__}.run' must be called from async code. Use `await Task.run(...)` inside a flow or test context."
                ) from exc

            execution_ctx = get_execution_context()
            if execution_ctx is None:
                raise RuntimeError(
                    f"PipelineTask '{task_cls.__name__}.run' called outside pipeline execution context. "
                    "Run tasks inside PipelineFlow/PipelineDeployment execution or pipeline_test_context()."
                )

            task = asyncio.create_task(task_cls._execute_invocation(arguments))
            handle = TaskHandle(
                _task=task,
                task_class=task_cls,
                input_arguments=MappingProxyType(dict(arguments)),
            )
            execution_ctx.active_task_handles.add(handle)
            task.add_done_callback(lambda _finished: execution_ctx.active_task_handles.discard(handle))
            return handle

        wrapped.__name__ = spec.user_run.__name__
        wrapped.__qualname__ = spec.user_run.__qualname__
        wrapped.__doc__ = spec.user_run.__doc__
        wrapped.__signature__ = cls._public_signature()  # type: ignore[attr-defined]
        return update_wrapper(wrapped, spec.user_run)

    @classmethod
    async def _persist_documents(
        cls,
        documents: tuple[Document, ...],
        task_ctx: TaskContext,
    ) -> tuple[Document, ...]:
        """Deduplicate, validate provenance, and persist documents to the database."""
        deduped = _TaskDocumentContext.deduplicate(list(documents))
        if not deduped:
            return ()

        lifecycle_ctx = _TaskDocumentContext(created=set(task_ctx.created))
        for orphan in lifecycle_ctx.finalize(deduped):
            logger.warning("[%s] Orphaned document %s — created but not returned", cls.__name__, orphan[:12])

        execution_ctx = get_execution_context()
        if execution_ctx is not None:
            await _persist_documents_to_database(
                deduped,
                execution_ctx.database,
                execution_ctx.deployment_id,
                execution_ctx.run_scope,
                producing_node_id=execution_ctx.current_node_id,
            )

        return tuple(deduped)

    @classmethod
    async def _run_with_retries(cls, arguments: Mapping[str, Any]) -> tuple[Document, ...]:
        attempts = cls.retries + 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            outcome = (
                await asyncio.gather(
                    _maybe_with_timeout(
                        cls.timeout_seconds,
                        lambda: cls._run_and_normalize(arguments),
                    ),
                    return_exceptions=True,
                )
            )[0]
            if isinstance(outcome, RETRY_CAPTURE_EXCEPTIONS):
                last_error = outcome
                if attempt < attempts - 1:
                    await asyncio.sleep(cls.retry_delay_seconds)
                continue
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        if last_error is None:
            raise RuntimeError(f"PipelineTask '{cls.__name__}' failed without raising a concrete exception.")
        raise last_error

    @classmethod
    async def _run_and_normalize(cls, arguments: Mapping[str, Any]) -> tuple[Document, ...]:
        result = await cls._run_spec.user_run(cls, **dict(arguments))
        return cls._normalize_result_documents(result)

    @classmethod
    def _normalize_result_documents(cls, result: Any) -> tuple[Document[Any], ...]:
        if result is None:
            return ()
        if isinstance(result, Document):
            raw_items = cast(Sequence[Any], (result,))
        elif isinstance(result, (list, tuple)):
            raw_items = cast(Sequence[Any], result)
        else:
            raise TypeError(
                f"PipelineTask '{cls.__name__}' returned {type(result).__name__}. "
                "run() must return Document, None, list[Document], tuple[Document, ...], or unions of those shapes."
            )

        normalized_docs: list[Document[Any]] = []
        bad_types: set[str] = set()
        for item in raw_items:
            if isinstance(item, Document):
                normalized_docs.append(cast(Document[Any], item))
                continue
            bad_types.add(_class_name(type(item)))

        if bad_types:
            bad_types_text = ", ".join(sorted(bad_types))
            raise TypeError(f"PipelineTask '{cls.__name__}' returned non-Document items ({bad_types_text}). run() must return only Document subclasses.")
        return tuple(normalized_docs)

    @staticmethod
    async def _emit_task_started(
        execution_ctx: ExecutionContext,
        flow_frame: FlowFrame | None,
        *,
        step: int,
        task_name: str,
        task_class_name: str,
        node_id: str,
    ) -> None:
        if flow_frame is None:
            return
        try:
            await execution_ctx.publisher.publish_task_started(
                TaskStartedEvent(
                    run_id=execution_ctx.run_id,
                    node_id=node_id,
                    root_deployment_id=str(execution_ctx.root_deployment_id or ""),
                    parent_deployment_task_id=str(execution_ctx.parent_deployment_task_id) if execution_ctx.parent_deployment_task_id else None,
                    flow_name=flow_frame.name,
                    step=step,
                    task_name=task_name,
                    task_class=task_class_name,
                )
            )
        except EVENT_PUBLISH_EXCEPTIONS as exc:
            logger.warning("Task started event publish failed for '%s': %s", task_name, exc)

    @staticmethod
    async def _emit_task_completed(
        execution_ctx: ExecutionContext,
        flow_frame: FlowFrame | None,
        *,
        step: int,
        task_name: str,
        task_class_name: str,
        node_id: str,
        start_time: float,
        output_documents: tuple[DocumentRef, ...],
    ) -> None:
        if flow_frame is None:
            return
        try:
            await execution_ctx.publisher.publish_task_completed(
                TaskCompletedEvent(
                    run_id=execution_ctx.run_id,
                    node_id=node_id,
                    root_deployment_id=str(execution_ctx.root_deployment_id or ""),
                    parent_deployment_task_id=str(execution_ctx.parent_deployment_task_id) if execution_ctx.parent_deployment_task_id else None,
                    flow_name=flow_frame.name,
                    step=step,
                    task_name=task_name,
                    task_class=task_class_name,
                    duration_ms=int((time.monotonic() - start_time) * MILLISECONDS_PER_SECOND),
                    output_documents=output_documents,
                )
            )
        except EVENT_PUBLISH_EXCEPTIONS as exc:
            logger.warning("Task completed event publish failed for '%s': %s", task_name, exc)

    @staticmethod
    async def _emit_task_failed(
        execution_ctx: ExecutionContext,
        flow_frame: FlowFrame | None,
        *,
        step: int,
        task_name: str,
        task_class_name: str,
        node_id: str,
        error_message: str,
    ) -> None:
        if flow_frame is None:
            return
        try:
            await execution_ctx.publisher.publish_task_failed(
                TaskFailedEvent(
                    run_id=execution_ctx.run_id,
                    node_id=node_id,
                    root_deployment_id=str(execution_ctx.root_deployment_id or ""),
                    parent_deployment_task_id=str(execution_ctx.parent_deployment_task_id) if execution_ctx.parent_deployment_task_id else None,
                    flow_name=flow_frame.name,
                    step=step,
                    task_name=task_name,
                    task_class=task_class_name,
                    error_message=error_message,
                )
            )
        except EVENT_PUBLISH_EXCEPTIONS as exc:
            logger.warning("Task failed event publish failed for '%s': %s", task_name, exc)

    @staticmethod
    def _collect_summaries(documents: Sequence[Document]) -> dict[DocumentSha256, str]:
        """Collect inline summaries already set on documents."""
        summary_by_sha: dict[DocumentSha256, str] = {}
        for document in documents:
            inline_summary = get_inline_summary(document.sha256)
            if inline_summary is not None:
                summary_by_sha[document.sha256] = inline_summary
        return summary_by_sha

    @staticmethod
    def _build_output_refs(
        documents: Sequence[Document],
        summary_by_sha: dict[DocumentSha256, str],
    ) -> tuple[DocumentRef, ...]:
        return tuple(
            DocumentRef(
                sha256=document.sha256,
                class_name=type(document).__name__,
                name=document.name,
                summary=summary_by_sha.get(document.sha256, ""),
                publicly_visible=getattr(type(document), "publicly_visible", False),
                derived_from=tuple(document.derived_from),
                triggered_by=tuple(document.triggered_by),
            )
            for document in documents
        )

    @staticmethod
    def _cleanup_task_artifacts(task_ctx: TaskContext) -> None:
        for created_sha in task_ctx.created:
            pop_inline_summary(created_sha)

    @classmethod
    async def _execute_lifecycle(
        cls,
        arguments: Mapping[str, Any],
        *,
        execution_ctx: ExecutionContext,
        flow_frame: FlowFrame | None,
        task_ctx: TaskContext,
        task_name: str,
        node_id: str,
        flow_step: int,
        start_time: float,
    ) -> tuple[Document, ...]:
        """Execute task lifecycle with events and persistence."""
        await cls._emit_task_started(
            execution_ctx,
            flow_frame,
            step=flow_step,
            task_name=task_name,
            task_class_name=cls.__name__,
            node_id=node_id,
        )
        record_lifecycle_event(
            "task.started",
            f"Starting task {task_name}",
            task_name=task_name,
            task_class=cls.__name__,
            flow_name=flow_frame.name if flow_frame is not None else "",
            step=flow_step,
        )
        try:
            documents = await cls._run_with_retries(arguments)

            summary_by_sha = cls._collect_summaries(documents)
            persisted_docs = await cls._persist_documents(documents, task_ctx)

            await cls._emit_task_completed(
                execution_ctx,
                flow_frame,
                step=flow_step,
                task_name=task_name,
                task_class_name=cls.__name__,
                node_id=node_id,
                start_time=start_time,
                output_documents=cls._build_output_refs(persisted_docs, summary_by_sha),
            )
            record_lifecycle_event(
                "task.completed",
                f"Completed task {task_name}",
                task_name=task_name,
                task_class=cls.__name__,
                flow_name=flow_frame.name if flow_frame is not None else "",
                step=flow_step,
                output_count=len(persisted_docs),
            )
            return persisted_docs
        except TASK_EXECUTION_EXCEPTIONS as exc:
            cls._cleanup_task_artifacts(task_ctx)
            await cls._emit_task_failed(
                execution_ctx,
                flow_frame,
                step=flow_step,
                task_name=task_name,
                task_class_name=cls.__name__,
                node_id=node_id,
                error_message=str(exc),
            )
            record_lifecycle_event(
                "task.failed",
                f"Task {task_name} failed",
                task_name=task_name,
                task_class=cls.__name__,
                flow_name=flow_frame.name if flow_frame is not None else "",
                step=flow_step,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise

    @classmethod
    async def _execute_invocation(cls, arguments: Mapping[str, Any]) -> tuple[Document, ...]:
        """Execute task lifecycle with events, summaries, and persistence."""
        execution_ctx = get_execution_context()
        if execution_ctx is None:
            raise RuntimeError(
                f"PipelineTask '{cls.__name__}.run' called outside pipeline execution context. "
                "Run tasks inside PipelineFlow/PipelineDeployment execution or pipeline_test_context()."
            )

        parent_task = execution_ctx.task_frame
        task_node_id = uuid4()
        task_function_path = f"{cls.__module__}:{cls.__qualname__}"
        task_frame = TaskFrame(
            task_class_name=cls.__name__,
            task_id=str(task_node_id),
            depth=(parent_task.depth + 1) if parent_task else 0,
            parent=parent_task,
        )
        try:
            task_replay_payload = {
                "version": 1,
                "payload_type": "pipeline_task",
                "function_path": task_function_path,
                "arguments": serialize_kwargs(dict(arguments)),
                "original": {},
            }
        except REPLAY_CAPTURE_EXCEPTIONS:
            task_replay_payload = {}

        # Insert task node into execution DAG (fire-and-forget)
        database = execution_ctx.database
        task_payload = {
            "function_path": task_function_path,
            "expected_cost": cls.expected_cost,
            "replay_payload": task_replay_payload,
        }
        if database is not None and execution_ctx.deployment_id is not None:
            parent_id = execution_ctx.current_node_id or NULL_PARENT
            task_node = ExecutionNode(
                node_id=task_node_id,
                node_kind=NodeKind.TASK,
                deployment_id=execution_ctx.deployment_id,
                root_deployment_id=execution_ctx.root_deployment_id or execution_ctx.deployment_id,
                parent_node_id=parent_id,
                run_id=execution_ctx.run_id,
                run_scope=RunScope(str(execution_ctx.run_scope)),
                deployment_name=execution_ctx.deployment_name,
                name=cls.name,
                sequence_no=execution_ctx.next_child_sequence(parent_id),
                task_class=cls.__name__,
                flow_class=execution_ctx.flow_frame.flow_class_name if execution_ctx.flow_frame else "",
                flow_id=execution_ctx.flow_node_id,
                status=NodeStatus.RUNNING,
                payload=task_payload,
            )
            await _insert_db_node_safe(database, task_node)

        # Update execution context with task node as current node
        task_exec_ctx = execution_ctx.with_task(task_frame).with_node(task_node_id)
        execution_token = set_execution_context(task_exec_ctx)
        task_ctx = TaskContext(task_class_name=cls.__name__)
        task_token = set_task_context(task_ctx)

        # Set up conversation turn capture
        conversation_turns: list[ConversationTurnData] = []
        turns_token = set_conversation_turns(conversation_turns)

        start_time = time.monotonic()
        task_name = cls.name
        flow_frame = execution_ctx.flow_frame

        try:

            async def _execute() -> tuple[Document, ...]:
                result = await cls._execute_lifecycle(
                    arguments,
                    execution_ctx=execution_ctx,
                    flow_frame=flow_frame,
                    task_ctx=task_ctx,
                    task_name=task_name,
                    node_id=task_frame.task_id,
                    flow_step=flow_frame.step if flow_frame is not None else 0,
                    start_time=start_time,
                )
                status = NodeStatus.COMPLETED
                output_shas = tuple(doc.sha256 for doc in result)
                input_docs = _input_documents(arguments)
                input_shas = tuple(doc.sha256 for doc in input_docs)
                task_turn_metrics = _aggregate_conversation_turn_metrics(conversation_turns)
                await _update_db_node_safe(
                    database,
                    task_node_id,
                    status=status,
                    ended_at=datetime.now(UTC),
                    input_document_shas=input_shas,
                    output_document_shas=output_shas,
                    **task_turn_metrics,
                    payload={**task_payload, "log_summary": _consume_log_summary(execution_ctx, task_node_id)},
                )

                return result

            try:
                return await _execute()
            except TASK_EXECUTION_EXCEPTIONS as exc:
                task_turn_metrics = _aggregate_conversation_turn_metrics(conversation_turns)
                # Update task node to FAILED
                await _update_db_node_safe(
                    database,
                    task_node_id,
                    status=NodeStatus.FAILED,
                    ended_at=datetime.now(UTC),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    **task_turn_metrics,
                    payload={**task_payload, "log_summary": _consume_log_summary(execution_ctx, task_node_id)},
                )
                raise
            finally:
                # Flush conversation turns on BOTH success and failure — partial
                # conversation data on failed tasks is valuable for debugging.
                if conversation_turns:
                    await _create_conversation_turn_nodes(conversation_turns, task_node_id, task_exec_ctx)
        finally:
            reset_conversation_turns(turns_token)
            reset_task_context(task_token)
            reset_execution_context(execution_token)
