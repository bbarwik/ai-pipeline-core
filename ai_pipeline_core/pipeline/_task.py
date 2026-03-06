"""Class-based pipeline task runtime.

Rules enforced at class definition time:
1. Subclasses must define ``@classmethod async def run(cls, ...)`` or inherit one from a parent task.
2. Every ``run()`` parameter after ``cls`` must use a supported annotation.
3. ``run()`` must return ``Document``, ``None``, ``list[Document]``, ``tuple[Document, ...]``, or unions of those shapes.
4. Bare ``Document`` is forbidden in both inputs and outputs; use concrete subclasses.
5. Class names must not start with ``Test``.
6. ``estimated_minutes`` must be >= 1, ``retries`` >= 0, and ``timeout_seconds`` positive when set.

Runtime behavior:
1. ``Task.run(...)`` returns an awaitable ``TaskHandle``.
2. ``await Task.run(...)`` executes the full lifecycle: tracing, retries, timeout, events, persistence, summaries, and replay capture.
3. Tasks must run inside an active pipeline execution context (or ``pipeline_test_context()`` in tests).
"""

import asyncio
import hashlib
import inspect
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import update_wrapper
from types import MappingProxyType
from typing import Any, ClassVar, cast
from uuid import uuid4

from lmnr import Laminar
from pydantic import BaseModel

from ai_pipeline_core.deployment._types import DocumentRef, TaskCompletedEvent, TaskFailedEvent, TaskStartedEvent
from ai_pipeline_core.document_store._protocol import get_document_store
from ai_pipeline_core.documents import Document, DocumentSha256
from ai_pipeline_core.documents._context import TaskContext, _TaskDocumentContext, reset_task_context, set_task_context
from ai_pipeline_core.documents.document import get_inline_summary, pop_inline_summary, set_inline_summary
from ai_pipeline_core.documents.utils import is_document_sha256
from ai_pipeline_core.llm.conversation import Conversation
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._document_tracking import track_task_io
from ai_pipeline_core.observability.tracing import TraceLevel, set_trace_cost, trace
from ai_pipeline_core.pipeline._execution_context import (
    ExecutionContext,
    FlowFrame,
    TaskFrame,
    get_execution_context,
    reset_execution_context,
    set_execution_context,
)
from ai_pipeline_core.pipeline._parallel import TaskHandle
from ai_pipeline_core.pipeline._type_validation import (
    is_already_traced,
    resolve_type_hints,
    validate_task_argument_value,
    validate_task_input_annotation,
    validate_task_return_annotation,
)
from ai_pipeline_core.replay._capture import serialize_kwargs

logger = get_pipeline_logger(__name__)

SUMMARY_EXCERPT_MAX_CHARS = 6000
MILLISECONDS_PER_SECOND = 1000
TASK_COMPLETION_KEY_PREFIX = "__task_completion__"
TASK_COMPLETION_HASH_CHARS = 24

EVENT_PUBLISH_EXCEPTIONS = (OSError, RuntimeError, ValueError, TypeError)
SUMMARY_GENERATION_EXCEPTIONS = (OSError, RuntimeError, ValueError, TypeError)
TRACKING_EXCEPTIONS = (RuntimeError, ValueError, TypeError)
SPAN_ATTRIBUTE_EXCEPTIONS = (OSError, RuntimeError, ValueError, TypeError)
TASK_EXECUTION_EXCEPTIONS = (Exception,)
PERSISTENCE_EXCEPTIONS = (Exception,)
REPLAY_CAPTURE_EXCEPTIONS = (Exception,)
RETRY_CAPTURE_EXCEPTIONS = (Exception,)

__all__ = ["PipelineTask"]


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


def _collect_documents(value: Any, documents: list[Document[Any]]) -> None:
    """Collect Documents nested in supported task input values."""
    if isinstance(value, Document):
        documents.append(value)
        return
    if isinstance(value, Conversation):
        for document in value.context:
            _collect_documents(document, documents)
        for message in value.messages:
            _collect_documents(message, documents)
        return
    if isinstance(value, (list, tuple)):
        for item in cast(Sequence[Any], value):
            _collect_documents(item, documents)
        return
    if isinstance(value, dict):
        for item in cast(dict[Any, Any], value).values():
            _collect_documents(item, documents)
        return
    if isinstance(value, BaseModel):
        for field_name in type(value).model_fields:
            _collect_documents(getattr(value, field_name), documents)


def _input_documents(arguments: Mapping[str, Any]) -> list[Document]:
    """Flatten Document inputs from task arguments while preserving order."""
    documents: list[Document] = []
    for value in arguments.values():
        _collect_documents(value, documents)
    deduped: dict[str, Document] = {}
    for document in documents:
        deduped.setdefault(document.sha256, document)
    return list(deduped.values())


def _task_arguments_fingerprint(arguments: Mapping[str, Any]) -> str:
    """Build a stable JSON fingerprint for task input arguments."""
    serialized = serialize_kwargs(dict(arguments))
    return json.dumps(serialized, sort_keys=True, separators=(",", ":"))


def _set_task_replay_payload(task_cls: type["PipelineTask"], arguments: Mapping[str, Any]) -> None:
    try:
        replay_payload = {
            "version": 1,
            "payload_type": "pipeline_task",
            "function_path": f"{task_cls.__module__}:{task_cls.__qualname__}",
            "arguments": serialize_kwargs(dict(arguments)),
        }
        Laminar.set_span_attributes({"replay.payload": json.dumps(replay_payload)})
    except REPLAY_CAPTURE_EXCEPTIONS:
        logger.debug("Failed to attach task replay payload for '%s'", task_cls.__name__, exc_info=True)


class PipelineTask:
    """Base class for pipeline tasks.

    Tasks are stateless units of work. Define ``run`` as a **@classmethod** because tasks
    carry no per-invocation instance state — all inputs arrive as arguments, all outputs
    are returned documents. The framework wraps ``run`` with tracing, retries, persistence,
    and event emission automatically.

    Minimal example::

        class SummarizeTask(PipelineTask):
            @classmethod
            async def run(cls, documents: list[ArticleDocument]) -> list[SummaryDocument]:
                conv = Conversation(model="gemini-3-flash").with_context(documents[0])
                conv = await conv.send("Summarize this article.")
                return [SummaryDocument.derive(from_documents=(documents[0],), name="summary.md", content=conv.content)]

    Calling ``await SummarizeTask.run([doc])`` dispatches the full lifecycle. Calling without
    ``await`` returns a ``TaskHandle`` for parallel execution via ``collect_tasks``.
    """

    name: ClassVar[str]
    estimated_minutes: ClassVar[float] = 1.0
    retries: ClassVar[int] = 0
    retry_delay_seconds: ClassVar[int] = 20
    timeout_seconds: ClassVar[int | None] = None
    cacheable: ClassVar[bool] = False

    trace_level: ClassVar[TraceLevel] = "always"
    trace_ignore_input: ClassVar[bool] = False
    trace_ignore_output: ClassVar[bool] = False
    trace_ignore_inputs: ClassVar[tuple[str, ...]] = ()
    trace_input_formatter: ClassVar[Callable[..., str] | None] = None
    trace_output_formatter: ClassVar[Callable[..., str] | None] = None
    expected_cost: ClassVar[float | None] = None
    trace_trim_documents: ClassVar[bool] = True
    trace_cost: ClassVar[float | None] = None

    input_document_types: ClassVar[list[type[Document]]] = []
    output_document_types: ClassVar[list[type[Document]]] = []
    _trace_decorator: ClassVar[Callable[[Callable[..., Any]], Callable[..., Any]]]
    _run_spec: ClassVar[_TaskRunSpec]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is PipelineTask:
            return

        cls._validate_class_config()
        cls._trace_decorator = cls._build_trace_decorator()

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

        user_run: Callable[..., Awaitable[Any]] = run_descriptor.__func__
        if is_already_traced(user_run):
            raise TypeError(
                f"PipelineTask '{cls.__name__}'.run is decorated with @trace. PipelineTask applies tracing automatically. Remove @trace from run()."
            )
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
    def _build_trace_decorator(cls) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return trace(
            level=cls.trace_level,
            name=cls.name,
            ignore_input=cls.trace_ignore_input,
            ignore_output=cls.trace_ignore_output,
            ignore_inputs=list(cls.trace_ignore_inputs) if cls.trace_ignore_inputs else None,
            input_formatter=cls.trace_input_formatter,
            output_formatter=cls.trace_output_formatter,
            trim_documents=cls.trace_trim_documents,
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
    def _build_run_wrapper(cls, spec: _TaskRunSpec) -> Callable[..., TaskHandle[list[Document]]]:
        def wrapped(task_cls: type["PipelineTask"], *args: Any, **kwargs: Any) -> TaskHandle[list[Document]]:
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
    def _task_completion_name(cls, arguments: Mapping[str, Any]) -> str:
        completion_hash = hashlib.sha256(f"{cls.__module__}:{cls.__qualname__}:{_task_arguments_fingerprint(arguments)}".encode()).hexdigest()
        return f"{TASK_COMPLETION_KEY_PREFIX}:{cls.__name__}:{completion_hash[:TASK_COMPLETION_HASH_CHARS]}"

    @classmethod
    async def _load_cached_documents(
        cls,
        output_sha256s: tuple[str, ...],
        store: Any,
        run_scope: Any,
    ) -> list[Document] | None:
        if not output_sha256s:
            return []

        by_sha: dict[DocumentSha256, Document] = {}
        target_sha256s = [DocumentSha256(value) for value in output_sha256s]
        for document_type in cls.output_document_types:
            loaded = await store.load_by_sha256s(target_sha256s, document_type, run_scope)
            for sha256, document in loaded.items():
                by_sha.setdefault(sha256, document)

        ordered: list[Document] = []
        for raw_sha in output_sha256s:
            sha = DocumentSha256(raw_sha)
            document = by_sha.get(sha)
            if document is None:
                logger.warning(
                    "Task completion cache miss for '%s': cached output %s not found. Re-executing task to rebuild completion record.",
                    cls.__name__,
                    raw_sha[:12],
                )
                return None
            ordered.append(document)
        return ordered

    @classmethod
    async def _load_task_completion_cache(cls, arguments: Mapping[str, Any]) -> list[Document] | None:
        if not cls.cacheable:
            return None

        execution_ctx = get_execution_context()
        run_scope = execution_ctx.run_scope if execution_ctx is not None else None
        store = execution_ctx.store if execution_ctx is not None else None
        if store is None:
            store = get_document_store()
        if store is None or run_scope is None:
            return None

        completion_name = cls._task_completion_name(arguments)
        completion = await store.get_flow_completion(run_scope, completion_name)
        if completion is None:
            return None
        return await cls._load_cached_documents(completion.output_sha256s, store, run_scope)

    @classmethod
    async def _save_task_completion_cache(
        cls,
        arguments: Mapping[str, Any],
        output_documents: list[Document],
    ) -> None:
        if not cls.cacheable:
            return

        execution_ctx = get_execution_context()
        run_scope = execution_ctx.run_scope if execution_ctx is not None else None
        store = execution_ctx.store if execution_ctx is not None else None
        if store is None:
            store = get_document_store()
        if store is None or run_scope is None:
            return

        completion_name = cls._task_completion_name(arguments)
        input_sha256s = tuple(document.sha256 for document in _input_documents(arguments))
        output_sha256s = tuple(document.sha256 for document in output_documents)
        try:
            await store.save_flow_completion(run_scope, completion_name, input_sha256s, output_sha256s)
        except PERSISTENCE_EXCEPTIONS:
            logger.warning("Failed to save task completion cache for '%s'", cls.__name__, exc_info=True)

    @staticmethod
    def _collect_reference_sha256s(documents: list[Document]) -> set[DocumentSha256]:
        reference_sha256s: set[DocumentSha256] = set()
        for document in documents:
            for source in document.derived_from:
                if is_document_sha256(source):
                    reference_sha256s.add(DocumentSha256(source))
            for trigger in document.triggered_by:
                reference_sha256s.add(DocumentSha256(trigger))
        return reference_sha256s

    @staticmethod
    async def _persist_inline_summaries(documents: list[Document], store: Any) -> None:
        for document in documents:
            inline_summary = pop_inline_summary(document.sha256)
            if inline_summary:
                await store.update_summary(document.sha256, inline_summary)

    @staticmethod
    def _clear_inline_summaries(documents: list[Document]) -> None:
        for document in documents:
            pop_inline_summary(document.sha256)

    @classmethod
    async def _persist_documents(
        cls,
        documents: list[Document],
        task_ctx: TaskContext,
        *,
        created_by_task: str,
    ) -> list[Document]:
        execution_ctx = get_execution_context()
        run_scope = execution_ctx.run_scope if execution_ctx is not None else None
        store = execution_ctx.store if execution_ctx is not None else None
        if store is None:
            store = get_document_store()

        deduped = _TaskDocumentContext.deduplicate(documents)
        if run_scope is None or store is None or not deduped:
            return deduped

        lifecycle_ctx = _TaskDocumentContext(created=set(task_ctx.created))
        reference_sha256s = cls._collect_reference_sha256s(deduped)
        existing: set[DocumentSha256] = await store.check_existing(sorted(reference_sha256s)) if reference_sha256s else set()

        for warning in lifecycle_ctx.validate_provenance(deduped, existing):
            logger.warning("[%s] %s", cls.__name__, warning)
        for orphan in lifecycle_ctx.finalize(deduped):
            logger.warning("[%s] Orphaned document %s — created but not returned", cls.__name__, orphan[:12])

        try:
            await store.save_batch(deduped, run_scope, created_by_task=created_by_task)
            await cls._persist_inline_summaries(deduped, store)
        except PERSISTENCE_EXCEPTIONS:
            cls._clear_inline_summaries(deduped)
            logger.warning("Failed to persist documents from PipelineTask '%s'", cls.__name__, exc_info=True)
        return deduped

    @classmethod
    async def _run_with_retries(cls, arguments: Mapping[str, Any]) -> list[Document]:
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
    async def _run_and_normalize(cls, arguments: Mapping[str, Any]) -> list[Document]:
        result = await cls._run_spec.user_run(cls, **dict(arguments))
        return cls._normalize_result_documents(result)

    @classmethod
    def _normalize_result_documents(cls, result: Any) -> list[Document[Any]]:
        documents: list[Document[Any]]
        if result is None:
            return []
        if isinstance(result, Document):
            documents = [result]
        elif isinstance(result, (list, tuple)):
            documents = list(result)
        else:
            raise TypeError(
                f"PipelineTask '{cls.__name__}' returned {type(result).__name__}. "
                "run() must return Document, None, list[Document], tuple[Document, ...], or unions of those shapes."
            )

        untyped_docs = cast(list[Any], documents)
        if any(not isinstance(document, Document) for document in untyped_docs):
            bad_types = sorted({_class_name(type(item)) for item in untyped_docs if not isinstance(item, Document)})
            raise TypeError(f"PipelineTask '{cls.__name__}' returned non-Document items ({', '.join(bad_types)}). run() must return only Document subclasses.")
        return documents

    @staticmethod
    def _track_task_io(arguments: Mapping[str, Any], output_documents: list[Document], task_name: str) -> None:
        try:
            track_task_io((), dict(arguments), output_documents)
        except TRACKING_EXCEPTIONS:
            logger.debug("Failed to track task IO for '%s'", task_name, exc_info=True)

    @staticmethod
    async def _emit_task_started(
        execution_ctx: ExecutionContext,
        flow_frame: FlowFrame | None,
        *,
        step: int,
        task_name: str,
        task_class_name: str,
        task_invocation_id: str,
        parent_task_name: str | None,
        task_depth: int,
    ) -> None:
        if flow_frame is None:
            return
        try:
            await execution_ctx.publisher.publish_task_started(
                TaskStartedEvent(
                    run_id=execution_ctx.run_id,
                    flow_name=flow_frame.name,
                    step=step,
                    task_name=task_name,
                    task_class=task_class_name,
                    task_invocation_id=task_invocation_id,
                    parent_task=parent_task_name,
                    task_depth=task_depth,
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
        task_invocation_id: str,
        parent_task_name: str | None,
        task_depth: int,
        start_time: float,
        output_documents: list[DocumentRef],
    ) -> None:
        if flow_frame is None:
            return
        try:
            await execution_ctx.publisher.publish_task_completed(
                TaskCompletedEvent(
                    run_id=execution_ctx.run_id,
                    flow_name=flow_frame.name,
                    step=step,
                    task_name=task_name,
                    task_class=task_class_name,
                    task_invocation_id=task_invocation_id,
                    parent_task=parent_task_name,
                    task_depth=task_depth,
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
        task_invocation_id: str,
        parent_task_name: str | None,
        task_depth: int,
        error_message: str,
    ) -> None:
        if flow_frame is None:
            return
        try:
            await execution_ctx.publisher.publish_task_failed(
                TaskFailedEvent(
                    run_id=execution_ctx.run_id,
                    flow_name=flow_frame.name,
                    step=step,
                    task_name=task_name,
                    task_class=task_class_name,
                    task_invocation_id=task_invocation_id,
                    parent_task=parent_task_name,
                    task_depth=task_depth,
                    error_message=error_message,
                )
            )
        except EVENT_PUBLISH_EXCEPTIONS as exc:
            logger.warning("Task failed event publish failed for '%s': %s", task_name, exc)

    @staticmethod
    async def _generate_summaries(
        documents: list[Document],
        execution_ctx: ExecutionContext,
    ) -> dict[DocumentSha256, str]:
        summary_by_sha: dict[DocumentSha256, str] = {}
        for document in documents:
            inline_summary = get_inline_summary(document.sha256)
            if inline_summary is not None:
                summary_by_sha[document.sha256] = inline_summary
                continue
            if execution_ctx.summary_generator is None or not document.is_text:
                continue
            try:
                generated = await execution_ctx.summary_generator(document.name, document.text[:SUMMARY_EXCERPT_MAX_CHARS])
                set_inline_summary(document.sha256, generated)
                summary_by_sha[document.sha256] = generated
            except SUMMARY_GENERATION_EXCEPTIONS as exc:
                logger.warning("Inline summary generation failed for '%s': %s", document.name, exc)
        return summary_by_sha

    @staticmethod
    def _build_output_refs(
        documents: list[Document],
        summary_by_sha: dict[DocumentSha256, str],
    ) -> list[DocumentRef]:
        return [
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
        ]

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
        task_invocation_id: str,
        parent_task_name: str | None,
        task_depth: int,
        flow_step: int,
        start_time: float,
    ) -> list[Document]:
        await cls._emit_task_started(
            execution_ctx,
            flow_frame,
            step=flow_step,
            task_name=task_name,
            task_class_name=cls.__name__,
            task_invocation_id=task_invocation_id,
            parent_task_name=parent_task_name,
            task_depth=task_depth,
        )
        try:
            cached_docs = await cls._load_task_completion_cache(arguments)
            if cached_docs is not None:
                cls._track_task_io(arguments, cached_docs, task_name)
                summary_by_sha = {document.sha256: (get_inline_summary(document.sha256) or "") for document in cached_docs}
                await cls._emit_task_completed(
                    execution_ctx,
                    flow_frame,
                    step=flow_step,
                    task_name=task_name,
                    task_class_name=cls.__name__,
                    task_invocation_id=task_invocation_id,
                    parent_task_name=parent_task_name,
                    task_depth=task_depth,
                    start_time=start_time,
                    output_documents=cls._build_output_refs(cached_docs, summary_by_sha),
                )
                _set_task_replay_payload(cls, arguments)
                return cached_docs

            documents = await cls._run_with_retries(arguments)
            cls._track_task_io(arguments, documents, task_name)

            summary_by_sha = await cls._generate_summaries(documents, execution_ctx)
            persisted_docs = await cls._persist_documents(documents, task_ctx, created_by_task=cls.__name__)
            await cls._save_task_completion_cache(arguments, persisted_docs)

            await cls._emit_task_completed(
                execution_ctx,
                flow_frame,
                step=flow_step,
                task_name=task_name,
                task_class_name=cls.__name__,
                task_invocation_id=task_invocation_id,
                parent_task_name=parent_task_name,
                task_depth=task_depth,
                start_time=start_time,
                output_documents=cls._build_output_refs(persisted_docs, summary_by_sha),
            )
            _set_task_replay_payload(cls, arguments)
            return persisted_docs
        except TASK_EXECUTION_EXCEPTIONS as exc:
            cls._cleanup_task_artifacts(task_ctx)
            await cls._emit_task_failed(
                execution_ctx,
                flow_frame,
                step=flow_step,
                task_name=task_name,
                task_class_name=cls.__name__,
                task_invocation_id=task_invocation_id,
                parent_task_name=parent_task_name,
                task_depth=task_depth,
                error_message=str(exc),
            )
            raise

    @classmethod
    async def _execute_invocation(cls, arguments: Mapping[str, Any]) -> list[Document]:
        """Execute task lifecycle with tracing, events, summaries, and persistence."""
        execution_ctx = get_execution_context()
        if execution_ctx is None:
            raise RuntimeError(
                f"PipelineTask '{cls.__name__}.run' called outside pipeline execution context. "
                "Run tasks inside PipelineFlow/PipelineDeployment execution or pipeline_test_context()."
            )

        parent_task = execution_ctx.task_frame
        task_frame = TaskFrame(
            task_class_name=cls.__name__,
            task_id=uuid4().hex,
            depth=(parent_task.depth + 1) if parent_task else 0,
            parent=parent_task,
        )
        execution_token = set_execution_context(execution_ctx.with_task(task_frame))
        task_ctx = TaskContext(task_class_name=cls.__name__)
        task_token = set_task_context(task_ctx)
        start_time = time.monotonic()
        task_name = cls.name
        parent_task_name = parent_task.task_class_name if parent_task else None
        flow_frame = execution_ctx.flow_frame

        try:

            async def _execute() -> list[Document]:
                try:
                    Laminar.set_span_attributes({"pipeline.task_class": cls.__name__})
                    if flow_frame is not None:
                        Laminar.set_span_attributes({"pipeline.flow_step": flow_frame.step})
                except SPAN_ATTRIBUTE_EXCEPTIONS:
                    logger.debug("Failed to set task span attributes", exc_info=True)
                if cls.expected_cost is not None:
                    try:
                        Laminar.set_span_attributes({"expected_cost": cls.expected_cost})
                    except SPAN_ATTRIBUTE_EXCEPTIONS:
                        logger.debug("Failed to set expected_cost span attribute", exc_info=True)
                result = await cls._execute_lifecycle(
                    arguments,
                    execution_ctx=execution_ctx,
                    flow_frame=flow_frame,
                    task_ctx=task_ctx,
                    task_name=task_name,
                    task_invocation_id=task_frame.task_id,
                    parent_task_name=parent_task_name,
                    task_depth=task_frame.depth,
                    flow_step=flow_frame.step if flow_frame is not None else 0,
                    start_time=start_time,
                )
                if cls.trace_cost is not None and cls.trace_cost > 0:
                    set_trace_cost(cls.trace_cost)
                return result

            traced_execute = cls._trace_decorator(_execute)
            return await traced_execute()
        finally:
            reset_task_context(task_token)
            reset_execution_context(execution_token)
