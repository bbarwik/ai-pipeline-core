"""Pipeline decorators with Prefect integration, tracing, and document lifecycle.

Wrappers around Prefect's @task and @flow that add Laminar tracing,
enforce async-only execution, and auto-save documents to the DocumentStore.
"""

import datetime
import inspect
import types
from collections.abc import Callable, Coroutine, Iterable
from functools import wraps
from typing import (
    Any,
    Protocol,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from lmnr import Laminar
from prefect.assets import Asset
from prefect.cache_policies import CachePolicy
from prefect.context import TaskRunContext
from prefect.flows import FlowStateHook
from prefect.flows import flow as _prefect_flow
from prefect.futures import PrefectFuture
from prefect.results import ResultSerializer, ResultStorage
from prefect.task_runners import TaskRunner
from prefect.tasks import task as _prefect_task
from prefect.utilities.annotations import NotSet
from pydantic import BaseModel

from ai_pipeline_core.document_store import get_document_store
from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents._types import DocumentSha256, RunScope
from ai_pipeline_core.documents.context import (
    RunContext,
    TaskDocumentContext,
    get_run_context,
    reset_run_context,
    reset_task_context,
    set_run_context,
    set_task_context,
)
from ai_pipeline_core.documents.utils import is_document_sha256
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._document_tracking import get_current_span_id, track_flow_io, track_task_io
from ai_pipeline_core.observability._initialization import get_tracking_service
from ai_pipeline_core.observability._tracking._models import DocumentEventType
from ai_pipeline_core.observability.tracing import TraceLevel, set_trace_cost, trace
from ai_pipeline_core.pipeline.options import FlowOptions

logger = get_pipeline_logger(__name__)

# --------------------------------------------------------------------------- #
# Public callback aliases (Prefect stubs omit these exact types)
# --------------------------------------------------------------------------- #
type RetryConditionCallable = Callable[[Any, Any, Any], bool]
type StateHookCallable = Callable[[Any, Any, Any], None]
type TaskRunNameValueOrCallable = str | Callable[[], str]

# --------------------------------------------------------------------------- #
# Typing helpers
# --------------------------------------------------------------------------- #
R_co = TypeVar("R_co", covariant=True)
FO_contra = TypeVar("FO_contra", bound=FlowOptions, contravariant=True)


class _TaskLike(Protocol[R_co]):
    """Protocol for type-safe Prefect task representation."""

    def __call__(self, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, R_co]: ...

    submit: Callable[..., Any]
    map: Callable[..., Any]
    name: str | None
    estimated_minutes: int

    def __getattr__(self, name: str) -> Any: ...


class _FlowLike(Protocol[FO_contra]):
    """Protocol for decorated flow objects returned by @pipeline_flow."""

    def __call__(
        self,
        project_name: str,
        documents: list[Document],
        flow_options: FO_contra,
    ) -> Coroutine[Any, Any, list[Document]]: ...

    name: str | None
    input_document_types: list[type[Document]]
    output_document_types: list[type[Document]]
    estimated_minutes: int

    def __getattr__(self, name: str) -> Any: ...


# --------------------------------------------------------------------------- #
# Annotation parsing helpers
# --------------------------------------------------------------------------- #
def _flatten_union(tp: Any) -> list[Any]:
    """Flatten Union / X | Y annotations into a list of constituent types."""
    origin = get_origin(tp)
    if origin is Union or isinstance(tp, types.UnionType):
        result: list[Any] = []
        for arg in get_args(tp):
            result.extend(_flatten_union(arg))
        return result
    return [tp]


def _find_non_document_leaves(tp: Any) -> list[Any]:
    """Walk a return type annotation and collect leaf types that are not Document subclasses or NoneType.

    Returns empty list when all leaf types are valid (Document subclasses or None).
    Used by @pipeline_task to validate return annotations at decoration time.
    """
    if tp is type(None) or (isinstance(tp, type) and issubclass(tp, Document)):
        return []

    origin = get_origin(tp)

    # Union / X | Y: all branches must be valid
    if origin is Union or isinstance(tp, types.UnionType):
        return [leaf for arg in get_args(tp) for leaf in _find_non_document_leaves(arg)]

    # list[X]: recurse into element type
    if origin is list:
        args = get_args(tp)
        return _find_non_document_leaves(args[0]) if args else [tp]

    # tuple[X, Y] or tuple[X, ...]
    if origin is tuple:
        args = get_args(tp)
        if not args:
            return [tp]
        elements = (args[0],) if (len(args) == 2 and args[1] is Ellipsis) else args
        return [leaf for arg in elements for leaf in _find_non_document_leaves(arg)]

    # Everything else is invalid (int, str, Any, object, dict, etc.)
    return [tp]


def _parse_document_types_from_annotation(annotation: Any) -> list[type[Document]]:
    """Extract Document subclasses from a list[...] type annotation.

    Handles list[DocA], list[DocA | DocB], list[Union[DocA, DocB]].
    Returns empty list if annotation is not a list of Document subclasses.
    """
    origin = get_origin(annotation)
    if origin is not list:
        return []

    args = get_args(annotation)
    if not args:
        return []

    inner = args[0]
    flat = _flatten_union(inner)

    return [t for t in flat if isinstance(t, type) and issubclass(t, Document)]


def _resolve_type_hints(fn: Callable[..., Any]) -> dict[str, Any]:
    """Safely resolve type hints, falling back to empty dict on failure."""
    try:
        return get_type_hints(fn, include_extras=True)
    except Exception:
        logger.warning(
            "Failed to resolve type hints for '%s'. Ensure all annotations are valid and importable.",
            _callable_name(fn, "unknown"),
        )
        return {}


# --------------------------------------------------------------------------- #
# Document extraction helper
# --------------------------------------------------------------------------- #
def _extract_documents(result: Any) -> list[Document]:
    """Recursively extract unique Document instances from a result value.

    Walks tuples, lists, dicts, and Pydantic BaseModel fields.
    Deduplicates by object identity (same instance appearing multiple times
    is collected only once). Checks Document before BaseModel since Document
    IS a BaseModel subclass.
    """
    docs: list[Document] = []
    seen: set[int] = set()

    def _walk(value: Any) -> None:
        obj_id = id(value)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if isinstance(value, Document):
            docs.append(value)
            return
        if isinstance(value, (list, tuple)):
            for item in cast(Iterable[Any], value):
                _walk(item)
            return
        if isinstance(value, dict):
            for v in cast(Iterable[Any], value.values()):
                _walk(v)
            return
        if isinstance(value, BaseModel):
            for field_name in type(value).model_fields:
                _walk(getattr(value, field_name))
            return

    _walk(result)
    return docs


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _callable_name(obj: Any, fallback: str) -> str:
    """Safely extract callable's name for error messages."""
    try:
        n = getattr(obj, "__name__", None)
        return n if isinstance(n, str) else fallback
    except Exception:
        return fallback


def _is_already_traced(func: Callable[..., Any]) -> bool:
    """Check if a function has already been wrapped by the trace decorator."""
    if hasattr(func, "__is_traced__") and func.__is_traced__:  # type: ignore[attr-defined]
        return True

    current = func
    depth = 0
    while hasattr(current, "__wrapped__") and depth < 10:
        wrapped = current.__wrapped__  # type: ignore[attr-defined]
        if hasattr(wrapped, "__is_traced__") and wrapped.__is_traced__:
            return True
        current = wrapped
        depth += 1
    return False


# --------------------------------------------------------------------------- #
# Tracking helpers
# --------------------------------------------------------------------------- #
def _resolve_label(user_summary: str | bool, fn: Callable[..., Any], kwargs: dict[str, Any]) -> str:
    """Resolve user_summary to a label string."""
    if isinstance(user_summary, str):
        try:
            return user_summary.format(**kwargs)
        except (KeyError, IndexError):
            return user_summary
    return _callable_name(fn, "task").replace("_", " ").title()


def _build_output_hint(result: object) -> str:
    """Build a privacy-safe metadata string describing a task's output."""
    if result is None:
        return "None"
    if isinstance(result, list) and result and isinstance(result[0], Document):
        doc_list = cast(list[Document], result)
        class_counts: dict[str, int] = {}
        total_size = 0
        for doc in doc_list:
            cls_name = type(doc).__name__
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            total_size += len(doc.content)
        parts = [f"{name} x{count}" for name, count in class_counts.items()]
        return f"{len(doc_list)} documents ({', '.join(parts)}), total {total_size / 1024:.0f}KB"
    if isinstance(result, Document):
        size = len(result.content)
        return f"{type(result).__name__}(name={result.name!r}, {size / 1024:.0f}KB, {result.mime_type})"
    if isinstance(result, (list, tuple)):
        seq = cast(list[Any] | tuple[Any, ...], result)
        return f"{type(seq).__name__} with {len(seq)} items"
    if isinstance(result, str):
        return f"str ({len(result)} chars)"
    return type(result).__name__


# --------------------------------------------------------------------------- #
# Store event emission helper
# --------------------------------------------------------------------------- #
def _emit_store_events(documents: list[Document], event_type: DocumentEventType) -> None:
    """Emit store lifecycle events for documents. No-op if tracking is not available."""
    try:
        service = get_tracking_service()
        if service is None:
            return
        span_id = get_current_span_id()
        for doc in documents:
            service.track_document_event(
                document_sha256=doc.sha256,
                span_id=span_id,
                event_type=event_type,
            )
    except Exception:
        logger.debug("Failed to emit store events", exc_info=True)


# --------------------------------------------------------------------------- #
# Document persistence helper (used by @pipeline_task and @pipeline_flow)
# --------------------------------------------------------------------------- #
async def _persist_documents(
    documents: list[Document],
    label: str,
    ctx: "TaskDocumentContext",
    *,
    check_created: bool = False,
) -> None:
    """Validate provenance, deduplicate, and save documents to the store.

    Silently skips if no store or no run context is configured.
    Logs warnings on persistence failure (graceful degradation).

    Args:
        check_created: When True, warn if a returned document was not created in this
            context. Enabled for @pipeline_task, disabled for @pipeline_flow (flows
            delegate creation to nested tasks).
    """
    run_ctx = get_run_context()
    store = get_document_store()
    if run_ctx is None or store is None:
        return

    if not documents:
        return

    deduped: list[Document] = []
    try:
        # Collect all SHA256 references (sources + origins) for existence check
        ref_sha256s: set[DocumentSha256] = set()
        for doc in documents:
            for src in doc.sources:
                if is_document_sha256(src):
                    ref_sha256s.add(DocumentSha256(src))
            for origin in doc.origins:
                ref_sha256s.add(DocumentSha256(origin))

        existing: set[DocumentSha256] = set()
        if ref_sha256s:
            existing = await store.check_existing(sorted(ref_sha256s))

        provenance_warnings = ctx.validate_provenance(documents, existing, check_created=check_created)
        for warning in provenance_warnings:
            logger.warning("[%s] %s", label, warning)

        # Deduplicate and save
        deduped = TaskDocumentContext.deduplicate(documents)
        await store.save_batch(deduped, run_ctx.run_scope)

        _emit_store_events(deduped, DocumentEventType.STORE_SAVED)

        # Finalize: warn about created-but-not-returned documents
        finalize_warnings = ctx.finalize(documents)
        for warning in finalize_warnings:
            logger.warning("[%s] %s", label, warning)
    except Exception:
        _emit_store_events(deduped or documents, DocumentEventType.STORE_SAVE_FAILED)
        logger.warning("Failed to persist documents from '%s'", label, exc_info=True)


# --------------------------------------------------------------------------- #
# @pipeline_task — async-only, traced, auto-persists documents
# --------------------------------------------------------------------------- #
@overload
def pipeline_task(__fn: Callable[..., Coroutine[Any, Any, R_co]], /) -> _TaskLike[R_co]: ...  # noqa: UP047
@overload
def pipeline_task(
    *,
    # tracing
    trace_level: TraceLevel = "always",
    trace_ignore_input: bool = False,
    trace_ignore_output: bool = False,
    trace_ignore_inputs: list[str] | None = None,
    trace_input_formatter: Callable[..., str] | None = None,
    trace_output_formatter: Callable[..., str] | None = None,
    trace_cost: float | None = None,
    expected_cost: float | None = None,
    trace_trim_documents: bool = True,
    # tracking
    user_summary: bool | str = False,
    # document lifecycle
    estimated_minutes: int = 1,
    persist: bool = True,
    # prefect passthrough
    name: str | None = None,
    description: str | None = None,
    tags: Iterable[str] | None = None,
    version: str | None = None,
    cache_policy: CachePolicy | type[NotSet] = NotSet,
    cache_key_fn: Callable[[TaskRunContext, dict[str, Any]], str | None] | None = None,
    cache_expiration: datetime.timedelta | None = None,
    task_run_name: TaskRunNameValueOrCallable | None = None,
    retries: int | None = None,
    retry_delay_seconds: int | float | list[float] | Callable[[int], list[float]] | None = None,
    retry_jitter_factor: float | None = None,
    persist_result: bool | None = None,
    result_storage: ResultStorage | str | None = None,
    result_serializer: ResultSerializer | str | None = None,
    result_storage_key: str | None = None,
    cache_result_in_memory: bool = True,
    timeout_seconds: int | float | None = None,
    log_prints: bool | None = False,
    refresh_cache: bool | None = None,
    on_completion: list[StateHookCallable] | None = None,
    on_failure: list[StateHookCallable] | None = None,
    retry_condition_fn: RetryConditionCallable | None = None,
    viz_return_value: bool | None = None,
    asset_deps: list[str | Asset] | None = None,
) -> Callable[[Callable[..., Coroutine[Any, Any, R_co]]], _TaskLike[R_co]]: ...


def pipeline_task(  # noqa: UP047
    __fn: Callable[..., Coroutine[Any, Any, R_co]] | None = None,
    /,
    *,
    # tracing
    trace_level: TraceLevel = "always",
    trace_ignore_input: bool = False,
    trace_ignore_output: bool = False,
    trace_ignore_inputs: list[str] | None = None,
    trace_input_formatter: Callable[..., str] | None = None,
    trace_output_formatter: Callable[..., str] | None = None,
    trace_cost: float | None = None,
    expected_cost: float | None = None,
    trace_trim_documents: bool = True,
    # tracking
    user_summary: bool | str = False,
    # document lifecycle
    estimated_minutes: int = 1,
    persist: bool = True,
    # prefect passthrough
    name: str | None = None,
    description: str | None = None,
    tags: Iterable[str] | None = None,
    version: str | None = None,
    cache_policy: CachePolicy | type[NotSet] = NotSet,
    cache_key_fn: Callable[[TaskRunContext, dict[str, Any]], str | None] | None = None,
    cache_expiration: datetime.timedelta | None = None,
    task_run_name: TaskRunNameValueOrCallable | None = None,
    retries: int | None = None,
    retry_delay_seconds: int | float | list[float] | Callable[[int], list[float]] | None = None,
    retry_jitter_factor: float | None = None,
    persist_result: bool | None = None,
    result_storage: ResultStorage | str | None = None,
    result_serializer: ResultSerializer | str | None = None,
    result_storage_key: str | None = None,
    cache_result_in_memory: bool = True,
    timeout_seconds: int | float | None = None,
    log_prints: bool | None = False,
    refresh_cache: bool | None = None,
    on_completion: list[StateHookCallable] | None = None,
    on_failure: list[StateHookCallable] | None = None,
    retry_condition_fn: RetryConditionCallable | None = None,
    viz_return_value: bool | None = None,
    asset_deps: list[str | Asset] | None = None,
) -> _TaskLike[R_co] | Callable[[Callable[..., Coroutine[Any, Any, R_co]]], _TaskLike[R_co]]:
    """Decorate an async function as a traced Prefect task with document auto-save.

    After the wrapped function returns, if documents are found in the result
    and a DocumentStore + RunContext are available, documents are validated
    for provenance, deduplicated by SHA256, and saved to the store.

    When persist=True (default), the return type annotation is validated at
    decoration time. Allowed return types::

        -> MyDocument                           # single Document
        -> list[DocA]  /  list[DocA | DocB]     # list of Documents
        -> tuple[DocA, DocB]                    # tuple of Documents
        -> tuple[list[DocA], list[DocB]]        # tuple of lists
        -> tuple[DocA, ...]                     # variable-length tuple
        -> None                                 # side-effect tasks
        -> DocA | None                          # optional Document

    Use persist=False for tasks returning non-document values (tracing and
    retries still apply, but no return type validation or document auto-save).

    Args:
        __fn: Function to decorate (when used without parentheses).
        trace_level: When to trace ("always", "debug", "off").
        trace_ignore_input: Don't trace input arguments.
        trace_ignore_output: Don't trace return value.
        trace_ignore_inputs: List of parameter names to exclude from tracing.
        trace_input_formatter: Custom formatter for input tracing.
        trace_output_formatter: Custom formatter for output tracing.
        trace_cost: Optional cost value to track in metadata.
        expected_cost: Optional expected cost budget for this task.
        trace_trim_documents: Trim document content in traces (default True).
        user_summary: Enable LLM-generated span summaries.
        estimated_minutes: Estimated duration for progress tracking (must be > 0).
        persist: Auto-save returned documents to the store (default True).
        name: Task name (defaults to function name).
        description: Human-readable task description.
        tags: Tags for organization and filtering.
        version: Task version string.
        cache_policy: Caching policy for task results.
        cache_key_fn: Custom cache key generation.
        cache_expiration: How long to cache results.
        task_run_name: Dynamic or static run name.
        retries: Number of retry attempts (default 0).
        retry_delay_seconds: Delay between retries.
        retry_jitter_factor: Random jitter for retry delays.
        persist_result: Whether to persist results.
        result_storage: Where to store results.
        result_serializer: How to serialize results.
        result_storage_key: Custom storage key.
        cache_result_in_memory: Keep results in memory.
        timeout_seconds: Task execution timeout.
        log_prints: Capture print() statements.
        refresh_cache: Force cache refresh.
        on_completion: Hooks for successful completion.
        on_failure: Hooks for task failure.
        retry_condition_fn: Custom retry condition.
        viz_return_value: Include return value in visualization.
        asset_deps: Upstream asset dependencies.
    """
    if estimated_minutes < 1:
        raise ValueError(f"estimated_minutes must be >= 1, got {estimated_minutes}")

    task_decorator: Callable[..., Any] = _prefect_task

    def _apply(fn: Callable[..., Coroutine[Any, Any, R_co]]) -> _TaskLike[R_co]:
        fname = _callable_name(fn, "task")

        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"@pipeline_task target '{fname}' must be 'async def'")

        if _is_already_traced(fn):
            raise TypeError(
                f"@pipeline_task target '{fname}' is already decorated "
                f"with @trace. Remove the @trace decorator - @pipeline_task includes "
                f"tracing automatically."
            )

        # Reject stale DocumentList references in annotations
        for ann_name, ann_value in getattr(fn, "__annotations__", {}).items():
            if "DocumentList" in str(ann_value):
                label = "return type" if ann_name == "return" else f"parameter '{ann_name}'"
                raise TypeError(f"@pipeline_task '{fname}' {label} references 'DocumentList' which has been removed. Use 'list[Document]' instead.")

        # Validate return type annotation when persist=True
        if persist:
            hints = _resolve_type_hints(fn)
            if "return" not in hints:
                raise TypeError(
                    f"@pipeline_task '{fname}': missing return type annotation. "
                    f"Persisted tasks must return Document types "
                    f"(Document, list[Document], tuple[Document, ...], or None). "
                    f"Add a return annotation or use persist=False."
                )
            bad_types = _find_non_document_leaves(hints["return"])
            if bad_types:
                bad_names = ", ".join(getattr(t, "__name__", str(t)) for t in bad_types)
                raise TypeError(
                    f"@pipeline_task '{fname}': return type contains non-Document types: {bad_names}. "
                    f"Persisted tasks must return Document, list[Document], "
                    f"tuple[Document, ...], or None. "
                    f"Use persist=False for tasks returning non-document values."
                )

        @wraps(fn)
        async def _wrapper(*args: Any, **kwargs: Any) -> R_co:
            attrs: dict[str, Any] = {}
            if description:
                attrs["description"] = description
            if expected_cost is not None:
                attrs["expected_cost"] = expected_cost
            if attrs:
                try:
                    Laminar.set_span_attributes(attrs)  # pyright: ignore[reportArgumentType]
                except Exception:
                    logger.debug("Failed to set span attributes", exc_info=True)

            # Set up TaskDocumentContext BEFORE calling fn() so Document.__init__ can register
            ctx: TaskDocumentContext | None = None
            task_token = None
            if persist and get_run_context() is not None and get_document_store() is not None:
                ctx = TaskDocumentContext()
                task_token = set_task_context(ctx)

            try:
                result = await fn(*args, **kwargs)
            finally:
                if task_token is not None:
                    reset_task_context(task_token)

            if trace_cost is not None and trace_cost > 0:
                set_trace_cost(trace_cost)

            # Track task I/O and schedule summaries
            try:
                track_task_io(args, kwargs, result)
            except Exception:
                logger.debug("Failed to track task IO", exc_info=True)

            if user_summary:
                try:
                    service = get_tracking_service()
                    if service is not None:
                        span_id = get_current_span_id()
                        if span_id:
                            label = _resolve_label(user_summary, fn, kwargs)
                            output_hint = _build_output_hint(result)
                            service.schedule_summary(span_id, label, output_hint)
                except Exception:
                    logger.debug("Failed to schedule user summary", exc_info=True)

            # Document auto-save
            if persist and ctx is not None:
                await _persist_documents(_extract_documents(result), fname, ctx, check_created=True)

            return result

        traced_fn = trace(
            level=trace_level,
            name=name or fname,
            ignore_input=trace_ignore_input,
            ignore_output=trace_ignore_output,
            ignore_inputs=trace_ignore_inputs,
            input_formatter=trace_input_formatter,
            output_formatter=trace_output_formatter,
            trim_documents=trace_trim_documents,
        )(_wrapper)

        task_obj = cast(
            _TaskLike[R_co],
            task_decorator(
                name=name or fname,
                description=description,
                tags=tags,
                version=version,
                cache_policy=cache_policy,
                cache_key_fn=cache_key_fn,
                cache_expiration=cache_expiration,
                task_run_name=task_run_name or name or fname,
                retries=0 if retries is None else retries,
                retry_delay_seconds=retry_delay_seconds,
                retry_jitter_factor=retry_jitter_factor,
                persist_result=persist_result,
                result_storage=result_storage,
                result_serializer=result_serializer,
                result_storage_key=result_storage_key,
                cache_result_in_memory=cache_result_in_memory,
                timeout_seconds=timeout_seconds,
                log_prints=log_prints,
                refresh_cache=refresh_cache,
                on_completion=on_completion,
                on_failure=on_failure,
                retry_condition_fn=retry_condition_fn,
                viz_return_value=viz_return_value,
                asset_deps=asset_deps,
            )(traced_fn),
        )
        task_obj.estimated_minutes = estimated_minutes
        return task_obj

    return _apply(__fn) if __fn else _apply


# --------------------------------------------------------------------------- #
# @pipeline_flow — async-only, traced, annotation-driven document types
# --------------------------------------------------------------------------- #
def pipeline_flow(
    *,
    # tracing
    trace_level: TraceLevel = "always",
    trace_ignore_input: bool = False,
    trace_ignore_output: bool = False,
    trace_ignore_inputs: list[str] | None = None,
    trace_input_formatter: Callable[..., str] | None = None,
    trace_output_formatter: Callable[..., str] | None = None,
    trace_cost: float | None = None,
    expected_cost: float | None = None,
    trace_trim_documents: bool = True,
    # tracking
    user_summary: bool | str = False,
    # document type specification
    estimated_minutes: int = 1,
    # prefect passthrough
    name: str | None = None,
    version: str | None = None,
    flow_run_name: Callable[[], str] | str | None = None,
    retries: int | None = None,
    retry_delay_seconds: int | float | None = None,
    task_runner: TaskRunner[PrefectFuture[Any]] | None = None,
    description: str | None = None,
    timeout_seconds: int | float | None = None,
    validate_parameters: bool = True,
    persist_result: bool | None = None,
    result_storage: ResultStorage | str | None = None,
    result_serializer: ResultSerializer | str | None = None,
    cache_result_in_memory: bool = True,
    log_prints: bool | None = None,
    on_completion: list[FlowStateHook[Any, Any]] | None = None,
    on_failure: list[FlowStateHook[Any, Any]] | None = None,
    on_cancellation: list[FlowStateHook[Any, Any]] | None = None,
    on_crashed: list[FlowStateHook[Any, Any]] | None = None,
    on_running: list[FlowStateHook[Any, Any]] | None = None,
) -> Callable[[Callable[..., Coroutine[Any, Any, list[Document]]]], _FlowLike[Any]]:
    """Decorate an async function as a traced Prefect flow with annotation-driven document types.

    Extracts input/output document types from the function's type annotations
    at decoration time and attaches them as ``input_document_types`` and
    ``output_document_types`` attributes on the returned flow object.

    Required function signature::

        @pipeline_flow(estimated_minutes=30)
        async def my_flow(
            project_name: str,
            documents: list[DocA | DocB],
            flow_options: FlowOptions,
        ) -> list[OutputDoc]:
            ...

    Args:
        user_summary: Enable LLM-generated span summaries.
        estimated_minutes: Estimated duration for progress tracking (must be >= 1).

    Returns:
        Decorator that produces a _FlowLike object with ``input_document_types``,
        ``output_document_types``, and ``estimated_minutes`` attributes.

    Raises:
        TypeError: If the function is not async, has wrong parameter count/types,
            missing return annotation, or output types overlap input types.
        ValueError: If estimated_minutes < 1.
    """
    if estimated_minutes < 1:
        raise ValueError(f"estimated_minutes must be >= 1, got {estimated_minutes}")

    flow_decorator: Callable[..., Any] = _prefect_flow

    def _apply(fn: Callable[..., Coroutine[Any, Any, list[Document]]]) -> _FlowLike[Any]:
        fname = _callable_name(fn, "flow")

        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"@pipeline_flow '{fname}' must be declared with 'async def'")

        if _is_already_traced(fn):
            raise TypeError(
                f"@pipeline_flow target '{fname}' is already decorated "
                f"with @trace. Remove the @trace decorator - @pipeline_flow includes "
                f"tracing automatically."
            )

        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if len(params) != 3:
            raise TypeError(
                f"@pipeline_flow '{fname}' must have exactly 3 parameters "
                f"(project_name: str, documents: list[...], flow_options: FlowOptions), got {len(params)}"
            )

        # Resolve document types from annotations
        hints = _resolve_type_hints(fn)

        # Validate first parameter is str
        if params[0].name in hints and hints[params[0].name] is not str:
            raise TypeError(f"@pipeline_flow '{fname}': first parameter '{params[0].name}' must be annotated as 'str'")

        # Validate third parameter is FlowOptions or subclass
        if params[2].name in hints:
            p2_type = hints[params[2].name]
            if not (isinstance(p2_type, type) and issubclass(p2_type, FlowOptions)):
                raise TypeError(f"@pipeline_flow '{fname}': third parameter '{params[2].name}' must be FlowOptions or subclass, got {p2_type}")

        # Extract input types from documents parameter annotation
        resolved_input_types: list[type[Document]]
        if params[1].name in hints:
            resolved_input_types = _parse_document_types_from_annotation(hints[params[1].name])
        else:
            resolved_input_types = []

        # Extract output types from return annotation
        resolved_output_types: list[type[Document]]
        if "return" in hints:
            resolved_output_types = _parse_document_types_from_annotation(hints["return"])
        else:
            resolved_output_types = []

        # Validate return annotation contains Document subclasses
        if "return" in hints and not resolved_output_types:
            raise TypeError(
                f"@pipeline_flow '{fname}': return annotation does not contain "
                f"Document subclasses. Flows must return list[SomeDocument]. "
                f"Got: {hints['return']}."
            )

        # Output types must not overlap input types (skip for base Document used in generic flows)
        if resolved_output_types and resolved_input_types:
            overlap = set(resolved_output_types) & set(resolved_input_types) - {Document}
            if overlap:
                names = ", ".join(t.__name__ for t in overlap)
                raise TypeError(f"@pipeline_flow '{fname}': output types [{names}] cannot also be input types")

        @wraps(fn)
        async def _wrapper(
            project_name: str,
            documents: list[Document],
            flow_options: Any,
        ) -> list[Document]:
            attrs: dict[str, Any] = {}
            if description:
                attrs["description"] = description
            if expected_cost is not None:
                attrs["expected_cost"] = expected_cost
            if attrs:
                try:
                    Laminar.set_span_attributes(attrs)  # pyright: ignore[reportArgumentType]
                except Exception:
                    logger.debug("Failed to set span attributes", exc_info=True)

            # Set RunContext for nested tasks (only if not already set by deployment)
            existing_ctx = get_run_context()
            run_token = None
            if existing_ctx is None:
                run_scope = RunScope(f"{project_name}/{name or fname}")
                run_token = set_run_context(RunContext(run_scope=run_scope))

            # Set up TaskDocumentContext for flow-level document lifecycle
            ctx: TaskDocumentContext | None = None
            task_token = None
            if get_run_context() is not None and get_document_store() is not None:
                ctx = TaskDocumentContext()
                task_token = set_task_context(ctx)

            try:
                result = await fn(project_name, documents, flow_options)
            finally:
                if task_token is not None:
                    reset_task_context(task_token)
                if run_token is not None:
                    reset_run_context(run_token)

            if trace_cost is not None and trace_cost > 0:
                set_trace_cost(trace_cost)
            if not isinstance(result, list):  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime guard
                raise TypeError(f"Flow '{fname}' must return list[Document], got {type(result).__name__}")

            # Track flow I/O
            try:
                track_flow_io(documents, result)
            except Exception:
                logger.debug("Failed to track flow IO", exc_info=True)

            if user_summary:
                try:
                    service = get_tracking_service()
                    if service is not None:
                        span_id = get_current_span_id()
                        if span_id:
                            label = _resolve_label(user_summary, fn, {"project_name": project_name, "flow_options": flow_options})
                            output_hint = _build_output_hint(result)
                            service.schedule_summary(span_id, label, output_hint)
                except Exception:
                    logger.debug("Failed to schedule user summary", exc_info=True)

            # Document auto-save
            if ctx is not None:
                await _persist_documents(result, fname, ctx)

            return result

        traced = trace(
            level=trace_level,
            name=name or fname,
            ignore_input=trace_ignore_input,
            ignore_output=trace_ignore_output,
            ignore_inputs=trace_ignore_inputs,
            input_formatter=trace_input_formatter,
            output_formatter=trace_output_formatter,
            trim_documents=trace_trim_documents,
        )(_wrapper)

        flow_obj = cast(
            _FlowLike[Any],
            flow_decorator(
                name=name or fname,
                version=version,
                flow_run_name=flow_run_name or name or fname,
                retries=0 if retries is None else retries,
                retry_delay_seconds=retry_delay_seconds,
                task_runner=task_runner,
                description=description,
                timeout_seconds=timeout_seconds,
                validate_parameters=validate_parameters,
                persist_result=persist_result,
                result_storage=result_storage,
                result_serializer=result_serializer,
                cache_result_in_memory=cache_result_in_memory,
                log_prints=log_prints,
                on_completion=on_completion,
                on_failure=on_failure,
                on_cancellation=on_cancellation,
                on_crashed=on_crashed,
                on_running=on_running,
            )(traced),
        )
        flow_obj.input_document_types = resolved_input_types
        flow_obj.output_document_types = resolved_output_types
        flow_obj.estimated_minutes = estimated_minutes
        return flow_obj

    return _apply


__all__ = ["pipeline_flow", "pipeline_task"]
