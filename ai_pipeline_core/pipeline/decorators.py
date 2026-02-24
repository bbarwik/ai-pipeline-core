"""Pipeline decorators with Prefect integration, tracing, and document lifecycle.

Wrappers around Prefect's @task and @flow that add Laminar tracing,
enforce async-only execution, and auto-save documents to the DocumentStore.
"""

import datetime
import inspect
from collections.abc import Callable, Coroutine, Iterable
from functools import wraps
from typing import (
    Any,
    Protocol,
    TypeVar,
    cast,
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

from ai_pipeline_core.document_store._protocol import get_document_store
from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents._context_vars import (
    TaskContext,
    reset_task_context,
    set_task_context,
)
from ai_pipeline_core.documents.context import (
    RunContext,
    TaskDocumentContext,
    get_run_context,
    reset_run_context,
    set_run_context,
)
from ai_pipeline_core.documents.types import DocumentSha256, RunScope
from ai_pipeline_core.documents.utils import is_document_sha256
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._document_tracking import get_current_span_id, track_flow_io, track_task_io
from ai_pipeline_core.observability._initialization import get_tracking_service
from ai_pipeline_core.observability._tracking._models import DocumentEventType
from ai_pipeline_core.observability.tracing import TraceLevel, set_trace_cost, trace
from ai_pipeline_core.pipeline._type_validation import (
    callable_name,
    find_non_document_leaves,
    is_already_traced,
    parse_document_types_from_annotation,
    resolve_type_hints,
    validate_input_types,
)
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
        run_id: str,
        documents: list[Document],
        flow_options: FO_contra,
    ) -> Coroutine[Any, Any, list[Document]]: ...

    name: str | None
    input_document_types: list[type[Document]]
    output_document_types: list[type[Document]]
    estimated_minutes: int
    stub: bool

    def __getattr__(self, name: str) -> Any: ...


# --------------------------------------------------------------------------- #
# Document extraction helper
# --------------------------------------------------------------------------- #
def _extract_documents(result: Any) -> list[Document]:
    """Extract Document instances from a task/flow result.

    Only handles the return types validated at decoration time:
    Document, list[Document], tuple[Document, ...], or None.
    """
    if result is None:
        return []
    if isinstance(result, Document):
        return [result]
    if isinstance(result, (list, tuple)):
        return [item for item in cast(Iterable[Any], result) if isinstance(item, Document)]
    return []


# --------------------------------------------------------------------------- #
# Shared wrapper helpers (used by both @pipeline_task and @pipeline_flow)
# --------------------------------------------------------------------------- #
def _set_span_attrs(description: str | None, expected_cost: float | None) -> None:
    """Set Laminar span attributes. No-op on failure."""
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
) -> None:
    """Validate provenance, deduplicate, and save documents to the store.

    Silently skips if no store or no run context is configured.
    Logs warnings on persistence failure (graceful degradation).
    """
    run_ctx = get_run_context()
    store = get_document_store()
    if run_ctx is None or store is None:
        return

    if not documents:
        return

    deduped: list[Document] = []
    try:
        # Collect all SHA256 references (derived_from + triggered_by) for existence check
        ref_sha256s: set[DocumentSha256] = set()
        for doc in documents:
            for src in doc.derived_from:
                if is_document_sha256(src):
                    ref_sha256s.add(DocumentSha256(src))
            for trigger in doc.triggered_by:
                ref_sha256s.add(DocumentSha256(trigger))

        existing: set[DocumentSha256] = set()
        if ref_sha256s:
            existing = await store.check_existing(sorted(ref_sha256s))

        provenance_warnings = ctx.validate_provenance(documents, existing)
        for warning in provenance_warnings:
            logger.warning("[%s] %s", label, warning)

        # Detect orphaned documents (created but not returned)
        orphans = ctx.finalize(documents)
        for orphan in orphans:
            logger.warning("[%s] Orphaned document %s — created but not returned", label, orphan[:12])

        # Deduplicate and save
        deduped = TaskDocumentContext.deduplicate(documents)
        await store.save_batch(deduped, run_ctx.run_scope)

        _emit_store_events(deduped, DocumentEventType.STORE_SAVED)
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
    # document lifecycle
    estimated_minutes: int = 1,
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
    # document lifecycle
    estimated_minutes: int = 1,
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

    The return type annotation is validated at decoration time.
    Allowed return types::

        -> MyDocument                           # single Document
        -> list[DocA]  /  list[DocA | DocB]     # list of Documents
        -> tuple[DocA, DocB]                    # tuple of Documents
        -> tuple[list[DocA], list[DocB]]        # tuple of lists
        -> tuple[DocA, ...]                     # variable-length tuple
        -> None                                 # side-effect tasks
        -> DocA | None                          # optional Document

    For non-document functions, use plain ``async def`` with ``@trace`` instead.

    Document is the universal container for pipeline data. Any structured data
    (Pydantic models, dicts, lists) can be wrapped via Document.create():
    ``MyDoc.create(name='output.json', content=model, derived_from=(input.sha256,))``
    and retrieved via ``doc.parse(MyModel)``. There is no need for custom return
    types or ``persist=False`` — wrap everything in a Document.

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
        estimated_minutes: Estimated duration for progress tracking (must be > 0).
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
        fname = callable_name(fn, "task")

        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"@pipeline_task target '{fname}' must be 'async def'")

        if is_already_traced(fn):
            raise TypeError(
                f"@pipeline_task target '{fname}' is already decorated "
                f"with @trace. Remove the @trace decorator - @pipeline_task includes "
                f"tracing automatically."
            )

        # Validate input and return type annotations
        hints = resolve_type_hints(fn)
        validate_input_types(fn, hints)
        if "return" not in hints:
            raise TypeError(
                f"@pipeline_task '{fname}': missing return type annotation. "
                f"Pipeline tasks must return Document types "
                f"(Document, list[Document], tuple[Document, ...], or None). "
                f"Add a return type annotation."
            )
        bad_types = find_non_document_leaves(hints["return"])
        if bad_types:
            bad_names = ", ".join(getattr(t, "__name__", str(t)) for t in bad_types)
            raise TypeError(
                f"@pipeline_task '{fname}': return type contains non-Document types: {bad_names}. "
                f"Pipeline tasks must return Document, list[Document], "
                f"tuple[Document, ...], or None.\n"
                f"FIX: Wrap your result in a Document:\n"
                f"  return MyDocument.create(name='output.json', content=my_data, derived_from=(...))\n"
                f"Document.create() auto-serializes str, bytes, dict, list, and BaseModel.\n"
                f"For non-document functions, use plain async def with @trace instead of @pipeline_task."
            )

        @wraps(fn)
        async def _wrapper(*args: Any, **kwargs: Any) -> R_co:
            _set_span_attrs(description, expected_cost)

            # Set up task context for document lifecycle tracking
            task_ctx = TaskContext()
            task_token = set_task_context(task_ctx)
            try:
                result = await fn(*args, **kwargs)
            finally:
                reset_task_context(task_token)

            if trace_cost is not None and trace_cost > 0:
                set_trace_cost(trace_cost)

            # Track task I/O
            try:
                track_task_io(args, kwargs, result)
            except Exception:
                logger.debug("Failed to track task IO", exc_info=True)

            # Document auto-save
            if get_run_context() is not None and get_document_store() is not None:
                ctx = TaskDocumentContext(created=task_ctx.created)
                docs = _extract_documents(result)
                await _persist_documents(docs, fname, ctx)

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
    # document type specification
    estimated_minutes: int = 1,
    stub: bool = False,
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
            run_id: str,
            documents: list[DocA | DocB],
            flow_options: FlowOptions,
        ) -> list[OutputDoc]:
            ...

    Args:
        estimated_minutes: Weight for progress bar calculation only (must be >= 1).
            Does not affect execution timeout or scheduling.

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
        fname = callable_name(fn, "flow")

        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"@pipeline_flow '{fname}' must be declared with 'async def'")

        if is_already_traced(fn):
            raise TypeError(
                f"@pipeline_flow target '{fname}' is already decorated "
                f"with @trace. Remove the @trace decorator - @pipeline_flow includes "
                f"tracing automatically."
            )

        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if len(params) != 3:
            raise TypeError(
                f"@pipeline_flow '{fname}' must have exactly 3 parameters (run_id: str, documents: list[...], flow_options: FlowOptions), got {len(params)}"
            )

        # Resolve and validate type annotations
        hints = resolve_type_hints(fn)
        validate_input_types(fn, hints)

        # Validate first parameter is str
        if params[0].name in hints and hints[params[0].name] is not str:
            raise TypeError(f"@pipeline_flow '{fname}': first parameter '{params[0].name}' must be annotated as 'str'")

        # Validate first parameter is named 'run_id' or '_run_id'
        first_param_name = next(iter(sig.parameters.keys()))
        if first_param_name not in {"run_id", "_run_id"}:
            raise TypeError(f"@pipeline_flow '{fname}': first parameter must be named 'run_id' or '_run_id', got '{first_param_name}'")

        # Validate third parameter is FlowOptions or subclass
        if params[2].name in hints:
            p2_type = hints[params[2].name]
            if not (isinstance(p2_type, type) and issubclass(p2_type, FlowOptions)):
                raise TypeError(f"@pipeline_flow '{fname}': third parameter '{params[2].name}' must be FlowOptions or subclass, got {p2_type}")

        # Extract input types from documents parameter annotation
        resolved_input_types: list[type[Document]]
        if params[1].name in hints:
            resolved_input_types = parse_document_types_from_annotation(hints[params[1].name])
        else:
            resolved_input_types = []

        # Extract output types from return annotation
        resolved_output_types: list[type[Document]]
        if "return" in hints:
            resolved_output_types = parse_document_types_from_annotation(hints["return"])
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
            run_id: str,
            documents: list[Document],
            flow_options: Any,
        ) -> list[Document]:
            _set_span_attrs(description, expected_cost)

            # Set RunContext for nested tasks (only if not already set by deployment)
            existing_ctx = get_run_context()
            run_token = None
            if existing_ctx is None:
                run_scope = RunScope(f"{run_id}/{name or fname}")
                run_token = set_run_context(RunContext(run_scope=run_scope))

            # Set up task context for document lifecycle tracking
            task_ctx = TaskContext()
            task_token = set_task_context(task_ctx)
            try:
                result = await fn(run_id, documents, flow_options)
            finally:
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

            # Document auto-save
            if get_run_context() is not None and get_document_store() is not None:
                ctx = TaskDocumentContext(created=task_ctx.created)
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
        flow_obj.stub = stub
        return flow_obj

    return _apply


__all__ = ["pipeline_flow", "pipeline_task"]
