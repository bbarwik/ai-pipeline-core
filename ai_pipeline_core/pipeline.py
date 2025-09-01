"""Pipeline decorators with Prefect integration and tracing.

@public

Wrappers around Prefect's @task and @flow that add Laminar tracing
and enforce async-only execution for consistency.
"""

from __future__ import annotations

import datetime
import inspect
from functools import wraps
from typing import (
    Any,
    Callable,
    Coroutine,
    Iterable,
    Protocol,
    TypeVar,
    Union,
    cast,
    overload,
)

from prefect.assets import Asset
from prefect.cache_policies import CachePolicy
from prefect.context import TaskRunContext
from prefect.flows import FlowStateHook
from prefect.flows import flow as _prefect_flow  # public import
from prefect.futures import PrefectFuture
from prefect.results import ResultSerializer, ResultStorage
from prefect.task_runners import TaskRunner
from prefect.tasks import task as _prefect_task  # public import
from prefect.utilities.annotations import NotSet
from typing_extensions import TypeAlias

from ai_pipeline_core.documents import DocumentList
from ai_pipeline_core.flow.options import FlowOptions
from ai_pipeline_core.tracing import TraceLevel, set_trace_cost, trace

# --------------------------------------------------------------------------- #
# Public callback aliases (Prefect stubs omit these exact types)
# --------------------------------------------------------------------------- #
RetryConditionCallable: TypeAlias = Callable[[Any, Any, Any], bool]
StateHookCallable: TypeAlias = Callable[[Any, Any, Any], None]
TaskRunNameValueOrCallable: TypeAlias = Union[str, Callable[[], str]]

# --------------------------------------------------------------------------- #
# Typing helpers
# --------------------------------------------------------------------------- #
R_co = TypeVar("R_co", covariant=True)
FO_contra = TypeVar("FO_contra", bound=FlowOptions, contravariant=True)
"""Flow options are an *input* type, so contravariant fits the callable model."""


class _TaskLike(Protocol[R_co]):
    """Protocol for type-safe Prefect task representation.

    Defines the minimal interface for a Prefect task as seen by
    type checkers. Ensures tasks are awaitable and have common
    Prefect task methods.

    Type Parameter:
        R_co: Covariant return type of the task.

    Methods:
        __call__: Makes the task awaitable.
        submit: Submit task for asynchronous execution.
        map: Map task over multiple inputs.

    Attributes:
        name: Optional task name.

    Note:
        This is a typing Protocol, not a runtime class.
        __getattr__ allows accessing Prefect-specific helpers.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, R_co]: ...

    submit: Callable[..., Any]
    map: Callable[..., Any]
    name: str | None

    def __getattr__(self, name: str) -> Any: ...  # allow unknown helpers without type errors


class _DocumentsFlowCallable(Protocol[FO_contra]):
    """Protocol for user-defined flow functions.

    Defines the required signature for functions that will be
    decorated with @pipeline_flow. Enforces the standard parameters
    for document processing flows.

    Type Parameter:
        FO_contra: Contravariant FlowOptions type (or subclass).

    Required Parameters:
        project_name: Name of the project/pipeline.
        documents: Input DocumentList to process.
        flow_options: Configuration options (FlowOptions or subclass).
        *args, **kwargs: Additional flow-specific parameters.

    Returns:
        DocumentList: Processed documents.

    Note:
        Functions must be async and return DocumentList.
    """

    def __call__(
        self,
        project_name: str,
        documents: DocumentList,
        flow_options: FO_contra,
        *args: Any,
        **kwargs: Any,
    ) -> Coroutine[Any, Any, DocumentList]: ...


class _FlowLike(Protocol[FO_contra]):
    """Protocol for decorated flow objects returned to users.

    Represents the callable object returned by @pipeline_flow,
    which wraps the original flow function with Prefect and
    tracing capabilities.

    Type Parameter:
        FO_contra: Contravariant FlowOptions type.

    Callable Signature:
        Same as _DocumentsFlowCallable - accepts project_name,
        documents, flow_options, plus additional arguments.

    Attributes:
        name: Optional flow name from decorator.

    Note:
        __getattr__ provides access to all Prefect flow methods
        without explicit typing (e.g., .serve(), .deploy()).
    """

    def __call__(
        self,
        project_name: str,
        documents: DocumentList,
        flow_options: FO_contra,
        *args: Any,
        **kwargs: Any,
    ) -> Coroutine[Any, Any, DocumentList]: ...

    name: str | None

    def __getattr__(self, name: str) -> Any: ...  # allow unknown helpers without type errors


# --------------------------------------------------------------------------- #
# Small helper: safely get a callable's name without upsetting the type checker
# --------------------------------------------------------------------------- #
def _callable_name(obj: Any, fallback: str) -> str:
    """Safely extract callable's name for error messages.

    Args:
        obj: Any object that might have a __name__ attribute.
        fallback: Default name if extraction fails.

    Returns:
        The callable's __name__ if available, fallback otherwise.

    Note:
        Internal helper that never raises exceptions.
    """
    try:
        n = getattr(obj, "__name__", None)
        return n if isinstance(n, str) else fallback
    except Exception:
        return fallback


def _is_already_traced(func: Callable[..., Any]) -> bool:
    """Check if a function has already been wrapped by the trace decorator.

    This checks both for the explicit __is_traced__ marker and walks
    the __wrapped__ chain to detect nested trace decorations.

    Args:
        func: Function to check for existing trace decoration.

    Returns:
        True if the function is already traced, False otherwise.
    """
    # Check for explicit marker
    if hasattr(func, "__is_traced__") and func.__is_traced__:  # type: ignore[attr-defined]
        return True

    # Walk the __wrapped__ chain to detect nested traces
    current = func
    depth = 0
    max_depth = 10  # Prevent infinite loops

    while hasattr(current, "__wrapped__") and depth < max_depth:
        wrapped = current.__wrapped__  # type: ignore[attr-defined]
        # Check if the wrapped function has the trace marker
        if hasattr(wrapped, "__is_traced__") and wrapped.__is_traced__:  # type: ignore[attr-defined]
            return True
        current = wrapped
        depth += 1

    return False


# --------------------------------------------------------------------------- #
# @pipeline_task — async-only, traced, returns Prefect's Task object
# --------------------------------------------------------------------------- #
@overload
def pipeline_task(__fn: Callable[..., Coroutine[Any, Any, R_co]], /) -> _TaskLike[R_co]: ...
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


def pipeline_task(
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
    """Decorate an async function as a traced Prefect task.

    @public

    Wraps an async function with both Prefect task functionality and
    LMNR tracing. The function MUST be async (declared with 'async def').

    IMPORTANT: Never combine with @trace decorator - this includes tracing automatically.
    The framework will raise TypeError if you try to use both decorators together.

    Best Practice - Use Defaults:
        For 90% of use cases, use this decorator WITHOUT any parameters.
        Only specify parameters when you have EXPLICIT requirements.

    Args:
        __fn: Function to decorate (when used without parentheses).

        Tracing parameters:
        trace_level: When to trace ("always", "debug", "off").
                    - "always": Always trace (default)
                    - "debug": Only trace when LMNR_DEBUG="true"
                    - "off": Disable tracing
        trace_ignore_input: Don't trace input arguments.
        trace_ignore_output: Don't trace return value.
        trace_ignore_inputs: List of parameter names to exclude from tracing.
        trace_input_formatter: Custom formatter for input tracing.
        trace_output_formatter: Custom formatter for output tracing.
        trace_cost: Optional cost value to track in metadata. When provided and > 0,
             sets gen_ai.usage.output_cost, gen_ai.usage.cost, and cost metadata.
             Also forces trace level to "always" if not already set.

        Prefect task parameters:
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

    Returns:
        Decorated task callable that is awaitable and has Prefect
        task methods (submit, map, etc.).

    Example:
        >>> # RECOMMENDED - No parameters needed!
        >>> @pipeline_task
        >>> async def process_document(doc: Document) -> Document:
        ...     result = await analyze(doc)
        ...     return result
        >>>
        >>> # With parameters (only when necessary):
        >>> @pipeline_task(retries=5)  # Only for known flaky operations
        >>> async def unreliable_api_call(url: str) -> dict:
        ...     # This API fails often, needs extra retries
        ...     return await fetch_with_retry(url)
        >>>
        >>> # AVOID specifying defaults - they're already optimal:
        >>> # - Automatic task naming
        >>> # - Standard retry policy
        >>> # - Sensible timeout
        >>> # - Full observability

    Performance:
        - Task decoration overhead: ~1-2ms
        - Tracing overhead: ~1-2ms per call
        - Prefect state tracking: ~5-10ms

    Note:
        Tasks are automatically traced with LMNR and appear in
        both Prefect and LMNR dashboards.

    See Also:
        - pipeline_flow: For flow-level decoration
        - trace: Lower-level tracing decorator
        - prefect.task: Standard Prefect task (no tracing)
    """
    task_decorator: Callable[..., Any] = _prefect_task  # helps the type checker

    def _apply(fn: Callable[..., Coroutine[Any, Any, R_co]]) -> _TaskLike[R_co]:
        """Apply pipeline_task decorator to async function.

        Returns:
            Wrapped task with tracing and Prefect functionality.

        Raises:
            TypeError: If function is not async or already traced.
        """
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(
                f"@pipeline_task target '{_callable_name(fn, 'task')}' must be 'async def'"
            )

        # Check if function is already traced
        if _is_already_traced(fn):
            raise TypeError(
                f"@pipeline_task target '{_callable_name(fn, 'task')}' is already decorated "
                f"with @trace. Remove the @trace decorator - @pipeline_task includes "
                f"tracing automatically."
            )

        fname = _callable_name(fn, "task")

        # Create wrapper to handle trace_cost if provided
        @wraps(fn)
        async def _wrapper(*args: Any, **kwargs: Any) -> R_co:
            result = await fn(*args, **kwargs)
            if trace_cost is not None and trace_cost > 0:
                set_trace_cost(trace_cost)
            return result

        # Preserve the original function name for Prefect
        _wrapper.__name__ = fname
        _wrapper.__qualname__ = getattr(fn, "__qualname__", fname)

        traced_fn = trace(
            level=trace_level,
            name=name or fname,
            ignore_input=trace_ignore_input,
            ignore_output=trace_ignore_output,
            ignore_inputs=trace_ignore_inputs,
            input_formatter=trace_input_formatter,
            output_formatter=trace_output_formatter,
        )(_wrapper)

        return cast(
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

    return _apply(__fn) if __fn else _apply


# --------------------------------------------------------------------------- #
# @pipeline_flow — async-only, traced, returns Prefect's flow wrapper
# --------------------------------------------------------------------------- #
@overload
def pipeline_flow(__fn: _DocumentsFlowCallable[FO_contra], /) -> _FlowLike[FO_contra]: ...
@overload
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
    # prefect passthrough
    name: str | None = None,
    version: str | None = None,
    flow_run_name: Union[Callable[[], str], str] | None = None,
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
) -> Callable[[_DocumentsFlowCallable[FO_contra]], _FlowLike[FO_contra]]: ...


def pipeline_flow(
    __fn: _DocumentsFlowCallable[FO_contra] | None = None,
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
    # prefect passthrough
    name: str | None = None,
    version: str | None = None,
    flow_run_name: Union[Callable[[], str], str] | None = None,
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
) -> _FlowLike[FO_contra] | Callable[[_DocumentsFlowCallable[FO_contra]], _FlowLike[FO_contra]]:
    """Decorate an async flow for document processing.

    @public

    Wraps an async function as a Prefect flow with tracing and type safety.
    The decorated function MUST be async and follow the required signature.

    IMPORTANT: Never combine with @trace decorator - this includes tracing automatically.
    The framework will raise TypeError if you try to use both decorators together.

    Best Practice - Use Defaults:
        For 90% of use cases, use this decorator WITHOUT any parameters.
        Only specify parameters when you have EXPLICIT requirements.

    Required function signature:
        async def flow_fn(
            project_name: str,         # Project/pipeline identifier
            documents: DocumentList,   # Input documents to process
            flow_options: FlowOptions, # Configuration (or subclass)
            *args,                     # Additional positional args for custom parameters
            **kwargs                   # Additional keyword args for custom parameters
        ) -> DocumentList             # Must return DocumentList

    Note: *args and **kwargs allow for defining custom parameters on your flow
    function, which can be passed during execution for flow-specific needs.

    Args:
        __fn: Function to decorate (when used without parentheses).

        Tracing parameters:
        trace_level: When to trace ("always", "debug", "off").
                    - "always": Always trace (default)
                    - "debug": Only trace when LMNR_DEBUG="true"
                    - "off": Disable tracing
        trace_ignore_input: Don't trace input arguments.
        trace_ignore_output: Don't trace return value.
        trace_ignore_inputs: Parameter names to exclude from tracing.
        trace_input_formatter: Custom input formatter.
        trace_output_formatter: Custom output formatter.
        trace_cost: Optional cost value to track in metadata. When provided and > 0,
             sets gen_ai.usage.output_cost, gen_ai.usage.cost, and cost metadata.
             Also forces trace level to "always" if not already set.

        Prefect flow parameters:
        name: Flow name (defaults to function name).
        version: Flow version identifier.
        flow_run_name: Static or dynamic run name.
        retries: Number of flow retry attempts (default 0).
        retry_delay_seconds: Delay between flow retries.
        task_runner: Task execution strategy (sequential/concurrent).
        description: Human-readable flow description.
        timeout_seconds: Flow execution timeout.
        validate_parameters: Validate input parameters.
        persist_result: Persist flow results.
        result_storage: Where to store results.
        result_serializer: How to serialize results.
        cache_result_in_memory: Keep results in memory.
        log_prints: Capture print() statements.
        on_completion: Hooks for successful completion.
        on_failure: Hooks for flow failure.
        on_cancellation: Hooks for flow cancellation.
        on_crashed: Hooks for flow crashes.
        on_running: Hooks for flow start.

    Returns:
        Decorated flow callable that maintains Prefect flow interface
        while enforcing document processing conventions.

    Example:
        >>> from ai_pipeline_core import FlowOptions
        >>>
        >>> # RECOMMENDED - No parameters needed!
        >>> @pipeline_flow
        >>> async def analyze_documents(
        ...     project_name: str,
        ...     documents: DocumentList,
        ...     flow_options: FlowOptions
        >>> ) -> DocumentList:
        ...     # Process each document
        ...     results = []
        ...     for doc in documents:
        ...         result = await process(doc)
        ...         results.append(result)
        ...     return DocumentList(results)
        >>>
        >>> # With parameters (only when necessary):
        >>> @pipeline_flow(retries=2)  # Only for flows that need retry logic
        >>> async def critical_flow(
        ...     project_name: str,
        ...     documents: DocumentList,
        ...     flow_options: FlowOptions
        >>> ) -> DocumentList:
        ...     # Critical processing that might fail
        ...     return await process_critical(documents)
        >>>
        >>> # AVOID specifying defaults - they're already optimal:
        >>> # - Automatic flow naming
        >>> # - Standard retry policy
        >>> # - Full observability

    Note:
        - Flow is wrapped with both Prefect and LMNR tracing
        - Return type is validated at runtime
        - FlowOptions can be subclassed for custom configuration
        - All Prefect flow methods (.serve(), .deploy()) are available

    See Also:
        - pipeline_task: For task-level decoration
        - FlowConfig: Type-safe flow configuration
        - FlowOptions: Base class for flow options
        - simple_runner.run_pipeline: Execute flows locally
    """
    flow_decorator: Callable[..., Any] = _prefect_flow

    def _apply(fn: _DocumentsFlowCallable[FO_contra]) -> _FlowLike[FO_contra]:
        """Apply pipeline_flow decorator to flow function.

        Returns:
            Wrapped flow with tracing and Prefect functionality.

        Raises:
            TypeError: If function is not async, already traced, doesn't have
                      required parameters, or doesn't return DocumentList.
        """
        fname = _callable_name(fn, "flow")

        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"@pipeline_flow '{fname}' must be declared with 'async def'")

        # Check if function is already traced
        if _is_already_traced(fn):
            raise TypeError(
                f"@pipeline_flow target '{fname}' is already decorated "
                f"with @trace. Remove the @trace decorator - @pipeline_flow includes "
                f"tracing automatically."
            )

        if len(inspect.signature(fn).parameters) < 3:
            raise TypeError(
                f"@pipeline_flow '{fname}' must accept "
                "'project_name, documents, flow_options' as its first three parameters"
            )

        async def _wrapper(
            project_name: str,
            documents: DocumentList,
            flow_options: FO_contra,
            *args: Any,
            **kwargs: Any,
        ) -> DocumentList:
            result = await fn(project_name, documents, flow_options, *args, **kwargs)
            if trace_cost is not None and trace_cost > 0:
                set_trace_cost(trace_cost)
            if not isinstance(result, DocumentList):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(
                    f"Flow '{fname}' must return DocumentList, got {type(result).__name__}"
                )
            return result

        # Preserve the original function name for Prefect
        _wrapper.__name__ = fname
        _wrapper.__qualname__ = getattr(fn, "__qualname__", fname)

        traced = trace(
            level=trace_level,
            name=name or fname,
            ignore_input=trace_ignore_input,
            ignore_output=trace_ignore_output,
            ignore_inputs=trace_ignore_inputs,
            input_formatter=trace_input_formatter,
            output_formatter=trace_output_formatter,
        )(_wrapper)

        return cast(
            _FlowLike[FO_contra],
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

    return _apply(__fn) if __fn else _apply


__all__ = ["pipeline_task", "pipeline_flow"]
