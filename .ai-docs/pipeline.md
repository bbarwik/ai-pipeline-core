# MODULE: pipeline
# CLASSES: LimitKind, PipelineLimit, FlowOptions
# DEPENDS: BaseSettings, StrEnum
# PURPOSE: Pipeline framework primitives — decorators, flow options, and concurrency limits.
# VERSION: 0.10.5
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import FlowOptions, LimitKind, PipelineLimit, pipeline_concurrency, pipeline_flow, pipeline_task, safe_gather, safe_gather_indexed
```

## Rules

1. Never inherit from FlowOptions for task-level options, writer configs, or programmatically-constructed parameter objects — use BaseModel instead. FlowOptions fields are always subject to env var override, which causes silent, hard-to-debug behavior when field names collide with common env vars (MODE, HOST, PORT, etc.).

## Types & Constants

```python
type RetryConditionCallable = Callable[[Any, Any, Any], bool]

type StateHookCallable = Callable[[Any, Any, Any], None]

type TaskRunNameValueOrCallable = str | Callable[[], str]

```

## Internal Types

```python
# Protocol — implement in concrete class
class _FlowLike(Protocol[FO_contra]):
    """Protocol for decorated flow objects returned by @pipeline_flow."""
    name: str | None
    input_document_types: list[type[Document]]
    output_document_types: list[type[Document]]
    estimated_minutes: int
    stub: bool

    def __call__(
        self,
        run_id: str,
        documents: Sequence[Document],
        flow_options: FO_contra,
    ) -> Coroutine[Any, Any, list[Document]]: ...

    def __getattr__(self, name: str) -> Any: ...


# Protocol — implement in concrete class
class _TaskLike(Protocol[R_co]):
    """Protocol for type-safe Prefect task representation."""
    submit: Callable[..., Any]
    map: Callable[..., Any]
    name: str | None
    estimated_minutes: int

    def __call__(self, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, R_co]: ...

    def __getattr__(self, name: str) -> Any: ...


```

## Public API

```python
# Enum
class LimitKind(StrEnum):
    """Kind of concurrency/rate limit.

CONCURRENT: Slots held for duration of operation (lease-based).
    limit=500 means at most 500 simultaneous operations across all runs.

PER_MINUTE: Token bucket with limit/60 decay per second.
    Allows bursting up to `limit` immediately, then refills gradually.
    NOT a sliding window.

PER_HOUR: Token bucket with limit/3600 decay per second. Same burst semantics."""
    CONCURRENT = 'concurrent'
    PER_MINUTE = 'per_minute'
    PER_HOUR = 'per_hour'


@dataclass(frozen=True, slots=True)
class PipelineLimit:
    """Concurrency/rate limit configuration.

limit: Maximum slots. For CONCURRENT: max simultaneous operations.
       For PER_MINUTE/PER_HOUR: token bucket capacity (burst size).
kind: Type of limit enforcement.
timeout: Max seconds to wait for slot acquisition."""
    limit: int
    kind: LimitKind = LimitKind.CONCURRENT
    timeout: int = 600

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError(f"limit must be >= 1, got {self.limit}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {self.timeout}")


class FlowOptions(BaseSettings):
    """Base configuration for pipeline flows. Uses pydantic-settings.

Every field defined on a FlowOptions subclass is automatically
overridable via environment variables (e.g., a field named 'mode'
reads from the MODE env var at instantiation time).

Use FlowOptions for deployment/environment configuration that may
differ between environments (dev/staging/production).

Never inherit from FlowOptions for task-level options, writer configs,
or programmatically-constructed parameter objects — use BaseModel instead.
FlowOptions fields are always subject to env var override, which causes
silent, hard-to-debug behavior when field names collide with common
env vars (MODE, HOST, PORT, etc.)."""
    model_config = SettingsConfigDict(frozen=True, extra='forbid')


```

## Functions

```python
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
        if contains_bare_document(hints["return"]):
            raise TypeError(f"@pipeline_task '{fname}' uses bare 'Document' class in return type. Use specific Document subclasses (e.g., MyDocument) instead.")

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
) -> Callable[[Callable[..., Coroutine[Any, Any, Sequence[Document]]]], _FlowLike[Any]]:
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

    def _apply(fn: Callable[..., Coroutine[Any, Any, Sequence[Document]]]) -> _FlowLike[Any]:
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
            input_annotation = hints[params[1].name]
            if contains_bare_document(input_annotation):
                raise TypeError(
                    f"@pipeline_flow '{fname}' uses bare 'Document' class in input annotation. "
                    f"Use specific Document subclasses (e.g., list[MyDocument]) instead."
                )
            resolved_input_types = parse_document_types_from_annotation(input_annotation)
        else:
            resolved_input_types = []

        # Extract output types from return annotation
        resolved_output_types: list[type[Document]]
        if "return" in hints:
            return_annotation = hints["return"]
            if contains_bare_document(return_annotation):
                raise TypeError(
                    f"@pipeline_flow '{fname}' uses bare 'Document' class in return annotation. "
                    f"Use specific Document subclasses (e.g., list[MyDocument]) instead."
                )
            resolved_output_types = parse_document_types_from_annotation(return_annotation)
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
            if not isinstance(result, list):
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

async def safe_gather[T](
    *coroutines: Coroutine[Any, Any, T],
    label: str = "",
    raise_if_all_fail: bool = True,
) -> list[T]:
    """Execute coroutines in parallel, returning successes and logging failures.

    Uses return_exceptions=True internally. Filters failures with BaseException
    (catches CancelledError). Logs each failure with context.

    Returns:
        List of successful results (failures filtered out). Relative order of
        successes is preserved, but indices shift. Use safe_gather_indexed
        for positional correspondence.
    """
    if not coroutines:
        return []

    results, failures = await _execute_gather(*coroutines, label=label)
    failure_indices = {i for i, _ in failures}
    successes: list[T] = [r for i, r in enumerate(results) if i not in failure_indices]

    if not successes and raise_if_all_fail and failures:
        first_error = failures[0][1]
        raise RuntimeError(f"All {len(failures)} tasks failed{f' in {label!r}' if label else ''}. First error: {first_error}") from first_error

    return successes

async def safe_gather_indexed[T](
    *coroutines: Coroutine[Any, Any, T],
    label: str = "",
    raise_if_all_fail: bool = True,
) -> list[T | None]:
    """Execute coroutines in parallel, preserving positional correspondence.

    Like safe_gather, but returns a list with the same length as the input.
    Failed positions contain None. Useful when results must correspond to
    specific inputs by index.

    Returns:
        List matching input length. Successful results at their original index,
        None at positions where the coroutine failed.
    """
    if not coroutines:
        return []

    results, failures = await _execute_gather(*coroutines, label=label)
    failure_indices = {i for i, _ in failures}
    output: list[T | None] = [None if i in failure_indices else r for i, r in enumerate(results)]

    if len(failures) == len(results) and raise_if_all_fail:
        first_error = failures[0][1]
        raise RuntimeError(f"All {len(failures)} tasks failed{f' in {label!r}' if label else ''}. First error: {first_error}") from first_error

    return output

@asynccontextmanager
async def pipeline_concurrency(
    name: str,
    *,
    timeout: int | None = None,
) -> AsyncGenerator[None, None]:
    """Acquire a concurrency/rate-limit slot for an operation.

    For CONCURRENT limits: slot held during block, released on exit.
    For PER_MINUTE/PER_HOUR: slot acquired (decays automatically), exit is no-op.

    Proceeds unthrottled when Prefect is unavailable.
    Timeout always raises AcquireConcurrencySlotTimeoutError.
    """
    state = _limits_state.get()
    cfg = state.limits.get(name)
    if cfg is None:
        available = ", ".join(sorted(state.limits)) or "(none)"
        raise KeyError(f"pipeline_concurrency({name!r}) not registered. Available limits: {available}. Declare it on PipelineDeployment.concurrency_limits.")

    # Prefect unavailable — proceed unthrottled
    if not state.status.prefect_available:
        yield
        return

    effective_timeout = timeout if timeout is not None else cfg.timeout
    t0 = time.monotonic()

    def _warn_if_slow() -> None:
        wait_seconds = time.monotonic() - t0
        if wait_seconds > _CACHE_TTL_WARNING_THRESHOLD:
            logger.warning(
                "Slot wait for %r took %.1fs — exceeds %ds threshold. "
                "LLM cache TTL (default 300s) may expire before execution. "
                "Consider increasing concurrency limit or reducing parallelism.",
                name,
                wait_seconds,
                _CACHE_TTL_WARNING_THRESHOLD,
            )

    # Prefect available — use global concurrency/rate limiting
    try:
        match cfg.kind:
            case LimitKind.CONCURRENT:
                async with concurrency(name, occupy=1, timeout_seconds=effective_timeout, strict=False):
                    _warn_if_slow()
                    yield
            case LimitKind.PER_MINUTE | LimitKind.PER_HOUR:
                await rate_limit(name, occupy=1, timeout_seconds=effective_timeout, strict=False)
                _warn_if_slow()
                yield
    except AcquireConcurrencySlotTimeoutError:
        raise
    except ConcurrencySlotAcquisitionError as e:
        logger.warning("Prefect concurrency unavailable for %r, proceeding unthrottled: %s", name, e)
        state.status.prefect_available = False
        yield

```

## Examples

**Pipeline flow deduplicates returned documents** (`tests/pipeline/test_flow_storage.py:125`)

```python
@pytest.mark.asyncio
async def test_pipeline_flow_deduplicates_returned_documents(prefect_test_fixture, memory_store: MemoryDocumentStore, run_context):
    """Test that @pipeline_flow deduplicates returned documents by SHA256."""
    doc = StorageOutputDoc(name="output.txt", content=b"test output")

    @pipeline_flow()
    async def test_flow(run_id: str, documents: list[StorageInputDoc], flow_options: FlowOptions) -> list[StorageOutputDoc]:
        return [doc, doc]  # Same document twice

    await test_flow("test-project", [], FlowOptions())

    loaded = await memory_store.load("test-project", [StorageOutputDoc])
    assert len(loaded) == 1
```

**Pipeline flow preserves existing run context** (`tests/pipeline/test_flow_storage.py:89`)

```python
@pytest.mark.asyncio
async def test_pipeline_flow_preserves_existing_run_context(prefect_test_fixture, memory_store, run_context):
    """Test that pipeline_flow does not override RunContext set by deployment."""
    from ai_pipeline_core.documents.context import get_run_context

    captured_ctx = None

    @pipeline_flow()
    async def test_flow(run_id: str, documents: list[StorageInputDoc], flow_options: FlowOptions) -> list[StorageOutputDoc]:
        nonlocal captured_ctx
        captured_ctx = get_run_context()
        return []

    await test_flow("my-project", [], FlowOptions())

    assert captured_ctx is not None
    # Should use the deployment-level context, not create a new one
    assert captured_ctx.run_scope == "test-project"
```

**Pipeline flow returns documents with store configured** (`tests/pipeline/test_flow_storage.py:39`)

```python
@pytest.mark.asyncio
async def test_pipeline_flow_returns_documents_with_store_configured(prefect_test_fixture, memory_store, run_context):
    """Test that pipeline_flow returns documents correctly when a store is configured."""

    @pipeline_flow()
    async def test_flow(run_id: str, documents: list[StorageInputDoc], flow_options: FlowOptions) -> list[StorageOutputDoc]:
        return [StorageOutputDoc(name="output.txt", content=b"test output")]

    input_docs = [StorageInputDoc(name="input.txt", content=b"test input")]

    result = await test_flow("test-project", input_docs, FlowOptions())

    assert len(result) == 1
    assert isinstance(result[0], StorageOutputDoc)
```

**Pipeline flow saves returned documents** (`tests/pipeline/test_flow_storage.py:109`)

```python
@pytest.mark.asyncio
async def test_pipeline_flow_saves_returned_documents(prefect_test_fixture, memory_store: MemoryDocumentStore, run_context):
    """Test that @pipeline_flow saves returned documents to the store."""

    @pipeline_flow()
    async def test_flow(run_id: str, documents: list[StorageInputDoc], flow_options: FlowOptions) -> list[StorageOutputDoc]:
        return [StorageOutputDoc(name="output.txt", content=b"test output")]

    input_docs = [StorageInputDoc(name="input.txt", content=b"test input")]
    await test_flow("test-project", input_docs, FlowOptions())

    loaded = await memory_store.load("test-project", [StorageOutputDoc])
    assert len(loaded) == 1
    assert loaded[0].name == "output.txt"
```

**Pipeline flow sets run context when missing** (`tests/pipeline/test_flow_storage.py:70`)

```python
@pytest.mark.asyncio
async def test_pipeline_flow_sets_run_context_when_missing(prefect_test_fixture, memory_store):
    """Test that pipeline_flow sets RunContext if none exists."""
    from ai_pipeline_core.documents.context import get_run_context

    captured_ctx = None

    @pipeline_flow()
    async def test_flow(run_id: str, documents: list[StorageInputDoc], flow_options: FlowOptions) -> list[StorageOutputDoc]:
        nonlocal captured_ctx
        captured_ctx = get_run_context()
        return []

    await test_flow("my-project", [], FlowOptions())

    assert captured_ctx is not None
    assert captured_ctx.run_scope == "my-project/test_flow"
```


## Error Examples

**Pipeline task then trace raises error** (`tests/pipeline/test_decorators.py:832`)

```python
def test_pipeline_task_then_trace_raises_error(self):
    from ai_pipeline_core import trace

    with pytest.raises(TypeError, match=r"already decorated with @pipeline_task"):

        @trace
        @pipeline_task
        async def my_task() -> None:  # pyright: ignore[reportUnusedFunction]
            pass
```

**Pipeline flow then trace raises error** (`tests/pipeline/test_decorators.py:854`)

```python
def test_pipeline_flow_then_trace_raises_error(self):
    from ai_pipeline_core import trace

    with pytest.raises(TypeError, match=r"already decorated with @pipeline"):

        @trace
        @pipeline_flow()
        async def my_flow(  # pyright: ignore[reportUnusedFunction]
            run_id: str, documents: list[InputDocument], flow_options: FlowOptions
        ) -> list[OutputDocument]:
            return list([OutputDocument(name="output.txt", content=b"output")])
```

**Sync function with pipeline task raises error** (`tests/pipeline/test_decorators.py:778`)

```python
def test_sync_function_with_pipeline_task_raises_error(self):
    from typing import Any, cast

    with pytest.raises(TypeError, match="must be 'async def'"):

        @cast(Any, pipeline_task)
        def sync_task(x: int) -> int:  # pyright: ignore[reportUnusedFunction]
            return x * 2
```

**Sync function with pipeline task with params raises error** (`tests/pipeline/test_decorators.py:787`)

```python
def test_sync_function_with_pipeline_task_with_params_raises_error(self):
    from typing import Any, cast

    with pytest.raises(TypeError, match="must be 'async def'"):

        @cast(Any, pipeline_task(retries=3, trace_level="debug"))
        def sync_task(x: int) -> int:  # pyright: ignore[reportUnusedFunction]
            return x * 2
```

**Trace then pipeline task raises error** (`tests/pipeline/test_decorators.py:822`)

```python
def test_trace_then_pipeline_task_raises_error(self):
    from ai_pipeline_core import trace

    with pytest.raises(TypeError, match=r"already decorated.*with @trace"):

        @pipeline_task
        @trace
        async def my_task(x: int) -> int:  # pyright: ignore[reportUnusedFunction]
            return x * 2
```
