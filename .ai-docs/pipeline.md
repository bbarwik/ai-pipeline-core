# MODULE: pipeline
# CLASSES: FlowOptions
# DEPENDS: BaseSettings
# PURPOSE: Pipeline framework primitives — decorators and flow options.
# SIZE: ~28KB

# === IMPORTS ===
from ai_pipeline_core import FlowOptions, pipeline_flow, pipeline_task

# === TYPES & CONSTANTS ===

type RetryConditionCallable = Callable[[Any, Any, Any], bool]

type StateHookCallable = Callable[[Any, Any, Any], None]

type TaskRunNameValueOrCallable = str | Callable[[], str]


# === INTERNAL TYPES (referenced by public API) ===

# Protocol — implement in concrete class
class _FlowLike(Protocol[FO_contra]):
    """Protocol for decorated flow objects returned by @pipeline_flow."""
    name: str | None
    input_document_types: list[type[Document]]
    output_document_types: list[type[Document]]
    estimated_minutes: int

    def __call__(
        self,
        project_name: str,
        documents: list[Document],
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


# === PUBLIC API ===

class FlowOptions(BaseSettings):
    """Base configuration for pipeline flows.

Subclass to add flow-specific parameters. Uses pydantic-settings
for environment variable overrides. Immutable after creation."""
    model_config = SettingsConfigDict(frozen=True, extra='allow')


# === FUNCTIONS ===

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

# === EXAMPLES (from tests/) ===

# Example: Pipeline flow deduplicates returned documents
# Source: tests/pipeline/test_flow_storage.py:125
@pytest.mark.asyncio
async def test_pipeline_flow_deduplicates_returned_documents(prefect_test_fixture, memory_store: MemoryDocumentStore, run_context):
    """Test that @pipeline_flow deduplicates returned documents by SHA256."""
    doc = StorageOutputDoc(name="output.txt", content=b"test output")

    @pipeline_flow()
    async def test_flow(project_name: str, documents: list[StorageInputDoc], flow_options: FlowOptions) -> list[StorageOutputDoc]:
        return [doc, doc]  # Same document twice

    await test_flow("test-project", [], FlowOptions())

    loaded = await memory_store.load("test-project", [StorageOutputDoc])
    assert len(loaded) == 1

# Example: Pipeline flow preserves existing run context
# Source: tests/pipeline/test_flow_storage.py:89
@pytest.mark.asyncio
async def test_pipeline_flow_preserves_existing_run_context(prefect_test_fixture, memory_store, run_context):
    """Test that pipeline_flow does not override RunContext set by deployment."""
    from ai_pipeline_core.documents.context import get_run_context

    captured_ctx = None

    @pipeline_flow()
    async def test_flow(project_name: str, documents: list[StorageInputDoc], flow_options: FlowOptions) -> list[StorageOutputDoc]:
        nonlocal captured_ctx
        captured_ctx = get_run_context()
        return []

    await test_flow("my-project", [], FlowOptions())

    assert captured_ctx is not None
    # Should use the deployment-level context, not create a new one
    assert captured_ctx.run_scope == "test-project"

# Example: Pipeline flow returns documents with store configured
# Source: tests/pipeline/test_flow_storage.py:39
@pytest.mark.asyncio
async def test_pipeline_flow_returns_documents_with_store_configured(prefect_test_fixture, memory_store, run_context):
    """Test that pipeline_flow returns documents correctly when a store is configured."""

    @pipeline_flow()
    async def test_flow(project_name: str, documents: list[StorageInputDoc], flow_options: FlowOptions) -> list[StorageOutputDoc]:
        return [StorageOutputDoc(name="output.txt", content=b"test output")]

    input_docs = [StorageInputDoc(name="input.txt", content=b"test input")]

    result = await test_flow("test-project", input_docs, FlowOptions())

    assert len(result) == 1
    assert isinstance(result[0], StorageOutputDoc)

# Example: Pipeline flow saves returned documents
# Source: tests/pipeline/test_flow_storage.py:109
@pytest.mark.asyncio
async def test_pipeline_flow_saves_returned_documents(prefect_test_fixture, memory_store: MemoryDocumentStore, run_context):
    """Test that @pipeline_flow saves returned documents to the store."""

    @pipeline_flow()
    async def test_flow(project_name: str, documents: list[StorageInputDoc], flow_options: FlowOptions) -> list[StorageOutputDoc]:
        return [StorageOutputDoc(name="output.txt", content=b"test output")]

    input_docs = [StorageInputDoc(name="input.txt", content=b"test input")]
    await test_flow("test-project", input_docs, FlowOptions())

    loaded = await memory_store.load("test-project", [StorageOutputDoc])
    assert len(loaded) == 1
    assert loaded[0].name == "output.txt"

# === ERROR EXAMPLES (What NOT to Do) ===

# Error: Pipeline task then trace raises error
# Source: tests/pipeline/test_decorators.py:879
def test_pipeline_task_then_trace_raises_error(self):
    from ai_pipeline_core import trace

    with pytest.raises(TypeError, match=r"already decorated with @pipeline_task"):

        @trace
        @pipeline_task(persist=False)
        async def my_task(x: int) -> int:  # pyright: ignore[reportUnusedFunction]
            return x * 2

# Error: Pipeline flow then trace raises error
# Source: tests/pipeline/test_decorators.py:901
def test_pipeline_flow_then_trace_raises_error(self):
    from ai_pipeline_core import trace

    with pytest.raises(TypeError, match=r"already decorated with @pipeline"):

        @trace
        @pipeline_flow()
        async def my_flow(  # pyright: ignore[reportUnusedFunction]
            project_name: str, documents: list[Document], flow_options: FlowOptions
        ) -> list[Document]:
            return list([OutputDocument(name="output.txt", content=b"output")])

# Error: Sync function with pipeline task raises error
# Source: tests/pipeline/test_decorators.py:825
def test_sync_function_with_pipeline_task_raises_error(self):
    from typing import Any, cast

    with pytest.raises(TypeError, match="must be 'async def'"):

        @cast(Any, pipeline_task)
        def sync_task(x: int) -> int:  # pyright: ignore[reportUnusedFunction]
            return x * 2

# Error: Sync function with pipeline task with params raises error
# Source: tests/pipeline/test_decorators.py:834
def test_sync_function_with_pipeline_task_with_params_raises_error(self):
    from typing import Any, cast

    with pytest.raises(TypeError, match="must be 'async def'"):

        @cast(Any, pipeline_task(retries=3, trace_level="debug"))
        def sync_task(x: int) -> int:  # pyright: ignore[reportUnusedFunction]
            return x * 2
