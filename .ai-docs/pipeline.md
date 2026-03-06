# MODULE: pipeline
# CLASSES: LimitKind, PipelineLimit, FlowOptions, PipelineFlow, TaskHandle, TaskBatch, PipelineTask
# DEPENDS: BaseSettings, StrEnum
# PURPOSE: Pipeline framework primitives.
# VERSION: 0.13.0
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import FlowOptions, LimitKind, PipelineFlow, PipelineLimit, PipelineTask, TaskBatch, TaskHandle, as_task_completed, collect_tasks, pipeline_concurrency, pipeline_test_context, run_tasks_until, safe_gather, safe_gather_indexed
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

Must use names matching ``[a-zA-Z0-9_-]+`` in PipelineDeployment.concurrency_limits (validated at class definition time)."""
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


class PipelineFlow:
    """Base class for pipeline flows.

Flows are the unit of resume, progress tracking, and document hand-off in a deployment.
Define ``run`` as an **instance method** (not @classmethod) because flows can carry
per-instance configuration passed via ``build_flows()``::

    class TranslateFlow(PipelineFlow):
        target_language: str = "en"

        async def run(self, run_id: str, documents: list[SourceDoc], options: FlowOptions) -> list[TranslatedDoc]:
            return await TranslateTask.run(documents, language=self.target_language)

The deployment creates flow instances with constructor kwargs::

    def build_flows(self, options):
        return [TranslateFlow(target_language="fr"), TranslateFlow(target_language="de")]

Each instance runs independently with its own parameters, resume record, and progress.
Constructor kwargs are captured for replay serialization via ``get_params()``.

Signature must be exactly ``(self, run_id: str, documents: list[DocType], options: FlowOptions)``
and is validated at class definition time by ``__init_subclass__``."""
    name: ClassVar[str]
    estimated_minutes: ClassVar[float] = 1.0
    input_document_types: ClassVar[list[type[Document]]] = []
    output_document_types: ClassVar[list[type[Document]]] = []
    task_graph: ClassVar[list[tuple[str, str]]] = []

    def __init__(self, **kwargs: Any) -> None:
        """Constructor for per-flow instance configuration."""
        cls = type(self)
        known_params: set[str] = set()
        for klass in cls.__mro__:
            known_params.update(name for name in getattr(klass, "__annotations__", {}) if not name.startswith("_"))
            known_params.update(
                name
                for name, value in vars(klass).items()
                if not name.startswith("_") and not callable(value) and not isinstance(value, (classmethod, staticmethod, property))
            )
        unknown = sorted(key for key in kwargs if key not in known_params)
        if unknown:
            allowed = ", ".join(sorted(known_params)) or "(none)"
            raise TypeError(f"PipelineFlow '{cls.__name__}' got unknown init parameter(s): {', '.join(unknown)}. Allowed parameters: {allowed}.")
        self._params: dict[str, Any] = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def expected_tasks(cls) -> list[str]:
        """Return expected task names extracted from run() AST."""
        return [name for name, _mode in cls.task_graph]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is PipelineFlow:
            return

        cls._validate_class_config()
        run_fn, hints, params = cls._validate_run_signature()
        input_types, output_types = cls._extract_document_types(hints, params)
        cls.input_document_types = input_types
        cls.output_document_types = output_types
        cls.task_graph = cls._parse_task_graph(run_fn)

    def get_params(self) -> dict[str, Any]:
        """Return constructor params for flow plan serialization."""
        return dict(getattr(self, "_params", {}))

    async def run(self, run_id: str, documents: Any, options: Any) -> Sequence[Document]:
        """Execute the flow. Must be overridden by subclasses (enforced at class definition time by ``__init_subclass__``)."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True, eq=False)
class TaskHandle:
    """Handle for an executing pipeline task."""
    task_class: type[Any] | None
    input_arguments: Mapping[str, Any]

    @property
    def done(self) -> bool:
        """Whether the underlying task has finished."""
        return self._task.done()

    def __await__(self):
        return self._task.__await__()

    def cancel(self) -> None:
        """Cancel the underlying task."""
        self._task.cancel()

    async def result(self) -> T:
        """Await the underlying task result."""
        return await self._task


@dataclass(frozen=True, slots=True)
class TaskBatch:
    """Collected task results and handles that did not complete successfully."""
    completed: list[list[Document]]
    incomplete: list[TaskHandle[list[Document]]]


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
``await`` returns a ``TaskHandle`` for parallel execution via ``collect_tasks``."""
    name: ClassVar[str]
    estimated_minutes: ClassVar[float] = 1.0
    retries: ClassVar[int] = 0
    retry_delay_seconds: ClassVar[int] = 20
    timeout_seconds: ClassVar[int | None] = None
    cacheable: ClassVar[bool] = False
    trace_level: ClassVar[TraceLevel] = 'always'
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


```

## Functions

```python
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
    Logs a warning if slot acquisition takes longer than 120 seconds.
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

@contextmanager
def pipeline_test_context(
    run_id: str = "test-run",
    store: DocumentStore | None = None,
    publisher: ResultPublisher | None = None,
) -> Generator[ExecutionContext, None, None]:
    """Set up an execution + task context for tests without full deployment wiring.

    Yields:
        The active execution context for the test scope.
    """
    owns_store = store is None
    active_store = store or MemoryDocumentStore()
    ctx = ExecutionContext(
        run_id=run_id,
        run_scope=RunScope(f"{run_id}/test"),
        execution_id=None,
        store=active_store,
        publisher=publisher or _NoopPublisher(),
        summary_generator=None,
        limits=MappingProxyType({}),
        limits_status=_SharedStatus(),
    )
    ctx_token = set_execution_context(ctx)
    task_token = set_task_context(TaskContext(scope_kind="test", task_class_name="pipeline_test_context"))
    try:
        yield ctx
    finally:
        reset_task_context(task_token)
        reset_execution_context(ctx_token)
        if owns_store:
            active_store.shutdown()

async def collect_tasks(
    *handles: TaskAwaitableGroup,
    deadline_seconds: float | None = None,
) -> TaskBatch:
    """Await task handles with an optional deadline and split completed/incomplete."""
    ordered_handles = _normalize_handles(handles)
    if not ordered_handles:
        return TaskBatch(completed=[], incomplete=[])

    completed: list[list[Document]] = []
    incomplete: list[TaskHandle[list[Document]]] = []
    by_task: dict[asyncio.Task[list[Document]], TaskHandle[list[Document]]] = {handle._task: handle for handle in ordered_handles}
    pending: set[asyncio.Task[list[Document]]] = set(by_task.keys())
    deadline_at = (time.monotonic() + deadline_seconds) if deadline_seconds is not None else None

    while pending:
        timeout: float | None = None
        if deadline_at is not None:
            timeout = max(0.0, deadline_at - time.monotonic())
            if timeout <= 0.0:
                break
        done, pending = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if not done:
            break
        for finished in done:
            handle = by_task[finished]
            outcome = (await asyncio.gather(handle.result(), return_exceptions=True))[0]
            if isinstance(outcome, BaseException):
                incomplete.append(handle)
                continue
            completed.append(outcome)

    incomplete.extend(by_task[still_pending] for still_pending in pending)
    return TaskBatch(completed=completed, incomplete=incomplete)

async def as_task_completed(*handles: TaskAwaitableGroup) -> AsyncIterator[TaskHandle[list[Document]]]:
    """Yield task handles in completion order."""
    ordered_handles = _normalize_handles(handles)
    if not ordered_handles:
        return

    by_task: dict[asyncio.Task[list[Document]], TaskHandle[list[Document]]] = {handle._task: handle for handle in ordered_handles}
    pending: set[asyncio.Task[list[Document]]] = set(by_task.keys())
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for finished in done:
            yield by_task[finished]

async def run_tasks_until(
    task_cls: type[Any],
    argument_groups: Sequence[tuple[tuple[Any, ...], dict[str, Any]]],
    *,
    deadline_seconds: float | None = None,
) -> TaskBatch:
    """Launch ``task_cls.run(*args, **kwargs)`` for each argument group and collect the handles."""
    handles = [task_cls.run(*args, **kwargs) for args, kwargs in argument_groups]
    return await collect_tasks(handles, deadline_seconds=deadline_seconds)

```

## Examples

**Name with dashes and underscores** (`tests/pipeline/test_limits.py:155`)

```python
def test_name_with_dashes_and_underscores(self):
    raw = {"my-limit_v2": PipelineLimit(10)}
    result = _validate_concurrency_limits("TestDeploy", raw)
    assert "my-limit_v2" in result
```

**Collect tasks empty returns empty batch** (`tests/pipeline/test_task_constraints.py:93`)

```python
@pytest.mark.asyncio
async def test_collect_tasks_empty_returns_empty_batch() -> None:
    """collect_tasks with no handles returns empty batch."""
    batch = await collect_tasks()
    assert batch.completed == []
    assert batch.incomplete == []
```

**Pipeline test context sets and restores** (`tests/pipeline/test_execution_context.py:94`)

```python
def test_pipeline_test_context_sets_and_restores() -> None:
    before = get_execution_context()
    with pipeline_test_context(run_id="ctx-test") as ctx:
        assert get_execution_context() is ctx
        assert ctx.run_id == "ctx-test"
        assert ctx.store is not None
    assert get_execution_context() is before
```

**Pipeline test context with custom publisher** (`tests/pipeline/test_execution_context.py:103`)

```python
def test_pipeline_test_context_with_custom_publisher() -> None:
    pub = _NoopPublisher()
    with pipeline_test_context(publisher=pub) as ctx:
        assert ctx.publisher is pub
```

**As task completed yields handles** (`tests/pipeline/test_parallel_primitives.py:154`)

```python
@pytest.mark.asyncio
async def test_as_task_completed_yields_handles() -> None:
    with pipeline_test_context() as ctx:
        token = set_execution_context(ctx.with_flow(_make_flow_frame()))
        try:
            h1 = _FastTask.run([_make_doc("1")])
            h2 = _FastTask.run([_make_doc("2")])
            yielded = [handle async for handle in as_task_completed(h1, h2)]
        finally:
            reset_execution_context(token)

    assert len(yielded) == 2
    assert all(isinstance(handle, TaskHandle) for handle in yielded)
```

**As task completed yields results** (`tests/pipeline/test_flow_resume.py:26`)

```python
@pytest.mark.asyncio
async def test_as_task_completed_yields_results() -> None:
    first = InputDoc.create_root(name="1.txt", content="a", reason="test input")
    second = InputDoc.create_root(name="2.txt", content="b", reason="test input")

    names: list[str] = []
    with pipeline_test_context():
        async for handle in as_task_completed(EchoTask.run([first]), EchoTask.run([second])):
            docs = await handle.result()
            names.extend(doc.name for doc in docs)

    assert set(names) == {"out_1.txt", "out_2.txt"}
```

**Collect tasks accepts list** (`tests/pipeline/test_parallel_primitives.py:140`)

```python
@pytest.mark.asyncio
async def test_collect_tasks_accepts_list() -> None:
    with pipeline_test_context() as ctx:
        token = set_execution_context(ctx.with_flow(_make_flow_frame()))
        try:
            handles = [_FastTask.run([_make_doc("x")]) for _ in range(3)]
            batch = await collect_tasks(handles)
        finally:
            reset_execution_context(token)

    assert len(batch.completed) == 3
    assert batch.incomplete == []
```

**Collect tasks all complete** (`tests/pipeline/test_parallel_primitives.py:91`)

```python
@pytest.mark.asyncio
async def test_collect_tasks_all_complete() -> None:
    with pipeline_test_context() as ctx:
        token = set_execution_context(ctx.with_flow(_make_flow_frame()))
        try:
            h1 = _FastTask.run([_make_doc("a")])
            h2 = _FastTask.run([_make_doc("b")])
            batch = await collect_tasks(h1, h2)
        finally:
            reset_execution_context(token)

    assert isinstance(batch, TaskBatch)
    assert len(batch.completed) == 2
    assert batch.incomplete == []
```


## Error Examples

**Invalid name pattern** (`tests/pipeline/test_limits.py:136`)

```python
def test_invalid_name_pattern(self):
    with pytest.raises(TypeError, match="invalid name"):
        _validate_concurrency_limits("TestDeploy", {"bad name!": PipelineLimit(10)})
```

**Base flow options rejects extra** (`tests/pipeline/test_options.py:29`)

```python
def test_base_flow_options_rejects_extra(self):
    """Test that base FlowOptions rejects extra fields (extra='forbid')."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        FlowOptions(unknown_field="value")
```

**Flow options is frozen** (`tests/pipeline/test_options.py:34`)

```python
def test_flow_options_is_frozen(self):
    """Test that FlowOptions instances are immutable."""

    class SimpleOptions(FlowOptions):
        core_model: str = "default"

    options = SimpleOptions()
    with pytest.raises(ValidationError):
        options.core_model = "new-model"
```

**Inherited flow options maintains frozen** (`tests/pipeline/test_options.py:111`)

```python
def test_inherited_flow_options_maintains_frozen(self):
    """Test that inherited classes maintain frozen configuration."""

    class CustomFlowOptions(FlowOptions):
        custom_field: str = "default"

    options = CustomFlowOptions()
    with pytest.raises(ValidationError):
        options.custom_field = "new_value"
```

**Invalid kind type** (`tests/pipeline/test_limits.py:144`)

```python
def test_invalid_kind_type(self):
    """Test that kind must be LimitKind enum instance."""
    # Create a PipelineLimit-like object with wrong kind type
    limit = PipelineLimit.__new__(PipelineLimit)
    object.__setattr__(limit, "limit", 10)
    object.__setattr__(limit, "kind", "concurrent")  # str, not LimitKind
    object.__setattr__(limit, "timeout", 600)
    with pytest.raises(TypeError, match="kind must be LimitKind"):
        _validate_concurrency_limits("TestDeploy", {"test": limit})
```

**Limit must be positive** (`tests/pipeline/test_limits.py:64`)

```python
def test_limit_must_be_positive(self):
    with pytest.raises(ValueError, match="limit must be >= 1"):
        PipelineLimit(limit=0)
    with pytest.raises(ValueError, match="limit must be >= 1"):
        PipelineLimit(limit=-5)
```

**Missing build flows raises** (`tests/pipeline/test_static_validation.py:789`)

```python
def test_missing_build_flows_raises(self):
    """Deployment without build_flows raises TypeError."""
    with pytest.raises(TypeError, match="must implement build_flows"):

        class NoFlows(PipelineDeployment[FlowOptions, SampleResult]):
            @staticmethod
            def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> SampleResult:
                return SampleResult(success=True)
```

**Missing build result raises** (`tests/pipeline/test_static_validation.py:781`)

```python
def test_missing_build_result_raises(self):
    """Deployment without build_result raises TypeError."""
    with pytest.raises(TypeError, match=r"must implement.*build_result"):

        class NoBuild(PipelineDeployment[FlowOptions, SampleResult]):
            def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
                return [AlphaToBetaFlow()]
```
