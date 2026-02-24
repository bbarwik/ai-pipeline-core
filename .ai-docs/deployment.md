# MODULE: deployment
# CLASSES: DeploymentContext, DeploymentResult, PipelineDeployment, RunState, FlowStatus, PendingRun, ProgressRun, DeploymentResultData, CompletedRun, FailedRun, RemoteDeployment
# DEPENDS: BaseModel, Generic, StrEnum
# PURPOSE: Pipeline deployment utilities for unified, type-safe deployments.
# VERSION: 0.10.1
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import DeploymentContext, DeploymentResult, PipelineDeployment, RemoteDeployment, run_remote_deployment
from ai_pipeline_core.deployment import CompletedRun, DeploymentResultData, FailedRun, FlowStatus, PendingRun, ProgressCallback, ProgressRun, RunResponse, RunState, progress_update
```

## Types & Constants

```python
RunResponse = Annotated[
    PendingRun | ProgressRun | CompletedRun | FailedRun,
    Discriminator("type"),
]

ProgressCallback = Callable[[float, str], Awaitable[None]]

```

## Internal Types

```python
class _OutputDocument(BaseModel):
    """Document metadata in deployment results. Binary content is None."""
    sha256: DocumentSha256
    name: str
    class_name: str
    mime_type: str
    size: int
    description: str = ''
    content: str | None = None
    attachments: tuple[OutputAttachment, ...] = ()
    model_config = ConfigDict(frozen=True)


```

## Public API

```python
class DeploymentContext(BaseModel):
    """Infrastructure configuration for deployments. Progress is tracked via Prefect labels (pub/sub)."""
    model_config = ConfigDict(frozen=True, extra='forbid')


class DeploymentResult(BaseModel):
    """Base class for deployment results."""
    success: bool
    error: str | None = None
    documents: tuple[_OutputDocument, ...] = ()
    model_config = ConfigDict(frozen=True)


class PipelineDeployment(Generic[TOptions, TResult]):
    """Base class for pipeline deployments.

Features enabled by default:
- Per-flow resume: Skip flows if outputs exist in DocumentStore
- Per-flow uploads: Upload documents after each flow
- Progress tracking via Prefect labels (pub/sub)
- Upload on failure: Save partial results if pipeline fails"""
    flows: ClassVar[list[Any]]
    name: ClassVar[str]
    options_type: ClassVar[type[FlowOptions]]
    result_type: ClassVar[type[DeploymentResult]]
    cache_ttl: ClassVar[timedelta | None] = timedelta(hours=24)
    concurrency_limits: ClassVar[Mapping[str, PipelineLimit]] = MappingProxyType({})

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "flows"):
            return

        if cls.__name__.startswith("Test"):
            raise TypeError(f"Deployment class name cannot start with 'Test': {cls.__name__}")

        cls.name = class_name_to_deployment_name(cls.__name__)

        generic_args = extract_generic_params(cls, PipelineDeployment)
        if len(generic_args) < 2:
            raise TypeError(f"{cls.__name__} must specify Generic parameters: class {cls.__name__}(PipelineDeployment[MyOptions, MyResult])")
        options_type, result_type = generic_args[0], generic_args[1]

        cls.options_type = options_type
        cls.result_type = result_type

        if not cls.flows:
            raise TypeError(f"{cls.__name__}.flows cannot be empty")

        # build_result must be implemented (not still abstract from PipelineDeployment)
        build_result_fn = getattr(cls, "build_result", None)
        if build_result_fn is None or getattr(build_result_fn, "__isabstractmethod__", False):
            raise TypeError(f"{cls.__name__} must implement 'build_result' static method")

        # No duplicate flows (by identity)
        seen_ids: set[int] = set()
        for flow_fn in cls.flows:
            fid = id(flow_fn)
            if fid in seen_ids:
                flow_name = getattr(flow_fn, "name", getattr(flow_fn, "__name__", str(flow_fn)))
                raise TypeError(f"{cls.__name__}.flows contains duplicate flow '{flow_name}'")
            seen_ids.add(fid)

        # Flow type chain validation: simulate a type pool
        _validate_flow_chain(cls.__name__, cls.flows)

        # Concurrency limits validation
        cls.concurrency_limits = _validate_concurrency_limits(cls.__name__, getattr(cls, "concurrency_limits", MappingProxyType({})))

    @final
    def as_prefect_flow(self) -> Callable[..., Any]:
        """Generate a Prefect flow for production deployment.

        Returns:
            Async Prefect flow callable that initializes DocumentStore from settings.
        """
        deployment = self

        async def _deployment_flow(
            run_id: str,
            documents: list[DocumentInput],
            options: FlowOptions,
            context: DeploymentContext,
        ) -> DeploymentResult:
            # Initialize observability for remote workers
            init_observability_best_effort()

            # Set session ID from Prefect flow run for trace grouping
            flow_run_id = str(runtime.flow_run.get_id()) if runtime.flow_run else str(uuid4())  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
            os.environ["LMNR_SESSION_ID"] = flow_run_id

            publisher = _create_publisher(settings)
            store = create_document_store(
                settings,
                summary_generator=_build_summary_generator(),
            )
            set_document_store(store)
            try:
                # Create parent span to group all traces under a single deployment trace
                with Laminar.start_as_current_span(
                    name=f"{deployment.name}-{run_id}",
                    input={"run_id": run_id, "options": options.model_dump()},
                    session_id=flow_run_id,
                ):
                    # Resolve DocumentInput (inline + URL references) into typed Documents
                    start_step_input_types: list[type[Document]] = getattr(deployment.flows[0], "input_document_types", [])
                    typed_docs = await resolve_document_inputs(
                        documents,
                        deployment._all_document_types(),
                        start_step_input_types=start_step_input_types,
                    )
                    result = await deployment.run(run_id, typed_docs, cast(Any, options), context, publisher=publisher)
                    Laminar.set_span_output(result.model_dump())
                    return result
            finally:
                await publisher.close()
                store.shutdown()
                set_document_store(None)

        # Override generic annotations with concrete types for Prefect parameter schema generation
        _deployment_flow.__annotations__["options"] = self.options_type
        _deployment_flow.__annotations__["return"] = self.result_type

        return flow(
            name=self.name,
            flow_run_name=f"{self.name}-{{run_id}}",
            persist_result=True,
            result_serializer="json",
        )(_deployment_flow)

    @staticmethod
    @abstractmethod
    def build_result(run_id: str, documents: list[Document], options: TOptions) -> TResult:
        """Extract typed result from pipeline documents loaded from DocumentStore.

        Called for both full runs and partial runs (--start/--end). For partial runs,
        _build_partial_result() delegates here by default — override _build_partial_result()
        to customize partial run results.
        """
        ...

    @final
    async def run(
        self,
        run_id: str,
        documents: list[Document],
        options: TOptions,
        context: DeploymentContext,
        publisher: ResultPublisher | None = None,
        start_step: int = 1,
        end_step: int | None = None,
    ) -> TResult:
        """Execute flows with resume, per-flow uploads, and step control.

        Args:
            run_id: Unique identifier for this pipeline run (used as run_scope).
            documents: Initial input documents for the first flow.
            options: Flow options passed to each flow.
            context: Deployment context.
            publisher: Lifecycle event publisher (defaults to NoopPublisher).
            start_step: First flow to execute (1-indexed, default 1).
            end_step: Last flow to execute (inclusive, default all flows).

        Returns:
            Typed deployment result built from all pipeline documents.
        """
        if publisher is None:
            publisher = NoopPublisher()
        store = get_document_store()
        total_steps = len(self.flows)

        if end_step is None:
            end_step = total_steps
        if start_step < 1 or start_step > total_steps:
            raise ValueError(f"start_step must be 1-{total_steps}, got {start_step}")
        if end_step < start_step or end_step > total_steps:
            raise ValueError(f"end_step must be {start_step}-{total_steps}, got {end_step}")

        flow_run_id: str = str(runtime.flow_run.get_id() or "") if runtime.flow_run else ""  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]

        # Write identity labels for polling endpoint
        flow_run_uuid = _safe_uuid(flow_run_id) if flow_run_id else None
        if flow_run_uuid is not None:
            try:
                async with get_client() as client:
                    await client.update_flow_run_labels(
                        flow_run_id=flow_run_uuid,
                        labels={_LABEL_RUN_ID: run_id},
                    )
            except Exception as e:
                logger.warning("Identity label update failed: %s", e)

        input_docs = list(documents)
        run_scope = _compute_run_scope(run_id, input_docs, options)

        if not store and total_steps > 1:
            logger.warning("No DocumentStore configured for multi-step pipeline — intermediate outputs will not accumulate between flows")

        # Tracking lifecycle
        tracking_svc = None
        run_uuid: UUID | None = None
        run_failed = False
        try:
            tracking_svc = get_tracking_service()
            if tracking_svc:
                run_uuid = (_safe_uuid(flow_run_id) if flow_run_id else None) or uuid4()
                tracking_svc.set_run_context(execution_id=run_uuid, run_id=run_id, flow_name=self.name, run_scope=run_scope)
                tracking_svc.track_run_start(execution_id=run_uuid, run_id=run_id, flow_name=self.name, run_scope=run_scope)
        except Exception as e:
            logger.warning("Tracking service initialization failed: %s", e)
            tracking_svc = None

        # Set concurrency limits and RunContext for the entire pipeline run
        failed_published = False
        heartbeat_task: asyncio.Task[None] | None = None
        limits_token = _set_limits_state(_LimitsState(limits=self.concurrency_limits, status=_SharedStatus()))
        run_token = set_run_context(RunContext(run_scope=run_scope))
        try:
            # Publish task.started event (inside try so failures still hit finally cleanup)
            await publisher.publish_started(StartedEvent(run_id=run_id, flow_run_id=flow_run_id, run_scope=str(run_scope)))

            # Start heartbeat background task
            heartbeat_task = asyncio.create_task(_heartbeat_loop(publisher, run_id))

            await _ensure_concurrency_limits(self.concurrency_limits)

            # Save initial input documents to store
            if store and input_docs:
                await store.save_batch(input_docs, run_scope)

            # Precompute flow minutes for progress calculation
            flow_minutes = tuple(getattr(f, "estimated_minutes", 1) for f in self.flows)

            for i in range(start_step - 1, end_step):
                step = i + 1
                flow_fn = self.flows[i]
                flow_name = getattr(flow_fn, "name", flow_fn.__name__)
                # Re-read flow_run_id in case Prefect subflow changes it
                flow_run_id = str(runtime.flow_run.get_id() or "") if runtime.flow_run else flow_run_id  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]

                # Resume check: skip if flow completed successfully in a previous run
                if store:
                    completion = await store.get_flow_completion(run_scope, flow_name, max_age=self.cache_ttl)
                    if completion is not None:
                        logger.info("[%d/%d] Resume: skipping %s (completion record found)", step, total_steps, flow_name)
                        cached_msg = f"Resumed from store: {flow_name}"
                        await self._update_progress_labels(
                            flow_run_id,
                            run_id,
                            step,
                            total_steps,
                            flow_name,
                            FlowStatus.CACHED,
                            step_progress=1.0,
                            message=cached_msg,
                        )
                        await publisher.publish_progress(
                            self._build_progress_event(
                                run_id,
                                flow_run_id,
                                flow_name,
                                step,
                                total_steps,
                                flow_minutes,
                                FlowStatus.CACHED,
                                1.0,
                                cached_msg,
                            )
                        )
                        continue

                started_msg = f"Starting: {flow_name}"
                await self._update_progress_labels(
                    flow_run_id,
                    run_id,
                    step,
                    total_steps,
                    flow_name,
                    FlowStatus.STARTED,
                    step_progress=0.0,
                    message=started_msg,
                )
                await publisher.publish_progress(
                    self._build_progress_event(
                        run_id,
                        flow_run_id,
                        flow_name,
                        step,
                        total_steps,
                        flow_minutes,
                        FlowStatus.STARTED,
                        0.0,
                        started_msg,
                    )
                )
                logger.info("[%d/%d] Starting: %s", step, total_steps, flow_name)

                # Load input documents from store
                input_types = getattr(flow_fn, "input_document_types", [])
                if store and input_types:
                    current_docs = await store.load(run_scope, input_types)
                else:
                    current_docs = input_docs

                # Set up intra-flow progress context so progress_update() works inside flows
                completed_mins = sum(flow_minutes[: max(step - 1, 0)])

                with _flow_context(
                    run_id=run_id,
                    flow_run_id=flow_run_id,
                    flow_name=flow_name,
                    step=step,
                    total_steps=total_steps,
                    flow_minutes=flow_minutes,
                    completed_minutes=completed_mins,
                    publisher=publisher,
                ):
                    await flow_fn(run_id, current_docs, options)

                # Record flow completion for resume (only after successful execution)
                if store:
                    output_types = getattr(flow_fn, "output_document_types", [])
                    input_sha256s = tuple(d.sha256 for d in current_docs)
                    output_docs = await store.load(run_scope, output_types) if output_types else []
                    output_sha256s = tuple(d.sha256 for d in output_docs)
                    await store.save_flow_completion(run_scope, flow_name, input_sha256s, output_sha256s)

                completed_msg = f"Completed: {flow_name}"
                await self._update_progress_labels(
                    flow_run_id,
                    run_id,
                    step,
                    total_steps,
                    flow_name,
                    FlowStatus.COMPLETED,
                    step_progress=1.0,
                    message=completed_msg,
                )
                await publisher.publish_progress(
                    self._build_progress_event(
                        run_id,
                        flow_run_id,
                        flow_name,
                        step,
                        total_steps,
                        flow_minutes,
                        FlowStatus.COMPLETED,
                        1.0,
                        completed_msg,
                    )
                )
                logger.info("[%d/%d] Completed: %s", step, total_steps, flow_name)

            # Build result from all documents in store
            if store:
                all_docs = await store.load(run_scope, self._all_document_types())
            else:
                all_docs = input_docs

            is_partial_run = end_step < total_steps
            if is_partial_run:
                logger.info("Partial run (steps %d-%d of %d) — skipping build_result", start_step, end_step, total_steps)
                result = self._build_partial_result(run_id, all_docs, options)
            else:
                result = self.build_result(run_id, all_docs, options)

            # Populate output documents
            output_docs = tuple(build_output_document(doc) for doc in all_docs)
            result = result.model_copy(update={"documents": output_docs})  # nosemgrep: no-document-model-copy

            # Compute chain_context from final flow output documents
            final_output_docs: list[Document] = []
            if store:
                last_flow = self.flows[end_step - 1]
                last_output_types = getattr(last_flow, "output_document_types", [])
                if last_output_types:
                    final_output_docs = await store.load(run_scope, last_output_types)

            chain_context = {
                "version": 1,
                "run_scope": str(run_scope),
                "output_document_refs": [doc.sha256 for doc in final_output_docs],
            }

            # Publish task.completed event
            await publisher.publish_completed(
                CompletedEvent(
                    run_id=run_id,
                    flow_run_id=flow_run_id,
                    result=result.model_dump(),
                    chain_context=chain_context,
                    actual_cost=0.0,
                )
            )

            return result

        except (Exception, asyncio.CancelledError) as exc:
            run_failed = True
            if not failed_published:
                failed_published = True
                try:
                    await publisher.publish_failed(
                        FailedEvent(
                            run_id=run_id,
                            flow_run_id=flow_run_id,
                            error_code=_classify_error(exc),
                            error_message=str(exc),
                        )
                    )
                except Exception as pub_err:
                    logger.warning("Failed to publish failure event: %s", pub_err)
            raise
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task
            reset_run_context(run_token)
            _reset_limits_state(limits_token)
            store = get_document_store()
            if store:
                try:
                    store.flush()
                except Exception as e:
                    logger.warning("Store flush failed: %s", e)
            if (svc := tracking_svc) is not None and run_uuid is not None:
                try:
                    svc.track_run_end(execution_id=run_uuid, status=RunStatus.FAILED if run_failed else RunStatus.COMPLETED)
                    svc.flush()
                except Exception as e:
                    logger.warning("Tracking shutdown failed: %s", e)

    @final
    def run_cli(
        self,
        initializer: Callable[[TOptions], tuple[str, list[Document]]] | None = None,
        trace_name: str | None = None,
        cli_mixin: type[BaseSettings] | None = None,
    ) -> None:
        """Execute pipeline from CLI arguments with --start/--end step control.

        Args:
            initializer: Optional callback returning (run_id, documents) from options.
            trace_name: Optional Laminar trace span name prefix.
            cli_mixin: Optional BaseSettings subclass with CLI-only fields mixed into options.
        """
        from ._cli import run_cli_for_deployment

        run_cli_for_deployment(self, initializer, trace_name, cli_mixin)

    @final
    def run_local(
        self,
        run_id: str,
        documents: list[Document],
        options: TOptions,
        context: DeploymentContext | None = None,
        publisher: ResultPublisher | None = None,
        output_dir: Path | None = None,
    ) -> TResult:
        """Run locally with Prefect test harness and in-memory document store.

        Args:
            run_id: Pipeline run identifier.
            documents: Initial input documents.
            options: Flow options.
            context: Optional deployment context (defaults to empty).
            publisher: Optional lifecycle event publisher (defaults to NoopPublisher).
            output_dir: Optional directory for writing result.json.

        Returns:
            Typed deployment result.
        """
        if context is None:
            context = DeploymentContext()

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        store = MemoryDocumentStore()
        set_document_store(store)
        try:
            with prefect_test_harness(), disable_run_logger():
                result = asyncio.run(self.run(run_id, documents, options, context, publisher=publisher))
        finally:
            store.shutdown()
            set_document_store(None)

        if output_dir:
            (output_dir / "result.json").write_text(result.model_dump_json(indent=2))

        return result


# Enum
class RunState(StrEnum):
    """Pipeline run lifecycle state."""
    PENDING = 'PENDING'
    RUNNING = 'RUNNING'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'
    CRASHED = 'CRASHED'
    CANCELLED = 'CANCELLED'


# Enum
class FlowStatus(StrEnum):
    """Individual flow step status within a pipeline run."""
    STARTED = 'started'
    COMPLETED = 'completed'
    CACHED = 'cached'
    PROGRESS = 'progress'


class PendingRun(_RunBase):
    """Pipeline queued or running but no progress reported yet."""
    type: Literal['pending'] = 'pending'


class ProgressRun(_RunBase):
    """Pipeline running with step-level progress data."""
    type: Literal['progress'] = 'progress'
    step: int
    total_steps: int
    flow_name: str
    status: FlowStatus
    progress: float  # overall 0.0-1.0
    step_progress: float  # within step 0.0-1.0
    message: str


class DeploymentResultData(BaseModel):
    """Typed result payload — always has success + optional error."""
    success: bool
    error: str | None = None
    model_config = ConfigDict(frozen=True, extra='allow')


class CompletedRun(_RunBase):
    """Pipeline finished (Prefect COMPLETED). Check result.success for business outcome."""
    type: Literal['completed'] = 'completed'
    result: DeploymentResultData


class FailedRun(_RunBase):
    """Pipeline crashed — execution error, not business logic."""
    type: Literal['failed'] = 'failed'
    error: str
    result: DeploymentResultData | None = None


class RemoteDeployment(Generic[TDoc, TOptions, TResult]):
    """Typed client for calling a remote PipelineDeployment via Prefect.

Name your client class identically to the server's PipelineDeployment
subclass so the auto-derived deployment name matches.

Generic parameters:
    TDoc: Document types accepted as input (single type or union).
    TOptions: FlowOptions subclass for the deployment.
    TResult: DeploymentResult subclass returned by the deployment.

Mirror type contract:
    The client defines local Document subclasses ('mirror types') whose class_name must
    match the remote pipeline's document types exactly. When the remote returns documents,
    they are deserialized using the local mirror types. If class names don't match,
    documents fail to deserialize.

    Tasks that return mirror-typed remote results should use persist_result=False
    to avoid polluting the DocumentStore with unknown class_name entries."""
    name: ClassVar[str]
    options_type: ClassVar[type[FlowOptions]]
    result_type: ClassVar[type[DeploymentResult]]
    trace_level: ClassVar[TraceLevel] = 'always'
    trace_cost: ClassVar[float | None] = None

    @property
    def deployment_path(self) -> str:
        """Full Prefect deployment path: '{flow_name}/{deployment_name}'."""
        return f"{self.name}/{self.name.replace('-', '_')}"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Auto-derive name unless explicitly set in class body
        if "name" not in cls.__dict__:
            cls.name = class_name_to_deployment_name(cls.__name__)

        # Extract Generic params: (TDoc, TOptions, TResult)
        generic_args = extract_generic_params(cls, RemoteDeployment)
        if len(generic_args) < 3:
            raise TypeError(f"{cls.__name__} must specify 3 Generic parameters: class {cls.__name__}(RemoteDeployment[DocType, OptionsType, ResultType])")

        doc_type, options_type, result_type = generic_args

        _validate_document_type(cls.__name__, doc_type)

        if not isinstance(options_type, type) or not issubclass(options_type, FlowOptions):
            raise TypeError(f"{cls.__name__}: second Generic param must be a FlowOptions subclass, got {options_type}")
        if not isinstance(result_type, type) or not issubclass(result_type, DeploymentResult):
            raise TypeError(f"{cls.__name__}: third Generic param must be a DeploymentResult subclass, got {result_type}")

        cls.options_type = options_type
        cls.result_type = result_type

        # Apply @trace to _execute: combined guard prevents no-op and double-wrap
        trace_level = getattr(cls, "trace_level", "always")
        if trace_level != "off" and not is_already_traced(cls._execute):
            cls._execute = trace(name=cls.name, level=trace_level)(cls._execute)  # type: ignore[assignment]

    @final
    async def run(
        self,
        run_id: str,
        documents: list[TDoc],
        options: TOptions,
        context: DeploymentContext | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> TResult:
        """Execute the remote deployment via Prefect."""
        return await self._execute(
            run_id,
            documents,
            options,
            context if context is not None else DeploymentContext(),
            on_progress,
        )


```

## Functions

```python
async def progress_update(fraction: float, message: str = "") -> None:
    """Report intra-flow progress (0.0-1.0). No-op without context.

    Publishes a ProgressEvent via the publisher and updates Prefect flow run
    labels (if flow_run_id available) so poll consumers see progress and
    staleness detection stays current.
    """
    ctx = _context.get()
    if ctx is None:
        return

    fraction = max(0.0, min(1.0, fraction))
    overall = _compute_weighted_progress(ctx.completed_minutes, ctx.current_flow_minutes, fraction, ctx.total_minutes)
    step_progress = round(fraction, 4)

    # Fire-and-forget progress event publish to avoid blocking flow execution
    if ctx.publisher is not None:
        event = ProgressEvent(
            run_id=ctx.run_id,
            flow_run_id=ctx.flow_run_id,
            flow_name=ctx.flow_name,
            step=ctx.step,
            total_steps=ctx.total_steps,
            progress=overall,
            step_progress=step_progress,
            status=FlowStatus.PROGRESS,
            message=message,
        )
        task = asyncio.create_task(ctx.publisher.publish_progress(event))
        task.add_done_callback(_on_publish_done)

    await _emit_progress(
        flow_run_id=ctx.flow_run_id,
        step=ctx.step,
        total_steps=ctx.total_steps,
        flow_name=ctx.flow_name,
        status=FlowStatus.PROGRESS,
        progress=overall,
        step_progress=step_progress,
        message=message,
    )

async def run_remote_deployment(
    deployment_name: str,
    parameters: dict[str, Any],
    on_progress: ProgressCallback | None = None,
) -> Any:
    """Run a remote Prefect deployment with optional progress callback.

    Creates the remote flow run immediately (timeout=0) then polls its state,
    invoking on_progress(fraction, message) on each poll cycle if provided.
    """

    async def _create_and_poll(client: PrefectClient, as_subflow: bool) -> Any:
        fr: FlowRun = await run_deployment(  # type: ignore[assignment]
            client=client,
            name=deployment_name,
            parameters=parameters,
            as_subflow=as_subflow,
            timeout=0,
        )
        return await _poll_remote_flow_run(client, cast(UUID, fr.id), deployment_name, on_progress=on_progress)

    async with get_client() as client:
        try:
            await client.read_deployment_by_name(name=deployment_name)
            return await _create_and_poll(client, True)  # noqa: FBT003
        except ObjectNotFound:
            pass

    if not settings.prefect_api_url:
        raise ValueError(f"{deployment_name} not found, PREFECT_API_URL not set")

    async with PrefectClient(
        api=settings.prefect_api_url,
        api_key=settings.prefect_api_key,
        auth_string=settings.prefect_api_auth_string,
    ) as client:
        try:
            await client.read_deployment_by_name(name=deployment_name)
            ctx = AsyncClientContext.model_construct(client=client, _httpx_settings=None, _context_stack=0)
            with ctx:
                return await _create_and_poll(client, False)  # noqa: FBT003
        except ObjectNotFound:
            pass

    raise ValueError(f"{deployment_name} deployment not found")

```

## Examples

**Default creation** (`tests/deployment/test_deployment_base.py:72`)

```python
def test_default_creation(self):
    """Test default context creates successfully."""
    ctx = DeploymentContext()
    assert ctx is not None
```

**Deployment result data** (`tests/deployment/test_deployment_base.py:179`)

```python
def test_deployment_result_data(self):
    """Test DeploymentResultData."""
    data = DeploymentResultData(success=True, error=None)
    assert data.success is True
    dumped = data.model_dump()
    assert "success" in dumped
```

**Subclass specific trace names** (`tests/deployment/test_remote_deployment.py:441`)

```python
def test_subclass_specific_trace_names(self):
    class PipelineA(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
        pass

    class PipelineB(RemoteDeployment[BetaDoc, FlowOptions, SimpleResult]):
        pass

    assert PipelineA._execute is not PipelineB._execute
```

**Three args returned by helper** (`tests/deployment/test_remote_deployment.py:94`)

```python
def test_three_args_returned_by_helper(self):
    class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
        trace_level: ClassVar[TraceLevel] = "off"

    args = extract_generic_params(Foo, RemoteDeployment)
    assert len(args) == 3
    assert args[0] is AlphaDoc
    assert args[1] is FlowOptions
    assert args[2] is SimpleResult
```

**Three params from remote deployment** (`tests/deployment/test_remote_deployment.py:612`)

```python
def test_three_params_from_remote_deployment(self):
    class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
        trace_level: ClassVar[TraceLevel] = "off"

    result = extract_generic_params(Foo, RemoteDeployment)
    assert len(result) == 3
    assert result[0] is AlphaDoc
    assert result[1] is FlowOptions
    assert result[2] is SimpleResult
```

**Union doc arg is union type** (`tests/deployment/test_remote_deployment.py:104`)

```python
def test_union_doc_arg_is_union_type(self):
    class Foo(RemoteDeployment[AlphaDoc | BetaDoc, FlowOptions, SimpleResult]):
        trace_level: ClassVar[TraceLevel] = "off"

    args = extract_generic_params(Foo, RemoteDeployment)
    assert isinstance(args[0], types.UnionType)
    assert set(args[0].__args__) == {AlphaDoc, BetaDoc}
```

**Union in first position** (`tests/deployment/test_remote_deployment.py:622`)

```python
def test_union_in_first_position(self):
    class Foo(RemoteDeployment[AlphaDoc | BetaDoc, FlowOptions, SimpleResult]):
        trace_level: ClassVar[TraceLevel] = "off"

    result = extract_generic_params(Foo, RemoteDeployment)
    assert isinstance(result[0], types.UnionType)
    assert result[1] is FlowOptions
    assert result[2] is SimpleResult
```

**Union with three plus doc types** (`tests/deployment/test_remote_deployment.py:577`)

```python
def test_union_with_three_plus_doc_types(self):
    class Foo(RemoteDeployment[AlphaDoc | BetaDoc | GammaDoc, FlowOptions, SimpleResult]):
        trace_level: ClassVar[TraceLevel] = "off"

    args = extract_generic_params(Foo, RemoteDeployment)
    assert isinstance(args[0], types.UnionType)
    assert len(args[0].__args__) == 3
```


## Error Examples

**Rejects extra fields** (`tests/deployment/test_deployment_base.py:77`)

```python
def test_rejects_extra_fields(self):
    """Test context rejects unknown fields (extra='forbid')."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DeploymentContext(unknown_field="value")  # type: ignore[call-arg]
```

**Rejects int** (`tests/deployment/test_remote_deployment.py:150`)

```python
def test_rejects_int(self):
    with pytest.raises(TypeError, match="Document subclass"):

        class Bad(RemoteDeployment[int, FlowOptions, SimpleResult]):  # type: ignore[type-var]
            trace_level: ClassVar[TraceLevel] = "off"
```

**Rejects no generic params** (`tests/deployment/test_remote_deployment.py:180`)

```python
def test_rejects_no_generic_params(self):
    with pytest.raises(TypeError, match="must specify 3 Generic parameters"):

        class Bad(RemoteDeployment):  # type: ignore[type-arg]
            trace_level: ClassVar[TraceLevel] = "off"
```

**Rejects non deployment result** (`tests/deployment/test_remote_deployment.py:169`)

```python
def test_rejects_non_deployment_result(self):
    class NotAResult(BaseModel):
        x: int = 1

    with pytest.raises(TypeError, match="DeploymentResult subclass"):

        class Bad(RemoteDeployment[AlphaDoc, FlowOptions, NotAResult]):  # type: ignore[type-var]
            trace_level: ClassVar[TraceLevel] = "off"
```

**Rejects non document in union** (`tests/deployment/test_remote_deployment.py:144`)

```python
def test_rejects_non_document_in_union(self):
    with pytest.raises(TypeError, match="Document subclass"):

        class Bad(RemoteDeployment[AlphaDoc | str, FlowOptions, SimpleResult]):  # type: ignore[type-var]
            trace_level: ClassVar[TraceLevel] = "off"
```

**Rejects non document type** (`tests/deployment/test_remote_deployment.py:138`)

```python
def test_rejects_non_document_type(self):
    with pytest.raises(TypeError, match="Document subclass"):

        class Bad(RemoteDeployment[str, FlowOptions, SimpleResult]):  # type: ignore[type-var]
            trace_level: ClassVar[TraceLevel] = "off"
```

**Rejects non flow options** (`tests/deployment/test_remote_deployment.py:158`)

```python
def test_rejects_non_flow_options(self):
    class NotFlowOptions(BaseModel):
        x: int = 1

    with pytest.raises(TypeError, match="FlowOptions subclass"):

        class Bad(RemoteDeployment[AlphaDoc, NotFlowOptions, SimpleResult]):  # type: ignore[type-var]
            trace_level: ClassVar[TraceLevel] = "off"
```

**Rejects two generic params** (`tests/deployment/test_remote_deployment.py:186`)

```python
def test_rejects_two_generic_params(self):
    with pytest.raises(TypeError):

        class Bad(RemoteDeployment[FlowOptions, SimpleResult]):  # type: ignore[type-arg]
            trace_level: ClassVar[TraceLevel] = "off"
```
