# MODULE: deployment
# CLASSES: DeploymentResult, PipelineDeployment, RunState, FlowStatus, PendingRun, ProgressRun, DeploymentResultData, CompletedRun, FailedRun, RemoteDeployment, OutputDocument
# DEPENDS: BaseModel, Generic, StrEnum
# PURPOSE: Pipeline deployment utilities for unified, type-safe deployments.
# VERSION: 0.12.4
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import DeploymentResult, PipelineDeployment, RemoteDeployment
from ai_pipeline_core.deployment import CompletedRun, DeploymentResultData, FailedRun, FlowStatus, OutputDocument, PendingRun, ProgressCallback, ProgressRun, RunResponse, RunState, progress_update
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
# Protocol — implement in concrete class
@runtime_checkable
class _ResultPublisher(Protocol):
    """Publishes pipeline lifecycle events to external consumers."""
    async def publish_started(self, event: _StartedEvent) -> None:
        """Publish a pipeline started event."""
        ...

    async def publish_progress(self, event: _ProgressEvent) -> None:
        """Publish a flow progress event."""
        ...

    async def publish_heartbeat(self, run_id: str) -> None:
        """Publish a heartbeat signal."""
        ...

    async def publish_completed(self, event: _CompletedEvent) -> None:
        """Publish a pipeline completed event."""
        ...

    async def publish_failed(self, event: _FailedEvent) -> None:
        """Publish a pipeline failed event."""
        ...

    async def close(self) -> None:
        """Release resources held by the publisher."""
        ...


```

## Public API

```python
class DeploymentResult(BaseModel):
    """Base class for deployment results."""
    success: bool
    error: str | None = None
    documents: tuple[OutputDocument, ...] = ()
    model_config = ConfigDict(frozen=True)


class PipelineDeployment(Generic[TOptions, TResult]):
    """Base class for pipeline deployments with three execution modes.

- ``run_cli()``: DualDocumentStore (ClickHouse + local) or local-only
- ``run_local()``: MemoryDocumentStore (ephemeral)
- ``as_prefect_flow()``: auto-configured from settings"""
    flows: ClassVar[list[Any]]
    name: ClassVar[str]
    options_type: ClassVar[type[FlowOptions]]
    result_type: ClassVar[type[DeploymentResult]]
    pubsub_service_type: ClassVar[str] = ''
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
        """Generate a Prefect flow for production deployment via ``ai-pipeline-deploy`` CLI."""
        deployment = self

        async def _deployment_flow(
            run_id: str,
            documents: list[DocumentInput],
            options: FlowOptions,
            parent_execution_id: str | None = None,
            parent_span_id: str | None = None,
        ) -> DeploymentResult:
            # Initialize observability for remote workers
            init_observability_best_effort()
            ensure_tracking_processor()

            # Set session ID from Prefect flow run for trace grouping
            flow_run_id = str(runtime.flow_run.get_id()) if runtime.flow_run else str(uuid4())  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
            os.environ["LMNR_SESSION_ID"] = flow_run_id

            publisher = _create_publisher(settings, deployment.pubsub_service_type)
            task_result_store = _create_task_result_store(settings)
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
                    parent_uuid = UUID(parent_execution_id) if parent_execution_id else None
                    result = await deployment.run(
                        run_id,
                        typed_docs,
                        cast(Any, options),
                        publisher=publisher,
                        task_result_store=task_result_store,
                        parent_execution_id=parent_uuid,
                        parent_span_id=parent_span_id,
                    )
                    Laminar.set_span_output(result.model_dump())
                    return result
            finally:
                await publisher.close()
                if task_result_store:
                    task_result_store.shutdown()
                store.shutdown()
                set_document_store(None)

        # Override generic annotations with concrete types for Prefect parameter schema generation
        _deployment_flow.__annotations__["options"] = self.options_type
        _deployment_flow.__annotations__["return"] = self.result_type

        # Attach integration metadata for deploy-time schema enrichment
        _deployment_flow._integration_meta = self._build_integration_meta()  # type: ignore[attr-defined]

        return flow(
            name=self.name,
            flow_run_name=f"{self.name}-{{run_id}}",
            persist_result=True,
            result_serializer="json",
        )(_deployment_flow)

    def build_partial_result(self, run_id: str, documents: list[Document], options: TOptions) -> TResult:
        """Build a result for partial pipeline runs (--start/--end that don't reach the last step).

        Override this method to customize partial run results. Default delegates to build_result.
        """
        return self.build_result(run_id, documents, options)

    @staticmethod
    @abstractmethod
    def build_result(run_id: str, documents: list[Document], options: TOptions) -> TResult:
        """Extract typed result from pipeline documents loaded from DocumentStore.

        Called for both full runs and partial runs (--start/--end). For partial runs,
        build_partial_result() delegates here by default — override build_partial_result()
        to customize partial run results.

        The base ``documents`` field on ``DeploymentResult`` is populated automatically
        by the framework after this method returns — only set fields defined on your
        custom result subclass.
        """
        ...

    @final
    async def run(
        self,
        run_id: str,
        documents: Sequence[Document],
        options: TOptions,
        publisher: _ResultPublisher | None = None,
        start_step: int = 1,
        end_step: int | None = None,
        task_result_store: TaskResultStore | None = None,
        parent_execution_id: UUID | None = None,
        parent_span_id: str | None = None,
    ) -> TResult:
        """Execute flows with resume, per-flow uploads, and step control.

        run_id must match ``[a-zA-Z0-9_-]+``, max 100 chars.
        """
        validate_run_id(run_id)

        if publisher is None:
            publisher = _NoopPublisher()
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
        ch_backend = None
        run_uuid: UUID | None = None
        run_start_time = None
        run_failed = False
        try:
            ch_backend = get_clickhouse_backend()
            if ch_backend:
                run_uuid = (_safe_uuid(flow_run_id) if flow_run_id else None) or uuid4()
                run_start_time = ch_backend.track_run_start(
                    execution_id=run_uuid,
                    run_id=run_id,
                    flow_name=self.name,
                    run_scope=str(run_scope),
                    parent_execution_id=parent_execution_id,
                    parent_span_id=parent_span_id,
                )
            # Set run context on processors so child spans inherit execution_id
            _set_run_context_on_processors(run_uuid or uuid4(), run_id, self.name, str(run_scope))
        except Exception as e:
            logger.warning("Tracking initialization failed: %s", e)
            ch_backend = None

        # Set concurrency limits and RunContext for the entire pipeline run
        failed_published = False
        heartbeat_task: asyncio.Task[None] | None = None
        limits_token = _set_limits_state(_LimitsState(limits=self.concurrency_limits, status=_SharedStatus()))
        run_token = _set_run_context(RunContext(run_scope=run_scope, execution_id=run_uuid))
        try:
            # Publish task.started event (inside try so failures still hit finally cleanup)
            await publisher.publish_started(_StartedEvent(run_id=run_id, flow_run_id=flow_run_id, run_scope=str(run_scope)))

            # Start heartbeat background task
            heartbeat_task = asyncio.create_task(_heartbeat_loop(publisher, run_id))

            await _ensure_concurrency_limits(self.concurrency_limits)

            # Save initial input documents to store
            if store and input_docs:
                await store.save_batch(input_docs, run_scope)

            # Precompute flow minutes for progress calculation
            flow_minutes = tuple(getattr(f, "estimated_minutes", 1) for f in self.flows)

            # Resume tracking: accumulate output SHA256s from skipped flows
            # so the first non-skipped flow loads by SHA256 instead of by type
            resumed_sha256s: set[str] | None = None
            executed_output_sha256s: set[str] = set()
            last_flow_output_sha256s: tuple[str, ...] = ()

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
                        if resumed_sha256s is None:
                            resumed_sha256s = {d.sha256 for d in input_docs}
                            resumed_sha256s.update(executed_output_sha256s)
                        resumed_sha256s.update(completion.output_sha256s)
                        last_flow_output_sha256s = completion.output_sha256s
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
                if resumed_sha256s is not None and store and input_types:
                    current_docs = await _load_documents_by_sha256s(  # pyright: ignore[reportUnreachable]  # reachable on 2nd+ loop iteration after resume skip
                        store,
                        resumed_sha256s,
                        input_types,
                        self._all_document_types(),
                        run_scope,
                    )
                    resumed_sha256s = None
                elif store and input_types:
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
                    result_docs = await flow_fn(run_id, current_docs, options)

                # Record flow completion for resume (only after successful execution)
                output_sha256s = tuple(d.sha256 for d in result_docs)
                executed_output_sha256s.update(output_sha256s)
                last_flow_output_sha256s = output_sha256s
                if store:
                    input_sha256s = tuple(d.sha256 for d in current_docs)
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
                result = self.build_partial_result(run_id, all_docs, options)
            else:
                result = self.build_result(run_id, all_docs, options)

            # Populate output documents
            output_docs = tuple(build_output_document(doc) for doc in all_docs)
            result = result.model_copy(update={"documents": output_docs})  # nosemgrep: no-document-model-copy

            # Compute chain_context from the last flow's actual outputs
            chain_context = {
                "version": 1,
                "run_scope": str(run_scope),
                "output_document_refs": list(last_flow_output_sha256s),
            }

            # Persist result to ClickHouse for remote caller fallback (independent of publisher)
            if task_result_store:
                try:
                    result_json = json.dumps(result.model_dump(), default=str)
                    chain_context_json = json.dumps(chain_context, default=str)
                    await task_result_store.write_result(run_id, result_json, chain_context_json)
                except Exception as e:
                    logger.warning("Task result store write failed for run_id=%s (non-blocking): %s", run_id, e)

            # Publish task.completed event
            await publisher.publish_completed(
                _CompletedEvent(
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
                        _FailedEvent(
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
            _reset_run_context(run_token)
            _reset_limits_state(limits_token)
            store = get_document_store()
            if store:
                try:
                    store.flush()
                except Exception as e:
                    logger.warning("Store flush failed: %s", e)
            _clear_run_context_on_processors()
            if ch_backend is not None and run_uuid is not None and run_start_time is not None:
                try:
                    ch_backend.track_run_end(
                        execution_id=run_uuid,
                        run_id=run_id,
                        flow_name=self.name,
                        run_scope=str(run_scope),
                        status=RunStatus.FAILED if run_failed else RunStatus.COMPLETED,
                        start_time=run_start_time,
                    )
                    ch_backend.flush()
                except Exception as e:
                    logger.warning("Tracking shutdown failed: %s", e)

    @final
    def run_cli(
        self,
        initializer: Callable[[TOptions], tuple[str, list[Document]]] | None = None,
        trace_name: str | None = None,
        cli_mixin: type[BaseSettings] | None = None,
    ) -> None:
        """Execute pipeline from CLI with positional working_directory and --start/--end/--no-trace flags."""
        from ._cli import run_cli_for_deployment

        run_cli_for_deployment(self, initializer, trace_name, cli_mixin)

    @final
    def run_local(
        self,
        run_id: str,
        documents: Sequence[Document],
        options: TOptions,
        publisher: _ResultPublisher | None = None,
        output_dir: Path | None = None,
    ) -> TResult:
        """Run locally with Prefect test harness and in-memory document store.

        Args:
            run_id: Pipeline run identifier.
            documents: Initial input documents.
            options: Flow options.
            publisher: Optional lifecycle event publisher (defaults to _NoopPublisher).
            output_dir: Optional directory for writing result.json.

        Returns:
            Typed deployment result.
        """
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        store = MemoryDocumentStore()
        set_document_store(store)
        try:
            with prefect_test_harness(), disable_run_logger():
                result = asyncio.run(self.run(run_id, documents, options, publisher=publisher))
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

Derives worker run_id as ``{run_id}-{fingerprint[:8]}`` from input documents
and options for resume and collision prevention.

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
    documents are silently skipped with a warning log.

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
        on_progress: ProgressCallback | None = None,
    ) -> TResult:
        """Execute the remote deployment via Prefect."""
        return await self._execute(
            run_id,
            documents,
            options,
            on_progress,
        )

    @final
    async def run_traced(
        self,
        run_id: str,
        documents: list[TDoc],
        options: TOptions,
        output_dir: Path,
        on_progress: ProgressCallback | None = None,
    ) -> TResult:
        """Execute with local .trace/ debug tracing at output_dir/.trace.

        Sets up FilesystemBackend + PipelineSpanProcessor for local debug
        output, matching the CLI pipeline tracing pattern. Use when running
        RemoteDeployment standalone (not inside a caller pipeline with tracing).
        """
        from ai_pipeline_core.deployment._cli import _init_debug_tracing
        from ai_pipeline_core.observability._initialization import get_clickhouse_backend

        init_observability_best_effort()
        debug_backend = _init_debug_tracing(output_dir)
        try:
            return await self._execute(run_id, documents, options, on_progress)
        finally:
            ch_backend = get_clickhouse_backend()
            if ch_backend:
                ch_backend.shutdown()
            if debug_backend:
                debug_backend.shutdown()


class OutputDocument(BaseModel):
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

## Functions

```python
async def progress_update(fraction: float, message: str = "") -> None:
    """Report intra-flow progress (0.0-1.0). No-op without context.

    Publishes a _ProgressEvent via the publisher and updates Prefect flow run
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
        event = _ProgressEvent(
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

```

## Examples

**Format starts with base run id** (`tests/deployment/test_remote_deployment.py:699`)

```python
def test_format_starts_with_base_run_id(self):
    """Derived run_id starts with the user's base run_id."""
    doc = AlphaDoc.create_root(name="test.txt", content="hello", reason="test")
    derived = _derive_remote_run_id("my-project", [doc], FlowOptions())
    assert derived.startswith("my-project-")
```

**Execute passes derived run id to prefect** (`tests/deployment/test_remote_deployment.py:730`)

```python
async def test_execute_passes_derived_run_id_to_prefect(self):
    """_execute() passes derived (not raw) run_id in Prefect parameters."""

    class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
        trace_level: ClassVar[TraceLevel] = "off"

    doc = AlphaDoc.create_root(name="test.txt", content="hello world", reason="test input")
    with patch(_REMOTE_RUN) as mock_run:
        mock_run.return_value = SimpleResult(success=True)
        await Foo().run("my-project", [doc], FlowOptions())

    params = mock_run.call_args[0][1]
    # run_id in parameters should NOT be the raw "my-project" but derived
    assert params["run_id"] != "my-project"
    assert params["run_id"].startswith("my-project-")
    assert len(params["run_id"]) > len("my-project-")
    # run_id kwarg must also be passed for ClickHouse fallback wiring
    assert mock_run.call_args.kwargs["run_id"] == params["run_id"]
```

**Progress update events have correct flow name** (`tests/deployment/test_pubsub_progress.py:261`)

```python
async def test_progress_update_events_have_correct_flow_name(
    self,
    real_publisher: PublisherWithStore,
    pubsub_topic: PubSubTestResources,
    pubsub_memory_store: MemoryDocumentStore,
):
    """progress_update() events carry the correct flow_name."""
    deployment = _ProgressReportingDeployment()
    await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store)

    await asyncio.sleep(FIRE_AND_FORGET_FLUSH_SECONDS)

    events = pull_events(pubsub_topic, expected_count=PROGRESS_REPORTING_EVENT_COUNT)
    progress = _progress_events(events)

    intra_flow = [e for e in progress if e.data["status"] == "progress"]
    for evt in intra_flow:
        assert evt.data["flow_name"] == "_progress_reporting_flow"
```

**Progress update events arrive on pubsub** (`tests/deployment/test_pubsub_progress.py:213`)

```python
async def test_progress_update_events_arrive_on_pubsub(
    self,
    real_publisher: PublisherWithStore,
    pubsub_topic: PubSubTestResources,
    pubsub_memory_store: MemoryDocumentStore,
):
    """progress_update() calls inside a flow publish PROGRESS events to real Pub/Sub."""
    deployment = _ProgressReportingDeployment()
    await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store)

    # Yield event loop so fire-and-forget publish tasks complete before
    # we block with synchronous pull_events()
    await asyncio.sleep(FIRE_AND_FORGET_FLUSH_SECONDS)

    events = pull_events(pubsub_topic, expected_count=PROGRESS_REPORTING_EVENT_COUNT)
    progress = _progress_events(events)

    # 5 progress events: STARTED + 3x PROGRESS + COMPLETED
    intra_flow = [e for e in progress if e.data["status"] == "progress"]
    assert len(intra_flow) == 3

    messages = [e.data["message"] for e in intra_flow]
    assert set(messages) == {"quarter done", "halfway there", "almost done"}
```

**Progress update step progress values** (`tests/deployment/test_pubsub_progress.py:237`)

```python
async def test_progress_update_step_progress_values(
    self,
    real_publisher: PublisherWithStore,
    pubsub_topic: PubSubTestResources,
    pubsub_memory_store: MemoryDocumentStore,
):
    """progress_update() events carry correct step_progress fractions."""
    deployment = _ProgressReportingDeployment()
    await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store)

    await asyncio.sleep(FIRE_AND_FORGET_FLUSH_SECONDS)

    events = pull_events(pubsub_topic, expected_count=PROGRESS_REPORTING_EVENT_COUNT)
    progress = _progress_events(events)

    intra_flow = [e for e in progress if e.data["status"] == "progress"]
    step_progresses = sorted(e.data["step_progress"] for e in intra_flow)
    assert step_progresses == [0.25, 0.5, 0.75]

    # All report step=1 (single-flow deployment) and total_steps=1
    for evt in intra_flow:
        assert evt.data["step"] == 1
        assert evt.data["total_steps"] == 1
```

**Deployment result data** (`tests/deployment/test_deployment_base.py:159`)

```python
def test_deployment_result_data(self):
    """Test DeploymentResultData."""
    data = DeploymentResultData(success=True, error=None)
    assert data.success is True
    dumped = data.model_dump()
    assert "success" in dumped
```

**Subclass specific trace names** (`tests/deployment/test_remote_deployment.py:419`)

```python
def test_subclass_specific_trace_names(self):
    class PipelineA(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
        pass

    class PipelineB(RemoteDeployment[BetaDoc, FlowOptions, SimpleResult]):
        pass

    assert PipelineA._execute is not PipelineB._execute
```

**Three args returned by helper** (`tests/deployment/test_remote_deployment.py:97`)

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

**Three params from remote deployment** (`tests/deployment/test_remote_deployment.py:590`)

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


## Error Examples

**Remote deployment rejects empty run id** (`tests/deployment/test_remote_deployment.py:900`)

```python
async def test_remote_deployment_rejects_empty_run_id(self):
    """RemoteDeployment._execute validates empty run_id."""

    class Wired2(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
        trace_level: ClassVar[TraceLevel] = "off"

    doc = AlphaDoc.create_root(name="test.txt", content="hello", reason="test")
    with pytest.raises(ValueError, match="must not be empty"):
        await Wired2().run("", [doc], FlowOptions())
```

**Remote deployment rejects invalid base run id** (`tests/deployment/test_remote_deployment.py:890`)

```python
async def test_remote_deployment_rejects_invalid_base_run_id(self):
    """RemoteDeployment._execute validates the base run_id before derivation."""

    class Wired(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
        trace_level: ClassVar[TraceLevel] = "off"

    doc = AlphaDoc.create_root(name="test.txt", content="hello", reason="test")
    with pytest.raises(ValueError, match="contains invalid characters"):
        await Wired().run("invalid run id with spaces", [doc], FlowOptions())
```

**Rejects int** (`tests/deployment/test_remote_deployment.py:153`)

```python
def test_rejects_int(self):
    with pytest.raises(TypeError, match="Document subclass"):

        class Bad(RemoteDeployment[int, FlowOptions, SimpleResult]):  # type: ignore[type-var]
            trace_level: ClassVar[TraceLevel] = "off"
```

**Rejects no generic params** (`tests/deployment/test_remote_deployment.py:183`)

```python
def test_rejects_no_generic_params(self):
    with pytest.raises(TypeError, match="must specify 3 Generic parameters"):

        class Bad(RemoteDeployment):  # type: ignore[type-arg]
            trace_level: ClassVar[TraceLevel] = "off"
```

**Rejects non deployment result** (`tests/deployment/test_remote_deployment.py:172`)

```python
def test_rejects_non_deployment_result(self):
    class NotAResult(BaseModel):
        x: int = 1

    with pytest.raises(TypeError, match="DeploymentResult subclass"):

        class Bad(RemoteDeployment[AlphaDoc, FlowOptions, NotAResult]):  # type: ignore[type-var]
            trace_level: ClassVar[TraceLevel] = "off"
```

**Rejects non document in union** (`tests/deployment/test_remote_deployment.py:147`)

```python
def test_rejects_non_document_in_union(self):
    with pytest.raises(TypeError, match="Document subclass"):

        class Bad(RemoteDeployment[AlphaDoc | str, FlowOptions, SimpleResult]):  # type: ignore[type-var]
            trace_level: ClassVar[TraceLevel] = "off"
```

**Rejects non document type** (`tests/deployment/test_remote_deployment.py:141`)

```python
def test_rejects_non_document_type(self):
    with pytest.raises(TypeError, match="Document subclass"):

        class Bad(RemoteDeployment[str, FlowOptions, SimpleResult]):  # type: ignore[type-var]
            trace_level: ClassVar[TraceLevel] = "off"
```
