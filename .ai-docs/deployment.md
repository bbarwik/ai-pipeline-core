# MODULE: deployment
# CLASSES: DeploymentContext, DeploymentResult, PipelineDeployment, PendingRun, ProgressRun, DeploymentResultData, CompletedRun, FailedRun, Deployer, ProgressContext, RemoteDeployment
# DEPENDS: BaseModel, Generic
# SIZE: ~31KB

# === DEPENDENCIES (Resolved) ===

class BaseModel:
    """Pydantic base model. Fields are typed class attributes."""
    ...

class Generic:
    """Python generic base class for parameterized types."""
    ...

# === PUBLIC API ===

class DeploymentContext(BaseModel):
    """Infrastructure configuration for deployments.

Webhooks are optional - provide URLs to enable:
- progress_webhook_url: Per-flow progress (started/completed/cached)
- status_webhook_url: Prefect state transitions (RUNNING/FAILED/etc)
- completion_webhook_url: Final result when deployment ends"""
    input_documents_urls: tuple[str, ...] = Field(default_factory=tuple)
    output_documents_urls: dict[str, str] = Field(default_factory=dict)
    progress_webhook_url: str = ''
    status_webhook_url: str = ''
    completion_webhook_url: str = ''
    model_config = ConfigDict(frozen=True, extra='forbid')


class DeploymentResult(BaseModel):
    """Base class for deployment results."""
    success: bool
    error: str | None = None
    model_config = ConfigDict(frozen=True)


class PipelineDeployment(Generic[TOptions, TResult]):
    """Base class for pipeline deployments.

Features enabled by default:
- Per-flow resume: Skip flows if outputs exist in DocumentStore
- Per-flow uploads: Upload documents after each flow
- Prefect hooks: Attach state hooks if status_webhook_url provided
- Upload on failure: Save partial results if pipeline fails"""
    flows: ClassVar[list[Any]]
    name: ClassVar[str]
    options_type: ClassVar[type[FlowOptions]]
    result_type: ClassVar[type[DeploymentResult]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "flows"):
            return

        if cls.__name__.startswith("Test"):
            raise TypeError(f"Deployment class name cannot start with 'Test': {cls.__name__}")

        cls.name = class_name_to_deployment_name(cls.__name__)

        options_type, result_type = extract_generic_params(cls, PipelineDeployment)
        if options_type is None or result_type is None:
            raise TypeError(f"{cls.__name__} must specify Generic parameters: class {cls.__name__}(PipelineDeployment[MyOptions, MyResult])")

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

    @final
    def as_prefect_flow(self) -> Callable[..., Any]:
        """Generate a Prefect flow for production deployment.

        Returns:
            Async Prefect flow callable that initializes DocumentStore from settings.
        """
        deployment = self

        async def _deployment_flow(
            project_name: str,
            documents: list[dict[str, Any]],
            options: FlowOptions,
            context: DeploymentContext,
        ) -> DeploymentResult:
            # Initialize observability for remote workers
            try:
                initialize_observability()
            except Exception as e:
                logger.warning("Failed to initialize observability: %s", e)
                try:
                    from ai_pipeline_core.observability import tracing

                    tracing._initialise_laminar()
                except Exception as e2:
                    logger.warning("Laminar fallback initialization failed: %s", e2)

            # Set session ID from Prefect flow run for trace grouping
            flow_run_id = str(runtime.flow_run.get_id()) if runtime.flow_run else str(uuid4())  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
            os.environ["LMNR_SESSION_ID"] = flow_run_id

            store = create_document_store(
                settings,
                summary_generator=_build_summary_generator(),
            )
            set_document_store(store)
            try:
                # Create parent span to group all traces under a single deployment trace
                with Laminar.start_as_current_span(
                    name=f"{deployment.name}-{project_name}",
                    input={"project_name": project_name, "options": options.model_dump()},
                    session_id=flow_run_id,
                ):
                    typed_docs = _reconstruct_documents(documents, deployment._all_document_types())
                    return await deployment.run(project_name, typed_docs, cast(Any, options), context)
            finally:
                store.shutdown()
                set_document_store(None)

        # Patch annotations so Prefect generates the parameter schema from the concrete types
        _deployment_flow.__annotations__["options"] = self.options_type
        _deployment_flow.__annotations__["return"] = self.result_type

        return flow(
            name=self.name,
            flow_run_name=f"{self.name}-{{project_name}}",
            persist_result=True,
            result_serializer="json",
        )(_deployment_flow)

    @staticmethod
    @abstractmethod
    def build_result(project_name: str, documents: list[Document], options: TOptions) -> TResult:
        """Extract typed result from pipeline documents loaded from DocumentStore."""
        ...

    @final
    async def run(
        self,
        project_name: str,
        documents: list[Document],
        options: TOptions,
        context: DeploymentContext,
        start_step: int = 1,
        end_step: int | None = None,
    ) -> TResult:
        """Execute flows with resume, per-flow uploads, webhooks, and step control.

        Args:
            project_name: Unique identifier for this pipeline run (used as run_scope).
            documents: Initial input documents for the first flow.
            options: Flow options passed to each flow.
            context: Deployment context with webhook URLs and document upload config.
            start_step: First flow to execute (1-indexed, default 1).
            end_step: Last flow to execute (inclusive, default all flows).

        Returns:
            Typed deployment result built from all pipeline documents.
        """
        store = get_document_store()
        total_steps = len(self.flows)

        if end_step is None:
            end_step = total_steps
        if start_step < 1 or start_step > total_steps:
            raise ValueError(f"start_step must be 1-{total_steps}, got {start_step}")
        if end_step < start_step or end_step > total_steps:
            raise ValueError(f"end_step must be {start_step}-{total_steps}, got {end_step}")

        flow_run_id: str = (runtime.flow_run.get_id() or "") if runtime.flow_run else ""  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]

        # Write identity labels for polling endpoint
        flow_run_uuid = _safe_uuid(flow_run_id) if flow_run_id else None
        if flow_run_uuid is not None:
            try:
                async with get_client() as client:
                    await client.update_flow_run_labels(
                        flow_run_id=flow_run_uuid,
                        labels={"pipeline.project_name": project_name},
                    )
            except Exception as e:
                logger.warning("Identity label update failed: %s", e)

        # Download additional input documents
        input_docs = list(documents)
        if context.input_documents_urls:
            downloaded = await download_documents(list(context.input_documents_urls))
            input_docs.extend(downloaded)

        # Compute run scope AFTER downloads so the fingerprint includes all inputs
        run_scope = _compute_run_scope(project_name, input_docs, options)

        if not store and total_steps > 1:
            logger.warning("No DocumentStore configured for multi-step pipeline — intermediate outputs will not accumulate between flows")

        completion_sent = False

        # Tracking lifecycle
        tracking_svc = None
        run_uuid: UUID | None = None
        run_failed = False
        try:
            tracking_svc = get_tracking_service()
            if tracking_svc:
                run_uuid = (_safe_uuid(flow_run_id) if flow_run_id else None) or uuid4()
                tracking_svc.set_run_context(run_id=run_uuid, project_name=project_name, flow_name=self.name, run_scope=run_scope)
                tracking_svc.track_run_start(run_id=run_uuid, project_name=project_name, flow_name=self.name, run_scope=run_scope)
        except Exception as e:
            logger.warning("Tracking service initialization failed: %s", e)
            tracking_svc = None

        # Set RunContext for the entire pipeline run
        run_token = set_run_context(RunContext(run_scope=run_scope))
        try:
            # Save initial input documents to store
            if store and input_docs:
                await store.save_batch(input_docs, run_scope)

            for i in range(start_step - 1, end_step):
                step = i + 1
                flow_fn = self.flows[i]
                flow_name = getattr(flow_fn, "name", flow_fn.__name__)
                # Re-read flow_run_id in case Prefect subflow changes it
                flow_run_id = (runtime.flow_run.get_id() or "") if runtime.flow_run else flow_run_id  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]

                # Resume check: skip if output documents already exist in store
                output_types = getattr(flow_fn, "output_document_types", [])
                if store and output_types:
                    all_outputs_exist = all([await store.has_documents(run_scope, ot) for ot in output_types])
                    if all_outputs_exist:
                        logger.info("[%d/%d] Resume: skipping %s (outputs exist)", step, total_steps, flow_name)
                        await self._send_progress(
                            context,
                            flow_run_id,
                            project_name,
                            step,
                            total_steps,
                            flow_name,
                            "cached",
                            step_progress=1.0,
                            message=f"Resumed from store: {flow_name}",
                        )
                        continue

                # Prefect state hooks (conditional on status_webhook_url)
                active_flow = flow_fn
                if context.status_webhook_url:
                    hooks = self._build_status_hooks(context, flow_run_id, project_name, step, total_steps, flow_name)
                    active_flow = flow_fn.with_options(**hooks)
                    _reattach_flow_metadata(flow_fn, active_flow)

                # Progress: started
                await self._send_progress(
                    context,
                    flow_run_id,
                    project_name,
                    step,
                    total_steps,
                    flow_name,
                    "started",
                    step_progress=0.0,
                    message=f"Starting: {flow_name}",
                )

                logger.info("[%d/%d] Starting: %s", step, total_steps, flow_name)

                # Load input documents from store
                input_types = getattr(flow_fn, "input_document_types", [])
                if store and input_types:
                    current_docs = await store.load(run_scope, input_types)
                else:
                    current_docs = input_docs

                # Set up intra-flow progress context so progress_update() works inside flows
                flow_minutes = tuple(getattr(f, "estimated_minutes", 1) for f in self.flows)
                completed_mins = sum(flow_minutes[: max(step - 1, 0)])
                wh_url = context.progress_webhook_url or ""

                with flow_context(
                    webhook_url=wh_url,
                    project_name=project_name,
                    flow_run_id=flow_run_id,
                    flow_name=flow_name,
                    step=step,
                    total_steps=total_steps,
                    flow_minutes=flow_minutes,
                    completed_minutes=completed_mins,
                ):
                    try:
                        await active_flow(project_name, current_docs, options)
                    except Exception as e:
                        # Upload partial results on failure
                        if context.output_documents_urls and store:
                            all_docs = await store.load(run_scope, self._all_document_types())
                            await upload_documents(all_docs, context.output_documents_urls)
                        await self._send_completion(context, flow_run_id, project_name, result=None, error=str(e))
                        completion_sent = True
                        raise

                # Per-flow upload (load from store since @pipeline_flow saves there)
                if context.output_documents_urls and store and output_types:
                    flow_docs = await store.load(run_scope, output_types)
                    await upload_documents(flow_docs, context.output_documents_urls)

                # Progress: completed
                await self._send_progress(
                    context,
                    flow_run_id,
                    project_name,
                    step,
                    total_steps,
                    flow_name,
                    "completed",
                    step_progress=1.0,
                    message=f"Completed: {flow_name}",
                )

                logger.info("[%d/%d] Completed: %s", step, total_steps, flow_name)

            # Build result from all documents in store
            if store:
                all_docs = await store.load(run_scope, self._all_document_types())
            else:
                all_docs = input_docs
            result = self.build_result(project_name, all_docs, options)
            await self._send_completion(context, flow_run_id, project_name, result=result, error=None)
            return result

        except Exception as e:
            run_failed = True
            if not completion_sent:
                await self._send_completion(context, flow_run_id, project_name, result=None, error=str(e))
            raise
        finally:
            reset_run_context(run_token)
            store = get_document_store()
            if store:
                try:
                    store.flush()
                except Exception as e:
                    logger.warning("Store flush failed: %s", e)
            if (svc := tracking_svc) is not None and run_uuid is not None:
                try:
                    svc.track_run_end(run_id=run_uuid, status=RunStatus.FAILED if run_failed else RunStatus.COMPLETED)
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
            initializer: Optional callback returning (project_name, documents) from options.
            trace_name: Optional Laminar trace span name prefix.
            cli_mixin: Optional BaseSettings subclass with CLI-only fields mixed into options.
        """
        from ._cli import run_cli_for_deployment

        run_cli_for_deployment(self, initializer, trace_name, cli_mixin)

    @final
    def run_local(
        self,
        project_name: str,
        documents: list[Document],
        options: TOptions,
        context: DeploymentContext | None = None,
        output_dir: Path | None = None,
    ) -> TResult:
        """Run locally with Prefect test harness and in-memory document store.

        Args:
            project_name: Pipeline run identifier.
            documents: Initial input documents.
            options: Flow options.
            context: Optional deployment context (defaults to empty).
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
                result = asyncio.run(self.run(project_name, documents, options, context))
        finally:
            store.shutdown()
            set_document_store(None)

        if output_dir:
            (output_dir / "result.json").write_text(result.model_dump_json(indent=2))

        return result


class PendingRun(_RunBase):
    """Pipeline queued or running but no progress reported yet."""
    type: Literal['pending'] = 'pending'


class ProgressRun(_RunBase):
    """Pipeline running with step-level progress data."""
    type: Literal['progress'] = 'progress'
    step: int
    total_steps: int
    flow_name: str
    status: str
    progress: float
    step_progress: float
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


class Deployer:
    """Deploy Prefect flows using the RunnerDeployment pattern.

Handles flow registration, deployment creation/updates, and all edge cases
using the official Prefect approach."""
    def __init__(self):
        self.config = self._load_config()
        self._validate_prefect_settings()

    async def run(self) -> None:
        """Execute the complete deployment pipeline: build, upload, deploy."""
        print("=" * 70)
        print(f"Prefect Deployment: {self.config['name']} v{self.config['version']}")
        print(f"Target: gs://{self.config['bucket']}/{self.config['folder']}")
        print("=" * 70)
        print()

        tarball = self._build_package()
        await self._upload_package(tarball)
        await self._deploy_via_api()

        print()
        print("=" * 70)
        self._success("Deployment complete!")
        print("=" * 70)


@dataclass(frozen=True, slots=True)
class ProgressContext:
    """Internal context holding state for progress calculation and webhook delivery."""
    webhook_url: str
    project_name: str
    flow_run_id: str
    flow_name: str
    step: int
    total_steps: int
    total_minutes: float
    completed_minutes: float
    current_flow_minutes: float


class RemoteDeployment(Generic[TDoc, TOptions, TResult]):
    """Typed client for calling a remote PipelineDeployment via Prefect.

Name your client class identically to the server's PipelineDeployment
subclass so the auto-derived deployment name matches.

Generic parameters:
    TDoc: Document types accepted as input (single type or union).
    TOptions: FlowOptions subclass for the deployment.
    TResult: DeploymentResult subclass returned by the deployment.

Usage::

    class AiResearch(RemoteDeployment[
        ResearchTaskDocument | ContextDocument,
        FlowOptions,
        AiResearchResult,
    ]): pass

    _client = AiResearch()
    result = await _client.run("project", docs, FlowOptions())"""
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
        if len(generic_args) < 3 or any(a is None for a in generic_args):
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
        if trace_level != "off" and not _is_already_traced(cls._execute):
            cls._execute = trace(name=cls.name, level=trace_level)(cls._execute)  # type: ignore[assignment]

    @final
    async def run(
        self,
        project_name: str,
        documents: list[TDoc],
        options: TOptions,
        context: DeploymentContext | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> TResult:
        """Execute the remote deployment via Prefect."""
        return await self._execute(
            project_name,
            documents,
            options,
            context if context is not None else DeploymentContext(),
            on_progress,
        )


# === FUNCTIONS ===

async def update(fraction: float, message: str = "") -> None:
    """Report intra-flow progress (0.0-1.0). No-op without context.

    Sends webhook payload (if webhook_url configured) AND updates Prefect
    flow run labels (if flow_run_id available) so both push and poll consumers
    see progress, and staleness detection stays current.
    """
    ctx = _context.get()
    if ctx is None:
        return

    fraction = max(0.0, min(1.0, fraction))

    if ctx.total_minutes > 0:
        overall = (ctx.completed_minutes + ctx.current_flow_minutes * fraction) / ctx.total_minutes
    else:
        overall = fraction
    overall = round(max(0.0, min(1.0, overall)), 4)
    step_progress = round(fraction, 4)

    run_uuid = _safe_uuid(ctx.flow_run_id) if ctx.flow_run_id else None

    if ctx.webhook_url:
        payload = ProgressRun(
            flow_run_id=run_uuid or _ZERO_UUID,
            project_name=ctx.project_name,
            state="RUNNING",
            timestamp=datetime.now(UTC),
            step=ctx.step,
            total_steps=ctx.total_steps,
            flow_name=ctx.flow_name,
            status="progress",
            progress=overall,
            step_progress=step_progress,
            message=message,
        )
        try:
            await send_webhook(ctx.webhook_url, payload, _PROGRESS_WEBHOOK_MAX_RETRIES, _PROGRESS_WEBHOOK_RETRY_DELAY)
        except Exception as e:
            logger.warning("Progress webhook failed: %s", e)

    if run_uuid is not None:
        try:
            async with get_client() as client:
                await client.update_flow_run_labels(
                    flow_run_id=run_uuid,
                    labels={
                        "progress.step": ctx.step,
                        "progress.total_steps": ctx.total_steps,
                        "progress.flow_name": ctx.flow_name,
                        "progress.status": "progress",
                        "progress.progress": overall,
                        "progress.step_progress": step_progress,
                        "progress.message": message,
                    },
                )
        except Exception as e:
            logger.warning("Progress label update failed: %s", e)

@contextmanager
def flow_context(
    webhook_url: str,
    project_name: str,
    flow_run_id: str,
    flow_name: str,
    step: int,
    *,
    total_steps: int,
    flow_minutes: tuple[float, ...],
    completed_minutes: float,
) -> Generator[None, None, None]:
    """Set up progress context for a flow. Framework internal use."""
    current_flow_minutes = flow_minutes[step - 1] if step <= len(flow_minutes) else 1.0
    total_minutes = sum(flow_minutes) if flow_minutes else current_flow_minutes
    ctx = ProgressContext(
        webhook_url=webhook_url,
        project_name=project_name,
        flow_run_id=flow_run_id,
        flow_name=flow_name,
        step=step,
        total_steps=total_steps,
        total_minutes=total_minutes,
        completed_minutes=completed_minutes,
        current_flow_minutes=current_flow_minutes,
    )
    token = _context.set(ctx)
    try:
        yield
    finally:
        _context.reset(token)

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
        return await _poll_remote_flow_run(client, fr.id, deployment_name, on_progress=on_progress)

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

# === EXAMPLES (from tests/) ===

# Example: Path format matches deployer
# Source: tests/deployment/test_remote_deployment.py:213
def test_path_format_matches_deployer(self):
    class SamplePipeline(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
        trace_level: ClassVar[TraceLevel] = "off"

    path = SamplePipeline().deployment_path
    flow_name, deployment_name = path.split("/")
    assert flow_name == "sample-pipeline"
    assert deployment_name == "sample_pipeline"
    assert "-" not in deployment_name

# Example: Creation
# Source: tests/deployment/test_progress.py:261
def test_creation(self):
    """Test ProgressContext creation."""
    ctx = ProgressContext(
        webhook_url="http://example.com",
        project_name="test",
        flow_run_id=str(UUID(int=1)),
        flow_name="flow",
        step=1,
        total_steps=3,
        total_minutes=6.0,
        completed_minutes=0.0,
        current_flow_minutes=1.0,
    )
    assert ctx.step == 1
    assert ctx.total_steps == 3

# Example: Default creation
# Source: tests/deployment/test_deployment_base.py:72
def test_default_creation(self):
    """Test default context has empty values."""
    ctx = DeploymentContext()
    assert ctx.progress_webhook_url == ""
    assert ctx.status_webhook_url == ""
    assert ctx.completion_webhook_url == ""
    assert ctx.input_documents_urls == ()
    assert ctx.output_documents_urls == {}

# Example: Deployment result data
# Source: tests/deployment/test_deployment_base.py:195
def test_deployment_result_data(self):
    """Test DeploymentResultData."""
    data = DeploymentResultData(success=True, error=None)
    assert data.success is True
    dumped = data.model_dump()
    assert "success" in dumped

# === ERROR EXAMPLES (What NOT to Do) ===

# Error: Frozen
# Source: tests/deployment/test_deployment_base.py:92
def test_frozen(self):
    """Test context is immutable."""
    from pydantic import ValidationError

    ctx = DeploymentContext()
    with pytest.raises(ValidationError):
        ctx.progress_webhook_url = "http://new"  # type: ignore[misc]

# Error: Frozen
# Source: tests/deployment/test_progress.py:277
def test_frozen(self):
    """Test ProgressContext is immutable."""
    ctx = ProgressContext(
        webhook_url="http://example.com",
        project_name="test",
        flow_run_id=str(UUID(int=1)),
        flow_name="flow",
        step=1,
        total_steps=1,
        total_minutes=1.0,
        completed_minutes=0.0,
        current_flow_minutes=1.0,
    )
    with pytest.raises(AttributeError):
        ctx.step = 2  # type: ignore[misc]

# Error: Rejects int
# Source: tests/deployment/test_remote_deployment.py:151
def test_rejects_int(self):
    with pytest.raises(TypeError, match="Document subclass"):

        class Bad(RemoteDeployment[int, FlowOptions, SimpleResult]):  # type: ignore[type-var]
            trace_level: ClassVar[TraceLevel] = "off"

# Error: Rejects no generic params
# Source: tests/deployment/test_remote_deployment.py:181
def test_rejects_no_generic_params(self):
    with pytest.raises(TypeError, match="must specify 3 Generic parameters"):

        class Bad(RemoteDeployment):  # type: ignore[type-arg]
            trace_level: ClassVar[TraceLevel] = "off"
