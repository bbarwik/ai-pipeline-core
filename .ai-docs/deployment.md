# MODULE: deployment
# CLASSES: DeploymentContext, DeploymentResult, FlowCallable, PipelineDeployment, PendingRun, ProgressRun, DeploymentResultData, CompletedRun, FailedRun, Deployer, DownloadedDocument, StatusPayload, DeploymentHookResult, DeploymentHook, ProgressContext
# DEPENDS: ABC, BaseModel, Generic, Protocol, TypedDict
# SIZE: ~45KB

# === DEPENDENCIES (Resolved) ===

class ABC:
    """Python abstract base class marker."""
    ...

class BaseModel:
    """Pydantic base model. Fields are typed class attributes."""
    ...

class Generic:
    """Python generic base class for parameterized types."""
    ...

class Protocol:
    """External base class (not fully documented)."""
    ...

class TypedDict:
    """External base class (not fully documented)."""
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
    model_config = ConfigDict(frozen=True, extra='forbid')


class FlowCallable(Protocol):
    """Protocol for @pipeline_flow decorated functions."""
    name: str
    __name__: str
    input_document_types: list[type[Document]]
    output_document_types: list[type[Document]]
    estimated_minutes: int

    def __call__(self, project_name: str, documents: list[Document], flow_options: FlowOptions | dict[str, Any]) -> Any:  # type: ignore[type-arg]
        """Execute the flow with standard pipeline signature."""
        ...

    def with_options(self, **kwargs: Any) -> "FlowCallable":
        """Return a copy with overridden Prefect flow options (e.g., hooks)."""
        ...


class PipelineDeployment(Generic[TOptions, TResult]):
    """Base class for pipeline deployments.

Features enabled by default:
- Per-flow resume: Skip flows if outputs exist in DocumentStore
- Per-flow uploads: Upload documents after each flow
- Prefect hooks: Attach state hooks if status_webhook_url provided
- Upload on failure: Save partial results if pipeline fails"""
    flows: ClassVar[list[FlowCallable]]
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
            documents: list[Document],
            options: FlowOptions,
            context: DeploymentContext,
        ) -> DeploymentResult:
            # Initialize observability for remote workers
            try:
                initialize_observability()
            except Exception as e:
                logger.warning(f"Failed to initialize observability: {e}")
                with contextlib.suppress(Exception):
                    # Use canonical initializer to ensure consistent Laminar setup
                    from ai_pipeline_core.observability import tracing

                    tracing._initialise_laminar()

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
                    return await deployment.run(project_name, documents, cast(Any, options), context)
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
    ) -> TResult:
        """Execute all flows with resume, per-flow uploads, and webhooks.

        Args:
            project_name: Unique identifier for this pipeline run (used as run_scope).
            documents: Initial input documents for the first flow.
            options: Flow options passed to each flow.
            context: Deployment context with webhook URLs and document upload config.

        Returns:
            Typed deployment result built from all pipeline documents.
        """
        store = get_document_store()
        total_steps = len(self.flows)
        flow_run_id: str = str(runtime.flow_run.get_id()) if runtime.flow_run else ""  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]

        # Write identity labels for polling endpoint
        if flow_run_id:
            try:
                async with get_client() as client:
                    await client.update_flow_run_labels(
                        flow_run_id=UUID(flow_run_id),
                        labels={"pipeline.project_name": project_name},
                    )
            except Exception as e:
                logger.warning(f"Identity label update failed: {e}")

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
                run_uuid = UUID(flow_run_id) if flow_run_id else uuid4()
                tracking_svc.set_run_context(run_id=run_uuid, project_name=project_name, flow_name=self.name, run_scope=run_scope)
                tracking_svc.track_run_start(run_id=run_uuid, project_name=project_name, flow_name=self.name, run_scope=run_scope)
        except Exception:
            tracking_svc = None

        # Set RunContext for the entire pipeline run
        run_token = set_run_context(RunContext(run_scope=run_scope))
        try:
            # Save initial input documents to store
            if store and input_docs:
                await store.save_batch(input_docs, run_scope)

            for step, flow_fn in enumerate(self.flows, start=1):
                flow_name = getattr(flow_fn, "name", flow_fn.__name__)
                flow_run_id = str(runtime.flow_run.get_id()) if runtime.flow_run else ""  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]

                # Resume check: skip if output documents already exist in store
                output_types = getattr(flow_fn, "output_document_types", [])
                if store and output_types:
                    all_outputs_exist = all([await store.has_documents(run_scope, ot) for ot in output_types])
                    if all_outputs_exist:
                        logger.info(f"[{step}/{total_steps}] Resume: skipping {flow_name} (outputs exist)")
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

                # Prefect state hooks
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

                logger.info(f"[{step}/{total_steps}] Starting: {flow_name}")

                # Load input documents from store
                input_types = getattr(flow_fn, "input_document_types", [])
                if store and input_types:
                    current_docs = await store.load(run_scope, input_types)
                else:
                    current_docs = input_docs

                # Set up intra-flow progress context so progress_update() works inside flows
                flow_minutes = tuple(getattr(f, "estimated_minutes", 1) for f in self.flows)
                completed_mins = sum(flow_minutes[: max(step - 1, 0)])
                progress_queue: asyncio.Queue[ProgressRun | None] = asyncio.Queue()
                wh_url = context.progress_webhook_url or ""
                worker = asyncio.create_task(webhook_worker(progress_queue, wh_url)) if wh_url else None

                with flow_context(
                    webhook_url=wh_url,
                    project_name=project_name,
                    run_id=flow_run_id,
                    flow_run_id=flow_run_id,
                    flow_name=flow_name,
                    step=step,
                    total_steps=total_steps,
                    flow_minutes=flow_minutes,
                    completed_minutes=completed_mins,
                    queue=progress_queue,
                ):
                    try:
                        await active_flow(project_name, current_docs, options.model_dump())
                    except Exception as e:
                        # Upload partial results on failure
                        if context.output_documents_urls and store:
                            all_docs = await store.load(run_scope, self._all_document_types())
                            await upload_documents(all_docs, context.output_documents_urls)
                        await self._send_completion(context, flow_run_id, project_name, result=None, error=str(e))
                        completion_sent = True
                        raise
                    finally:
                        progress_queue.put_nowait(None)
                        if worker:
                            await worker

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

                logger.info(f"[{step}/{total_steps}] Completed: {flow_name}")

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
                with contextlib.suppress(Exception):
                    store.flush()
            if (svc := tracking_svc) is not None and run_uuid is not None:
                with contextlib.suppress(Exception):
                    svc.track_run_end(run_id=run_uuid, status=RunStatus.FAILED if run_failed else RunStatus.COMPLETED)
                    svc.flush()

    @final
    def run_cli(
        self,
        initializer: Callable[[TOptions], tuple[str, list[Document]]] | None = None,
        trace_name: str | None = None,
    ) -> None:
        """Execute pipeline from CLI arguments with --start/--end step control.

        Args:
            initializer: Optional callback returning (project_name, documents) from options.
            trace_name: Optional Laminar trace span name prefix.
        """
        if len(sys.argv) == 1:
            sys.argv.append("--help")

        setup_logging()
        try:
            initialize_observability()
            logger.info("Observability initialized.")
        except Exception as e:
            logger.warning(f"Failed to initialize observability: {e}")
            with contextlib.suppress(Exception):
                # Use canonical initializer to ensure consistent Laminar setup
                from ai_pipeline_core.observability import tracing

                tracing._initialise_laminar()

        deployment = self

        class _CliOptions(
            deployment.options_type,
            cli_parse_args=True,
            cli_kebab_case=True,
            cli_exit_on_error=True,
            cli_prog_name=deployment.name,
            cli_use_class_docs_for_groups=True,
        ):
            working_directory: CliPositionalArg[Path]
            project_name: str | None = None
            start: int = 1
            end: int | None = None
            no_trace: bool = False

            model_config = SettingsConfigDict(frozen=True, extra="ignore")

        opts = cast(TOptions, _CliOptions())  # type: ignore[reportCallIssue]

        wd = cast(Path, opts.working_directory)  # pyright: ignore[reportAttributeAccessIssue]
        wd.mkdir(parents=True, exist_ok=True)

        project_name = cast(str, opts.project_name or wd.name)  # pyright: ignore[reportAttributeAccessIssue]
        start_step = getattr(opts, "start", 1)
        end_step = getattr(opts, "end", None)
        no_trace = getattr(opts, "no_trace", False)

        # Set up local debug tracing (writes to <working_dir>/.trace)
        debug_processor: LocalDebugSpanProcessor | None = None
        if not no_trace:
            try:
                trace_path = wd / ".trace"
                trace_path.mkdir(parents=True, exist_ok=True)
                debug_config = TraceDebugConfig(path=trace_path, max_traces=20)
                debug_writer = LocalTraceWriter(debug_config)
                debug_processor = LocalDebugSpanProcessor(debug_writer)
                provider: Any = otel_trace.get_tracer_provider()
                if hasattr(provider, "add_span_processor"):
                    provider.add_span_processor(debug_processor)
                    logger.info(f"Local debug tracing enabled at {trace_path}")
            except Exception as e:
                logger.warning(f"Failed to set up local debug tracing: {e}")
                debug_processor = None

        # Initialize document store — ClickHouse when configured, local filesystem otherwise
        summary_generator = _build_summary_generator()
        if settings.clickhouse_host:
            store = create_document_store(settings, summary_generator=summary_generator)
        else:
            store = LocalDocumentStore(base_path=wd, summary_generator=summary_generator)
        set_document_store(store)

        # Initialize documents (always run initializer for run scope fingerprinting,
        # even when start_step > 1, so --start N resumes find the correct scope)
        initial_documents: list[Document] = []
        if initializer:
            _, initial_documents = initializer(opts)

        context = DeploymentContext()

        with ExitStack() as stack:
            if trace_name:
                stack.enter_context(
                    Laminar.start_as_current_span(
                        name=f"{trace_name}-{project_name}",
                        input=[opts.model_dump_json()],
                    )
                )

            under_pytest = "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules
            if not settings.prefect_api_key and not under_pytest:
                stack.enter_context(prefect_test_harness())
                stack.enter_context(disable_run_logger())

            result = asyncio.run(
                self._run_with_steps(
                    project_name=project_name,
                    options=opts,
                    context=context,
                    start_step=start_step,
                    end_step=end_step,
                    initial_documents=initial_documents,
                )
            )

        result_file = wd / "result.json"
        result_file.write_text(result.model_dump_json(indent=2))
        logger.info(f"Result saved to {result_file}")

        # Shutdown background workers (debug tracing, document summaries, tracking)
        if debug_processor is not None:
            debug_processor.shutdown()
        store = get_document_store()
        if store:
            store.shutdown()
        tracking_svc = get_tracking_service()
        if tracking_svc:
            tracking_svc.shutdown()

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

This is the official Prefect approach that handles flow registration,
deployment creation/updates, and all edge cases automatically."""
    def __init__(self):
        """Initialize deployer."""
        self.config = self._load_config()
        self._validate_prefect_settings()

    async def run(self):
        """Execute the complete deployment pipeline."""
        print("=" * 70)
        print(f"Prefect Deployment: {self.config['name']} v{self.config['version']}")
        print(f"Target: gs://{self.config['bucket']}/{self.config['folder']}")
        print("=" * 70)
        print()

        # Phase 1: Build flow package
        tarball = self._build_package()

        # Phase 2: Run deployment hooks (if configured)
        hook_result = await self._run_deployment_hooks()

        # Phase 3: Build agent bundles (legacy path, only if no hooks produced agents)
        agent_builds: dict[str, dict[str, Any]] = {}
        if hook_result is None or not any(path.startswith("agents/") for path, _ in hook_result.artifacts):
            agent_builds = self._build_agents()

        # Phase 4: Build vendor packages from [tool.deploy].vendor_packages
        vendor_wheels = self._build_vendor_packages()

        # Build cli-agents wheel if source is configured — it's a private package
        # not on PyPI, so the worker needs the wheel even when no agents are deployed
        cli_agents_source = self._get_cli_agents_source()
        if cli_agents_source:
            cli_dir = Path(cli_agents_source).resolve()
            if (cli_dir / "pyproject.toml").exists():
                cli_wheel = self._build_wheel_from_source(cli_dir)
                if cli_wheel.name not in {w.name for w in vendor_wheels}:
                    vendor_wheels.append(cli_wheel)
                    self._success(f"Built cli-agents vendor wheel: {cli_wheel.name}")

        # Phase 5: Upload flow package + vendor wheels
        await self._upload_package(tarball, vendor_wheels)

        # Phase 6: Upload agent bundles (legacy path)
        await self._upload_agents(agent_builds)

        # Phase 7: Upload hook artifacts
        if hook_result:
            await self._upload_hook_artifacts(hook_result)

        # Phase 8: Create/update Prefect deployment
        # Merge job_variables from hooks and legacy agent_builds
        combined_job_vars: dict[str, Any] = {}
        if hook_result and hook_result.job_variables:
            combined_job_vars = hook_result.job_variables
        await self._deploy_via_api(agent_builds, hook_job_variables=combined_job_vars)

        print()
        print("=" * 70)
        self._success("Deployment complete!")
        print("=" * 70)


class DownloadedDocument(Document):
    """Concrete document for downloaded content."""
    # [Inherited from Document]
    # __init__, __init_subclass__, approximate_tokens_count, as_json, as_pydantic_model, as_yaml, canonical_name, content_sha256, create, from_dict, get_expected_files, has_source, id, is_image, is_pdf, is_text, mime_type, model_convert, parse, serialize_content, serialize_model, sha256, size, source_documents, source_references, text, validate_content, validate_file_name, validate_name, validate_no_source_origin_overlap, validate_origins, validate_sources, validate_total_size


class StatusPayload(TypedDict):
    """Webhook payload for Prefect state transitions (sub-flow level)."""
    type: Literal['status']
    flow_run_id: str
    project_name: str
    step: int
    total_steps: int
    flow_name: str
    state: str
    state_name: str
    timestamp: str


@dataclass
class DeploymentHookResult:
    """Result from a deployment hook.

Attributes:
    artifacts: List of (relative_path, bytes) tuples to upload
    job_variables: Dict to deep-merge into Prefect job_variables,
        e.g. {"env": {"MY_VAR": "value"}}"""
    artifacts: list[tuple[str, bytes]] = field(default_factory=list)
    job_variables: dict[str, Any] = field(default_factory=dict)


class DeploymentHook(ABC):
    """Abstract base class for deployment hooks.

Implementations extend the deployment process with custom logic.
Each hook is called once during deployment and can:
- Build additional artifacts (agent bundles, config files, etc.)
- Add environment variables to Prefect worker configuration

Hooks are loaded explicitly from pyproject.toml configuration,
not auto-registered, for predictable behavior."""
    @property
    @abstractmethod
    def name(self) -> str:
        """Hook name for logging and debugging."""

    @abstractmethod
    async def process(
        self,
        project_root: Path,
        pyproject: dict[str, Any],
        build_dir: Path,
        upload_uri: str,
    ) -> DeploymentHookResult | None:
        """Process the deployment.

        Called during deployment after the main package is built.
        Return None to skip (e.g., if this hook doesn't apply),
        or DeploymentHookResult with artifacts and job_variables.

        Args:
            project_root: Root directory of the project being deployed
            pyproject: Parsed pyproject.toml contents
            build_dir: Temporary directory for build artifacts
            upload_uri: Base URI where artifacts will be uploaded
                (e.g., "gs://bucket/flows/my-project")

        Returns:
            DeploymentHookResult with artifacts and job_variables,
            or None to skip this hook
        """


@dataclass(frozen=True, slots=True)
class ProgressContext:
    """Internal context holding state for progress calculation and webhook delivery."""
    webhook_url: str
    project_name: str
    run_id: str
    flow_run_id: str
    flow_name: str
    step: int
    total_steps: int
    total_minutes: float
    completed_minutes: float
    current_flow_minutes: float
    queue: asyncio.Queue[ProgressRun | None]


# === FUNCTIONS ===

def class_name_to_deployment_name(class_name: str) -> str:
    """Convert PascalCase to kebab-case: ResearchPipeline -> research-pipeline."""
    name = re.sub(r"(?<!^)(?=[A-Z])", "-", class_name)
    return name.lower()

def extract_generic_params(cls: type, base_class: type) -> tuple[type | None, type | None]:
    """Extract TOptions and TResult from a generic base class's args."""
    for base in getattr(cls, "__orig_bases__", []):
        origin = getattr(base, "__origin__", None)
        if origin is base_class:
            args = getattr(base, "__args__", ())
            if len(args) == 2:
                return args[0], args[1]

    return None, None

async def download_documents(urls: list[str]) -> list[Document]:
    """Download documents from URLs."""
    documents: list[Document] = []
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        for url in urls:
            response = await client.get(url)
            response.raise_for_status()
            filename = url.split("/")[-1].split("?")[0] or "document"
            documents.append(DownloadedDocument(name=filename, content=response.content))
    return documents

async def upload_documents(documents: list[Document], url_mapping: dict[str, str]) -> None:
    """Upload documents to their mapped URLs."""
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        for doc in documents:
            if doc.name in url_mapping:
                response = await client.put(
                    url_mapping[doc.name],
                    content=doc.content,
                    headers={"Content-Type": doc.mime_type},
                )
                response.raise_for_status()

async def send_webhook(
    url: str,
    payload: ProgressRun | CompletedRun | FailedRun,
    max_retries: int = 3,
    retry_delay: float = 10.0,
) -> None:
    """Send webhook with retries."""
    data: dict[str, Any] = payload.model_dump(mode="json")
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(url, json=data, follow_redirects=True)
                response.raise_for_status()
            return
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Webhook retry {attempt + 1}/{max_retries}: {e}")
                await asyncio.sleep(retry_delay)
            else:
                logger.exception(f"Webhook failed after {max_retries} attempts")
                raise

def load_deployment_hooks(pyproject: dict[str, Any]) -> list[DeploymentHook]:
    """Load deployment hooks from pyproject.toml configuration.

    Hooks are specified as module paths in [tool.deploy.hooks].
    Each module must have a get_hook() function returning a DeploymentHook.

    Args:
        pyproject: Parsed pyproject.toml contents

    Returns:
        List of DeploymentHook instances

    Example pyproject.toml:
        [tool.deploy]
        hooks = ["cli_agents.pipeline_integration.deployment_hook"]
    """
    import importlib

    hook_modules = pyproject.get("tool", {}).get("deploy", {}).get("hooks", [])
    hooks: list[DeploymentHook] = []

    for module_path in hook_modules:
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, "get_hook"):
                hook = module.get_hook()
                if isinstance(hook, DeploymentHook):
                    hooks.append(hook)
                else:
                    raise TypeError(f"get_hook() must return DeploymentHook, got {type(hook).__name__}")
            else:
                raise AttributeError(f"Hook module {module_path} must have get_hook() function")
        except Exception as e:
            raise RuntimeError(f"Failed to load deployment hook '{module_path}': {e}") from e

    return hooks

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

    # Enqueue webhook payload for async delivery
    if ctx.webhook_url:
        payload = ProgressRun(
            flow_run_id=UUID(ctx.flow_run_id) if ctx.flow_run_id else UUID(int=0),
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
        ctx.queue.put_nowait(payload)

    # Update Prefect labels so polling consumers and staleness detection stay current
    if ctx.flow_run_id:
        try:
            async with get_client() as client:
                await client.update_flow_run_labels(
                    flow_run_id=UUID(ctx.flow_run_id),
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
            logger.warning(f"Progress label update failed: {e}")

async def webhook_worker(
    queue: asyncio.Queue[ProgressRun | None],
    webhook_url: str,
    max_retries: int = 3,
    retry_delay: float = 10.0,
) -> None:
    """Process webhooks sequentially with retries, preserving order."""
    while True:
        payload = await queue.get()
        if payload is None:
            queue.task_done()
            break

        with contextlib.suppress(Exception):
            await send_webhook(webhook_url, payload, max_retries, retry_delay)

        queue.task_done()

@contextmanager
def flow_context(  # noqa: PLR0917
    webhook_url: str,
    project_name: str,
    run_id: str,
    flow_run_id: str,
    flow_name: str,
    step: int,
    total_steps: int,
    flow_minutes: tuple[float, ...],
    completed_minutes: float,
    queue: asyncio.Queue[ProgressRun | None],
) -> Generator[None, None, None]:
    """Set up progress context for a flow. Framework internal use."""
    current_flow_minutes = flow_minutes[step - 1] if step <= len(flow_minutes) else 1.0
    total_minutes = sum(flow_minutes) if flow_minutes else current_flow_minutes
    ctx = ProgressContext(
        webhook_url=webhook_url,
        project_name=project_name,
        run_id=run_id,
        flow_run_id=flow_run_id,
        flow_name=flow_name,
        step=step,
        total_steps=total_steps,
        total_minutes=total_minutes,
        completed_minutes=completed_minutes,
        current_flow_minutes=current_flow_minutes,
        queue=queue,
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
        fr: FlowRun = await run_deployment(
            client=client,
            name=deployment_name,
            parameters=parameters,
            as_subflow=as_subflow,
            timeout=0,
        )  # type: ignore
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

def remote_deployment(
    deployment_class: type[PipelineDeployment[TOptions, TResult]],
    *,
    deployment_name: str | None = None,
    name: str | None = None,
    trace_level: TraceLevel = "always",
    trace_cost: float | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Coroutine[Any, Any, TResult]]]:
    """Decorator to call PipelineDeployment flows remotely with automatic serialization.

    The decorated function's body is never executed — it serves as a typed stub.
    The wrapper enforces the deployment contract: (project_name, documents, options, context).
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Coroutine[Any, Any, TResult]]:
        fname = getattr(func, "__name__", deployment_class.name)

        if _is_already_traced(func):
            raise TypeError(f"@remote_deployment target '{fname}' already has @trace")

        @wraps(func)
        async def _wrapper(
            project_name: str,
            documents: list[Document],
            options: TOptions,
            context: DeploymentContext | None = None,
            on_progress: ProgressCallback | None = None,
        ) -> TResult:
            parameters: dict[str, Any] = {
                "project_name": project_name,
                "documents": documents,
                "options": options,
                "context": context if context is not None else DeploymentContext(),
            }

            full_name = f"{deployment_class.name}/{deployment_name or deployment_class.name.replace('-', '_')}"

            result = await run_remote_deployment(full_name, parameters, on_progress=on_progress)

            if trace_cost is not None and trace_cost > 0:
                set_trace_cost(trace_cost)

            if isinstance(result, DeploymentResult):
                return cast(TResult, result)
            if isinstance(result, dict):
                return cast(TResult, deployment_class.result_type(**cast(dict[str, Any], result)))
            raise TypeError(f"Expected DeploymentResult, got {type(result).__name__}")

        traced_wrapper = trace(
            level=trace_level,
            name=name or deployment_class.name,
        )(_wrapper)

        return traced_wrapper

    return decorator

# === EXAMPLES (from tests/) ===

# Example: Basic remote deployment
# Source: tests/deployment/test_remote.py:61
async def test_basic_remote_deployment(self):
    """Test basic decorator usage returns correct result type."""

    @remote_deployment(SamplePipeline)
    async def my_flow(
        project_name: str,
        documents: list[Document],
        options: FlowOptions,
        context: DeploymentContext | None = None,
    ) -> SampleResult: ...

    with patch("ai_pipeline_core.deployment.remote.run_remote_deployment") as mock_run:
        mock_run.return_value = SampleResult(success=True, report="test")
        result = await my_flow("test-project", [], FlowOptions())

        assert isinstance(result, SampleResult)
        assert result.success is True
        assert result.report == "test"
        mock_run.assert_called_once()

# Example: Creation
# Source: tests/deployment/test_progress.py:265
def test_creation(self):
    """Test ProgressContext creation."""
    queue: asyncio.Queue[ProgressRun | None] = asyncio.Queue()
    ctx = ProgressContext(
        webhook_url="http://example.com",
        project_name="test",
        run_id="r1",
        flow_run_id=str(UUID(int=1)),
        flow_name="flow",
        step=1,
        total_steps=3,
        total_minutes=6.0,
        completed_minutes=0.0,
        current_flow_minutes=1.0,
        queue=queue,
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

# Example: Default name matches deployer convention
# Source: tests/deployment/test_remote.py:281
async def test_default_name_matches_deployer_convention(self):
    """Test that default deployment path matches what Deployer registers in Prefect.

    Deployer registers: flow_name=kebab-case, deployment_name=underscore (Python package name).
    remote_deployment must produce the same '{flow_name}/{package_name}' path.
    """

    @remote_deployment(SamplePipeline)
    async def my_flow(
        project_name: str,
        documents: list[Document],
        options: FlowOptions,
        context: DeploymentContext | None = None,
    ) -> SampleResult: ...

    with patch("ai_pipeline_core.deployment.remote.run_remote_deployment") as mock_run:
        mock_run.return_value = SampleResult(success=True)
        await my_flow("project", [], FlowOptions())

        full_name = mock_run.call_args[0][0]
        flow_name, deployment_name = full_name.split("/")
        # flow_name is kebab-case (class_name_to_deployment_name)
        assert flow_name == SamplePipeline.name
        assert "-" not in deployment_name, "deployment name must use underscores, not hyphens"
        assert deployment_name == SamplePipeline.name.replace("-", "_")

# === ERROR EXAMPLES (What NOT to Do) ===

# Error: Missing pyproject raises
# Source: tests/deployment/test_deploy.py:410
def test_missing_pyproject_raises(self, tmp_path: Path):
    """Should die if source dir has no pyproject.toml."""
    deployer = Deployer.__new__(Deployer)
    deployer._die = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))

    with pytest.raises(RuntimeError, match=r"No pyproject.toml"):
        deployer._build_wheel_from_source(tmp_path)

# Error: Already traced raises error
# Source: tests/deployment/test_remote.py:268
async def test_already_traced_raises_error(self):
    """Test that applying @trace before @remote_deployment raises TypeError."""
    with pytest.raises(TypeError, match="already has @trace"):

        @remote_deployment(SamplePipeline)
        @trace(level="always")
        async def my_flow(
            project_name: str,
            documents: list[Document],
            options: FlowOptions,
            context: DeploymentContext | None = None,
        ) -> SampleResult: ...

# Error: Dies when cli agents dir missing
# Source: tests/deployment/test_deploy.py:221
def test_dies_when_cli_agents_dir_missing(self, tmp_path: Path):
    """Should die when cli_agents_source points to non-existent dir."""
    deployer = Deployer.__new__(Deployer)
    deployer._pyproject_data = {
        "tool": {
            "deploy": {
                "cli_agents_source": str(tmp_path / "nonexistent"),
                "agents": {"my_agent": {"path": "/tmp"}},
            }
        },
    }
    deployer._die = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
    deployer._info = lambda *a, **k: None

    with pytest.raises(RuntimeError, match="cli-agents source not found"):
        deployer._build_agents()

# Error: Dies when no cli agents source
# Source: tests/deployment/test_deploy.py:210
def test_dies_when_no_cli_agents_source(self):
    """Should die when agents configured but cli_agents_source missing."""
    deployer = Deployer.__new__(Deployer)
    deployer._pyproject_data = {
        "tool": {"deploy": {"agents": {"my_agent": {"path": "/tmp"}}}},
    }
    deployer._die = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))

    with pytest.raises(RuntimeError, match="cli_agents_source is not set"):
        deployer._build_agents()
