"""Core classes for pipeline deployments.

Provides the PipelineDeployment base class and related types for
creating unified, type-safe pipeline deployments with:
- Per-flow resume (skip if outputs exist in DocumentStore)
- Per-flow uploads (immediate, not just at end)
- Prefect state hooks (on_running, on_completion, etc.)
- Upload on failure (partial results saved)
"""

import asyncio
import contextlib
import hashlib
import os
import sys
from abc import abstractmethod
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast, final
from uuid import UUID, uuid4

import httpx
from lmnr import Laminar
from opentelemetry import trace as otel_trace
from prefect import flow, get_client, runtime
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import CliPositionalArg, SettingsConfigDict

from ai_pipeline_core.document_store import SummaryGenerator, create_document_store, get_document_store, set_document_store
from ai_pipeline_core.document_store.local import LocalDocumentStore
from ai_pipeline_core.document_store.memory import MemoryDocumentStore
from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents.context import RunContext, reset_run_context, set_run_context
from ai_pipeline_core.logging import get_pipeline_logger, setup_logging
from ai_pipeline_core.observability._debug import LocalDebugSpanProcessor, LocalTraceWriter, TraceDebugConfig
from ai_pipeline_core.observability._initialization import get_tracking_service, initialize_observability
from ai_pipeline_core.observability._tracking._models import RunStatus
from ai_pipeline_core.pipeline.options import FlowOptions
from ai_pipeline_core.settings import settings
from ai_pipeline_core.testing import disable_run_logger, prefect_test_harness

from .contract import CompletedRun, DeploymentResultData, FailedRun, ProgressRun
from .helpers import (
    StatusPayload,
    class_name_to_deployment_name,
    download_documents,
    extract_generic_params,
    send_webhook,
    upload_documents,
)
from .progress import flow_context, webhook_worker

logger = get_pipeline_logger(__name__)


def _build_summary_generator() -> SummaryGenerator | None:
    """Build a summary generator callable from settings, or None if disabled/unavailable."""
    if not settings.doc_summary_enabled:
        return None

    from ai_pipeline_core.observability._summary import generate_document_summary

    model = settings.doc_summary_model

    async def _generator(name: str, excerpt: str) -> str:
        return await generate_document_summary(name, excerpt, model=model)

    return _generator


# Fields added by run_cli()'s _CliOptions that should not affect the run scope fingerprint
_CLI_FIELDS: set[str] = {"working_directory", "project_name", "start", "end", "no_trace"}


def _compute_run_scope(project_name: str, documents: list[Document], options: FlowOptions) -> str:
    """Compute a run scope that fingerprints inputs and options.

    Different inputs or options produce a different scope, preventing
    stale cache hits when re-running with the same project name.
    Falls back to just project_name when no documents are provided
    (e.g. --start N resume without initializer).
    """
    if not documents:
        return project_name
    sha256s = sorted(doc.sha256 for doc in documents)
    exclude = _CLI_FIELDS & set(type(options).model_fields)
    options_json = options.model_dump_json(exclude=exclude, exclude_none=True)
    fingerprint = hashlib.sha256(f"{':'.join(sha256s)}|{options_json}".encode()).hexdigest()[:16]
    return f"{project_name}:{fingerprint}"


class DeploymentContext(BaseModel):
    """Infrastructure configuration for deployments.

    Webhooks are optional - provide URLs to enable:
    - progress_webhook_url: Per-flow progress (started/completed/cached)
    - status_webhook_url: Prefect state transitions (RUNNING/FAILED/etc)
    - completion_webhook_url: Final result when deployment ends
    """

    input_documents_urls: tuple[str, ...] = Field(default_factory=tuple)
    output_documents_urls: dict[str, str] = Field(default_factory=dict)  # nosemgrep: mutable-field-on-frozen-pydantic-model

    progress_webhook_url: str = ""
    status_webhook_url: str = ""
    completion_webhook_url: str = ""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DeploymentResult(BaseModel):
    """Base class for deployment results."""

    success: bool
    error: str | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


TOptions = TypeVar("TOptions", bound=FlowOptions)
TResult = TypeVar("TResult", bound=DeploymentResult)


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


def _reattach_flow_metadata(original: FlowCallable, target: Any) -> None:
    """Reattach custom flow attributes that Prefect's with_options() may strip."""
    for attr in ("input_document_types", "output_document_types", "estimated_minutes"):
        if hasattr(original, attr) and not hasattr(target, attr):
            setattr(target, attr, getattr(original, attr))


@dataclass(slots=True)
class _StatusWebhookHook:
    """Prefect hook that sends status webhooks on state transitions."""

    webhook_url: str
    flow_run_id: str
    project_name: str
    step: int
    total_steps: int
    flow_name: str

    async def __call__(self, flow: Any, flow_run: Any, state: Any) -> None:
        payload: StatusPayload = {
            "type": "status",
            "flow_run_id": str(flow_run.id),
            "project_name": self.project_name,
            "step": self.step,
            "total_steps": self.total_steps,
            "flow_name": self.flow_name,
            "state": state.type.value if hasattr(state.type, "value") else str(state.type),
            "state_name": state.name or "",
            "timestamp": datetime.now(UTC).isoformat(),
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(self.webhook_url, json=payload)
        except Exception as e:
            logger.warning(f"Status webhook failed: {e}")


def _validate_flow_chain(deployment_name: str, flows: list[Any]) -> None:
    """Validate that each flow's input types are satisfiable by preceding flows' outputs.

    Simulates a type pool: starts with the first flow's input types, adds each flow's
    output types after processing. For subsequent flows, each required input type must
    be satisfiable by at least one type in the pool (via issubclass).
    """
    type_pool: set[type[Document]] = set()

    for i, flow_fn in enumerate(flows):
        input_types: list[type[Document]] = getattr(flow_fn, "input_document_types", [])
        output_types: list[type[Document]] = getattr(flow_fn, "output_document_types", [])
        flow_name = getattr(flow_fn, "name", getattr(flow_fn, "__name__", f"flow[{i}]"))

        if i == 0:
            # First flow: its input types seed the pool
            type_pool.update(input_types)
        elif input_types:
            # Subsequent flows: at least one declared input type must be satisfiable
            # from the pool (union semantics — flow accepts any of the declared types)
            any_satisfied = any(any(issubclass(available, t) for available in type_pool) for t in input_types)
            if not any_satisfied:
                input_names = sorted(t.__name__ for t in input_types)
                pool_names = sorted(t.__name__ for t in type_pool) if type_pool else ["(empty)"]
                raise TypeError(
                    f"{deployment_name}: flow '{flow_name}' (step {i + 1}) requires input types "
                    f"{input_names} but none are produced by preceding flows. "
                    f"Available types: {pool_names}"
                )

        type_pool.update(output_types)


class PipelineDeployment(Generic[TOptions, TResult]):
    """Base class for pipeline deployments.

    Features enabled by default:
    - Per-flow resume: Skip flows if outputs exist in DocumentStore
    - Per-flow uploads: Upload documents after each flow
    - Prefect hooks: Attach state hooks if status_webhook_url provided
    - Upload on failure: Save partial results if pipeline fails
    """

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

    @staticmethod
    @abstractmethod
    def build_result(project_name: str, documents: list[Document], options: TOptions) -> TResult:
        """Extract typed result from pipeline documents loaded from DocumentStore."""
        ...

    def _all_document_types(self) -> list[type[Document]]:
        """Collect all document types from all flows (inputs + outputs), deduplicated."""
        types: dict[str, type[Document]] = {}
        for flow_fn in self.flows:
            for t in getattr(flow_fn, "input_document_types", []):
                types[t.__name__] = t
            for t in getattr(flow_fn, "output_document_types", []):
                types[t.__name__] = t
        return list(types.values())

    def _build_status_hooks(
        self,
        context: DeploymentContext,
        flow_run_id: str,
        project_name: str,
        step: int,
        total_steps: int,
        flow_name: str,
    ) -> dict[str, list[Callable[..., Any]]]:
        """Build Prefect hooks for status webhooks."""
        hook = _StatusWebhookHook(
            webhook_url=context.status_webhook_url,
            flow_run_id=flow_run_id,
            project_name=project_name,
            step=step,
            total_steps=total_steps,
            flow_name=flow_name,
        )
        return {
            "on_running": [hook],
            "on_completion": [hook],
            "on_failure": [hook],
            "on_crashed": [hook],
            "on_cancellation": [hook],
        }

    async def _send_progress(
        self,
        context: DeploymentContext,
        flow_run_id: str,
        project_name: str,
        step: int,
        total_steps: int,
        flow_name: str,
        status: str,
        step_progress: float = 0.0,
        message: str = "",
    ) -> None:
        """Send progress webhook and update flow run labels."""
        # Use estimated_minutes for weighted progress calculation
        flow_minutes = [getattr(f, "estimated_minutes", 1) for f in self.flows]
        total_minutes = sum(flow_minutes) or 1
        completed_minutes = sum(flow_minutes[: max(step - 1, 0)])
        current_flow_minutes = flow_minutes[step - 1] if step - 1 < len(flow_minutes) else 1
        progress = round(max(0.0, min(1.0, (completed_minutes + current_flow_minutes * step_progress) / total_minutes)), 4)

        if context.progress_webhook_url:
            payload = ProgressRun(
                flow_run_id=UUID(flow_run_id) if flow_run_id else UUID(int=0),
                project_name=project_name,
                state="RUNNING",
                timestamp=datetime.now(UTC),
                step=step,
                total_steps=total_steps,
                flow_name=flow_name,
                status=status,
                progress=progress,
                step_progress=round(step_progress, 4),
                message=message,
            )
            try:
                await send_webhook(context.progress_webhook_url, payload)
            except Exception as e:
                logger.warning(f"Progress webhook failed: {e}")

        if flow_run_id:
            try:
                async with get_client() as client:
                    await client.update_flow_run_labels(
                        flow_run_id=UUID(flow_run_id),
                        labels={
                            "progress.step": step,
                            "progress.total_steps": total_steps,
                            "progress.flow_name": flow_name,
                            "progress.status": status,
                            "progress.progress": progress,
                            "progress.step_progress": round(step_progress, 4),
                            "progress.message": message,
                        },
                    )
            except Exception as e:
                logger.warning(f"Progress label update failed: {e}")

    async def _send_completion(
        self,
        context: DeploymentContext,
        flow_run_id: str,
        project_name: str,
        result: TResult | None,
        error: str | None,
    ) -> None:
        """Send completion webhook."""
        if not context.completion_webhook_url:
            return
        try:
            now = datetime.now(UTC)
            frid = UUID(flow_run_id) if flow_run_id else UUID(int=0)
            payload: CompletedRun | FailedRun
            if result is not None:
                payload = CompletedRun(
                    flow_run_id=frid,
                    project_name=project_name,
                    timestamp=now,
                    state="COMPLETED",
                    result=DeploymentResultData.model_validate(result.model_dump()),
                )
            else:
                payload = FailedRun(
                    flow_run_id=frid,
                    project_name=project_name,
                    timestamp=now,
                    state="FAILED",
                    error=error or "Unknown error",
                )
            await send_webhook(context.completion_webhook_url, payload)
        except Exception as e:
            logger.warning(f"Completion webhook failed: {e}")

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

    async def _run_with_steps(
        self,
        project_name: str,
        options: TOptions,
        context: DeploymentContext,
        start_step: int = 1,
        end_step: int | None = None,
        initial_documents: list[Document] | None = None,
    ) -> TResult:
        """Run pipeline with start/end step control and DocumentStore-based resume."""
        store = get_document_store()
        if end_step is None:
            end_step = len(self.flows)

        total_steps = len(self.flows)
        run_scope = _compute_run_scope(project_name, initial_documents or [], options)

        # Tracking lifecycle for CLI path
        tracking_svc = None
        run_uuid: UUID | None = None
        run_failed = False
        try:
            tracking_svc = get_tracking_service()
            if tracking_svc:
                run_uuid = uuid4()
                tracking_svc.set_run_context(run_id=run_uuid, project_name=project_name, flow_name=self.name, run_scope=run_scope)
                tracking_svc.track_run_start(run_id=run_uuid, project_name=project_name, flow_name=self.name, run_scope=run_scope)
        except Exception:
            tracking_svc = None

        # Set RunContext for the entire pipeline run
        run_token = set_run_context(RunContext(run_scope=run_scope))
        try:
            # Save initial documents to store
            if store and initial_documents:
                await store.save_batch(initial_documents, run_scope)

            for i in range(start_step - 1, end_step):
                step = i + 1
                flow_fn = self.flows[i]
                flow_name = getattr(flow_fn, "name", flow_fn.__name__)
                logger.info(f"--- [Step {step}/{total_steps}] {flow_name} ---")

                # Resume check: skip if output documents already exist
                output_types = getattr(flow_fn, "output_document_types", [])
                if store and output_types:
                    all_outputs_exist = all([await store.has_documents(run_scope, ot) for ot in output_types])
                    if all_outputs_exist:
                        logger.info(f"--- [Step {step}/{total_steps}] Skipping {flow_name} (outputs exist) ---")
                        continue

                # Load inputs from store
                input_types = getattr(flow_fn, "input_document_types", [])
                if store and input_types:
                    current_docs = await store.load(run_scope, input_types)
                else:
                    current_docs = initial_documents or []

                # Set up intra-flow progress context so progress_update() works inside flows
                flow_minutes = tuple(getattr(f, "estimated_minutes", 1) for f in self.flows)
                completed_mins = sum(flow_minutes[: max(step - 1, 0)])
                progress_queue: asyncio.Queue[ProgressRun | None] = asyncio.Queue()
                wh_url = context.progress_webhook_url or ""
                worker = asyncio.create_task(webhook_worker(progress_queue, wh_url)) if wh_url else None

                with flow_context(
                    webhook_url=wh_url,
                    project_name=project_name,
                    run_id=str(run_uuid) if run_uuid else "",
                    flow_run_id=str(run_uuid) if run_uuid else "",
                    flow_name=flow_name,
                    step=step,
                    total_steps=total_steps,
                    flow_minutes=flow_minutes,
                    completed_minutes=completed_mins,
                    queue=progress_queue,
                ):
                    try:
                        await flow_fn(project_name, current_docs, options)
                    finally:
                        progress_queue.put_nowait(None)
                        if worker:
                            await worker

            # Build result from all documents in store
            if store:
                all_docs = await store.load(run_scope, self._all_document_types())
            else:
                all_docs = initial_documents or []
            return self.build_result(project_name, all_docs, options)
        except Exception:
            run_failed = True
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


__all__ = [
    "DeploymentContext",
    "DeploymentResult",
    "PipelineDeployment",
]
