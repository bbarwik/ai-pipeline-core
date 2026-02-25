"""Core classes for pipeline deployments.

Provides the PipelineDeployment base class and related types for
creating unified, type-safe pipeline deployments with:
- Per-flow resume (skip if outputs exist in DocumentStore)
- Per-flow uploads (immediate, not just at end)
- Prefect flow-run label updates for progress tracking
- Upload on failure (partial results saved)
"""

import asyncio
import contextlib
import hashlib
import os
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Generic, TypeVar, cast, final
from uuid import UUID, uuid4

from lmnr import Laminar
from prefect import flow, get_client, runtime
from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings

from ai_pipeline_core.document_store._factory import create_document_store
from ai_pipeline_core.document_store._memory import MemoryDocumentStore
from ai_pipeline_core.document_store._protocol import get_document_store, set_document_store
from ai_pipeline_core.document_store._summary_worker import SummaryGenerator
from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents.context import RunContext, reset_run_context, set_run_context
from ai_pipeline_core.documents.types import RunScope
from ai_pipeline_core.exceptions import LLMError, PipelineCoreError
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._initialization import get_tracking_service
from ai_pipeline_core.observability._tracking._models import RunStatus
from ai_pipeline_core.pipeline.limits import (
    PipelineLimit,
    _ensure_concurrency_limits,
    _LimitsState,
    _reset_limits_state,
    _set_limits_state,
    _SharedStatus,
    _validate_concurrency_limits,
)
from ai_pipeline_core.pipeline.options import FlowOptions
from ai_pipeline_core.settings import Settings, settings

from ._helpers import (
    class_name_to_deployment_name,
    extract_generic_params,
    init_observability_best_effort,
)
from ._publishers import NoopPublisher
from ._resolve import DocumentInput, _OutputDocument, build_output_document, resolve_document_inputs
from ._types import (
    CompletedEvent,
    ErrorCode,
    FailedEvent,
    ProgressEvent,
    ResultPublisher,
    StartedEvent,
)
from .contract import FlowStatus
from .progress import _flow_context, _safe_uuid

logger = get_pipeline_logger(__name__)


def _classify_error(exc: BaseException) -> ErrorCode:
    """Map exception to ErrorCode enum value."""
    if isinstance(exc, LLMError):
        return ErrorCode.PROVIDER_ERROR
    if isinstance(exc, asyncio.CancelledError):
        return ErrorCode.CANCELLED
    if isinstance(exc, TimeoutError):
        return ErrorCode.DURATION_EXCEEDED
    if isinstance(exc, (ValueError, TypeError)):
        return ErrorCode.INVALID_INPUT
    if isinstance(exc, PipelineCoreError):
        return ErrorCode.PIPELINE_ERROR
    return ErrorCode.UNKNOWN


def _create_publisher(settings_obj: Settings) -> ResultPublisher:
    """Create publisher based on environment configuration.

    Returns PubSubPublisher when Pub/Sub is configured, NoopPublisher otherwise.
    """
    if settings_obj.pubsub_project_id and settings_obj.pubsub_topic_id:
        if not settings_obj.clickhouse_host:
            raise ValueError("PubSubPublisher requires CLICKHOUSE_HOST for task_results durability")
        if not settings_obj.service_type:
            raise ValueError("PubSubPublisher requires SERVICE_TYPE for CloudEvents source identification")
        from ._pubsub import PubSubPublisher
        from ._task_results import ClickHouseTaskResultStore

        result_store = ClickHouseTaskResultStore(
            host=settings_obj.clickhouse_host,
            port=settings_obj.clickhouse_port,
            database=settings_obj.clickhouse_database,
            username=settings_obj.clickhouse_user,
            password=settings_obj.clickhouse_password,
            secure=settings_obj.clickhouse_secure,
        )
        return PubSubPublisher(
            project_id=settings_obj.pubsub_project_id,
            topic_id=settings_obj.pubsub_topic_id,
            service_type=settings_obj.service_type,
            result_store=result_store,
        )
    return NoopPublisher()


_HEARTBEAT_INTERVAL_SECONDS = 30


def _build_summary_generator() -> SummaryGenerator | None:
    """Build a summary generator callable from settings, or None if disabled/unavailable."""
    if not settings.doc_summary_enabled:
        return None

    from ai_pipeline_core.document_store._summary_llm import generate_document_summary

    model = settings.doc_summary_model

    async def _generator(name: str, excerpt: str) -> str:
        return await generate_document_summary(name, excerpt, model=model)

    return _generator


# Fields added by run_cli()'s _CliOptions that should not affect the run scope fingerprint
_CLI_FIELDS: frozenset[str] = frozenset({"working_directory", "run_id", "start", "end", "no_trace"})


def _compute_run_scope(run_id: str, documents: list[Document], options: FlowOptions) -> RunScope:
    """Compute a run scope that fingerprints inputs and options.

    Different inputs or options produce a different scope, preventing
    stale cache hits when re-running with the same run_id.
    """
    exclude = set(_CLI_FIELDS & set(type(options).model_fields))
    options_json = options.model_dump_json(exclude=exclude, exclude_none=True)

    if not documents:
        fingerprint = hashlib.sha256(options_json.encode()).hexdigest()[:16]
        return RunScope(f"{run_id}:{fingerprint}")

    sha256s = sorted(doc.sha256 for doc in documents)
    fingerprint = hashlib.sha256(f"{':'.join(sha256s)}|{options_json}".encode()).hexdigest()[:16]
    return RunScope(f"{run_id}:{fingerprint}")


async def _heartbeat_loop(publisher: ResultPublisher, run_id: str) -> None:
    """Publish heartbeat signals at regular intervals until cancelled."""
    while True:
        await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)
        try:
            await publisher.publish_heartbeat(run_id)
        except Exception as e:
            logger.warning("Heartbeat publish failed: %s", e)


class DeploymentContext(BaseModel):
    """Infrastructure configuration for deployments. Progress is tracked via Prefect labels (pub/sub)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DeploymentResult(BaseModel):
    """Base class for deployment results."""

    success: bool
    error: str | None = None
    documents: tuple[_OutputDocument, ...] = ()

    model_config = ConfigDict(frozen=True)


TOptions = TypeVar("TOptions", bound=FlowOptions)
TResult = TypeVar("TResult", bound=DeploymentResult)

_LABEL_RUN_ID = "pipeline.run_id"


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
    - Progress tracking via Prefect labels (pub/sub)
    - Upload on failure: Save partial results if pipeline fails
    """

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

    @staticmethod
    @abstractmethod
    def build_result(run_id: str, documents: list[Document], options: TOptions) -> TResult:
        """Extract typed result from pipeline documents loaded from DocumentStore.

        Called for both full runs and partial runs (--start/--end). For partial runs,
        _build_partial_result() delegates here by default — override _build_partial_result()
        to customize partial run results.
        """
        ...

    def _build_partial_result(self, run_id: str, documents: list[Document], options: TOptions) -> TResult:
        """Build a result for partial pipeline runs (--start/--end that don't reach the last step).

        Override this method to customize partial run results. Default delegates to build_result.
        """
        return self.build_result(run_id, documents, options)

    def _all_document_types(self) -> list[type[Document]]:
        """Collect all document types from all flows (inputs + outputs), deduplicated."""
        types: dict[str, type[Document]] = {}
        for flow_fn in self.flows:
            for t in getattr(flow_fn, "input_document_types", []):
                types[t.__name__] = t
            for t in getattr(flow_fn, "output_document_types", []):
                types[t.__name__] = t
        return list(types.values())

    async def _update_progress_labels(
        self,
        flow_run_id: str,
        run_id: str,
        step: int,
        total_steps: int,
        flow_name: str,
        status: FlowStatus,
        step_progress: float = 0.0,
        message: str = "",
    ) -> None:
        """Update Prefect flow run labels with progress information."""
        flow_minutes = [getattr(f, "estimated_minutes", 1) for f in self.flows]
        total_minutes = sum(flow_minutes) or 1
        completed_minutes = sum(flow_minutes[: max(step - 1, 0)])
        current_flow_minutes = flow_minutes[step - 1] if step - 1 < len(flow_minutes) else 1
        progress = round(max(0.0, min(1.0, (completed_minutes + current_flow_minutes * step_progress) / total_minutes)), 4)

        run_uuid = _safe_uuid(flow_run_id) if flow_run_id else None
        if run_uuid is None:
            return

        try:
            async with get_client() as client:
                await client.update_flow_run_labels(
                    flow_run_id=run_uuid,
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
            logger.warning("Progress label update failed: %s", e)

    def _build_progress_event(
        self,
        run_id: str,
        flow_run_id: str,
        flow_name: str,
        step: int,
        total_steps: int,
        flow_minutes: tuple[float, ...],
        status: FlowStatus,
        step_progress: float,
        message: str,
    ) -> ProgressEvent:
        """Build a ProgressEvent with computed overall progress."""
        total_mins = sum(flow_minutes) or 1
        completed_mins = sum(flow_minutes[: max(step - 1, 0)])
        current_flow_mins = flow_minutes[step - 1] if step - 1 < len(flow_minutes) else 1
        progress = round(max(0.0, min(1.0, (completed_mins + current_flow_mins * step_progress) / total_mins)), 4)
        return ProgressEvent(
            run_id=run_id,
            flow_run_id=flow_run_id,
            flow_name=flow_name,
            step=step,
            total_steps=total_steps,
            progress=progress,
            step_progress=round(step_progress, 4),
            status=status,
            message=message,
        )

    @final
    async def run(
        self,
        run_id: str,
        documents: Sequence[Document],
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
    def run_local(
        self,
        run_id: str,
        documents: Sequence[Document],
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


__all__ = [
    "DeploymentContext",
    "DeploymentResult",
    "PipelineDeployment",
]
