"""Core classes for pipeline deployments.

Provides the PipelineDeployment base class and related types for
creating unified, type-safe pipeline deployments with:
- Per-flow resume (skip if outputs exist in DocumentStore)
- Per-flow uploads (immediate, not just at end)
- Prefect state hooks (on_running, on_completion, etc.)
- Upload on failure (partial results saved)
"""

import asyncio
import hashlib
import os
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar, cast, final
from uuid import UUID, uuid4

import httpx
from lmnr import Laminar
from prefect import flow, get_client, runtime
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings

from ai_pipeline_core.document_store import SummaryGenerator, create_document_store, get_document_store, set_document_store
from ai_pipeline_core.document_store.memory import MemoryDocumentStore
from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents.context import RunContext, reset_run_context, set_run_context
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._initialization import get_tracking_service, initialize_observability
from ai_pipeline_core.observability._tracking._models import RunStatus
from ai_pipeline_core.pipeline.options import FlowOptions
from ai_pipeline_core.settings import settings
from ai_pipeline_core.testing import disable_run_logger, prefect_test_harness

from ._helpers import (
    StatusPayload,
    class_name_to_deployment_name,
    download_documents,
    extract_generic_params,
    send_webhook,
    upload_documents,
)
from .contract import CompletedRun, DeploymentResultData, FailedRun, ProgressRun
from .progress import _ZERO_UUID, _safe_uuid, flow_context

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
_CLI_FIELDS: frozenset[str] = frozenset({"working_directory", "project_name", "start", "end", "no_trace"})


def _compute_run_scope(project_name: str, documents: list[Document], options: FlowOptions) -> str:
    """Compute a run scope that fingerprints inputs and options.

    Different inputs or options produce a different scope, preventing
    stale cache hits when re-running with the same project name.
    """
    exclude = set(_CLI_FIELDS & set(type(options).model_fields))
    options_json = options.model_dump_json(exclude=exclude, exclude_none=True)

    if not documents:
        fingerprint = hashlib.sha256(options_json.encode()).hexdigest()[:16]
        return f"{project_name}:{fingerprint}"

    sha256s = sorted(doc.sha256 for doc in documents)
    fingerprint = hashlib.sha256(f"{':'.join(sha256s)}|{options_json}".encode()).hexdigest()[:16]
    return f"{project_name}:{fingerprint}"


def _reconstruct_documents(
    raw_docs: list[dict[str, Any]],
    known_types: list[type[Document]],
) -> list[Document]:
    """Reconstruct typed Documents from serialized dicts using class_name lookup."""
    if not raw_docs:
        return []
    type_map = {t.__name__: t for t in known_types}
    result: list[Document] = []
    for doc_dict in raw_docs:
        class_name = doc_dict.get("class_name", "")
        doc_type = type_map.get(class_name)
        if doc_type is None:
            if not known_types:
                raise ValueError(f"Cannot reconstruct document: unknown class_name '{class_name}'")
            doc_type = known_types[0]
            logger.warning("Unknown document class_name '%s', using fallback %s", class_name, doc_type.__name__)
        result.append(doc_type.from_dict(doc_dict))
    return result


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

    model_config = ConfigDict(frozen=True)


TOptions = TypeVar("TOptions", bound=FlowOptions)
TResult = TypeVar("TResult", bound=DeploymentResult)


def _reattach_flow_metadata(original: Any, target: Any) -> None:
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
            logger.warning("Status webhook failed: %s", e)


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

        run_uuid = _safe_uuid(flow_run_id) if flow_run_id else None

        if context.progress_webhook_url:
            payload = ProgressRun(
                flow_run_id=run_uuid or _ZERO_UUID,
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
                logger.warning("Progress webhook failed: %s", e)

        if run_uuid is not None:
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
            frid = (_safe_uuid(flow_run_id) if flow_run_id else None) or _ZERO_UUID
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
            logger.warning("Completion webhook failed: %s", e)

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


__all__ = [
    "DeploymentContext",
    "DeploymentResult",
    "PipelineDeployment",
]
