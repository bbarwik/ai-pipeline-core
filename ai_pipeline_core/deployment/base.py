"""Core classes for pipeline deployments.

Provides the PipelineDeployment base class and related types for
creating unified, type-safe pipeline deployments with:
- Per-flow resume (skip when a FlowCompletion record exists in DocumentStore)
- Per-flow uploads (immediate, not just at end)
- Prefect flow-run label updates for progress tracking
- Upload on failure (partial results saved)
"""

import asyncio
import contextlib
import json
import os
import time
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
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

from ai_pipeline_core.document_store._memory import MemoryDocumentStore
from ai_pipeline_core.document_store._protocol import create_document_store, get_document_store, set_document_store
from ai_pipeline_core.documents import Document, DocumentSha256, RunContext, RunScope
from ai_pipeline_core.documents._context import TaskContext, _reset_run_context, _set_run_context, reset_task_context, set_task_context
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._initialization import ensure_tracking_processor, get_clickhouse_backend
from ai_pipeline_core.observability._tracking._models import RunStatus
from ai_pipeline_core.pipeline._execution_context import ExecutionContext, FlowFrame, get_execution_context, reset_execution_context, set_execution_context
from ai_pipeline_core.pipeline._flow import PipelineFlow
from ai_pipeline_core.pipeline._parallel import TaskHandle
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
from ai_pipeline_core.settings import settings

from ._contract import FlowStatus
from ._helpers import (
    _HANDLE_CANCEL_GRACE_SECONDS,
    _MILLISECONDS_PER_SECOND,
    _build_summary_generator,
    _classify_error,
    _clear_run_context_on_processors,
    _compute_run_scope,
    _create_publisher,
    _create_task_result_store,
    _heartbeat_loop,
    _set_flow_replay_payload,
    _set_run_context_on_processors,
    class_name_to_deployment_name,
    extract_generic_params,
    init_observability_best_effort,
    validate_run_id,
)
from ._resolve import DocumentInput, OutputDocument, build_output_document, resolve_document_inputs
from ._types import (
    DocumentRef,
    FlowCompletedEvent,
    FlowFailedEvent,
    FlowSkippedEvent,
    FlowStartedEvent,
    ProgressEvent,
    ResultPublisher,
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    TaskResultStore,
    _NoopPublisher,
)
from .progress import _compute_progress_from_flow_minutes, _flow_context, _safe_uuid

logger = get_pipeline_logger(__name__)


async def _load_documents_by_sha256s(
    store: Any,
    sha256s: set[str],
    input_types: list[type[Document]],
    all_types: list[type[Document]],
    run_scope: RunScope,
) -> list[Document]:
    """Load documents by SHA256, filtering to input types and preserving subclass identity.

    Uses load_nodes_by_sha256s for class_name metadata, then load_by_sha256s per type
    to ensure correct Document subclass construction across all store backends.
    """
    if not sha256s or not input_types:
        return []

    type_by_name = {t.__name__: t for t in all_types}
    input_type_names = {t.__name__ for t in input_types}

    sha256_list = [DocumentSha256(s) for s in sha256s]
    nodes = await store.load_nodes_by_sha256s(sha256_list)

    by_type: dict[type[Document], list[DocumentSha256]] = {}
    for sha, node in nodes.items():
        if node.class_name in input_type_names:
            doc_type = type_by_name.get(node.class_name)
            if doc_type is not None:
                by_type.setdefault(doc_type, []).append(sha)

    result: list[Document] = []
    for doc_type, type_sha256s in by_type.items():
        loaded = await store.load_by_sha256s(type_sha256s, doc_type, run_scope)
        result.extend(loaded.values())

    return result


async def _cancel_dispatched_handles(
    active_handles: set[object],
    *,
    baseline_handles: set[object],
) -> None:
    """Cancel handles dispatched within a flow and wait briefly for shutdown."""
    new_handles: list[TaskHandle[list[Document]]] = [
        handle for handle in list(active_handles) if handle not in baseline_handles and isinstance(handle, TaskHandle)
    ]
    if not new_handles:
        return

    for handle in new_handles:
        handle.cancel()

    pending_tasks: list[asyncio.Task[list[Document]]] = [handle._task for handle in new_handles if not handle.done]
    if pending_tasks:
        _done, pending = await asyncio.wait(pending_tasks, timeout=_HANDLE_CANCEL_GRACE_SECONDS)
        if pending:
            for task in pending:
                task.cancel()
            with contextlib.suppress(Exception):
                await asyncio.gather(*pending, return_exceptions=True)

    for handle in new_handles:
        active_handles.discard(handle)


class DeploymentResult(BaseModel):
    """Base class for deployment results."""

    success: bool
    error: str | None = None
    documents: tuple[OutputDocument, ...] = ()

    model_config = ConfigDict(frozen=True)


TOptions = TypeVar("TOptions", bound=FlowOptions)
TResult = TypeVar("TResult", bound=DeploymentResult)

_LABEL_RUN_ID = "pipeline.run_id"


class FlowAction(StrEnum):
    """Directive action for dynamic flow control."""

    CONTINUE = "continue"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class FlowDirective:
    """Flow planning directive returned by plan_next_flow()."""

    action: FlowAction = FlowAction.CONTINUE
    reason: str = ""


def _validate_flow_chain(deployment_name: str, flows: Sequence[PipelineFlow]) -> None:
    """Validate that each flow's input types are satisfiable by preceding flows' outputs.

    Simulates a type pool: starts with the first flow's input types, adds each flow's
    output types after processing. For subsequent flows, each required input type must
    be satisfiable by at least one type in the pool (via issubclass).
    """
    type_pool: set[type[Document]] = set()

    for i, flow_instance in enumerate(flows):
        flow_cls = type(flow_instance)
        input_types = flow_cls.input_document_types
        output_types = flow_cls.output_document_types
        flow_name = flow_instance.name

        if i == 0:
            type_pool.update(input_types)
        elif input_types:
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
    """Base class for pipeline deployments with three execution modes.

    - ``run_cli()``: DualDocumentStore (ClickHouse + local) or local-only
    - ``run_local()``: MemoryDocumentStore (ephemeral)
    - ``as_prefect_flow()``: auto-configured from settings
    """

    name: ClassVar[str]
    options_type: ClassVar[type[FlowOptions]]
    result_type: ClassVar[type[DeploymentResult]]
    # Sets CloudEvents ``source`` attribute (e.g. ``ai-{service_type}-worker``).
    # Does not affect topic routing. Requires PUBSUB_PROJECT_ID + PUBSUB_TOPIC_ID. Empty = _NoopPublisher.
    pubsub_service_type: ClassVar[str] = ""
    cache_ttl: ClassVar[timedelta | None] = timedelta(hours=24)
    concurrency_limits: ClassVar[Mapping[str, PipelineLimit]] = MappingProxyType({})

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if cls.__name__.startswith("Test"):
            raise TypeError(f"Deployment class name cannot start with 'Test': {cls.__name__}")

        if "name" not in cls.__dict__:
            cls.name = class_name_to_deployment_name(cls.__name__)

        generic_args = extract_generic_params(cls, PipelineDeployment)
        if len(generic_args) < 2:
            raise TypeError(f"{cls.__name__} must specify Generic parameters: class {cls.__name__}(PipelineDeployment[MyOptions, MyResult])")
        options_type, result_type = generic_args[0], generic_args[1]

        cls.options_type = options_type
        cls.result_type = result_type

        # build_result must be implemented (not still abstract from PipelineDeployment)
        build_result_fn = getattr(cls, "build_result", None)
        if build_result_fn is None or getattr(build_result_fn, "__isabstractmethod__", False):
            raise TypeError(f"{cls.__name__} must implement 'build_result' static method")

        if cls.build_flows is PipelineDeployment.build_flows:
            raise TypeError(f"{cls.__name__} must implement build_flows(options) -> Sequence[PipelineFlow]. Decorator-based `flows = [...]` is removed.")

        # Concurrency limits validation
        cls.concurrency_limits = _validate_concurrency_limits(cls.__name__, getattr(cls, "concurrency_limits", MappingProxyType({})))

    def build_flows(self, options: TOptions) -> Sequence[PipelineFlow]:
        """Build flow instances for this run."""
        raise NotImplementedError(f"{type(self).__name__}.build_flows() must return a sequence of PipelineFlow.")

    def plan_next_flow(
        self,
        flow_class: type[PipelineFlow],
        plan: Sequence[PipelineFlow],
        output_documents: list[Document],
    ) -> FlowDirective:
        """Optionally skip future instances of a flow class."""
        _ = (flow_class, plan, output_documents)
        return FlowDirective()

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

    def build_partial_result(self, run_id: str, documents: list[Document], options: TOptions) -> TResult:
        """Build a result for partial pipeline runs (--start/--end that don't reach the last step).

        Override this method to customize partial run results. Default delegates to build_result.
        """
        return self.build_result(run_id, documents, options)

    def _all_document_types(self, flows: Sequence[PipelineFlow]) -> list[type[Document]]:
        """Collect all document types from all flows (inputs + outputs), deduplicated."""
        types: dict[str, type[Document]] = {}
        for flow_inst in flows:
            flow_cls = type(flow_inst)
            for t in flow_cls.input_document_types:
                types[t.__name__] = t
            for t in flow_cls.output_document_types:
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
        flow_minutes: tuple[float, ...],
        step_progress: float = 0.0,
        message: str = "",
    ) -> None:
        """Update Prefect flow run labels with progress information."""
        progress = _compute_progress_from_flow_minutes(flow_minutes, step, step_progress)

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
        progress = _compute_progress_from_flow_minutes(flow_minutes, step, step_progress)
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
        publisher: ResultPublisher | None = None,
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
        flows = self.build_flows(options)
        if not flows:
            raise ValueError(f"{type(self).__name__}.build_flows() returned an empty list. Provide at least one PipelineFlow.")
        for flow_item in cast(Sequence[Any], flows):
            if not isinstance(flow_item, PipelineFlow):
                raise TypeError(f"{type(self).__name__}.build_flows() must return PipelineFlow instances, got {type(flow_item).__name__}.")
        _validate_flow_chain(type(self).__name__, flows)

        store = get_document_store()
        total_steps = len(flows)

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
        flow_plan = [
            {
                "name": flow_instance.name,
                "flow_class": type(flow_instance).__name__,
                "step": idx + 1,
                "estimated_minutes": flow_instance.estimated_minutes,
                "params": flow_instance.get_params(),
                "expected_tasks": type(flow_instance).expected_tasks(),
            }
            for idx, flow_instance in enumerate(flows)
        ]

        if not store and total_steps > 1:
            logger.warning("No DocumentStore configured for multi-step pipeline — intermediate outputs will not accumulate between flows")

        # Tracking lifecycle
        ch_backend = None
        run_uuid: UUID | None = None
        run_execution_id = uuid4()
        run_start_time = None
        run_failed = False
        try:
            ch_backend = get_clickhouse_backend()
            if ch_backend:
                run_uuid = (_safe_uuid(flow_run_id) if flow_run_id else None) or uuid4()
                run_execution_id = run_uuid
                run_start_time = ch_backend.track_run_start(
                    execution_id=run_uuid,
                    run_id=run_id,
                    flow_name=self.name,
                    run_scope=str(run_scope),
                    parent_execution_id=parent_execution_id,
                    parent_span_id=parent_span_id,
                    metadata={"flow_plan": flow_plan},
                )
            # Set run context on processors so child spans inherit execution_id
            _set_run_context_on_processors(run_execution_id, run_id, self.name, str(run_scope))
        except Exception as e:
            logger.warning("Tracking initialization failed: %s", e)
            ch_backend = None

        # Set concurrency limits and run context for the entire pipeline run
        failed_published = False
        heartbeat_task: asyncio.Task[None] | None = None
        limits_status = _SharedStatus()
        limits_token = _set_limits_state(_LimitsState(limits=self.concurrency_limits, status=limits_status))
        run_token = _set_run_context(RunContext(run_scope=run_scope, execution_id=run_execution_id))
        execution_token = set_execution_context(
            ExecutionContext(
                run_id=run_id,
                run_scope=run_scope,
                execution_id=run_execution_id,
                store=store,
                publisher=publisher,
                summary_generator=_build_summary_generator(),
                limits=self.concurrency_limits,
                limits_status=limits_status,
            )
        )
        try:
            all_document_types = self._all_document_types(flows)
            await publisher.publish_run_started(
                RunStartedEvent(
                    run_id=run_id,
                    flow_run_id=flow_run_id,
                    run_scope=str(run_scope),
                    flow_plan=flow_plan,
                )
            )

            # Start heartbeat background task
            heartbeat_task = asyncio.create_task(_heartbeat_loop(publisher, run_id))

            await _ensure_concurrency_limits(self.concurrency_limits)

            # Save initial input documents to store
            if store and input_docs:
                await store.save_batch(input_docs, run_scope, created_by_task="")

            # Precompute flow minutes for progress calculation
            flow_minutes = tuple(flow_instance.estimated_minutes for flow_instance in flows)

            # Resume tracking: accumulate output SHA256s from skipped flows
            # so the first non-skipped flow loads by SHA256 instead of by type
            resumed_sha256s: set[str] | None = None
            executed_output_sha256s: set[str] = set()
            last_flow_output_sha256s: tuple[str, ...] = ()
            skipped_classes: set[type[PipelineFlow]] = set()
            previous_output_documents: list[Document] = []

            for i in range(start_step - 1, end_step):
                step = i + 1
                flow_instance = flows[i]
                flow_class = type(flow_instance)
                flow_name = flow_instance.name
                completion_name = f"{flow_name}:{step}"
                # Re-read flow_run_id in case Prefect subflow changes it
                flow_run_id = str(runtime.flow_run.get_id() or "") if runtime.flow_run else flow_run_id  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]

                if flow_class in skipped_classes:
                    skipped_msg = f"Skipped by plan_next_flow: {flow_name}"
                    await publisher.publish_flow_skipped(
                        FlowSkippedEvent(
                            run_id=run_id,
                            flow_name=flow_name,
                            step=step,
                            total_steps=total_steps,
                            reason="skipped",
                        )
                    )
                    await self._update_progress_labels(
                        flow_run_id,
                        run_id,
                        step,
                        total_steps,
                        flow_name,
                        FlowStatus.CACHED,
                        flow_minutes=flow_minutes,
                        step_progress=1.0,
                        message=skipped_msg,
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
                            skipped_msg,
                        )
                    )
                    previous_output_documents = []
                    continue

                directive = self.plan_next_flow(flow_class, flows, previous_output_documents)
                if directive.action is FlowAction.SKIP:
                    skipped_classes.add(flow_class)
                    skip_reason = directive.reason or "skipped"
                    skipped_msg = f"Skipped by plan_next_flow: {flow_name}"
                    await publisher.publish_flow_skipped(
                        FlowSkippedEvent(
                            run_id=run_id,
                            flow_name=flow_name,
                            step=step,
                            total_steps=total_steps,
                            reason=skip_reason,
                        )
                    )
                    await self._update_progress_labels(
                        flow_run_id,
                        run_id,
                        step,
                        total_steps,
                        flow_name,
                        FlowStatus.CACHED,
                        flow_minutes=flow_minutes,
                        step_progress=1.0,
                        message=skipped_msg,
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
                            skipped_msg,
                        )
                    )
                    previous_output_documents = []
                    continue

                # Resume check: skip if flow completed successfully in a previous run
                if store:
                    completion = await store.get_flow_completion(run_scope, completion_name, max_age=self.cache_ttl)
                    if completion is not None:
                        logger.info("[%d/%d] Resume: skipping %s (completion record found)", step, total_steps, flow_name)
                        if resumed_sha256s is None:
                            resumed_sha256s = {d.sha256 for d in input_docs}
                            resumed_sha256s.update(executed_output_sha256s)
                        resumed_sha256s.update(completion.output_sha256s)
                        last_flow_output_sha256s = completion.output_sha256s
                        cached_msg = f"Resumed from store: {flow_name}"
                        await publisher.publish_flow_skipped(
                            FlowSkippedEvent(
                                run_id=run_id,
                                flow_name=flow_name,
                                step=step,
                                total_steps=total_steps,
                                reason="completed",
                            )
                        )
                        await self._update_progress_labels(
                            flow_run_id,
                            run_id,
                            step,
                            total_steps,
                            flow_name,
                            FlowStatus.CACHED,
                            flow_minutes=flow_minutes,
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
                        previous_output_documents = []
                        if completion.output_sha256s:
                            previous_output_documents = await _load_documents_by_sha256s(
                                store,
                                set(completion.output_sha256s),
                                flow_class.output_document_types,
                                all_document_types,
                                run_scope,
                            )
                        continue

                await publisher.publish_flow_started(
                    FlowStartedEvent(
                        run_id=run_id,
                        flow_name=flow_name,
                        flow_class=flow_class.__name__,
                        step=step,
                        total_steps=total_steps,
                        expected_tasks=flow_class.expected_tasks(),
                        flow_params=flow_instance.get_params(),
                    )
                )
                started_msg = f"Starting: {flow_name}"
                await self._update_progress_labels(
                    flow_run_id,
                    run_id,
                    step,
                    total_steps,
                    flow_name,
                    FlowStatus.STARTED,
                    flow_minutes=flow_minutes,
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
                input_types = flow_class.input_document_types
                pending_resume_sha256s = resumed_sha256s
                if pending_resume_sha256s is not None and store and input_types:
                    current_docs = await _load_documents_by_sha256s(
                        store,
                        pending_resume_sha256s,
                        input_types,
                        all_document_types,
                        run_scope,
                    )
                    resumed_sha256s = None
                elif store and input_types:
                    current_docs = await store.load(run_scope, input_types)
                else:
                    current_docs = input_docs

                # Set up intra-flow progress context so progress_update() works inside flows
                completed_mins = sum(flow_minutes[: max(step - 1, 0)])
                flow_started_at = time.monotonic()

                # Set FlowFrame on ExecutionContext so tasks inside this flow can emit task-level events
                flow_frame = FlowFrame(
                    name=flow_name,
                    flow_class_name=flow_class.__name__,
                    step=step,
                    total_steps=total_steps,
                    flow_minutes=flow_minutes,
                    completed_minutes=completed_mins,
                    flow_params=flow_instance.get_params(),
                )
                current_exec_ctx = get_execution_context()
                flow_exec_token = set_execution_context(current_exec_ctx.with_flow(flow_frame)) if current_exec_ctx is not None else None
                flow_task_token = set_task_context(TaskContext(scope_kind="flow", task_class_name=flow_class.__name__))
                active_handles_before: set[object] = set(current_exec_ctx.active_task_handles) if current_exec_ctx is not None else set()

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
                    try:
                        _set_flow_replay_payload(flow_instance, run_id, current_docs, options)
                        result_docs = await flow_instance.run(run_id, current_docs, options)
                    except (Exception, asyncio.CancelledError) as flow_exc:
                        if current_exec_ctx is not None:
                            await _cancel_dispatched_handles(current_exec_ctx.active_task_handles, baseline_handles=active_handles_before)
                        try:
                            await publisher.publish_flow_failed(
                                FlowFailedEvent(
                                    run_id=run_id,
                                    flow_name=flow_name,
                                    flow_class=flow_class.__name__,
                                    step=step,
                                    total_steps=total_steps,
                                    error_message=str(flow_exc),
                                )
                            )
                        except Exception as pub_err:
                            logger.warning("Failed to publish flow.failed event: %s", pub_err)
                        raise
                    finally:
                        reset_task_context(flow_task_token)
                        if flow_exec_token is not None:
                            reset_execution_context(flow_exec_token)

                # Cancel handles dispatched during this flow that weren't awaited
                if current_exec_ctx is not None:
                    leaked: list[TaskHandle[list[Document]]] = [
                        h for h in current_exec_ctx.active_task_handles if h not in active_handles_before and isinstance(h, TaskHandle) and not h.done
                    ]
                    if leaked:
                        logger.warning(
                            "PipelineFlow '%s' returned with %d un-awaited dispatched task(s). Cancelling to prevent post-flow writes.",
                            flow_class.__name__,
                            len(leaked),
                        )
                        await _cancel_dispatched_handles(
                            current_exec_ctx.active_task_handles,
                            baseline_handles=active_handles_before,
                        )

                untyped_result = cast(Any, result_docs)
                if not isinstance(untyped_result, list) or any(not isinstance(d, Document) for d in untyped_result):
                    raise TypeError(f"PipelineFlow '{flow_class.__name__}' returned invalid output. run() must return list[Document].")
                validated_docs: list[Document] = list(result_docs)

                # Record flow completion for resume (only after successful execution)
                output_sha256s = tuple(d.sha256 for d in validated_docs)
                executed_output_sha256s.update(output_sha256s)
                last_flow_output_sha256s = output_sha256s
                previous_output_documents = validated_docs
                if store:
                    await store.save_batch(validated_docs, run_scope, created_by_task="")
                    input_sha256s = tuple(d.sha256 for d in current_docs)
                    await store.save_flow_completion(run_scope, completion_name, input_sha256s, output_sha256s)

                completed_msg = f"Completed: {flow_name}"
                await self._update_progress_labels(
                    flow_run_id,
                    run_id,
                    step,
                    total_steps,
                    flow_name,
                    FlowStatus.COMPLETED,
                    flow_minutes=flow_minutes,
                    step_progress=1.0,
                    message=completed_msg,
                )
                completed_progress_event = self._build_progress_event(
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
                await publisher.publish_progress(completed_progress_event)
                summary_lookup: dict[DocumentSha256, str] = {}
                if store and output_sha256s:
                    summary_lookup = await store.load_summaries([DocumentSha256(sha) for sha in output_sha256s])
                output_refs = [
                    DocumentRef(
                        sha256=doc.sha256,
                        class_name=type(doc).__name__,
                        name=doc.name,
                        summary=summary_lookup.get(DocumentSha256(doc.sha256), "") or "",
                        publicly_visible=getattr(type(doc), "publicly_visible", False),
                        derived_from=tuple(doc.derived_from),
                        triggered_by=tuple(doc.triggered_by),
                    )
                    for doc in validated_docs
                ]
                await publisher.publish_flow_completed(
                    FlowCompletedEvent(
                        run_id=run_id,
                        flow_name=flow_name,
                        flow_class=flow_class.__name__,
                        step=step,
                        total_steps=total_steps,
                        duration_ms=int((time.monotonic() - flow_started_at) * _MILLISECONDS_PER_SECOND),
                        output_documents=output_refs,
                        progress=completed_progress_event.progress,
                    )
                )
                logger.info("[%d/%d] Completed: %s", step, total_steps, flow_name)

            # Build result from all documents in store
            if store:
                all_docs = await store.load(run_scope, all_document_types)
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

            await publisher.publish_run_completed(
                RunCompletedEvent(
                    run_id=run_id,
                    flow_run_id=flow_run_id,
                    result=result.model_dump(),
                    chain_context=chain_context,
                    actual_cost=ch_backend.run_total_cost if ch_backend is not None else 0.0,
                )
            )

            return result

        except (Exception, asyncio.CancelledError) as exc:
            run_failed = True
            current_exec_ctx = get_execution_context()
            if current_exec_ctx is not None:
                await _cancel_dispatched_handles(current_exec_ctx.active_task_handles, baseline_handles=set())
            if not failed_published:
                failed_published = True
                try:
                    await publisher.publish_run_failed(
                        RunFailedEvent(
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
            reset_execution_context(execution_token)
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
    def run_local(
        self,
        run_id: str,
        documents: Sequence[Document],
        options: TOptions,
        publisher: ResultPublisher | None = None,
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

    def _build_integration_meta(self) -> dict[str, Any]:
        """Build integration metadata for schema enrichment at deploy time.

        Collected once in as_prefect_flow() and injected into Prefect's
        parameter_openapi_schema.definitions by the deploy script.
        """
        flows: Sequence[PipelineFlow]
        try:
            options = cast(TOptions, self.options_type.model_construct())
            flows = self.build_flows(options)
        except Exception as exc:
            logger.warning("Failed to build flow metadata for %s: %s", type(self).__name__, exc)
            flows = []

        if not flows:
            return {
                "input_document_types": [],
                "all_document_types": [],
                "flow_chain": [],
            }

        first_flow = flows[0]
        input_types: list[type[Document]] = type(first_flow).input_document_types

        input_document_types = [{"class_name": t.__name__, "description": (t.__doc__ or "").strip()} for t in input_types]

        all_document_types = [{"class_name": t.__name__, "description": (t.__doc__ or "").strip()} for t in self._all_document_types(flows)]

        flow_chain = [
            {
                "name": flow_instance.name,
                "input_types": [t.__name__ for t in type(flow_instance).input_document_types],
                "output_types": [t.__name__ for t in type(flow_instance).output_document_types],
                "estimated_minutes": flow_instance.estimated_minutes,
            }
            for flow_instance in flows
        ]

        return {
            "input_document_types": input_document_types,
            "all_document_types": all_document_types,
            "flow_chain": flow_chain,
        }

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
                    built_flows = deployment.build_flows(cast(TOptions, options))
                    if not built_flows:
                        raise ValueError(f"{type(deployment).__name__}.build_flows() returned an empty list.")
                    start_step_input_types = type(built_flows[0]).input_document_types
                    typed_docs = await resolve_document_inputs(
                        documents,
                        deployment._all_document_types(built_flows),
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


__all__ = [
    "DeploymentResult",
    "FlowAction",
    "FlowDirective",
    "PipelineDeployment",
]
