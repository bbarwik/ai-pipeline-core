"""Core classes for pipeline deployments.

Provides the PipelineDeployment base class and related types for
creating unified, type-safe pipeline deployments with:
- Per-flow resume (skip completed flows via execution DAG)
- Per-flow uploads (immediate, not just at end)
- Upload on failure (partial results saved)
"""

import asyncio
import contextlib
import time
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Generic, TypeVar, cast, final
from uuid import UUID, uuid4

from prefect import get_client, runtime
from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness
from pydantic import BaseModel, ConfigDict

from ai_pipeline_core.database import NULL_PARENT, Database, ExecutionNode, NodeKind, NodeStatus, create_database_from_settings
from ai_pipeline_core.database._documents import load_documents_from_database
from ai_pipeline_core.documents import Document, RunContext
from ai_pipeline_core.documents._context import TaskContext, _reset_run_context, _set_run_context, reset_task_context, set_task_context
from ai_pipeline_core.logging import ExecutionLogBuffer, get_pipeline_logger
from ai_pipeline_core.pipeline._execution_context import (
    ExecutionContext,
    FlowFrame,
    get_execution_context,
    record_lifecycle_event,
    reset_execution_context,
    set_execution_context,
)
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

from ._helpers import (
    _HANDLE_CANCEL_GRACE_SECONDS,
    _MILLISECONDS_PER_SECOND,
    _classify_error,
    _compute_run_scope,
    _consume_log_summary,
    _ensure_execution_log_handler_installed,
    _execution_log_flush_loop,
    _heartbeat_loop,
    _record_terminal_flow_node,
    class_name_to_deployment_name,
    extract_generic_params,
    validate_run_id,
)
from ._types import (
    DocumentRef,
    FlowCompletedEvent,
    FlowFailedEvent,
    FlowStartedEvent,
    ResultPublisher,
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    _NoopPublisher,
)

logger = get_pipeline_logger(__name__)


def _safe_uuid(value: str) -> UUID | None:
    """Parse a UUID string, returning None if invalid."""
    try:
        return UUID(value)
    except (ValueError, AttributeError):
        return None


async def _cancel_dispatched_handles(
    active_handles: set[object],
    *,
    baseline_handles: set[object],
) -> None:
    """Cancel handles dispatched within a flow and wait briefly for shutdown."""
    new_handles: list[TaskHandle[tuple[Document[Any], ...]]] = [
        handle for handle in list(active_handles) if handle not in baseline_handles and isinstance(handle, TaskHandle)
    ]
    if not new_handles:
        return

    for handle in new_handles:
        handle.cancel()

    pending_tasks: list[asyncio.Task[tuple[Document[Any], ...]]] = [handle._task for handle in new_handles if not handle.done]
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


def _deduplicate_documents_by_sha256(documents: Sequence[Document]) -> tuple[Document, ...]:
    """Deduplicate documents by SHA256 while preserving first-seen order."""
    deduped: dict[str, Document] = {}
    for document in documents:
        deduped.setdefault(document.sha256, document)
    return tuple(deduped.values())


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


def _first_declaring_class(cls: type, attribute_name: str) -> type | None:
    """Return the first class in the MRO that declares ``attribute_name``."""
    for base in cls.__mro__:
        if attribute_name in base.__dict__:
            return base
    return None


class PipelineDeployment(Generic[TOptions, TResult]):
    """Base class for pipeline deployments with three execution modes.

    - ``run_cli()``: Database-backed (ClickHouse or filesystem)
    - ``run_local()``: In-memory database (ephemeral)
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

        if _first_declaring_class(cls, "build_flows") is PipelineDeployment:
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
        output_documents: tuple[Document, ...],
    ) -> FlowDirective:
        """Optionally skip future instances of a flow class."""
        _ = (flow_class, plan, output_documents)
        return FlowDirective()

    @staticmethod
    @abstractmethod
    def build_result(run_id: str, documents: tuple[Document, ...], options: TOptions) -> TResult:
        """Extract typed result from pipeline documents.

        Called for both full runs and partial runs (--start/--end). For partial runs,
        build_partial_result() delegates here by default — override build_partial_result()
        to customize partial run results.
        """
        ...

    def build_partial_result(self, run_id: str, documents: tuple[Document, ...], options: TOptions) -> TResult:
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

    @staticmethod
    async def _insert_db_node(database: Database | None, node: ExecutionNode) -> None:
        """Insert an execution node, logging warnings on failure."""
        if database is None:
            return
        try:
            await database.insert_node(node)
        except Exception as exc:
            logger.warning("Database node insert failed for %s %s: %s", node.node_kind, node.node_id, exc)

    @staticmethod
    async def _update_db_node(database: Database | None, node_id: UUID, **updates: Any) -> None:
        """Update an execution node, logging warnings on failure."""
        if database is None:
            return
        try:
            await database.update_node(node_id, **updates)
        except Exception as exc:
            logger.warning("Database node update failed for %s: %s", node_id, exc)

    @staticmethod
    async def _shutdown_db(database: Database | None) -> None:
        """Flush and shut down database, logging warnings on failure."""
        if database is None:
            return
        try:
            await database.flush()
        except Exception as exc:
            logger.warning("Database flush failed: %s", exc)
        try:
            await database.shutdown()
        except Exception as exc:
            logger.warning("Database shutdown failed: %s", exc)

    @final
    async def _run_with_context(
        self,
        run_id: str,
        documents: Sequence[Document],
        options: TOptions,
        *,
        deployment_node_id: UUID,
        root_deployment_id: UUID,
        parent_deployment_task_id: UUID | None = None,
        publisher: ResultPublisher | None = None,
        start_step: int = 1,
        end_step: int | None = None,
        parent_execution_id: UUID | None = None,
        database: Database | None = None,
    ) -> TResult:
        """Internal entry point with pre-allocated DAG-linking parameters.

        Called by public run() (standalone), remote deployment (Prefect), and inline mode.
        """
        return await self._run_core(
            run_id=run_id,
            documents=documents,
            options=options,
            publisher=publisher,
            start_step=start_step,
            end_step=end_step,
            parent_execution_id=parent_execution_id,
            deployment_node_id=deployment_node_id,
            root_deployment_id=root_deployment_id,
            parent_deployment_task_id=parent_deployment_task_id,
            database=database,
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
        parent_execution_id: UUID | None = None,
        database: Database | None = None,
    ) -> TResult:
        """Execute flows with resume, per-flow uploads, and step control.

        run_id must match ``[a-zA-Z0-9_-]+``, max 100 chars.
        """
        deployment_node_id = uuid4()
        root_deployment_id = deployment_node_id
        return await self._run_with_context(
            run_id,
            documents,
            options,
            deployment_node_id=deployment_node_id,
            root_deployment_id=root_deployment_id,
            parent_deployment_task_id=None,
            publisher=publisher,
            start_step=start_step,
            end_step=end_step,
            parent_execution_id=parent_execution_id,
            database=database,
        )

    async def _run_core(
        self,
        run_id: str,
        documents: Sequence[Document],
        options: TOptions,
        *,
        deployment_node_id: UUID,
        root_deployment_id: UUID,
        parent_deployment_task_id: UUID | None = None,
        publisher: ResultPublisher | None = None,
        start_step: int = 1,
        end_step: int | None = None,
        parent_execution_id: UUID | None = None,
        database: Database | None = None,
    ) -> TResult:
        """Core deployment execution with database node tracking."""
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
        parent_execution_id_value = str(parent_execution_id) if parent_execution_id is not None else ""
        deployment_payload = {
            "flow_plan": flow_plan,
            "options": options.model_dump(),
            "parent_execution_id": parent_execution_id_value,
        }

        # Common event fields for this deployment
        node_id_str = str(deployment_node_id)
        root_id_str = str(root_deployment_id)
        parent_task_id_str = str(parent_deployment_task_id) if parent_deployment_task_id else None

        # Create database backend if not provided externally
        owns_database = database is None
        if owns_database:
            try:
                database = create_database_from_settings(settings)
            except Exception as exc:
                logger.warning("Database creation failed, continuing without execution DAG tracking: %s", exc)
                database = None

        log_buffer: ExecutionLogBuffer | None = None
        flush_event: asyncio.Event | None = None
        log_flush_task: asyncio.Task[None] | None = None
        if database is not None:
            _ensure_execution_log_handler_installed()
            flush_event = asyncio.Event()
            event_loop = asyncio.get_running_loop()

            def _request_log_flush() -> None:
                event_loop.call_soon_threadsafe(flush_event.set)

            log_buffer = ExecutionLogBuffer(
                request_flush=_request_log_flush,
            )

        # Insert deployment node into execution DAG
        deployment_node = ExecutionNode(
            node_id=deployment_node_id,
            node_kind=NodeKind.DEPLOYMENT,
            deployment_id=deployment_node_id,
            root_deployment_id=root_deployment_id,
            parent_node_id=NULL_PARENT,
            parent_deployment_task_id=parent_deployment_task_id,
            run_id=run_id,
            run_scope=run_scope,
            deployment_name=self.name,
            name=self.name,
            sequence_no=0,
            status=NodeStatus.RUNNING,
            payload=deployment_payload,
        )
        await self._insert_db_node(database, deployment_node)

        # Set concurrency limits and run context for the entire pipeline run
        run_execution_id = uuid4()
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
                publisher=publisher,
                limits=self.concurrency_limits,
                limits_status=limits_status,
                database=database,
                deployment_id=deployment_node_id,
                root_deployment_id=root_deployment_id,
                parent_deployment_task_id=parent_deployment_task_id,
                deployment_name=self.name,
                current_node_id=deployment_node_id,
                log_buffer=log_buffer,
            )
        )
        try:
            if flush_event is not None:
                log_flush_task = asyncio.create_task(_execution_log_flush_loop(database, log_buffer, flush_event))

            record_lifecycle_event(
                "deployment.started",
                "Starting deployment",
                deployment_name=self.name,
                total_steps=total_steps,
                start_step=start_step,
                end_step=end_step,
            )
            await publisher.publish_run_started(
                RunStartedEvent(
                    run_id=run_id,
                    node_id=node_id_str,
                    root_deployment_id=root_id_str,
                    parent_deployment_task_id=parent_task_id_str,
                    run_scope=str(run_scope),
                    flow_plan=flow_plan,
                )
            )

            # Start heartbeat background task
            heartbeat_task = asyncio.create_task(_heartbeat_loop(publisher, run_id))

            await _ensure_concurrency_limits(self.concurrency_limits)

            # Persist input documents to database
            if database is not None and input_docs:
                from ai_pipeline_core.pipeline._task import _persist_documents_to_database

                await _persist_documents_to_database(
                    input_docs,
                    database,
                    deployment_node_id,
                    run_scope,
                    producing_node_id=None,
                )

            # Precompute flow minutes for progress calculation
            flow_minutes = tuple(flow_instance.estimated_minutes for flow_instance in flows)

            # In-memory document accumulator: all documents available for subsequent flows
            accumulated_docs: list[Document] = list(_deduplicate_documents_by_sha256(input_docs))
            skipped_classes: set[type[PipelineFlow]] = set()
            last_flow_output_sha256s: tuple[str, ...] = ()
            previous_output_documents: tuple[Document, ...] = ()

            for i in range(start_step - 1, end_step):
                step = i + 1
                flow_instance = flows[i]
                flow_class = type(flow_instance)
                flow_name = flow_instance.name
                # Re-read flow_run_id in case Prefect subflow changes it
                flow_run_id = str(runtime.flow_run.get_id() or "") if runtime.flow_run else flow_run_id  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]

                if flow_class in skipped_classes:
                    await _record_terminal_flow_node(
                        insert_node=self._insert_db_node,
                        update_node=self._update_db_node,
                        database=database,
                        publisher=publisher,
                        deployment_name=self.name,
                        deployment_node_id=deployment_node_id,
                        root_deployment_id=root_deployment_id,
                        run_id=run_id,
                        run_scope=run_scope,
                        flow_name=flow_name,
                        flow_class_name=flow_class.__name__,
                        step=step,
                        total_steps=total_steps,
                        root_id_str=root_id_str,
                        parent_task_id_str=parent_task_id_str,
                        status=NodeStatus.SKIPPED,
                        publish_reason="skipped",
                        log_buffer=log_buffer,
                        lifecycle_event="flow.skipped",
                        lifecycle_message=f"Skipped flow {flow_name}",
                        lifecycle_fields={
                            "flow_name": flow_name,
                            "flow_class": flow_class.__name__,
                            "step": step,
                            "total_steps": total_steps,
                            "reason": "skipped by plan_next_flow",
                        },
                        payload={"skip_reason": "skipped by plan_next_flow"},
                    )
                    previous_output_documents = ()
                    continue

                directive = self.plan_next_flow(flow_class, flows, previous_output_documents)
                if directive.action is FlowAction.SKIP:
                    skipped_classes.add(flow_class)
                    skip_reason = directive.reason or "skipped"
                    await _record_terminal_flow_node(
                        insert_node=self._insert_db_node,
                        update_node=self._update_db_node,
                        database=database,
                        publisher=publisher,
                        deployment_name=self.name,
                        deployment_node_id=deployment_node_id,
                        root_deployment_id=root_deployment_id,
                        run_id=run_id,
                        run_scope=run_scope,
                        flow_name=flow_name,
                        flow_class_name=flow_class.__name__,
                        step=step,
                        total_steps=total_steps,
                        root_id_str=root_id_str,
                        parent_task_id_str=parent_task_id_str,
                        status=NodeStatus.SKIPPED,
                        publish_reason=skip_reason,
                        log_buffer=log_buffer,
                        lifecycle_event="flow.skipped",
                        lifecycle_message=f"Skipped flow {flow_name}",
                        lifecycle_fields={
                            "flow_name": flow_name,
                            "flow_class": flow_class.__name__,
                            "step": step,
                            "total_steps": total_steps,
                            "reason": skip_reason,
                        },
                        payload={"skip_reason": skip_reason},
                    )
                    previous_output_documents = ()
                    continue

                # Resume check: look for a completed flow node in the execution DAG
                if database is not None and self.cache_ttl is not None:
                    cache_key = f"flow:{run_scope}:{flow_name}:{step}"
                    cached_node = await database.get_cached_completion(cache_key, max_age=self.cache_ttl)
                    if cached_node is not None:
                        logger.info("[%d/%d] Resume: skipping %s (completion record found)", step, total_steps, flow_name)
                        # Reconstruct documents from cached output SHA256s
                        if cached_node.output_document_shas:
                            resumed_docs = await load_documents_from_database(
                                database,
                                set(cached_node.output_document_shas),
                                filter_types=flow_class.output_document_types or None,
                            )
                            accumulated_docs = list(_deduplicate_documents_by_sha256([*accumulated_docs, *resumed_docs]))
                            previous_output_documents = _deduplicate_documents_by_sha256(resumed_docs)
                        else:
                            previous_output_documents = ()
                        last_flow_output_sha256s = cached_node.output_document_shas
                        await _record_terminal_flow_node(
                            insert_node=self._insert_db_node,
                            update_node=self._update_db_node,
                            database=database,
                            publisher=publisher,
                            deployment_name=self.name,
                            deployment_node_id=deployment_node_id,
                            root_deployment_id=root_deployment_id,
                            run_id=run_id,
                            run_scope=run_scope,
                            flow_name=flow_name,
                            flow_class_name=flow_class.__name__,
                            step=step,
                            total_steps=total_steps,
                            root_id_str=root_id_str,
                            parent_task_id_str=parent_task_id_str,
                            status=NodeStatus.CACHED,
                            publish_reason="completed",
                            log_buffer=log_buffer,
                            lifecycle_event="flow.cached",
                            lifecycle_message=f"Reused cached flow output for {flow_name}",
                            lifecycle_fields={
                                "flow_name": flow_name,
                                "flow_class": flow_class.__name__,
                                "step": step,
                                "total_steps": total_steps,
                                "cache_key": cache_key,
                            },
                            output_document_shas=cached_node.output_document_shas,
                        )
                        continue

                # Insert flow node into execution DAG (RUNNING)
                flow_node_id = uuid4()
                flow_function_path = f"{flow_class.__module__}:{flow_class.__qualname__}"
                flow_params = flow_instance.get_params()
                expected_tasks = flow_class.expected_tasks()
                flow_options_payload = options.model_dump(exclude_defaults=True)
                await self._insert_db_node(
                    database,
                    ExecutionNode(
                        node_id=flow_node_id,
                        node_kind=NodeKind.FLOW,
                        deployment_id=deployment_node_id,
                        root_deployment_id=root_deployment_id,
                        parent_node_id=deployment_node_id,
                        run_id=run_id,
                        run_scope=run_scope,
                        deployment_name=self.name,
                        name=flow_name,
                        sequence_no=step,
                        flow_class=flow_class.__name__,
                        status=NodeStatus.RUNNING,
                        payload={
                            "function_path": flow_function_path,
                            "flow_params": flow_params,
                            "expected_tasks": expected_tasks,
                            "flow_options": flow_options_payload,
                        },
                    ),
                )

                await publisher.publish_flow_started(
                    FlowStartedEvent(
                        run_id=run_id,
                        node_id=str(flow_node_id),
                        root_deployment_id=root_id_str,
                        parent_deployment_task_id=parent_task_id_str,
                        flow_name=flow_name,
                        flow_class=flow_class.__name__,
                        step=step,
                        total_steps=total_steps,
                        expected_tasks=expected_tasks,
                        flow_params=flow_params,
                    )
                )
                logger.info("[%d/%d] Starting: %s", step, total_steps, flow_name)

                # Select input documents for this flow from accumulated docs
                input_types = flow_class.input_document_types
                if input_types:
                    input_type_set = set(input_types)
                    current_docs = [d for d in accumulated_docs if type(d) in input_type_set or any(issubclass(type(d), t) for t in input_type_set)]
                else:
                    current_docs = list(accumulated_docs)

                # Set up flow execution context
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
                    flow_params=flow_params,
                )
                current_exec_ctx = get_execution_context()
                flow_exec_ctx = (
                    replace(current_exec_ctx, flow_frame=flow_frame, task_frame=None, current_node_id=flow_node_id, flow_node_id=flow_node_id)
                    if current_exec_ctx is not None
                    else None
                )
                flow_exec_token = set_execution_context(flow_exec_ctx) if flow_exec_ctx is not None else None
                flow_task_token = set_task_context(TaskContext(scope_kind="flow", task_class_name=flow_class.__name__))
                active_handles_before: set[object] = set(current_exec_ctx.active_task_handles) if current_exec_ctx is not None else set()
                record_lifecycle_event(
                    "flow.started",
                    f"Starting flow {flow_name}",
                    flow_name=flow_name,
                    flow_class=flow_class.__name__,
                    step=step,
                    total_steps=total_steps,
                )
                flow_payload = {
                    "function_path": flow_function_path,
                    "flow_params": flow_params,
                    "expected_tasks": expected_tasks,
                    "flow_options": flow_options_payload,
                    "replay_payload": {
                        "version": 1,
                        "payload_type": "pipeline_flow",
                        "function_path": flow_function_path,
                        "run_id": run_id,
                        "documents": [{"$doc_ref": d.sha256, "class_name": type(d).__name__, "name": d.name} for d in current_docs],
                        "flow_options": flow_options_payload,
                        "flow_params": flow_params,
                        "original": {},
                    },
                }
                await self._update_db_node(
                    database,
                    flow_node_id,
                    payload=flow_payload,
                )

                try:
                    raw_flow_result = cast(object, await flow_instance.run(tuple(current_docs), options))
                    if not isinstance(raw_flow_result, tuple):
                        raise TypeError(
                            f"PipelineFlow '{flow_class.__name__}' returned {type(raw_flow_result).__name__}. "
                            f"run() must return tuple[Document, ...]. "
                            f"Hint: for single-document returns use (doc,) with trailing comma, "
                            f"or wrap a list: return tuple(results)"
                        )
                    raw_result_docs = cast(tuple[object, ...], raw_flow_result)
                    if any(not isinstance(document, Document) for document in raw_result_docs):
                        raise TypeError(f"PipelineFlow '{flow_class.__name__}' returned non-Document items in tuple. run() must return tuple[Document, ...].")
                    validated_docs = cast(tuple[Document, ...], raw_flow_result)
                except (Exception, asyncio.CancelledError) as flow_exc:
                    record_lifecycle_event(
                        "flow.failed",
                        f"Flow {flow_name} failed",
                        flow_name=flow_name,
                        flow_class=flow_class.__name__,
                        step=step,
                        total_steps=total_steps,
                        error_type=type(flow_exc).__name__,
                        error_message=str(flow_exc),
                    )
                    # Update flow node to FAILED
                    await self._update_db_node(
                        database,
                        flow_node_id,
                        status=NodeStatus.FAILED,
                        ended_at=datetime.now(UTC),
                        error_type=type(flow_exc).__name__,
                        error_message=str(flow_exc),
                        payload={**flow_payload, "log_summary": _consume_log_summary(log_buffer, flow_node_id)},
                    )
                    if current_exec_ctx is not None:
                        await _cancel_dispatched_handles(current_exec_ctx.active_task_handles, baseline_handles=active_handles_before)
                    try:
                        await publisher.publish_flow_failed(
                            FlowFailedEvent(
                                run_id=run_id,
                                node_id=str(flow_node_id),
                                root_deployment_id=root_id_str,
                                parent_deployment_task_id=parent_task_id_str,
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
                    leaked: list[TaskHandle[tuple[Document[Any], ...]]] = [
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

                # Accumulate output documents and record completion
                output_sha256s = tuple(d.sha256 for d in validated_docs)
                last_flow_output_sha256s = output_sha256s
                previous_output_documents = tuple(validated_docs)
                accumulated_docs = list(_deduplicate_documents_by_sha256([*accumulated_docs, *validated_docs]))

                # Update flow node to COMPLETED with cache_key for resume
                flow_duration_ms = int((time.monotonic() - flow_started_at) * _MILLISECONDS_PER_SECOND)
                cache_key = f"flow:{run_scope}:{flow_name}:{step}"
                input_shas = tuple(d.sha256 for d in current_docs)
                flow_completion_token = set_execution_context(flow_exec_ctx) if flow_exec_ctx is not None else None
                try:
                    record_lifecycle_event(
                        "flow.completed",
                        f"Completed flow {flow_name}",
                        flow_name=flow_name,
                        flow_class=flow_class.__name__,
                        step=step,
                        total_steps=total_steps,
                        duration_ms=flow_duration_ms,
                        output_count=len(validated_docs),
                    )
                    await self._update_db_node(
                        database,
                        flow_node_id,
                        status=NodeStatus.COMPLETED,
                        ended_at=datetime.now(UTC),
                        output_document_shas=output_sha256s,
                        input_document_shas=input_shas,
                        cache_key=cache_key,
                        payload={**flow_payload, "log_summary": _consume_log_summary(log_buffer, flow_node_id)},
                    )
                finally:
                    if flow_completion_token is not None:
                        reset_execution_context(flow_completion_token)

                output_refs = tuple(
                    DocumentRef(
                        sha256=doc.sha256,
                        class_name=type(doc).__name__,
                        name=doc.name,
                        summary="",
                        publicly_visible=getattr(type(doc), "publicly_visible", False),
                        derived_from=tuple(doc.derived_from),
                        triggered_by=tuple(doc.triggered_by),
                    )
                    for doc in validated_docs
                )
                await publisher.publish_flow_completed(
                    FlowCompletedEvent(
                        run_id=run_id,
                        node_id=str(flow_node_id),
                        root_deployment_id=root_id_str,
                        parent_deployment_task_id=parent_task_id_str,
                        flow_name=flow_name,
                        flow_class=flow_class.__name__,
                        step=step,
                        total_steps=total_steps,
                        duration_ms=flow_duration_ms,
                        output_documents=output_refs,
                    )
                )
                logger.info("[%d/%d] Completed: %s", step, total_steps, flow_name)

            # Build result from accumulated documents
            all_docs = _deduplicate_documents_by_sha256(accumulated_docs)

            is_partial_run = end_step < total_steps
            if is_partial_run:
                logger.info("Partial run (steps %d-%d of %d) — skipping build_result", start_step, end_step, total_steps)
                result = self.build_partial_result(run_id, tuple(all_docs), options)
            else:
                result = self.build_result(run_id, tuple(all_docs), options)

            # Update deployment node to COMPLETED
            record_lifecycle_event(
                "deployment.completed",
                "Completed deployment",
                deployment_name=self.name,
                total_steps=total_steps,
                output_count=len(all_docs),
                partial_run=is_partial_run,
            )
            await self._update_db_node(
                database,
                deployment_node_id,
                status=NodeStatus.COMPLETED,
                ended_at=datetime.now(UTC),
                output_document_shas=last_flow_output_sha256s,
                payload={
                    **deployment_payload,
                    "result": result.model_dump(),
                    "log_summary": _consume_log_summary(log_buffer, deployment_node_id),
                },
            )

            await publisher.publish_run_completed(
                RunCompletedEvent(
                    run_id=run_id,
                    node_id=node_id_str,
                    root_deployment_id=root_id_str,
                    parent_deployment_task_id=parent_task_id_str,
                    result=result.model_dump(),
                    output_document_sha256s=last_flow_output_sha256s,
                )
            )

            return result

        except (Exception, asyncio.CancelledError) as exc:
            record_lifecycle_event(
                "deployment.failed",
                "Deployment failed",
                deployment_name=self.name,
                total_steps=total_steps,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            # Update deployment node to FAILED
            await self._update_db_node(
                database,
                deployment_node_id,
                status=NodeStatus.FAILED,
                ended_at=datetime.now(UTC),
                error_type=type(exc).__name__,
                error_message=str(exc),
                payload={
                    **deployment_payload,
                    "error_code": _classify_error(exc),
                    "log_summary": _consume_log_summary(log_buffer, deployment_node_id),
                },
            )
            current_exec_ctx = get_execution_context()
            if current_exec_ctx is not None:
                await _cancel_dispatched_handles(current_exec_ctx.active_task_handles, baseline_handles=set())
            if not failed_published:
                failed_published = True
                try:
                    await publisher.publish_run_failed(
                        RunFailedEvent(
                            run_id=run_id,
                            node_id=node_id_str,
                            root_deployment_id=root_id_str,
                            parent_deployment_task_id=parent_task_id_str,
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
            if log_flush_task is not None:
                log_flush_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await log_flush_task
            reset_execution_context(execution_token)
            _reset_run_context(run_token)
            _reset_limits_state(limits_token)
            # Shut down database if we own it
            if owns_database:
                await self._shutdown_db(database)

    @final
    def run_local(
        self,
        run_id: str,
        documents: Sequence[Document],
        options: TOptions,
        publisher: ResultPublisher | None = None,
        output_dir: Path | None = None,
    ) -> TResult:
        """Run locally with Prefect test harness and in-memory database.

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

        with prefect_test_harness(), disable_run_logger():
            result = asyncio.run(self.run(run_id, documents, options, publisher=publisher))

        if output_dir:
            (output_dir / "result.json").write_text(result.model_dump_json(indent=2))

        return result

    @final
    def run_cli(
        self,
        initializer: Callable[[TOptions], tuple[str, tuple[Document, ...]]] | None = None,
        cli_mixin: type | None = None,
    ) -> None:
        """Execute pipeline from CLI with positional working_directory and --start/--end flags."""
        from ._cli import run_cli_for_deployment

        run_cli_for_deployment(self, initializer, cli_mixin)

    def _build_integration_meta(self) -> dict[str, Any]:
        """Build deploy-time schema metadata for the Prefect wrapper."""
        from ._prefect import build_integration_meta

        return build_integration_meta(self)

    @final
    def as_prefect_flow(self) -> Callable[..., Any]:
        """Generate a Prefect flow for production deployment via ``ai-pipeline-deploy`` CLI."""
        from ._prefect import build_prefect_flow

        return build_prefect_flow(self)


__all__ = [
    "DeploymentResult",
    "FlowAction",
    "FlowDirective",
    "PipelineDeployment",
]
