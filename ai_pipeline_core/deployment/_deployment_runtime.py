"""Private helpers for deployment execution and flow validation."""

import asyncio
import contextlib
import time
from collections.abc import Sequence
from datetime import timedelta
from typing import Any, cast
from uuid import UUID

from ai_pipeline_core.database import SpanKind, SpanRecord, SpanStatus
from ai_pipeline_core.database._documents import load_documents_from_database
from ai_pipeline_core.documents import Document
from ai_pipeline_core.logger import get_pipeline_logger
from ai_pipeline_core.pipeline._execution_context import (
    ExecutionContext,
    TaskContext,
    get_sinks,
    record_lifecycle_event,
    set_execution_context,
    set_task_context,
)
from ai_pipeline_core.pipeline._flow import PipelineFlow
from ai_pipeline_core.pipeline._track_span import track_span
from ai_pipeline_core.pipeline.options import FlowOptions

from ._helpers import _cancel_dispatched_handles
from ._types import DocumentRef, FlowCompletedEvent, FlowFailedEvent, FlowStartedEvent, ResultPublisher

__all__ = [
    "_deduplicate_documents_by_sha256",
    "_execute_flow_with_context",
    "_first_declaring_class",
    "_flow_class_path",
    "_reuse_cached_flow_output",
    "_safe_uuid",
    "_validate_flow_chain",
]

logger = get_pipeline_logger(__name__)
_PUBLISH_EXCEPTIONS = (OSError, RuntimeError, TypeError, ValueError)


def _flow_class_path(flow_class: type[PipelineFlow]) -> str:
    """Return the fully qualified flow class path recorded in span metadata."""
    return f"{flow_class.__module__}:{flow_class.__qualname__}"


async def _reuse_cached_flow_output(
    *,
    database: Any,
    cache_ttl: timedelta | None,
    flow_cache_key: str,
    flow_class: type[PipelineFlow],
    flow_name: str,
    step: int,
    total_steps: int,
    accumulated_docs: list[Document],
) -> tuple[SpanRecord, tuple[Document, ...], list[Document]] | None:
    """Reuse cached flow outputs and return the cached span plus hydrated documents."""
    if database is None or cache_ttl is None:
        return None

    cached_span = await database.get_cached_completion(flow_cache_key, max_age=cache_ttl)
    if cached_span is None:
        return None

    logger.info("[%d/%d] Resume: skipping %s (completion record found)", step, total_steps, flow_name)

    previous_output_documents: tuple[Document, ...] = ()
    updated_docs = list(accumulated_docs)
    if cached_span.output_document_shas:
        resumed_docs = await load_documents_from_database(
            database,
            set(cached_span.output_document_shas),
            filter_types=flow_class.output_document_types or None,
        )
        updated_docs = list(_deduplicate_documents_by_sha256([*accumulated_docs, *resumed_docs]))
        previous_output_documents = _deduplicate_documents_by_sha256(resumed_docs)

    return cached_span, previous_output_documents, updated_docs


async def _execute_flow_with_context(
    *,
    flow_instance: PipelineFlow,
    flow_class: type[PipelineFlow],
    flow_name: str,
    current_docs: list[Document],
    options: FlowOptions,
    flow_exec_ctx: ExecutionContext | None,
    current_exec_ctx: ExecutionContext | None,
    active_handles_before: set[object],
    database: Any,
    publisher: ResultPublisher,
    deployment_span_id: UUID,
    run_id: str,
    flow_span_id: UUID,
    flow_cache_key: str,
    flow_options_payload: dict[str, Any],
    expected_tasks: list[str],
    step: int,
    total_steps: int,
    root_id_str: str,
    parent_task_id_str: str | None,
) -> tuple[Document, ...]:
    """Execute one flow under flow/task context and record failure state on exceptions."""
    with contextlib.ExitStack() as flow_scope:
        if flow_exec_ctx is not None:
            flow_scope.enter_context(set_execution_context(flow_exec_ctx))
        flow_scope.enter_context(set_task_context(TaskContext(scope_kind="flow", task_class_name=flow_class.__name__)))
        flow_target = f"instance_method:{flow_class.__module__}:{flow_class.__qualname__}.run"
        flow_receiver = {"mode": "constructor_args", "value": flow_instance.get_params()}
        flow_input = {"documents": tuple(current_docs), "options": options}
        flow_input_preview = {
            "flow_class": flow_class.__name__,
            "flow_options": flow_options_payload,
            "input_documents": [document.name for document in current_docs],
        }
        flow_started_at = time.monotonic()

        input_doc_sha256s = tuple(doc.sha256 for doc in current_docs)
        deployment_span_id_str = str(deployment_span_id)

        async with track_span(
            SpanKind.FLOW,
            flow_name,
            flow_target,
            sinks=get_sinks(),
            span_id=flow_span_id,
            parent_span_id=deployment_span_id,
            encode_receiver=flow_receiver,
            encode_input=flow_input,
            db=database,
            input_preview=flow_input_preview,
        ) as span_ctx:
            if flow_class.BASE_COST_USD > 0:
                span_ctx._add_cost(flow_class.BASE_COST_USD)
            span_ctx.set_meta(
                step=step,
                total_steps=total_steps,
                estimated_minutes=flow_instance.estimated_minutes,
                expected_task_names=expected_tasks,
                cache_hit=False,
                cache_key=flow_cache_key,
            )
            await publisher.publish_flow_started(
                FlowStartedEvent(
                    run_id=run_id,
                    span_id=str(flow_span_id),
                    root_deployment_id=root_id_str,
                    parent_deployment_task_id=parent_task_id_str,
                    flow_name=flow_name,
                    flow_class=flow_class.__name__,
                    step=step,
                    total_steps=total_steps,
                    status=str(SpanStatus.RUNNING),
                    expected_tasks=expected_tasks,
                    flow_params=flow_instance.get_params(),
                    parent_span_id=deployment_span_id_str,
                    input_document_sha256s=input_doc_sha256s,
                )
            )
            record_lifecycle_event(
                "flow.started",
                f"Starting flow {flow_name}",
                flow_name=flow_name,
                flow_class=flow_class.__name__,
                step=step,
                total_steps=total_steps,
            )
            try:
                raw_flow_result = cast(object, await flow_instance.run(tuple(current_docs), options))
                if not isinstance(raw_flow_result, tuple):
                    raise TypeError(
                        f"PipelineFlow '{flow_class.__name__}' returned {type(raw_flow_result).__name__}. "
                        "run() must return tuple[Document, ...]. "
                        "Hint: for single-document returns use (doc,) with trailing comma, "
                        "or wrap a list: return tuple(results)"
                    )
                raw_result_docs = cast(tuple[object, ...], raw_flow_result)
                if any(not isinstance(document, Document) for document in raw_result_docs):
                    raise TypeError(f"PipelineFlow '{flow_class.__name__}' returned non-Document items. run() must return tuple[Document, ...].")
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
                if current_exec_ctx is not None:
                    await _cancel_dispatched_handles(current_exec_ctx.active_task_handles, baseline_handles=active_handles_before)
                try:
                    await publisher.publish_flow_failed(
                        FlowFailedEvent(
                            run_id=run_id,
                            span_id=str(flow_span_id),
                            root_deployment_id=root_id_str,
                            parent_deployment_task_id=parent_task_id_str,
                            flow_name=flow_name,
                            flow_class=flow_class.__name__,
                            step=step,
                            total_steps=total_steps,
                            status=str(SpanStatus.FAILED),
                            error_message=str(flow_exc),
                            parent_span_id=deployment_span_id_str,
                            input_document_sha256s=input_doc_sha256s,
                        )
                    )
                except _PUBLISH_EXCEPTIONS as publish_error:
                    logger.warning("Failed to publish flow.failed event: %s", publish_error)
                raise

            flow_duration_ms = int((time.monotonic() - flow_started_at) * 1000)
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
            output_refs = tuple(
                DocumentRef(
                    sha256=doc.sha256,
                    class_name=type(doc).__name__,
                    name=doc.name,
                    summary=doc.summary,
                    publicly_visible=getattr(type(doc), "publicly_visible", False),
                    derived_from=tuple(doc.derived_from),
                    triggered_by=tuple(doc.triggered_by),
                )
                for doc in validated_docs
            )
            await publisher.publish_flow_completed(
                FlowCompletedEvent(
                    run_id=run_id,
                    span_id=str(flow_span_id),
                    root_deployment_id=root_id_str,
                    parent_deployment_task_id=parent_task_id_str,
                    flow_name=flow_name,
                    flow_class=flow_class.__name__,
                    step=step,
                    total_steps=total_steps,
                    status=str(SpanStatus.COMPLETED),
                    duration_ms=flow_duration_ms,
                    output_documents=output_refs,
                    parent_span_id=deployment_span_id_str,
                    input_document_sha256s=input_doc_sha256s,
                )
            )
            span_ctx.set_output_preview({"documents": [document.name for document in validated_docs]})
            span_ctx._set_output_value(validated_docs)
            logger.info("[%d/%d] Completed: %s", step, total_steps, flow_name)
            return validated_docs


def _safe_uuid(value: str) -> UUID | None:
    """Parse a UUID string, returning None if invalid."""
    try:
        return UUID(value)
    except ValueError, AttributeError:
        return None


def _deduplicate_documents_by_sha256(documents: Sequence[Document]) -> tuple[Document, ...]:
    """Deduplicate documents by SHA256 while preserving first-seen order."""
    deduped: dict[str, Document] = {}
    for document in documents:
        deduped.setdefault(document.sha256, document)
    return tuple(deduped.values())


def _validate_flow_chain(deployment_name: str, flows: Sequence[PipelineFlow]) -> None:
    """Validate that each flow's input types are satisfiable by preceding flows' outputs."""
    type_pool: set[type[Document]] = set()

    for index, flow_instance in enumerate(flows):
        flow_cls = type(flow_instance)
        input_types = flow_cls.input_document_types
        output_types = flow_cls.output_document_types
        flow_name = flow_instance.name

        if index == 0:
            type_pool.update(input_types)
        elif input_types:
            any_satisfied = any(any(issubclass(available, required) for available in type_pool) for required in input_types)
            if not any_satisfied:
                input_names = sorted(document_type.__name__ for document_type in input_types)
                pool_names = sorted(document_type.__name__ for document_type in type_pool) if type_pool else ["(empty)"]
                raise TypeError(
                    f"{deployment_name}: flow '{flow_name}' (step {index + 1}) requires input types "
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
