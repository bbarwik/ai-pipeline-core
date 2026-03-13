"""Span sink protocol and database-backed sink implementation."""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from ai_pipeline_core.database import SpanKind, SpanRecord, SpanStatus
from ai_pipeline_core.database._json_helpers import json_dumps
from ai_pipeline_core.database._protocol import DatabaseWriter
from ai_pipeline_core.logger import get_pipeline_logger
from ai_pipeline_core.pipeline._execution_context import get_execution_context
from ai_pipeline_core.pipeline._span_types import SpanContext, SpanMetrics, SpanSink
from ai_pipeline_core.pipeline._task_runtime import _next_span_version

__all__ = [
    "DatabaseSpanSink",
    "SpanContext",
    "SpanMetrics",
    "SpanSink",
]

logger = get_pipeline_logger(__name__)
EMPTY_LOG_SUMMARY = {
    "total": 0,
    "warnings": 0,
    "errors": 0,
    "last_error": "",
}


@dataclass(frozen=True, slots=True)
class _StartedSpanState:
    parent_span_id: UUID | None
    kind: SpanKind
    name: str
    target: str
    started_at: datetime
    started_version: int
    sequence_no: int
    receiver_json: str
    input_json: str
    input_document_shas: tuple[str, ...]
    input_blob_shas: tuple[str, ...]
    previous_conversation_id: UUID | None


class DatabaseSpanSink:
    """Persist tracked span lifecycle updates into the active database backend."""

    def __init__(self, database: DatabaseWriter) -> None:
        self._database = database
        self._started: dict[UUID, _StartedSpanState] = {}

    async def on_span_started(
        self,
        *,
        span_id: UUID,
        parent_span_id: UUID | None,
        kind: SpanKind,
        name: str,
        target: str,
        started_at: datetime,
        receiver_json: str,
        input_json: str,
        input_document_shas: frozenset[str],
        input_blob_shas: frozenset[str],
        input_preview: Any | None,
    ) -> None:
        _ = input_preview
        execution_ctx = get_execution_context()
        if execution_ctx is None:
            return

        deployment_id = execution_ctx.deployment_id or UUID(int=0)
        root_deployment_id = execution_ctx.root_deployment_id or execution_ctx.deployment_id or UUID(int=0)
        sequence_no = execution_ctx.next_child_sequence(parent_span_id) if parent_span_id is not None else 0
        started_version = _next_span_version()
        previous_conversation_id = _extract_previous_conversation_id(kind, receiver_json)

        self._started[span_id] = _StartedSpanState(
            parent_span_id=parent_span_id,
            kind=kind,
            name=name,
            target=target,
            started_at=started_at,
            started_version=started_version,
            sequence_no=sequence_no,
            receiver_json=receiver_json,
            input_json=input_json,
            input_document_shas=tuple(sorted(input_document_shas)),
            input_blob_shas=tuple(sorted(input_blob_shas)),
            previous_conversation_id=previous_conversation_id,
        )

        await self._insert_span_safe(
            SpanRecord(
                span_id=span_id,
                parent_span_id=parent_span_id,
                deployment_id=deployment_id,
                root_deployment_id=root_deployment_id,
                run_id=execution_ctx.run_id,
                deployment_name=execution_ctx.deployment_name,
                kind=kind,
                name=name,
                description="",
                status=SpanStatus.RUNNING,
                sequence_no=sequence_no,
                started_at=started_at,
                version=started_version,
                target=target,
                receiver_json=receiver_json,
                input_json=input_json,
                input_document_shas=tuple(sorted(input_document_shas)),
                input_blob_shas=tuple(sorted(input_blob_shas)),
                previous_conversation_id=previous_conversation_id,
            )
        )

    async def on_span_finished(
        self,
        *,
        span_id: UUID,
        ended_at: datetime,
        output_json: str,
        error_json: str,
        output_document_shas: frozenset[str],
        output_blob_shas: frozenset[str],
        output_preview: Any | None,
        error: BaseException | None,
        metrics: SpanMetrics,
        meta: dict[str, Any],
    ) -> None:
        _ = output_preview
        execution_ctx = get_execution_context()
        if execution_ctx is None:
            return

        started_state = self._started.pop(span_id, None)
        if started_state is None:
            started_state = _StartedSpanState(
                parent_span_id=execution_ctx.parent_span_id,
                kind=SpanKind.OPERATION,
                name="",
                target="",
                started_at=ended_at,
                started_version=_next_span_version(),
                sequence_no=0,
                receiver_json="",
                input_json="",
                input_document_shas=(),
                input_blob_shas=(),
                previous_conversation_id=None,
            )

        deployment_id = execution_ctx.deployment_id or UUID(int=0)
        root_deployment_id = execution_ctx.root_deployment_id or execution_ctx.deployment_id or UUID(int=0)
        cost_usd = metrics.cost_usd or 0.0
        description = ""
        raw_description = meta.pop("description", "")
        raw_cache_key = meta.pop("cache_key", "")
        raw_status = meta.get("_span_status")
        if isinstance(raw_description, str):
            description = raw_description
        cache_key = raw_cache_key if isinstance(raw_cache_key, str) else ""
        meta_json = json_dumps(_strip_control_meta(meta))
        metrics_json = json_dumps(_metrics_to_json(metrics))
        status = _resolve_finished_status(error=error, raw_status=raw_status)

        await self._insert_span_safe(
            SpanRecord(
                span_id=span_id,
                parent_span_id=started_state.parent_span_id,
                deployment_id=deployment_id,
                root_deployment_id=root_deployment_id,
                run_id=execution_ctx.run_id,
                deployment_name=execution_ctx.deployment_name,
                kind=started_state.kind,
                name=started_state.name,
                description=description,
                status=status,
                sequence_no=started_state.sequence_no,
                started_at=started_state.started_at,
                ended_at=ended_at,
                version=_next_span_version(started_state.started_version),
                cost_usd=cost_usd,
                cache_key=cache_key,
                error_type=type(error).__name__ if error is not None else "",
                error_message=str(error) if error is not None else "",
                input_document_shas=started_state.input_document_shas,
                output_document_shas=tuple(sorted(output_document_shas)),
                target=started_state.target,
                receiver_json=started_state.receiver_json,
                input_json=started_state.input_json,
                output_json=output_json,
                error_json=error_json,
                meta_json=meta_json,
                metrics_json=metrics_json,
                input_blob_shas=started_state.input_blob_shas,
                output_blob_shas=tuple(sorted(output_blob_shas)),
                previous_conversation_id=started_state.previous_conversation_id,
            )
        )

    async def _insert_span_safe(self, span: SpanRecord) -> None:
        try:
            await self._database.insert_span(span)
        except BaseException as exc:
            if _must_reraise_sink_error(exc):
                raise
            execution_ctx = get_execution_context()
            if execution_ctx is not None:
                execution_ctx.recording_degraded = True
            if span.parent_span_id is None and execution_ctx is not None and execution_ctx.replay_root_span_id is None:
                execution_ctx.replay_root_span_id = span.span_id
            logger.warning(
                "Database span insert failed for %s %s: %s. Span recording is degraded for this execution.",
                span.kind,
                span.span_id,
                exc,
            )
            return
        execution_ctx = get_execution_context()
        if execution_ctx is not None and span.parent_span_id is None and execution_ctx.replay_root_span_id is None:
            execution_ctx.replay_root_span_id = span.span_id


def _must_reraise_sink_error(error: BaseException) -> bool:
    return isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit))


def _metrics_to_json(metrics: SpanMetrics) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "time_taken_ms": metrics.time_taken_ms,
        "log_summary": metrics.log_summary,
    }
    optional_fields = (
        ("tokens_input", metrics.tokens_input),
        ("tokens_output", metrics.tokens_output),
        ("tokens_cache_read", metrics.tokens_cache_read),
        ("tokens_reasoning", metrics.tokens_reasoning),
        ("cost_usd", metrics.cost_usd),
        ("first_token_ms", metrics.first_token_ms),
    )
    for field_name, value in optional_fields:
        if value is not None:
            payload.update({field_name: value})
    return payload


def _extract_previous_conversation_id(kind: SpanKind, receiver_json: str) -> UUID | None:
    if kind != SpanKind.CONVERSATION or not receiver_json:
        return None
    receiver = _parse_receiver_json(receiver_json)
    if receiver is None:
        return None
    conversation_id = _extract_conversation_id(receiver)
    if conversation_id is None:
        return None
    try:
        return UUID(conversation_id)
    except ValueError:
        return None


def _parse_receiver_json(receiver_json: str) -> dict[str, Any] | None:
    try:
        receiver = json.loads(receiver_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(receiver, dict) or receiver.get("mode") != "decoded_state":
        return None
    return receiver


def _extract_conversation_id(receiver: dict[str, Any]) -> str | None:
    value = receiver.get("value")
    if not isinstance(value, dict):
        return None
    data = value.get("data")
    if not isinstance(data, dict):
        return None
    conversation_id = data.get("_conversation_id")
    if not isinstance(conversation_id, str) or not conversation_id:
        return None
    return conversation_id


def _resolve_finished_status(*, error: BaseException | None, raw_status: Any) -> SpanStatus:
    if error is not None:
        return SpanStatus.FAILED
    if isinstance(raw_status, SpanStatus):
        return raw_status
    if isinstance(raw_status, str) and raw_status:
        return SpanStatus(raw_status)
    return SpanStatus.COMPLETED


def _strip_control_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in meta.items() if not key.startswith("_")}
