"""Span transport types shared across pipeline execution modules."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from ai_pipeline_core.database import SpanKind

__all__ = [
    "SpanContext",
    "SpanMetrics",
    "SpanSink",
]


@dataclass(frozen=True, slots=True)
class SpanMetrics:
    """Normalized metrics payload for span sinks."""

    time_taken_ms: int
    log_summary: dict[str, Any]
    tokens_input: int | None = None
    tokens_output: int | None = None
    tokens_cache_read: int | None = None
    tokens_reasoning: int | None = None
    cost_usd: float | None = None
    first_token_ms: int | None = None


class SpanSink(Protocol):
    """Lifecycle callbacks for execution span transport implementations."""

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
    ) -> None: ...

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
    ) -> None: ...


class SpanContext:
    """Mutable state populated by a tracked span body."""

    def __init__(self, *, span_id: UUID, parent_span_id: UUID | None, input_preview: Any | None = None) -> None:
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self._input_preview = input_preview
        self._output_preview: Any | None = None
        self._meta: dict[str, Any] = {}
        self._metrics_updates: dict[str, Any] = {}
        self._output_value: Any = None
        self._has_output_value = False
        self._status: str | None = None

    @property
    def input_preview(self) -> Any | None:
        return self._input_preview

    @property
    def output_preview(self) -> Any | None:
        return self._output_preview

    def set_input_preview(self, value: Any) -> None:
        self._input_preview = value

    def set_output_preview(self, value: Any) -> None:
        self._output_preview = value

    def set_meta(self, **values: Any) -> None:
        self._meta.update(values)

    def set_metrics(self, **values: Any) -> None:
        valid_fields = {field_name for field_name in SpanMetrics.__dataclass_fields__ if field_name != "log_summary"}
        unknown_fields = sorted(set(values) - valid_fields)
        if unknown_fields:
            names = ", ".join(unknown_fields)
            raise ValueError(f"Unknown span metric field(s): {names}. Use only fields declared on SpanMetrics.")
        self._metrics_updates.update(values)

    def _set_output_value(self, value: Any) -> None:
        self._output_value = value
        self._has_output_value = True

    def set_status(self, status: str) -> None:
        self._status = status

    def _build_metrics(self, *, ended_at: datetime, started_at: datetime, log_summary: dict[str, Any]) -> SpanMetrics:
        metrics_data = dict(self._metrics_updates)
        if "time_taken_ms" not in metrics_data:
            elapsed_ms = int((ended_at - started_at).total_seconds() * 1000)
            metrics_data["time_taken_ms"] = max(elapsed_ms, 0)
        return SpanMetrics(
            time_taken_ms=int(metrics_data["time_taken_ms"]),
            log_summary=log_summary,
            tokens_input=_optional_int(metrics_data.get("tokens_input")),
            tokens_output=_optional_int(metrics_data.get("tokens_output")),
            tokens_cache_read=_optional_int(metrics_data.get("tokens_cache_read")),
            tokens_reasoning=_optional_int(metrics_data.get("tokens_reasoning")),
            cost_usd=_optional_float(metrics_data.get("cost_usd")),
            first_token_ms=_optional_int(metrics_data.get("first_token_ms")),
        )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
