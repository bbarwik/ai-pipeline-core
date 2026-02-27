"""Immutable span data model shared by all backends.

SpanData is the single representation of a completed span, consumed by both
ClickHouseBackend and FilesystemBackend. Constructed either from an OTel
ReadableSpan (live) or from a ClickHouse row dict (download).
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Self
from uuid import UUID

from opentelemetry.trace import StatusCode

# OTel resource attribute keys set by PipelineDeployment.run()
RES_EXECUTION_ID = "pipeline.execution_id"
RES_RUN_ID = "pipeline.run_id"
RES_FLOW_NAME = "pipeline.flow_name"
RES_RUN_SCOPE = "pipeline.run_scope"

# OTel span attribute keys for document lineage
ATTR_INPUT_DOC_SHA256S = "pipeline.input_document_sha256s"
ATTR_OUTPUT_DOC_SHA256S = "pipeline.output_document_sha256s"

# Attributes extracted into dedicated SpanData fields, excluded from attributes_json
_CONTENT_ATTRS: frozenset[str] = frozenset({
    "lmnr.span.input",
    "lmnr.span.output",
    "replay.payload",
    RES_EXECUTION_ID,
    RES_RUN_ID,
    RES_FLOW_NAME,
    RES_RUN_SCOPE,
    ATTR_INPUT_DOC_SHA256S,
    ATTR_OUTPUT_DOC_SHA256S,
})


@dataclass(frozen=True, slots=True)
class SpanData:
    """All data for a completed span. Immutable, source-agnostic."""

    # Identity
    execution_id: UUID | None
    span_id: str
    trace_id: str
    parent_span_id: str | None
    name: str

    # Run context (from OTel resource attributes)
    run_id: str
    flow_name: str
    run_scope: str

    # Classification
    span_type: str  # task | flow | llm | trace
    status: str  # completed | failed

    # Timing
    start_time: datetime
    end_time: datetime
    duration_ms: int
    span_order: int

    # LLM metrics
    cost: float
    tokens_input: int
    tokens_output: int
    tokens_cached: int
    llm_model: str | None

    # Error
    error_message: str

    # Content
    input_json: str
    output_json: str
    replay_payload: str
    attributes: dict[str, Any]
    events: tuple[dict[str, Any], ...]

    # Document lineage
    input_doc_sha256s: tuple[str, ...]
    output_doc_sha256s: tuple[str, ...]

    @classmethod
    def from_otel_span(cls, span: Any, *, span_order: int) -> Self:
        """Extract all fields from an OTel ReadableSpan + resource attributes."""
        ctx = span.get_span_context()
        parent = span.parent
        attrs: dict[str, Any] = dict(span.attributes or {})
        res_attrs = dict(span.resource.attributes or {}) if hasattr(span, "resource") and span.resource else {}

        identity = _extract_identity(ctx, parent, span.name, attrs, res_attrs)
        timing = _extract_timing(span)
        llm = _extract_llm_metrics(attrs)
        content = _extract_content(attrs, span.status.description or "")

        return cls(
            **identity,
            span_type=_classify_span_type(attrs),
            status="failed" if span.status.status_code == StatusCode.ERROR else "completed",
            **timing,
            span_order=span_order,
            **llm,
            error_message=span.status.description or "",
            **content,
            events=tuple(
                {
                    "name": str(e.name),
                    "timestamp": int(e.timestamp) if e.timestamp else 0,
                    "attributes": {str(k): str(v) for k, v in (dict(e.attributes) if e.attributes else {}).items()},
                }
                for e in list(span.events or [])
            ),
            input_doc_sha256s=tuple(attrs.get(ATTR_INPUT_DOC_SHA256S) or ()),
            output_doc_sha256s=tuple(attrs.get(ATTR_OUTPUT_DOC_SHA256S) or ()),
        )

    @classmethod
    def from_clickhouse_row(cls, row: dict[str, Any]) -> Self:
        """Reconstruct from a pipeline_spans row dict."""
        return cls(
            execution_id=row.get("execution_id"),
            span_id=row["span_id"],
            trace_id=row["trace_id"],
            parent_span_id=row.get("parent_span_id"),
            name=row["name"],
            run_id=row.get("run_id", ""),
            flow_name=row.get("flow_name", ""),
            run_scope=row.get("run_scope", ""),
            span_type=row.get("span_type", "trace"),
            status=row.get("status", "completed"),
            start_time=row["start_time"],
            end_time=row["end_time"],
            duration_ms=row.get("duration_ms", 0),
            span_order=row.get("span_order", 0),
            cost=row.get("cost", 0.0),
            tokens_input=row.get("tokens_input", 0),
            tokens_output=row.get("tokens_output", 0),
            tokens_cached=row.get("tokens_cached", 0),
            llm_model=row.get("llm_model"),
            error_message=row.get("error_message", ""),
            input_json=row.get("input_json", ""),
            output_json=row.get("output_json", ""),
            replay_payload=row.get("replay_payload", ""),
            attributes=json.loads(row.get("attributes_json", "{}")),
            events=tuple(json.loads(row.get("events_json", "[]"))),
            input_doc_sha256s=tuple(row.get("input_doc_sha256s", ())),
            output_doc_sha256s=tuple(row.get("output_doc_sha256s", ())),
        )


def _extract_identity(ctx: Any, parent: Any, name: str, attrs: dict[str, Any], res_attrs: dict[str, Any]) -> dict[str, Any]:
    """Extract identity and run context fields from OTel span."""
    raw_exec_id = attrs.get(RES_EXECUTION_ID) or res_attrs.get(RES_EXECUTION_ID)
    return {
        "execution_id": UUID(str(raw_exec_id)) if raw_exec_id else None,
        "span_id": format(ctx.span_id, "016x"),
        "trace_id": format(ctx.trace_id, "032x"),
        "parent_span_id": format(parent.span_id, "016x") if parent and parent.span_id else None,
        "name": name,
        "run_id": str(attrs.get(RES_RUN_ID) or res_attrs.get(RES_RUN_ID, "")),
        "flow_name": str(attrs.get(RES_FLOW_NAME) or res_attrs.get(RES_FLOW_NAME, "")),
        "run_scope": str(attrs.get(RES_RUN_SCOPE) or res_attrs.get(RES_RUN_SCOPE, "")),
    }


def _extract_timing(span: Any) -> dict[str, Any]:
    """Extract timing fields from OTel span."""
    start_ns = span.start_time or 0
    end_ns = span.end_time or 0
    return {
        "start_time": datetime.fromtimestamp(start_ns / 1e9, tz=UTC),
        "end_time": datetime.fromtimestamp(end_ns / 1e9, tz=UTC),
        "duration_ms": max(0, (end_ns - start_ns) // 1_000_000),
    }


def _extract_llm_metrics(attrs: dict[str, Any]) -> dict[str, Any]:
    """Extract LLM cost/token metrics from span attributes."""
    model_str = str(attrs.get("gen_ai.request.model") or attrs.get("gen_ai.request_model") or "")
    return {
        "cost": float(attrs.get("gen_ai.usage.cost", 0.0)),
        "tokens_input": int(attrs.get("gen_ai.usage.input_tokens", 0)),
        "tokens_output": int(attrs.get("gen_ai.usage.output_tokens", 0)),
        "tokens_cached": int(attrs.get("gen_ai.usage.cache_read_input_tokens", 0)),
        "llm_model": model_str or None,
    }


def _extract_content(attrs: dict[str, Any], error_message: str) -> dict[str, Any]:
    """Extract content fields and filtered attributes from span attributes."""
    remaining_attrs = {k: v for k, v in attrs.items() if k not in _CONTENT_ATTRS}
    if error_message:
        remaining_attrs["status_description"] = error_message
    return {
        "input_json": _attr_to_json(attrs.get("lmnr.span.input")),
        "output_json": _attr_to_json(attrs.get("lmnr.span.output")),
        "replay_payload": _attr_to_json(attrs.get("replay.payload")),
        "attributes": remaining_attrs,
    }


def _classify_span_type(attrs: dict[str, Any]) -> str:
    """Classify span type from OTel attributes."""
    span_type_str = str(attrs.get("lmnr.span.type", ""))
    if span_type_str == "LLM":
        return "llm"
    if attrs.get("prefect.flow.name"):
        return "flow"
    if attrs.get("prefect.task.name"):
        return "task"
    return "trace"


def _attr_to_json(value: Any) -> str:
    """Convert an attribute value to a JSON string, or empty string if None."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)
