"""Pydantic row models and enums for ClickHouse tracking tables."""

from datetime import datetime
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RunStatus(StrEnum):
    """Pipeline run status."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SpanType(StrEnum):
    """Span type classification."""

    TASK = "task"
    FLOW = "flow"
    LLM = "llm"
    TRACE = "trace"


class DocumentEventType(StrEnum):
    """Document lifecycle event types."""

    TASK_INPUT = "task_input"
    TASK_OUTPUT = "task_output"
    FLOW_INPUT = "flow_input"
    FLOW_OUTPUT = "flow_output"
    LLM_CONTEXT = "llm_context"
    LLM_MESSAGE = "llm_message"
    STORE_SAVED = "store_saved"
    STORE_SAVE_FAILED = "store_save_failed"


# --- Table names ---

TABLE_PIPELINE_RUNS = "pipeline_runs"
TABLE_TRACKED_SPANS = "tracked_spans"
TABLE_DOCUMENT_EVENTS = "document_events"
TABLE_SPAN_EVENTS = "span_events"

# --- OTel span attribute names for document lineage ---

ATTR_INPUT_DOCUMENT_SHA256S = "pipeline.input_document_sha256s"
ATTR_OUTPUT_DOCUMENT_SHA256S = "pipeline.output_document_sha256s"


# --- Row models ---


class PipelineRunRow(BaseModel):
    """Row model for pipeline_runs table."""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    project_name: str
    flow_name: str
    run_scope: str = ""
    status: RunStatus
    start_time: datetime
    end_time: datetime | None = None
    total_cost: float = 0.0
    total_tokens: int = 0
    metadata: str = "{}"
    version: int = 1


class TrackedSpanRow(BaseModel):
    """Row model for tracked_spans table."""

    model_config = ConfigDict(frozen=True)

    span_id: str
    trace_id: str
    run_id: UUID
    parent_span_id: str | None = None
    name: str
    span_type: SpanType
    status: str
    start_time: datetime
    end_time: datetime | None = None
    duration_ms: int = 0
    cost: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    llm_model: str | None = None
    user_summary: str | None = None
    user_visible: bool = False
    user_label: str | None = None
    input_document_sha256s: tuple[str, ...] = Field(default_factory=tuple)
    output_document_sha256s: tuple[str, ...] = Field(default_factory=tuple)
    version: int = 1


class DocumentEventRow(BaseModel):
    """Row model for document_events table."""

    model_config = ConfigDict(frozen=True)

    event_id: UUID
    run_id: UUID
    document_sha256: str
    span_id: str
    event_type: DocumentEventType
    timestamp: datetime
    metadata: str = "{}"


class SpanEventRow(BaseModel):
    """Row model for span_events table."""

    model_config = ConfigDict(frozen=True)

    event_id: UUID
    run_id: UUID
    span_id: str
    name: str
    timestamp: datetime
    attributes: str = "{}"
    level: str | None = None


# --- Protocol for circular import resolution ---


class SummaryRowBuilder(Protocol):
    """Protocol satisfied by TrackingService for writer callback."""

    def build_span_summary_update(self, span_id: str, summary: str) -> TrackedSpanRow | None:
        """Build a replacement row with summary filled."""
        ...
