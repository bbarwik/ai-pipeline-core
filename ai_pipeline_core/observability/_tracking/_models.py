"""Pydantic row models and enums for ClickHouse tracking tables."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class RunStatus(StrEnum):
    """Pipeline run status."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# --- Table names ---

TABLE_PIPELINE_RUNS = "pipeline_runs"
TABLE_PIPELINE_SPANS = "pipeline_spans"


# --- Row models ---


class PipelineRunRow(BaseModel):
    """Row model for pipeline_runs table."""

    model_config = ConfigDict(frozen=True)

    execution_id: UUID
    run_id: str
    flow_name: str
    run_scope: str = ""
    status: RunStatus
    start_time: datetime
    end_time: datetime | None = None
    total_cost: float = 0.0
    total_tokens: int = 0
    metadata_json: str = "{}"
    version: int = 1
    parent_execution_id: UUID | None = None
    parent_span_id: str | None = None


class PipelineSpanRow(BaseModel):
    """Row model for pipeline_spans table. Written once on span end."""

    model_config = ConfigDict(frozen=True)

    # Identity
    execution_id: UUID
    span_id: str
    trace_id: str
    parent_span_id: str | None = None

    # Denormalized run context
    run_id: str = ""
    flow_name: str = ""
    run_scope: str = ""

    # Classification
    name: str
    span_type: str  # task | flow | llm | trace
    status: str  # completed | failed

    # Timing
    start_time: datetime
    end_time: datetime
    duration_ms: int = 0
    span_order: int = 0

    # LLM metrics
    cost: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cached: int = 0
    llm_model: str | None = None

    # Error
    error_message: str = ""

    # Content (ZSTD-compressed in ClickHouse)
    input_json: str = ""
    output_json: str = ""
    replay_payload: str = ""
    attributes_json: str = "{}"
    events_json: str = "[]"

    # Document lineage
    input_doc_sha256s: tuple[str, ...] = ()
    output_doc_sha256s: tuple[str, ...] = ()
