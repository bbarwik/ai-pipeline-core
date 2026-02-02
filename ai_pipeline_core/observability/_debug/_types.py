"""Shared data types for the debug tracing system.

Extracted to break the circular dependency between _writer.py and _summary.py:
_writer needs summary generation functions, _summary needs SpanInfo/TraceState.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class WriteJob:
    """Job for background writer thread."""

    trace_id: str
    span_id: str
    name: str
    parent_id: str | None
    attributes: dict[str, Any]
    events: list[Any]
    status_code: str  # "OK" | "ERROR" | "UNSET"
    status_description: str | None
    start_time_ns: int
    end_time_ns: int


@dataclass
class SpanInfo:
    """Information about a span for index building.

    Tracks execution details including timing, LLM metrics (tokens, cost, expected_cost, purpose),
    and Prefect context for observability and cost tracking across the trace hierarchy.
    """

    span_id: str
    parent_id: str | None
    name: str
    span_type: str
    status: str
    start_time: datetime
    path: Path  # Actual directory path for this span
    depth: int = 0  # Nesting depth (0 for root)
    order: int = 0  # Global execution order within trace
    end_time: datetime | None = None
    duration_ms: int = 0
    children: list[str] = field(default_factory=list)
    llm_info: dict[str, Any] | None = None
    prefect_info: dict[str, Any] | None = None
    description: str | None = None
    expected_cost: float | None = None


@dataclass
class TraceState:
    """State for an active trace.

    Maintains trace metadata and span hierarchy with accumulated cost
    metrics (total_cost, total_expected_cost) for monitoring resource
    usage and budget tracking during trace execution.
    """

    trace_id: str
    name: str
    path: Path
    start_time: datetime
    spans: dict[str, SpanInfo] = field(default_factory=dict)
    root_span_id: str | None = None
    total_tokens: int = 0
    total_cost: float = 0.0
    total_expected_cost: float = 0.0
    llm_call_count: int = 0
    span_counter: int = 0  # Global counter for ordering span directories
    merged_wrapper_ids: set[str] = field(default_factory=set)  # IDs of merged wrappers
