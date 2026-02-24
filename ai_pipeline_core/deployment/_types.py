"""Event types and protocols for pipeline lifecycle publishing."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from ai_pipeline_core.deployment.contract import FlowStatus


# Enum
class EventType(StrEnum):
    """Pipeline lifecycle event types for Pub/Sub publishing."""

    STARTED = "task.started"
    PROGRESS = "task.progress"
    HEARTBEAT = "task.heartbeat"
    COMPLETED = "task.completed"
    FAILED = "task.failed"


# Enum
class ErrorCode(StrEnum):
    """Classifies pipeline failure reason for task.failed events."""

    BUDGET_EXCEEDED = "budget_exceeded"
    DURATION_EXCEEDED = "duration_exceeded"
    PROVIDER_ERROR = "provider_error"
    PIPELINE_ERROR = "pipeline_error"
    INVALID_INPUT = "invalid_input"
    CRASHED = "crashed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class StartedEvent:
    """Pipeline execution started."""

    run_id: str
    flow_run_id: str
    run_scope: str


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """Flow-level or intra-flow progress."""

    run_id: str
    flow_run_id: str
    flow_name: str
    step: int
    total_steps: int
    progress: float
    step_progress: float
    status: FlowStatus
    message: str


@dataclass(frozen=True, slots=True)
class CompletedEvent:
    """Pipeline completed successfully."""

    run_id: str
    flow_run_id: str
    result: dict[str, Any]
    chain_context: dict[str, Any]
    actual_cost: float


@dataclass(frozen=True, slots=True)
class FailedEvent:
    """Pipeline execution failed."""

    run_id: str
    flow_run_id: str
    error_code: ErrorCode
    error_message: str


# Protocol
@runtime_checkable
class ResultPublisher(Protocol):
    """Publishes pipeline lifecycle events to external consumers."""

    async def publish_started(self, event: StartedEvent) -> None:
        """Publish a pipeline started event."""
        ...

    async def publish_progress(self, event: ProgressEvent) -> None:
        """Publish a flow progress event."""
        ...

    async def publish_heartbeat(self, run_id: str) -> None:
        """Publish a heartbeat signal."""
        ...

    async def publish_completed(self, event: CompletedEvent) -> None:
        """Publish a pipeline completed event."""
        ...

    async def publish_failed(self, event: FailedEvent) -> None:
        """Publish a pipeline failed event."""
        ...

    async def close(self) -> None:
        """Release resources held by the publisher."""
        ...


@dataclass(frozen=True, slots=True)
class TaskResultRecord:
    """Row from task_results ClickHouse table."""

    run_id: str
    result: str
    chain_context: str
    stored_at: datetime


# Protocol
@runtime_checkable
class TaskResultStore(Protocol):
    """Durability backup for completion results."""

    async def write_result(self, run_id: str, result: str, chain_context: str) -> None:
        """Write a completion result for durable backup."""
        ...

    async def read_result(self, run_id: str) -> TaskResultRecord | None:
        """Read a completion result by run_id."""
        ...

    def shutdown(self) -> None:
        """Release resources held by the store."""
        ...


__all__ = [
    "CompletedEvent",
    "ErrorCode",
    "EventType",
    "FailedEvent",
    "ProgressEvent",
    "ResultPublisher",
    "StartedEvent",
    "TaskResultRecord",
    "TaskResultStore",
]
