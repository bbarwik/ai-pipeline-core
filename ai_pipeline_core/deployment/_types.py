"""Event types and publisher protocols for pipeline lifecycle publishing."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from ai_pipeline_core._lifecycle_events import DocumentRef, TaskCompletedEvent, TaskFailedEvent, TaskStartedEvent


# Enum
class EventType(StrEnum):
    """Pipeline lifecycle event types for Pub/Sub publishing."""

    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_HEARTBEAT = "run.heartbeat"
    FLOW_STARTED = "flow.started"
    FLOW_COMPLETED = "flow.completed"
    FLOW_FAILED = "flow.failed"
    FLOW_SKIPPED = "flow.skipped"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"


# Enum
class ErrorCode(StrEnum):
    """Classifies pipeline failure reason for run.failed events."""

    BUDGET_EXCEEDED = "budget_exceeded"
    DURATION_EXCEEDED = "duration_exceeded"
    PROVIDER_ERROR = "provider_error"
    PIPELINE_ERROR = "pipeline_error"
    INVALID_INPUT = "invalid_input"
    CRASHED = "crashed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RunStartedEvent:
    """Pipeline execution started."""

    run_id: str
    span_id: str
    root_deployment_id: str
    parent_deployment_task_id: str | None
    input_fingerprint: str
    status: str
    deployment_name: str = ""
    deployment_class: str = ""
    flow_plan: list[dict[str, Any]] = field(default_factory=list)
    parent_span_id: str = ""
    input_document_sha256s: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RunCompletedEvent:
    """Pipeline completed successfully.

    output_document_sha256s contains the SHA256 hashes of the LAST flow's
    output documents — these are the pipeline's final deliverables. Intermediate
    documents from earlier flows are available via the database (ai-trace show).
    """

    run_id: str
    span_id: str
    root_deployment_id: str
    parent_deployment_task_id: str | None
    status: str
    result: dict[str, Any]
    deployment_name: str = ""
    deployment_class: str = ""
    duration_ms: int = 0
    output_document_sha256s: tuple[str, ...] = ()
    parent_span_id: str = ""


@dataclass(frozen=True, slots=True)
class RunFailedEvent:
    """Pipeline execution failed."""

    run_id: str
    span_id: str
    root_deployment_id: str
    parent_deployment_task_id: str | None
    status: str
    error_code: ErrorCode
    error_message: str
    deployment_name: str = ""
    deployment_class: str = ""
    duration_ms: int = 0
    parent_span_id: str = ""


@dataclass(frozen=True, slots=True)
class FlowStartedEvent:
    """Flow execution started."""

    run_id: str
    span_id: str
    root_deployment_id: str
    parent_deployment_task_id: str | None
    flow_name: str
    flow_class: str
    step: int
    total_steps: int
    status: str
    expected_tasks: list[dict[str, Any]] = field(default_factory=list)
    flow_params: dict[str, Any] = field(default_factory=dict)
    parent_span_id: str = ""
    input_document_sha256s: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FlowCompletedEvent:
    """Flow execution completed."""

    run_id: str
    span_id: str
    root_deployment_id: str
    parent_deployment_task_id: str | None
    flow_name: str
    flow_class: str
    step: int
    total_steps: int
    status: str
    duration_ms: int
    output_documents: tuple[DocumentRef, ...] = field(default_factory=tuple)
    parent_span_id: str = ""
    input_document_sha256s: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FlowFailedEvent:
    """Flow execution failed."""

    run_id: str
    span_id: str
    root_deployment_id: str
    parent_deployment_task_id: str | None
    flow_name: str
    flow_class: str
    step: int
    total_steps: int
    status: str
    error_message: str
    parent_span_id: str = ""
    input_document_sha256s: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FlowSkippedEvent:
    """Flow skipped because it was resumed or intentionally bypassed."""

    run_id: str
    span_id: str
    root_deployment_id: str
    parent_deployment_task_id: str | None
    flow_name: str
    flow_class: str
    step: int
    total_steps: int
    status: str
    reason: str
    parent_span_id: str = ""
    input_document_sha256s: tuple[str, ...] = ()


# Protocol
@runtime_checkable
class ResultPublisher(Protocol):
    """Publishes pipeline lifecycle events to external consumers."""

    async def publish_run_started(self, event: RunStartedEvent) -> None:
        """Publish a pipeline start event."""
        ...

    async def publish_run_completed(self, event: RunCompletedEvent) -> None:
        """Publish a pipeline completion event."""
        ...

    async def publish_run_failed(self, event: RunFailedEvent) -> None:
        """Publish a pipeline failure event."""
        ...

    async def publish_heartbeat(self, run_id: str, *, root_deployment_id: str = "", span_id: str = "") -> None:
        """Publish a heartbeat signal."""
        ...

    async def publish_flow_started(self, event: FlowStartedEvent) -> None:
        """Publish a flow start event."""
        ...

    async def publish_flow_completed(self, event: FlowCompletedEvent) -> None:
        """Publish a flow completion event."""
        ...

    async def publish_flow_failed(self, event: FlowFailedEvent) -> None:
        """Publish a flow failure event."""
        ...

    async def publish_flow_skipped(self, event: FlowSkippedEvent) -> None:
        """Publish a flow skipped event."""
        ...

    async def publish_task_started(self, event: TaskStartedEvent) -> None:
        """Publish a task start event."""
        ...

    async def publish_task_completed(self, event: TaskCompletedEvent) -> None:
        """Publish a task completion event."""
        ...

    async def publish_task_failed(self, event: TaskFailedEvent) -> None:
        """Publish a task failure event."""
        ...

    async def close(self) -> None:
        """Release resources held by the publisher."""
        ...


class _NoopPublisher:
    """Discards all lifecycle events. Default publisher for CLI and run_local."""

    async def publish_run_started(self, event: RunStartedEvent) -> None:
        """Accept and discard a run started event."""

    async def publish_run_completed(self, event: RunCompletedEvent) -> None:
        """Accept and discard a run completed event."""

    async def publish_run_failed(self, event: RunFailedEvent) -> None:
        """Accept and discard a run failed event."""

    async def publish_heartbeat(self, run_id: str, *, root_deployment_id: str = "", span_id: str = "") -> None:
        """Accept and discard a heartbeat."""

    async def publish_flow_started(self, event: FlowStartedEvent) -> None:
        """Accept and discard a flow started event."""

    async def publish_flow_completed(self, event: FlowCompletedEvent) -> None:
        """Accept and discard a flow completed event."""

    async def publish_flow_failed(self, event: FlowFailedEvent) -> None:
        """Accept and discard a flow failed event."""

    async def publish_flow_skipped(self, event: FlowSkippedEvent) -> None:
        """Accept and discard a flow skipped event."""

    async def publish_task_started(self, event: TaskStartedEvent) -> None:
        """Accept and discard a task started event."""

    async def publish_task_completed(self, event: TaskCompletedEvent) -> None:
        """Accept and discard a task completed event."""

    async def publish_task_failed(self, event: TaskFailedEvent) -> None:
        """Accept and discard a task failed event."""

    async def close(self) -> None:
        """No resources to release."""


class _MemoryPublisher:
    """Records all lifecycle events in-memory for test assertions."""

    def __init__(self) -> None:
        self.events: list[
            RunStartedEvent
            | RunCompletedEvent
            | RunFailedEvent
            | FlowStartedEvent
            | FlowCompletedEvent
            | FlowFailedEvent
            | FlowSkippedEvent
            | TaskStartedEvent
            | TaskCompletedEvent
            | TaskFailedEvent
        ] = []
        self.heartbeats: list[dict[str, str]] = []

    async def publish_run_started(self, event: RunStartedEvent) -> None:
        """Record a run started event."""
        self.events.append(event)

    async def publish_run_completed(self, event: RunCompletedEvent) -> None:
        """Record a run completed event."""
        self.events.append(event)

    async def publish_run_failed(self, event: RunFailedEvent) -> None:
        """Record a run failed event."""
        self.events.append(event)

    async def publish_heartbeat(self, run_id: str, *, root_deployment_id: str = "", span_id: str = "") -> None:
        """Record a heartbeat."""
        self.heartbeats.append({"run_id": run_id, "root_deployment_id": root_deployment_id, "span_id": span_id})

    async def publish_flow_started(self, event: FlowStartedEvent) -> None:
        """Record a flow started event."""
        self.events.append(event)

    async def publish_flow_completed(self, event: FlowCompletedEvent) -> None:
        """Record a flow completed event."""
        self.events.append(event)

    async def publish_flow_failed(self, event: FlowFailedEvent) -> None:
        """Record a flow failed event."""
        self.events.append(event)

    async def publish_flow_skipped(self, event: FlowSkippedEvent) -> None:
        """Record a flow skipped event."""
        self.events.append(event)

    async def publish_task_started(self, event: TaskStartedEvent) -> None:
        """Record a task started event."""
        self.events.append(event)

    async def publish_task_completed(self, event: TaskCompletedEvent) -> None:
        """Record a task completed event."""
        self.events.append(event)

    async def publish_task_failed(self, event: TaskFailedEvent) -> None:
        """Record a task failed event."""
        self.events.append(event)

    async def close(self) -> None:
        """No resources to release."""


__all__ = [
    "DocumentRef",
    "ErrorCode",
    "EventType",
    "FlowCompletedEvent",
    "FlowFailedEvent",
    "FlowSkippedEvent",
    "FlowStartedEvent",
    "ResultPublisher",
    "RunCompletedEvent",
    "RunFailedEvent",
    "RunStartedEvent",
    "TaskCompletedEvent",
    "TaskFailedEvent",
    "TaskStartedEvent",
    "_MemoryPublisher",
    "_NoopPublisher",
]
