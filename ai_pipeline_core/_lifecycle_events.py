"""Shared lifecycle event types used across pipeline and deployment modules."""

from dataclasses import dataclass, field

__all__ = [
    "DocumentRef",
    "TaskCompletedEvent",
    "TaskFailedEvent",
    "TaskStartedEvent",
]


@dataclass(frozen=True, slots=True)
class DocumentRef:
    """Lightweight document reference for event payloads."""

    sha256: str
    class_name: str
    name: str
    summary: str = ""
    publicly_visible: bool = False
    derived_from: tuple[str, ...] = ()
    triggered_by: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TaskStartedEvent:
    """Task execution started."""

    run_id: str
    span_id: str
    root_deployment_id: str
    parent_deployment_task_id: str | None
    flow_name: str
    step: int
    total_steps: int
    status: str
    task_name: str
    task_class: str
    parent_span_id: str = ""
    input_document_sha256s: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TaskCompletedEvent:
    """Task execution completed."""

    run_id: str
    span_id: str
    root_deployment_id: str
    parent_deployment_task_id: str | None
    flow_name: str
    step: int
    total_steps: int
    status: str
    task_name: str
    task_class: str
    duration_ms: int
    output_documents: tuple[DocumentRef, ...] = field(default_factory=tuple)
    parent_span_id: str = ""
    input_document_sha256s: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TaskFailedEvent:
    """Task execution failed."""

    run_id: str
    span_id: str
    root_deployment_id: str
    parent_deployment_task_id: str | None
    flow_name: str
    step: int
    total_steps: int
    status: str
    task_name: str
    task_class: str
    error_message: str
    parent_span_id: str = ""
    input_document_sha256s: tuple[str, ...] = ()
