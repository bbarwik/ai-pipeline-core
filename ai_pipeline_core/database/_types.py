"""Data models for the unified execution DAG and document storage."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from ai_pipeline_core.documents._context import DocumentSha256, RunScope

__all__ = [
    "NULL_PARENT",
    "BlobRecord",
    "DocumentRecord",
    "ExecutionLog",
    "ExecutionNode",
    "NodeKind",
    "NodeStatus",
    "RunScopeInfo",
]

NULL_PARENT = UUID(int=0)


class NodeKind(StrEnum):
    """Discriminator for execution DAG node types."""

    DEPLOYMENT = "deployment"
    FLOW = "flow"
    TASK = "task"
    CONVERSATION = "conversation"
    CONVERSATION_TURN = "conversation_turn"


class NodeStatus(StrEnum):
    """Lifecycle status of an execution node."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"
    SKIPPED = "skipped"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _empty_run_scope() -> RunScope:
    return RunScope("")


@dataclass(frozen=True, slots=True)
class ExecutionNode:
    """A single node in the execution DAG.

    Every entity in a pipeline run (deployment, flow, task, conversation,
    conversation_turn) is represented as one row in execution_nodes.
    """

    # Identity
    node_id: UUID
    node_kind: NodeKind

    # Hierarchy
    deployment_id: UUID
    root_deployment_id: UUID

    # Run context (denormalized on every row)
    run_id: str
    run_scope: RunScope
    deployment_name: str

    # Ordering
    name: str
    sequence_no: int

    # Parent (NULL_PARENT sentinel for root nodes)
    parent_node_id: UUID = field(default=NULL_PARENT)
    attempt: int = 0

    # Class names (for Pub/Sub event reconstruction)
    flow_class: str = ""
    task_class: str = ""

    # Lifecycle
    status: NodeStatus = NodeStatus.RUNNING
    started_at: datetime = field(default_factory=_utcnow)
    ended_at: datetime | None = None
    updated_at: datetime = field(default_factory=_utcnow)
    version: int = 1

    # LLM metrics (populated on conversation_turn nodes)
    model: str = ""
    cost_usd: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cache_read: int = 0
    tokens_cache_write: int = 0
    tokens_reasoning: int = 0
    turn_count: int = 0

    # Error
    error_type: str = ""
    error_message: str = ""

    # Cross-deployment linking
    remote_child_deployment_id: UUID | None = None
    parent_deployment_task_id: UUID | None = None

    # Cache/resume
    cache_key: str = ""
    input_fingerprint: str = ""

    # Document references (SHA256 arrays)
    input_document_shas: tuple[str, ...] = ()
    output_document_shas: tuple[str, ...] = ()
    context_document_shas: tuple[str, ...] = ()

    # Denormalized IDs for zero-JOIN filtering
    flow_id: UUID | None = None
    task_id: UUID | None = None
    conversation_id: UUID | None = None

    # Payload (node-kind-specific JSON data, ZSTD compressed in ClickHouse)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """Document metadata registry row. Content-addressed by document_sha256."""

    document_sha256: DocumentSha256
    content_sha256: str
    deployment_id: UUID
    producing_node_id: UUID | None
    document_type: str
    name: str
    run_scope: RunScope = field(default_factory=_empty_run_scope)

    description: str = ""
    mime_type: str = ""
    size_bytes: int = 0
    publicly_visible: bool = False

    derived_from: tuple[str, ...] = ()
    triggered_by: tuple[str, ...] = ()

    # Parallel arrays for attachment metadata
    attachment_names: tuple[str, ...] = ()
    attachment_descriptions: tuple[str, ...] = ()
    attachment_sha256s: tuple[str, ...] = ()
    attachment_mime_types: tuple[str, ...] = ()
    attachment_sizes: tuple[int, ...] = ()

    summary: str = ""
    metadata_json: str = "{}"
    created_at: datetime = field(default_factory=_utcnow)
    version: int = 1

    def __post_init__(self) -> None:
        lengths = [
            len(self.attachment_names),
            len(self.attachment_descriptions),
            len(self.attachment_sha256s),
            len(self.attachment_mime_types),
            len(self.attachment_sizes),
        ]
        if lengths and len(set(lengths)) > 1:
            names = ("attachment_names", "attachment_descriptions", "attachment_sha256s", "attachment_mime_types", "attachment_sizes")
            detail = ", ".join(f"{n}={length}" for n, length in zip(names, lengths, strict=True))
            msg = f"Attachment parallel arrays must have equal lengths: {detail}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class BlobRecord:
    """Content-addressed binary storage row."""

    content_sha256: str
    content: bytes
    size_bytes: int
    created_at: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True, slots=True)
class ExecutionLog:
    """Append-only execution log row linked to the execution DAG."""

    node_id: UUID
    deployment_id: UUID
    root_deployment_id: UUID
    flow_id: UUID | None
    task_id: UUID | None
    timestamp: datetime
    sequence_no: int
    level: str
    category: str
    logger_name: str
    message: str
    event_type: str = ""
    fields: str = "{}"
    exception_text: str = ""


@dataclass(frozen=True, slots=True)
class RunScopeInfo:
    """Aggregated metadata for a non-empty document run scope."""

    run_scope: RunScope
    document_count: int
    latest_created_at: datetime
