"""ClickHouse backend for the unified database.

Implements both DatabaseWriter and DatabaseReader protocols.
Uses a single-thread executor for all blocking ClickHouse operations,
ensuring circuit breaker state needs no locking.
"""

import asyncio
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID

from clickhouse_connect.driver.exceptions import DatabaseError as ClickHouseDatabaseError

from ai_pipeline_core.database._clickhouse_updates import (
    build_node_update_command,
    command_written_rows,
    next_node_version,
    update_node_optimistically,
)
from ai_pipeline_core.database._connection import ClickHouseCircuitBreaker
from ai_pipeline_core.database._ddl import (
    DDL_DOCUMENT_BLOBS,
    DDL_DOCUMENTS,
    DDL_EXECUTION_LOGS,
    DDL_EXECUTION_NODES,
    DOCUMENT_BLOBS_TABLE,
    DOCUMENTS_TABLE,
    EXECUTION_LOGS_TABLE,
    EXECUTION_NODES_TABLE,
)
from ai_pipeline_core.database._document_ancestry import collect_document_ancestry
from ai_pipeline_core.database._types import BlobRecord, DocumentRecord, ExecutionLog, ExecutionNode, NodeKind, NodeStatus, RunScopeInfo
from ai_pipeline_core.documents._context import DocumentSha256, RunScope
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.settings import Settings

logger = get_pipeline_logger(__name__)

__all__ = [
    "ClickHouseDatabase",
]

# Column names for execution_nodes table (order matches DDL)
_NODE_COLUMNS = (
    "node_id",
    "node_kind",
    "deployment_id",
    "parent_node_id",
    "root_deployment_id",
    "run_id",
    "run_scope",
    "deployment_name",
    "name",
    "sequence_no",
    "attempt",
    "flow_class",
    "task_class",
    "status",
    "started_at",
    "ended_at",
    "updated_at",
    "version",
    "model",
    "cost_usd",
    "tokens_input",
    "tokens_output",
    "tokens_cache_read",
    "tokens_cache_write",
    "tokens_reasoning",
    "turn_count",
    "error_type",
    "error_message",
    "remote_child_deployment_id",
    "parent_deployment_task_id",
    "cache_key",
    "input_fingerprint",
    "input_document_shas",
    "output_document_shas",
    "context_document_shas",
    "flow_id",
    "task_id",
    "conversation_id",
    "payload",
)

_DOCUMENT_COLUMNS = (
    "document_sha256",
    "content_sha256",
    "deployment_id",
    "producing_node_id",
    "document_type",
    "name",
    "run_scope",
    "description",
    "mime_type",
    "size_bytes",
    "publicly_visible",
    "derived_from",
    "triggered_by",
    "attachment_names",
    "attachment_descriptions",
    "attachment_sha256s",
    "attachment_mime_types",
    "attachment_sizes",
    "summary",
    "metadata_json",
    "created_at",
    "version",
)

_BLOB_COLUMNS = ("content_sha256", "content", "size_bytes", "created_at")
_LOG_COLUMNS = (
    "node_id",
    "deployment_id",
    "root_deployment_id",
    "flow_id",
    "task_id",
    "timestamp",
    "sequence_no",
    "level",
    "category",
    "logger_name",
    "message",
    "event_type",
    "fields",
    "exception_text",
)


def _node_to_row(node: ExecutionNode) -> list[Any]:
    """Convert ExecutionNode to a list of column values for INSERT."""
    return [
        node.node_id,
        node.node_kind.value,
        node.deployment_id,
        node.parent_node_id,
        node.root_deployment_id,
        node.run_id,
        node.run_scope,
        node.deployment_name,
        node.name,
        node.sequence_no,
        node.attempt,
        node.flow_class,
        node.task_class,
        node.status.value,
        node.started_at,
        node.ended_at,
        node.updated_at,
        node.version,
        node.model,
        node.cost_usd,
        node.tokens_input,
        node.tokens_output,
        node.tokens_cache_read,
        node.tokens_cache_write,
        node.tokens_reasoning,
        node.turn_count,
        node.error_type,
        node.error_message,
        node.remote_child_deployment_id,
        node.parent_deployment_task_id,
        node.cache_key,
        node.input_fingerprint,
        list(node.input_document_shas),
        list(node.output_document_shas),
        list(node.context_document_shas),
        node.flow_id,
        node.task_id,
        node.conversation_id,
        json.dumps(node.payload, default=str),
    ]


def _row_to_node(row: tuple[Any, ...]) -> ExecutionNode:
    """Convert a ClickHouse result row to ExecutionNode."""
    fields = dict(zip(_NODE_COLUMNS, row, strict=True))

    # Parse UUID fields
    for uuid_field in ("node_id", "deployment_id", "parent_node_id", "root_deployment_id"):
        v = fields[uuid_field]
        fields[uuid_field] = UUID(str(v)) if not isinstance(v, UUID) else v

    # Nullable UUID fields
    for uuid_field in ("remote_child_deployment_id", "parent_deployment_task_id", "flow_id", "task_id", "conversation_id"):
        v = fields[uuid_field]
        if v is not None:
            fields[uuid_field] = UUID(str(v)) if not isinstance(v, UUID) else v

    # Enums
    fields["node_kind"] = NodeKind(str(fields["node_kind"]))
    fields["status"] = NodeStatus(str(fields["status"]))

    # Tuples from arrays
    for arr_field in ("input_document_shas", "output_document_shas", "context_document_shas"):
        fields[arr_field] = _string_tuple(fields[arr_field])

    # Parse payload JSON
    payload_raw = fields["payload"]
    if isinstance(payload_raw, str) and payload_raw:
        fields["payload"] = json.loads(payload_raw)
    elif isinstance(payload_raw, bytes):
        fields["payload"] = json.loads(payload_raw.decode("utf-8"))
    else:
        fields["payload"] = {}

    # Ensure string fields
    for str_field in (
        "run_id",
        "run_scope",
        "deployment_name",
        "name",
        "flow_class",
        "task_class",
        "model",
        "error_type",
        "error_message",
        "cache_key",
        "input_fingerprint",
    ):
        v = fields[str_field]
        fields[str_field] = v.decode("utf-8") if isinstance(v, bytes) else str(v) if v is not None else ""
    fields["run_scope"] = RunScope(fields["run_scope"])

    # Datetime handling
    for dt_field in ("started_at", "updated_at"):
        v = fields[dt_field]
        if v is not None and not isinstance(v, datetime):
            fields[dt_field] = datetime.fromisoformat(str(v))
    v = fields["ended_at"]
    if v is not None and not isinstance(v, datetime):
        fields["ended_at"] = datetime.fromisoformat(str(v))

    # Ensure timezone awareness
    for dt_field in ("started_at", "updated_at", "ended_at"):
        v = fields[dt_field]
        if isinstance(v, datetime) and v.tzinfo is None:
            fields[dt_field] = v.replace(tzinfo=UTC)

    return ExecutionNode(**fields)


def _document_to_row(doc: DocumentRecord) -> list[Any]:
    """Convert DocumentRecord to column values."""
    return [
        doc.document_sha256,
        doc.content_sha256,
        doc.deployment_id,
        doc.producing_node_id,
        doc.document_type,
        doc.name,
        doc.run_scope,
        doc.description,
        doc.mime_type,
        doc.size_bytes,
        doc.publicly_visible,
        list(doc.derived_from),
        list(doc.triggered_by),
        list(doc.attachment_names),
        list(doc.attachment_descriptions),
        list(doc.attachment_sha256s),
        list(doc.attachment_mime_types),
        list(doc.attachment_sizes),
        doc.summary,
        doc.metadata_json,
        doc.created_at,
        doc.version,
    ]


def _row_to_document(row: tuple[Any, ...]) -> DocumentRecord:
    """Convert a ClickHouse result row to DocumentRecord."""
    fields = dict(zip(_DOCUMENT_COLUMNS, row, strict=True))

    # Decode bytes to str
    for str_field in ("document_sha256", "content_sha256", "document_type", "name", "run_scope", "description", "mime_type", "summary", "metadata_json"):
        v = fields[str_field]
        fields[str_field] = v.decode("utf-8") if isinstance(v, bytes) else str(v) if v is not None else ""
    fields["run_scope"] = RunScope(fields["run_scope"])

    # UUID
    v = fields["deployment_id"]
    fields["deployment_id"] = UUID(str(v)) if not isinstance(v, UUID) else v

    v = fields["producing_node_id"]
    if v is not None:
        fields["producing_node_id"] = UUID(str(v)) if not isinstance(v, UUID) else v

    # Tuples from arrays
    for arr_field in ("derived_from", "triggered_by", "attachment_names", "attachment_descriptions", "attachment_sha256s", "attachment_mime_types"):
        fields[arr_field] = _string_tuple(fields[arr_field])
    fields["attachment_sizes"] = _int_tuple(fields["attachment_sizes"])

    # Datetime
    v = fields["created_at"]
    if v is not None and not isinstance(v, datetime):
        fields["created_at"] = datetime.fromisoformat(str(v))
    if isinstance(fields["created_at"], datetime) and fields["created_at"].tzinfo is None:
        fields["created_at"] = fields["created_at"].replace(tzinfo=UTC)

    return DocumentRecord(**fields)


def _row_to_blob(row: tuple[Any, ...]) -> BlobRecord:
    """Convert a ClickHouse result row to BlobRecord."""
    fields = dict(zip(_BLOB_COLUMNS, row, strict=True))
    v = fields["content_sha256"]
    fields["content_sha256"] = v.decode("utf-8") if isinstance(v, bytes) else str(v)
    v = fields["content"]
    fields["content"] = v if isinstance(v, bytes) else v.encode("utf-8") if isinstance(v, str) else bytes(v)
    v = fields["created_at"]
    if v is not None and not isinstance(v, datetime):
        fields["created_at"] = datetime.fromisoformat(str(v))
    if isinstance(fields["created_at"], datetime) and fields["created_at"].tzinfo is None:
        fields["created_at"] = fields["created_at"].replace(tzinfo=UTC)
    return BlobRecord(**fields)


def _string_tuple(value: Any) -> tuple[str, ...]:
    """Normalize ClickHouse array-like values into an immutable tuple of strings."""
    if value is None:
        return ()
    items = (value,) if not isinstance(value, (list, tuple)) else cast(list[object] | tuple[object, ...], value)
    return tuple(item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in items)


def _int_tuple(value: Any) -> tuple[int, ...]:
    """Normalize ClickHouse array-like values into an immutable tuple of ints."""
    if value is None:
        return ()
    items = (value,) if not isinstance(value, (list, tuple)) else cast(list[object] | tuple[object, ...], value)
    return tuple(int(cast(Any, item)) for item in items)


def _as_list(value: Any) -> list[Any]:
    """Convert a tuple/list value into a plain list for ClickHouse parameters."""
    if not isinstance(value, (list, tuple)):
        msg = f"Expected tuple/list for ClickHouse array parameter, got {type(value).__name__}."
        raise TypeError(msg)
    return [*cast(list[Any] | tuple[Any, ...], value)]


def _log_to_row(log: ExecutionLog) -> list[Any]:
    """Convert ExecutionLog to a list of column values for INSERT."""
    return [
        log.node_id,
        log.deployment_id,
        log.root_deployment_id,
        log.flow_id,
        log.task_id,
        log.timestamp,
        log.sequence_no,
        log.level,
        log.category,
        log.logger_name,
        log.message,
        log.event_type,
        log.fields,
        log.exception_text,
    ]


def _row_to_log(row: tuple[Any, ...]) -> ExecutionLog:
    """Convert a ClickHouse result row to ExecutionLog."""
    fields = dict(zip(_LOG_COLUMNS, row, strict=True))

    for uuid_field in ("node_id", "deployment_id", "root_deployment_id"):
        value = fields[uuid_field]
        fields[uuid_field] = UUID(str(value)) if not isinstance(value, UUID) else value

    for uuid_field in ("flow_id", "task_id"):
        value = fields[uuid_field]
        if value is not None:
            fields[uuid_field] = UUID(str(value)) if not isinstance(value, UUID) else value

    for str_field in ("level", "category", "logger_name", "message", "event_type", "fields", "exception_text"):
        value = fields[str_field]
        fields[str_field] = value.decode("utf-8") if isinstance(value, bytes) else str(value) if value is not None else ""

    timestamp = fields["timestamp"]
    if timestamp is not None and not isinstance(timestamp, datetime):
        fields["timestamp"] = datetime.fromisoformat(str(timestamp))
    if isinstance(fields["timestamp"], datetime) and fields["timestamp"].tzinfo is None:
        fields["timestamp"] = fields["timestamp"].replace(tzinfo=UTC)

    return ExecutionLog(**fields)


class ClickHouseDatabase:
    """ClickHouse backend implementing DatabaseWriter and DatabaseReader.

    All blocking operations run on a single-thread executor.
    Circuit breaker protects against ClickHouse unavailability.
    """

    supports_remote = True

    def __init__(self, settings: Settings | None = None) -> None:
        self._cb = ClickHouseCircuitBreaker(settings)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ch-database")
        self._ddl = [DDL_EXECUTION_NODES, DDL_DOCUMENTS, DDL_DOCUMENT_BLOBS, DDL_EXECUTION_LOGS]

    async def _run(self, fn: Any, *args: Any) -> Any:
        """Run a sync function on the dedicated executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    def _ensure_tables(self) -> None:
        self._cb.ensure_tables(self._ddl)

    def _execute_sync(self, fn: Callable[[], Any]) -> Any:
        """Run a sync ClickHouse operation with circuit breaker protection.

        Checks circuit state, wraps fn() in try/except, and records
        success/failure on the circuit breaker.
        """
        if self._cb.is_open and not self._cb.try_reconnect():
            msg = "ClickHouse circuit breaker is open — refusing operation"
            raise ConnectionError(msg)
        try:
            result = fn()
            self._cb.record_success()
            return result
        except (ClickHouseDatabaseError, ConnectionError, OSError):
            self._cb.record_failure()
            raise

    # --- DatabaseWriter ---

    async def insert_node(self, node: ExecutionNode) -> None:
        """Insert a new execution node."""
        await self._run(self._insert_node_sync, node)

    async def update_node(self, node_id: UUID, **updates: Any) -> None:
        """Update fields on an existing execution node."""
        await self._run(self._update_node_sync, node_id, updates)

    async def save_document(self, record: DocumentRecord) -> None:
        """Persist a single document record."""
        await self._run(self._save_document_sync, record)

    async def save_document_batch(self, records: list[DocumentRecord]) -> None:
        """Persist multiple document records in one operation."""
        await self._run(self._save_document_batch_sync, records)

    async def save_blob(self, blob: BlobRecord) -> None:
        """Persist a single binary blob."""
        await self._run(self._save_blob_sync, blob)

    async def save_blob_batch(self, blobs: list[BlobRecord]) -> None:
        """Persist multiple binary blobs in one operation."""
        await self._run(self._save_blob_batch_sync, blobs)

    async def save_logs_batch(self, logs: list[ExecutionLog]) -> None:
        """Persist multiple execution logs in one operation."""
        await self._run(self._save_logs_batch_sync, logs)

    async def update_document_summary(self, document_sha256: str, summary: str) -> None:
        """Update the summary field of an existing document."""
        await self._run(self._update_document_summary_sync, document_sha256, summary)

    async def flush(self) -> None:
        """Flush any buffered writes to storage."""

    async def shutdown(self) -> None:
        """Release resources and close connections."""
        self._cb.close()
        self._executor.shutdown(wait=False)

    # --- DatabaseReader ---

    async def get_node(self, node_id: UUID) -> ExecutionNode | None:
        """Retrieve an execution node by its ID."""
        return await self._run(self._get_node_sync, node_id)

    async def get_children(self, parent_node_id: UUID) -> list[ExecutionNode]:
        """Retrieve all direct child nodes of a parent node."""
        return await self._run(self._get_children_sync, parent_node_id)

    async def get_deployment_tree(self, deployment_id: UUID) -> list[ExecutionNode]:
        """Retrieve all nodes belonging to a deployment."""
        return await self._run(self._get_deployment_tree_sync, deployment_id)

    async def get_deployment_by_run_id(self, run_id: str) -> ExecutionNode | None:
        """Find the deployment node for a given run ID."""
        return await self._run(self._get_deployment_by_run_id_sync, run_id)

    async def get_deployment_by_run_scope(self, run_scope: str) -> ExecutionNode | None:
        """Find the deployment node for a given run scope."""
        return await self._run(self._get_deployment_by_run_scope_sync, run_scope)

    async def get_document(self, document_sha256: str) -> DocumentRecord | None:
        """Retrieve a document record by its SHA256."""
        return await self._run(self._get_document_sync, document_sha256)

    async def find_document_by_name(self, name: str) -> DocumentRecord | None:
        """Find the newest document with an exact name match."""
        return await self._run(self._find_document_by_name_sync, name)

    async def get_documents_batch(self, sha256s: list[DocumentSha256]) -> dict[DocumentSha256, DocumentRecord]:
        """Retrieve multiple document records by their SHA256s."""
        return await self._run(self._get_documents_batch_sync, sha256s)

    async def get_blob(self, content_sha256: str) -> BlobRecord | None:
        """Retrieve a binary blob by its content SHA256."""
        return await self._run(self._get_blob_sync, content_sha256)

    async def get_blobs_batch(self, content_sha256s: list[str]) -> dict[str, BlobRecord]:
        """Retrieve multiple binary blobs by their content SHA256s."""
        return await self._run(self._get_blobs_batch_sync, content_sha256s)

    async def get_documents_by_deployment(self, deployment_id: UUID) -> list[DocumentRecord]:
        """Retrieve all documents belonging to a deployment chain."""
        return await self._run(self._get_documents_by_deployment_sync, deployment_id)

    async def get_documents_by_node(self, node_id: UUID) -> list[DocumentRecord]:
        """Retrieve all documents produced by a specific node."""
        return await self._run(self._get_documents_by_node_sync, node_id)

    async def get_all_document_shas_for_deployment(self, deployment_id: UUID) -> set[str]:
        """Retrieve all document SHA256s referenced by a deployment's nodes."""
        return await self._run(self._get_all_document_shas_for_deployment_sync, deployment_id)

    async def check_existing_documents(self, sha256s: list[DocumentSha256]) -> set[DocumentSha256]:
        """Return the subset of SHA256s that already exist in storage."""
        return await self._run(self._check_existing_documents_sync, sha256s)

    async def find_documents_by_source(self, source_sha256: str) -> list[DocumentRecord]:
        """Find documents derived from a given source SHA256."""
        return await self._run(self._find_documents_by_source_sync, source_sha256)

    async def get_document_ancestry(self, sha256: DocumentSha256) -> dict[str, DocumentRecord]:
        """Return all ancestor documents reachable from derived_from and triggered_by."""
        return await self._run(self._get_document_ancestry_sync, sha256)

    async def find_documents_by_origin(self, sha256: DocumentSha256) -> list[DocumentRecord]:
        """Find documents referencing a SHA256 in derived_from or triggered_by."""
        return await self._run(self._find_documents_by_origin_sync, sha256)

    async def list_run_scopes(self, limit: int) -> list[RunScopeInfo]:
        """List non-empty document run scopes ordered by latest activity."""
        return await self._run(self._list_run_scopes_sync, limit)

    async def search_documents(
        self,
        name: str | None,
        document_type: str | None,
        run_scope: str | None,
        limit: int,
        offset: int,
    ) -> list[DocumentRecord]:
        """Search documents by metadata with pagination."""
        return await self._run(self._search_documents_sync, name, document_type, run_scope, limit, offset)

    async def get_deployment_cost_totals(self, deployment_id: UUID) -> tuple[float, int]:
        """Return total conversation-turn cost and total tokens for a deployment."""
        return await self._run(self._get_deployment_cost_totals_sync, deployment_id)

    async def get_documents_by_run_scope(self, run_scope: str) -> list[DocumentRecord]:
        """Retrieve all documents for a run scope."""
        return await self._run(self._get_documents_by_run_scope_sync, run_scope)

    async def list_deployments(self, limit: int, status: str | None) -> list[ExecutionNode]:
        """List deployment nodes ordered by newest start time first."""
        return await self._run(self._list_deployments_sync, limit, status)

    async def get_cached_completion(self, cache_key: str, max_age: timedelta | None = None) -> ExecutionNode | None:
        """Find a completed node matching the cache key within the max age."""
        return await self._run(self._get_cached_completion_sync, cache_key, max_age)

    async def get_node_logs(
        self,
        node_id: UUID,
        *,
        level: str | None = None,
        category: str | None = None,
    ) -> list[ExecutionLog]:
        """Retrieve execution logs for a specific node."""
        return await self._run(self._get_node_logs_sync, node_id, level, category)

    async def get_deployment_logs(
        self,
        deployment_id: UUID,
        *,
        level: str | None = None,
        category: str | None = None,
    ) -> list[ExecutionLog]:
        """Retrieve execution logs for an entire deployment."""
        return await self._run(self._get_deployment_logs_sync, deployment_id, level, category)

    # --- Sync writer implementations ---

    def _insert_node_sync(self, node: ExecutionNode) -> None:
        self._ensure_tables()

        def _op() -> None:
            self._cb.client.insert(EXECUTION_NODES_TABLE, [_node_to_row(node)], column_names=list(_NODE_COLUMNS))

        self._execute_sync(_op)

    def _update_node_sync(self, node_id: UUID, updates: dict[str, Any]) -> None:
        """Update node via optimistic append-only INSERT...SELECT retries."""
        self._ensure_tables()

        def _write_attempt(expected_version: int, write_updates: dict[str, Any]) -> int:
            next_version = next_node_version(expected_version)
            select_expr, params = build_node_update_command(
                node_id=node_id,
                updates=write_updates,
                expected_version=expected_version,
                next_version=next_version,
                node_columns=_NODE_COLUMNS,
                clickhouse_param_type=_clickhouse_param_type,
                as_list=_as_list,
            )
            command = (
                f"INSERT INTO {EXECUTION_NODES_TABLE} ({', '.join(_NODE_COLUMNS)}) "
                f"SELECT {select_expr} FROM {EXECUTION_NODES_TABLE} FINAL "
                f"WHERE node_id = {{node_id:UUID}} AND version = {{expected_version:UInt64}}"
            )
            return command_written_rows(self._execute_sync(lambda command=command, params=params: self._cb.client.command(command, parameters=params)))

        update_node_optimistically(
            node_id=node_id,
            updates=updates,
            load_node=self._get_node_sync,
            write_attempt=_write_attempt,
        )

    def _save_document_sync(self, record: DocumentRecord) -> None:
        self._ensure_tables()

        def _op() -> None:
            self._cb.client.insert(DOCUMENTS_TABLE, [_document_to_row(record)], column_names=list(_DOCUMENT_COLUMNS))

        self._execute_sync(_op)

    def _save_document_batch_sync(self, records: list[DocumentRecord]) -> None:
        if not records:
            return
        self._ensure_tables()

        def _op() -> None:
            self._cb.client.insert(DOCUMENTS_TABLE, [_document_to_row(r) for r in records], column_names=list(_DOCUMENT_COLUMNS))

        self._execute_sync(_op)

    def _save_blob_sync(self, blob: BlobRecord) -> None:
        self._ensure_tables()

        def _op() -> None:
            self._cb.client.insert(
                DOCUMENT_BLOBS_TABLE,
                [[blob.content_sha256, blob.content, blob.size_bytes, blob.created_at]],
                column_names=list(_BLOB_COLUMNS),
            )

        self._execute_sync(_op)

    def _save_blob_batch_sync(self, blobs: list[BlobRecord]) -> None:
        if not blobs:
            return
        self._ensure_tables()

        def _op() -> None:
            self._cb.client.insert(
                DOCUMENT_BLOBS_TABLE,
                [[b.content_sha256, b.content, b.size_bytes, b.created_at] for b in blobs],
                column_names=list(_BLOB_COLUMNS),
            )

        self._execute_sync(_op)

    def _save_logs_batch_sync(self, logs: list[ExecutionLog]) -> None:
        if not logs:
            return
        self._ensure_tables()

        def _op() -> None:
            self._cb.client.insert(EXECUTION_LOGS_TABLE, [_log_to_row(log) for log in logs], column_names=list(_LOG_COLUMNS))

        self._execute_sync(_op)

    def _update_document_summary_sync(self, document_sha256: str, summary: str) -> None:
        self._ensure_tables()

        def _op() -> None:
            self._cb.client.command(
                f"INSERT INTO {DOCUMENTS_TABLE} "
                "(document_sha256, content_sha256, deployment_id, producing_node_id, "
                "document_type, name, run_scope, description, mime_type, size_bytes, publicly_visible, "
                "derived_from, triggered_by, "
                "attachment_names, attachment_descriptions, attachment_sha256s, attachment_mime_types, attachment_sizes, "
                "summary, metadata_json, created_at, version) "
                "SELECT document_sha256, content_sha256, deployment_id, producing_node_id, "
                "document_type, name, run_scope, description, mime_type, size_bytes, publicly_visible, "
                "derived_from, triggered_by, "
                "attachment_names, attachment_descriptions, attachment_sha256s, attachment_mime_types, attachment_sizes, "
                "{summary:String}, metadata_json, created_at, version + 1 "
                f"FROM {DOCUMENTS_TABLE} FINAL WHERE document_sha256 = {{sha:String}}",
                parameters={"summary": summary, "sha": document_sha256},
            )

        self._execute_sync(_op)

    # --- Sync reader implementations ---

    def _get_node_sync(self, node_id: UUID) -> ExecutionNode | None:
        self._ensure_tables()
        cols = ", ".join(_NODE_COLUMNS)

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT {cols} FROM {EXECUTION_NODES_TABLE} FINAL WHERE node_id = {{id:UUID}}",
                parameters={"id": node_id},
            )

        result = self._execute_sync(_op)
        if not result.result_rows:
            return None
        return _row_to_node(tuple(result.result_rows[0]))

    def _get_children_sync(self, parent_node_id: UUID) -> list[ExecutionNode]:
        self._ensure_tables()
        cols = ", ".join(_NODE_COLUMNS)

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT {cols} FROM {EXECUTION_NODES_TABLE} FINAL WHERE parent_node_id = {{id:UUID}} ORDER BY sequence_no, node_id",
                parameters={"id": parent_node_id},
            )

        result = self._execute_sync(_op)
        return [_row_to_node(tuple(row)) for row in result.result_rows]

    def _get_deployment_tree_sync(self, deployment_id: UUID) -> list[ExecutionNode]:
        self._ensure_tables()
        cols = ", ".join(_NODE_COLUMNS)

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT {cols} FROM {EXECUTION_NODES_TABLE} FINAL WHERE deployment_id = {{id:UUID}} ORDER BY sequence_no, node_id",
                parameters={"id": deployment_id},
            )

        result = self._execute_sync(_op)
        return [_row_to_node(tuple(row)) for row in result.result_rows]

    def _get_deployment_by_run_id_sync(self, run_id: str) -> ExecutionNode | None:
        self._ensure_tables()
        cols = ", ".join(_NODE_COLUMNS)

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT {cols} FROM {EXECUTION_NODES_TABLE} FINAL WHERE run_id = {{run_id:String}} AND node_kind = '{NodeKind.DEPLOYMENT.value}' LIMIT 1",
                parameters={"run_id": run_id},
            )

        result = self._execute_sync(_op)
        if not result.result_rows:
            return None
        return _row_to_node(tuple(result.result_rows[0]))

    def _get_deployment_by_run_scope_sync(self, run_scope: str) -> ExecutionNode | None:
        self._ensure_tables()
        cols = ", ".join(_NODE_COLUMNS)

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT {cols} FROM {EXECUTION_NODES_TABLE} FINAL WHERE run_scope = {{scope:String}} AND node_kind = '{NodeKind.DEPLOYMENT.value}' LIMIT 1",
                parameters={"scope": run_scope},
            )

        result = self._execute_sync(_op)
        if not result.result_rows:
            return None
        return _row_to_node(tuple(result.result_rows[0]))

    def _get_document_sync(self, document_sha256: str) -> DocumentRecord | None:
        self._ensure_tables()
        cols = ", ".join(_DOCUMENT_COLUMNS)

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT {cols} FROM {DOCUMENTS_TABLE} FINAL WHERE document_sha256 = {{sha:String}}",
                parameters={"sha": document_sha256},
            )

        result = self._execute_sync(_op)
        if not result.result_rows:
            return None
        return _row_to_document(tuple(result.result_rows[0]))

    def _find_document_by_name_sync(self, name: str) -> DocumentRecord | None:
        self._ensure_tables()
        cols = ", ".join(_DOCUMENT_COLUMNS)

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT {cols} FROM {DOCUMENTS_TABLE} FINAL WHERE name = {{name:String}} ORDER BY created_at DESC, document_sha256 DESC LIMIT 1",
                parameters={"name": name},
            )

        result = self._execute_sync(_op)
        if not result.result_rows:
            return None
        return _row_to_document(tuple(result.result_rows[0]))

    def _get_documents_batch_sync(self, sha256s: list[str]) -> dict[str, DocumentRecord]:
        if not sha256s:
            return {}
        self._ensure_tables()
        cols = ", ".join(_DOCUMENT_COLUMNS)

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT {cols} FROM {DOCUMENTS_TABLE} FINAL WHERE document_sha256 IN {{shas:Array(String)}}",
                parameters={"shas": sha256s},
            )

        result = self._execute_sync(_op)
        docs: dict[str, DocumentRecord] = {}
        for row in result.result_rows:
            doc = _row_to_document(tuple(row))
            docs[doc.document_sha256] = doc
        return docs

    def _get_blob_sync(self, content_sha256: str) -> BlobRecord | None:
        self._ensure_tables()

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT content_sha256, content, size_bytes, created_at FROM {DOCUMENT_BLOBS_TABLE} FINAL WHERE content_sha256 = {{sha:String}}",
                parameters={"sha": content_sha256},
            )

        result = self._execute_sync(_op)
        if not result.result_rows:
            return None
        return _row_to_blob(tuple(result.result_rows[0]))

    def _get_blobs_batch_sync(self, content_sha256s: list[str]) -> dict[str, BlobRecord]:
        if not content_sha256s:
            return {}
        self._ensure_tables()

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT content_sha256, content, size_bytes, created_at FROM {DOCUMENT_BLOBS_TABLE} FINAL WHERE content_sha256 IN {{shas:Array(String)}}",
                parameters={"shas": content_sha256s},
            )

        result = self._execute_sync(_op)
        blobs: dict[str, BlobRecord] = {}
        for row in result.result_rows:
            blob = _row_to_blob(tuple(row))
            blobs[blob.content_sha256] = blob
        return blobs

    def _get_documents_by_deployment_sync(self, deployment_id: UUID) -> list[DocumentRecord]:
        self._ensure_tables()
        cols = ", ".join(_DOCUMENT_COLUMNS)
        deployment_ids = self._get_deployment_chain_ids_sync(deployment_id)

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT {cols} FROM {DOCUMENTS_TABLE} FINAL WHERE deployment_id IN {{deployment_ids:Array(UUID)}}",
                parameters={"deployment_ids": deployment_ids},
            )

        result = self._execute_sync(_op)
        return [_row_to_document(tuple(row)) for row in result.result_rows]

    def _get_deployment_chain_ids_sync(self, deployment_id: UUID) -> list[UUID]:
        self._ensure_tables()

        def _root_op() -> Any:
            return self._cb.client.query(
                f"SELECT root_deployment_id FROM {EXECUTION_NODES_TABLE} FINAL "
                f"WHERE deployment_id = {{deployment_id:UUID}} AND node_kind = '{NodeKind.DEPLOYMENT.value}' LIMIT 1",
                parameters={"deployment_id": deployment_id},
            )

        root_result = self._execute_sync(_root_op)
        root_deployment_id = deployment_id
        if root_result.result_rows:
            root_value = root_result.result_rows[0][0]
            root_deployment_id = UUID(str(root_value)) if not isinstance(root_value, UUID) else root_value

        def _deployment_ids_op() -> Any:
            return self._cb.client.query(
                f"SELECT deployment_id FROM {EXECUTION_NODES_TABLE} FINAL "
                f"WHERE node_kind = '{NodeKind.DEPLOYMENT.value}' AND root_deployment_id = {{root_id:UUID}}",
                parameters={"root_id": root_deployment_id},
            )

        deployment_ids_result = self._execute_sync(_deployment_ids_op)
        deployment_ids = [UUID(str(row[0])) if not isinstance(row[0], UUID) else row[0] for row in deployment_ids_result.result_rows]
        return deployment_ids or [deployment_id]

    def _get_documents_by_node_sync(self, node_id: UUID) -> list[DocumentRecord]:
        self._ensure_tables()
        cols = ", ".join(_DOCUMENT_COLUMNS)

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT {cols} FROM {DOCUMENTS_TABLE} FINAL WHERE producing_node_id = {{id:UUID}}",
                parameters={"id": node_id},
            )

        result = self._execute_sync(_op)
        docs = [_row_to_document(tuple(row)) for row in result.result_rows]

        # For deployment nodes, include root input documents without a separate node lookup.
        def _op_root() -> Any:
            return self._cb.client.query(
                f"SELECT {cols} FROM {DOCUMENTS_TABLE} FINAL "
                "WHERE producing_node_id IS NULL "
                f"AND deployment_id = (SELECT deployment_id FROM {EXECUTION_NODES_TABLE} FINAL WHERE node_id = {{id:UUID}} LIMIT 1) "
                f"AND EXISTS(SELECT 1 FROM {EXECUTION_NODES_TABLE} FINAL WHERE node_id = {{id:UUID}} AND node_kind = '{NodeKind.DEPLOYMENT.value}')",
                parameters={"id": node_id},
            )

        root_result = self._execute_sync(_op_root)
        docs.extend(_row_to_document(tuple(row)) for row in root_result.result_rows)

        return docs

    def _get_all_document_shas_for_deployment_sync(self, deployment_id: UUID) -> set[str]:
        self._ensure_tables()

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT input_document_shas, output_document_shas, context_document_shas FROM {EXECUTION_NODES_TABLE} WHERE deployment_id = {{id:UUID}}",
                parameters={"id": deployment_id},
            )

        result = self._execute_sync(_op)
        shas: set[str] = set()
        for row in result.result_rows:
            for arr in row:
                if arr:
                    shas.update(str(s) if not isinstance(s, str) else s for s in arr)
        return shas

    def _check_existing_documents_sync(self, sha256s: list[str]) -> set[str]:
        if not sha256s:
            return set()
        self._ensure_tables()

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT document_sha256 FROM {DOCUMENTS_TABLE} WHERE document_sha256 IN {{shas:Array(String)}}",
                parameters={"shas": sha256s},
            )

        result = self._execute_sync(_op)
        return {(row[0].decode("utf-8") if isinstance(row[0], bytes) else str(row[0])) for row in result.result_rows}

    def _find_documents_by_source_sync(self, source_sha256: str) -> list[DocumentRecord]:
        self._ensure_tables()
        cols = ", ".join(_DOCUMENT_COLUMNS)

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT {cols} FROM {DOCUMENTS_TABLE} FINAL WHERE has(derived_from, {{sha:String}})",
                parameters={"sha": source_sha256},
            )

        result = self._execute_sync(_op)
        return [_row_to_document(tuple(row)) for row in result.result_rows]

    def _get_document_ancestry_sync(self, sha256: DocumentSha256) -> dict[str, DocumentRecord]:
        target = self._get_document_sync(sha256)
        if target is None:
            return {}

        self._ensure_tables()
        deployment_ids = self._get_deployment_chain_ids_sync(target.deployment_id)
        cols = ", ".join(_DOCUMENT_COLUMNS)

        def _chain_docs_op() -> Any:
            return self._cb.client.query(
                f"SELECT {cols} FROM {DOCUMENTS_TABLE} FINAL WHERE deployment_id IN {{deployment_ids:Array(UUID)}}",
                parameters={"deployment_ids": deployment_ids},
            )

        chain_docs_result = self._execute_sync(_chain_docs_op)
        docs_by_sha: dict[str, DocumentRecord] = {doc.document_sha256: doc for doc in (_row_to_document(tuple(row)) for row in chain_docs_result.result_rows)}
        return collect_document_ancestry(
            target=target,
            docs_by_sha=docs_by_sha,
            load_extra_documents=self._get_documents_batch_sync,
        )

    def _find_documents_by_origin_sync(self, sha256: DocumentSha256) -> list[DocumentRecord]:
        self._ensure_tables()
        cols = ", ".join(_DOCUMENT_COLUMNS)

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT {cols} FROM {DOCUMENTS_TABLE} FINAL "
                "WHERE has(derived_from, {sha:String}) OR has(triggered_by, {sha:String}) "
                "ORDER BY created_at DESC, document_sha256 DESC",
                parameters={"sha": sha256},
            )

        result = self._execute_sync(_op)
        return [_row_to_document(tuple(row)) for row in result.result_rows]

    def _list_run_scopes_sync(self, limit: int) -> list[RunScopeInfo]:
        self._ensure_tables()

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT run_scope, count(), max(created_at) FROM {DOCUMENTS_TABLE} FINAL "
                "WHERE run_scope != '' "
                "GROUP BY run_scope "
                "ORDER BY max(created_at) DESC "
                "LIMIT {limit:UInt32}",
                parameters={"limit": limit},
            )

        result = self._execute_sync(_op)
        infos: list[RunScopeInfo] = []
        for run_scope, document_count, latest_created_at in result.result_rows:
            latest = latest_created_at
            if not isinstance(latest, datetime):
                latest = datetime.fromisoformat(str(latest))
            if latest.tzinfo is None:
                latest = latest.replace(tzinfo=UTC)
            infos.append(
                RunScopeInfo(
                    run_scope=RunScope(run_scope.decode("utf-8") if isinstance(run_scope, bytes) else str(run_scope)),
                    document_count=int(document_count),
                    latest_created_at=latest,
                )
            )
        return infos

    def _search_documents_sync(
        self,
        name: str | None,
        document_type: str | None,
        run_scope: str | None,
        limit: int,
        offset: int,
    ) -> list[DocumentRecord]:
        self._ensure_tables()
        cols = ", ".join(_DOCUMENT_COLUMNS)
        filters: list[str] = ["1 = 1"]
        params: dict[str, Any] = {"limit": limit, "offset": offset}

        if name is not None:
            filters.append("positionCaseInsensitiveUTF8(name, {name:String}) > 0")
            params["name"] = name
        if document_type is not None:
            filters.append("document_type = {document_type:String}")
            params["document_type"] = document_type
        if run_scope is not None:
            filters.append("run_scope = {run_scope:String}")
            params["run_scope"] = run_scope

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT {cols} FROM {DOCUMENTS_TABLE} FINAL "
                f"WHERE {' AND '.join(filters)} "
                "ORDER BY created_at DESC, document_sha256 DESC "
                "LIMIT {limit:UInt32} OFFSET {offset:UInt32}",
                parameters=params,
            )

        result = self._execute_sync(_op)
        return [_row_to_document(tuple(row)) for row in result.result_rows]

    def _get_deployment_cost_totals_sync(self, deployment_id: UUID) -> tuple[float, int]:
        self._ensure_tables()

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT sum(cost_usd), sum(tokens_input + tokens_output) FROM {EXECUTION_NODES_TABLE} FINAL "
                "WHERE deployment_id = {deployment_id:UUID} "
                f"AND node_kind = '{NodeKind.CONVERSATION_TURN.value}'",
                parameters={"deployment_id": deployment_id},
            )

        result = self._execute_sync(_op)
        if not result.result_rows:
            return 0.0, 0
        total_cost, total_tokens = result.result_rows[0]
        return float(total_cost or 0.0), int(total_tokens or 0)

    def _get_documents_by_run_scope_sync(self, run_scope: str) -> list[DocumentRecord]:
        self._ensure_tables()
        cols = ", ".join(_DOCUMENT_COLUMNS)

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT {cols} FROM {DOCUMENTS_TABLE} FINAL WHERE run_scope = {{run_scope:String}} ORDER BY created_at DESC, document_sha256 DESC",
                parameters={"run_scope": run_scope},
            )

        result = self._execute_sync(_op)
        return [_row_to_document(tuple(row)) for row in result.result_rows]

    def _list_deployments_sync(self, limit: int, status: str | None) -> list[ExecutionNode]:
        self._ensure_tables()
        cols = ", ".join(_NODE_COLUMNS)
        filters = [f"node_kind = '{NodeKind.DEPLOYMENT.value}'"]
        params: dict[str, Any] = {"limit": limit}
        if status is not None:
            filters.append("status = {status:String}")
            params["status"] = status

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT {cols} FROM {EXECUTION_NODES_TABLE} FINAL WHERE {' AND '.join(filters)} ORDER BY started_at DESC, node_id DESC LIMIT {{limit:UInt32}}",
                parameters=params,
            )

        result = self._execute_sync(_op)
        return [_row_to_node(tuple(row)) for row in result.result_rows]

    def _get_cached_completion_sync(self, cache_key: str, max_age: timedelta | None) -> ExecutionNode | None:
        self._ensure_tables()
        cols = ", ".join(_NODE_COLUMNS)
        params: dict[str, Any] = {"key": cache_key}
        age_filter = ""
        if max_age is not None:
            params["cutoff"] = datetime.now(UTC) - max_age
            age_filter = " AND ended_at >= {cutoff:DateTime64(3, 'UTC')}"

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT {cols} FROM {EXECUTION_NODES_TABLE} FINAL "
                f"WHERE cache_key = {{key:String}} AND status = '{NodeStatus.COMPLETED.value}'{age_filter} "
                "ORDER BY ended_at DESC LIMIT 1",
                parameters=params,
            )

        result = self._execute_sync(_op)
        if not result.result_rows:
            return None
        return _row_to_node(tuple(result.result_rows[0]))

    def _get_node_logs_sync(self, node_id: UUID, level: str | None, category: str | None) -> list[ExecutionLog]:
        self._ensure_tables()
        cols = ", ".join(_LOG_COLUMNS)
        params: dict[str, Any] = {"node_id": node_id}
        filters = ["node_id = {node_id:UUID}"]

        if level is not None:
            filters.append("level = {level:String}")
            params["level"] = level
        if category is not None:
            filters.append("category = {category:String}")
            params["category"] = category

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT {cols} FROM {EXECUTION_LOGS_TABLE} WHERE {' AND '.join(filters)} ORDER BY sequence_no, timestamp",
                parameters=params,
            )

        result = self._execute_sync(_op)
        return [_row_to_log(tuple(row)) for row in result.result_rows]

    def _get_deployment_logs_sync(self, deployment_id: UUID, level: str | None, category: str | None) -> list[ExecutionLog]:
        self._ensure_tables()
        cols = ", ".join(_LOG_COLUMNS)
        params: dict[str, Any] = {"deployment_id": deployment_id}
        filters = ["deployment_id = {deployment_id:UUID}"]

        if level is not None:
            filters.append("level = {level:String}")
            params["level"] = level
        if category is not None:
            filters.append("category = {category:String}")
            params["category"] = category

        def _op() -> Any:
            return self._cb.client.query(
                f"SELECT {cols} FROM {EXECUTION_LOGS_TABLE} WHERE {' AND '.join(filters)} ORDER BY timestamp, sequence_no, node_id",
                parameters=params,
            )

        result = self._execute_sync(_op)
        return [_row_to_log(tuple(row)) for row in result.result_rows]


def _clickhouse_param_type(col: str, value: Any) -> str:
    """Map column name and value to ClickHouse parameter type string."""
    if isinstance(value, UUID):
        return "UUID"
    if isinstance(value, datetime):
        return "DateTime64(3, 'UTC')"
    if isinstance(value, bool):
        return "Bool"
    if isinstance(value, int):
        if col in {"sequence_no", "attempt"}:
            return "Int32"
        if col == "turn_count":
            return "UInt16"
        if col == "version":
            return "UInt64"
        return "UInt32"
    if isinstance(value, float):
        return "Float64"
    if isinstance(value, list):
        return "Array(String)"
    if value is None:
        nullable_types = {
            "ended_at": "Nullable(DateTime64(3, 'UTC'))",
            "remote_child_deployment_id": "Nullable(UUID)",
            "parent_deployment_task_id": "Nullable(UUID)",
            "flow_id": "Nullable(UUID)",
            "task_id": "Nullable(UUID)",
            "conversation_id": "Nullable(UUID)",
        }
        return nullable_types.get(col, "Nullable(String)")
    return "String"
