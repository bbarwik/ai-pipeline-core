"""Filesystem backend for local CLI and debug use.

Stores execution nodes as JSON files in a hierarchical directory structure.
Human-browsable layout with run-partitioned directories and shared blobs.

Layout:
    {base_path}/
      runs/
        {date}_{deployment_name}_{run_id_short}/
          deployment.json
          documents/             # root input documents
          flows/
            01_FlowName_{node_id8}/
              flow.json
              tasks/
                01_TaskName_{node_id8}/
                  task.json
                  documents/
                  conversations/
                    01_name_{node_id8}/
                      conversation.json
                      turns/
                        00.json
      blobs/
        {prefix2}/{content_sha256}
"""

import asyncio
import contextlib
import fcntl
import json
import os
import re
import tempfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from ai_pipeline_core.database._types import NULL_PARENT, BlobRecord, DocumentRecord, ExecutionLog, ExecutionNode, NodeKind, NodeStatus, RunScopeInfo
from ai_pipeline_core.documents._context import DocumentSha256, RunScope
from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)

__all__ = [
    "FilesystemDatabase",
]

_MAX_DIR_NAME_LENGTH = 100
_RUN_ID_SHORT_LENGTH = 8
_NODE_ID_SHORT_LENGTH = 8


def _sanitize_dir_name(name: str) -> str:
    """Sanitize a name for filesystem use: alphanumeric + dashes, max length."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    safe = safe.strip("_- ")
    if len(safe) > _MAX_DIR_NAME_LENGTH:
        safe = safe[:_MAX_DIR_NAME_LENGTH]
    return safe.rstrip("_- ") or "unnamed"


def _serialize_uuid(v: UUID | None) -> str | None:
    return str(v) if v is not None else None


def _serialize_datetime(v: datetime | None) -> str | None:
    return v.isoformat() if v is not None else None


def _as_json_list(value: Any) -> list[Any]:
    """Convert tuple/list values into plain JSON-serializable lists."""
    if not isinstance(value, (list, tuple)):
        msg = f"Expected tuple/list for JSON serialization, got {type(value).__name__}."
        raise TypeError(msg)
    items = cast(list[Any] | tuple[Any, ...], value)
    return [*items]


def _serialize_node(node: ExecutionNode) -> dict[str, Any]:
    """Serialize ExecutionNode to a JSON-compatible dict."""
    return {
        "node_id": str(node.node_id),
        "node_kind": node.node_kind.value,
        "deployment_id": str(node.deployment_id),
        "parent_node_id": str(node.parent_node_id),
        "root_deployment_id": str(node.root_deployment_id),
        "run_id": node.run_id,
        "run_scope": node.run_scope,
        "deployment_name": node.deployment_name,
        "name": node.name,
        "sequence_no": node.sequence_no,
        "attempt": node.attempt,
        "flow_class": node.flow_class,
        "task_class": node.task_class,
        "status": node.status.value,
        "started_at": _serialize_datetime(node.started_at),
        "ended_at": _serialize_datetime(node.ended_at),
        "updated_at": _serialize_datetime(node.updated_at),
        "version": node.version,
        "model": node.model,
        "cost_usd": node.cost_usd,
        "tokens_input": node.tokens_input,
        "tokens_output": node.tokens_output,
        "tokens_cache_read": node.tokens_cache_read,
        "tokens_cache_write": node.tokens_cache_write,
        "tokens_reasoning": node.tokens_reasoning,
        "turn_count": node.turn_count,
        "error_type": node.error_type,
        "error_message": node.error_message,
        "remote_child_deployment_id": _serialize_uuid(node.remote_child_deployment_id),
        "parent_deployment_task_id": _serialize_uuid(node.parent_deployment_task_id),
        "cache_key": node.cache_key,
        "input_fingerprint": node.input_fingerprint,
        "input_document_shas": list(node.input_document_shas),
        "output_document_shas": list(node.output_document_shas),
        "context_document_shas": list(node.context_document_shas),
        "flow_id": _serialize_uuid(node.flow_id),
        "task_id": _serialize_uuid(node.task_id),
        "conversation_id": _serialize_uuid(node.conversation_id),
        "payload": node.payload,
    }


def _deserialize_node(data: dict[str, Any]) -> ExecutionNode:
    """Deserialize a dict (from JSON) to ExecutionNode."""

    def _parse_uuid(v: str | None) -> UUID | None:
        return UUID(v) if v is not None else None

    def _parse_dt(v: str | None) -> datetime | None:
        if v is None:
            return None
        dt = datetime.fromisoformat(v)
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)

    return ExecutionNode(
        node_id=UUID(data["node_id"]),
        node_kind=NodeKind(data["node_kind"]),
        deployment_id=UUID(data["deployment_id"]),
        parent_node_id=UUID(data["parent_node_id"]),
        root_deployment_id=UUID(data["root_deployment_id"]),
        run_id=data["run_id"],
        run_scope=RunScope(data["run_scope"]),
        deployment_name=data["deployment_name"],
        name=data["name"],
        sequence_no=data["sequence_no"],
        attempt=data.get("attempt", 0),
        flow_class=data.get("flow_class", ""),
        task_class=data.get("task_class", ""),
        status=NodeStatus(data["status"]),
        started_at=_parse_dt(data["started_at"]) or datetime.now(UTC),
        ended_at=_parse_dt(data.get("ended_at")),
        updated_at=_parse_dt(data["updated_at"]) or datetime.now(UTC),
        version=data.get("version", 1),
        model=data.get("model", ""),
        cost_usd=data.get("cost_usd", 0.0),
        tokens_input=data.get("tokens_input", 0),
        tokens_output=data.get("tokens_output", 0),
        tokens_cache_read=data.get("tokens_cache_read", 0),
        tokens_cache_write=data.get("tokens_cache_write", 0),
        tokens_reasoning=data.get("tokens_reasoning", 0),
        turn_count=data.get("turn_count", 0),
        error_type=data.get("error_type", ""),
        error_message=data.get("error_message", ""),
        remote_child_deployment_id=_parse_uuid(data.get("remote_child_deployment_id")),
        parent_deployment_task_id=_parse_uuid(data.get("parent_deployment_task_id")),
        cache_key=data.get("cache_key", ""),
        input_fingerprint=data.get("input_fingerprint", ""),
        input_document_shas=tuple(data.get("input_document_shas", ())),
        output_document_shas=tuple(data.get("output_document_shas", ())),
        context_document_shas=tuple(data.get("context_document_shas", ())),
        flow_id=_parse_uuid(data.get("flow_id")),
        task_id=_parse_uuid(data.get("task_id")),
        conversation_id=_parse_uuid(data.get("conversation_id")),
        payload=data.get("payload", {}),
    )


def _serialize_document(doc: DocumentRecord) -> dict[str, Any]:
    """Serialize DocumentRecord to JSON-compatible dict."""
    return {
        "document_sha256": doc.document_sha256,
        "content_sha256": doc.content_sha256,
        "deployment_id": str(doc.deployment_id),
        "producing_node_id": _serialize_uuid(doc.producing_node_id),
        "document_type": doc.document_type,
        "name": doc.name,
        "run_scope": doc.run_scope,
        "description": doc.description,
        "mime_type": doc.mime_type,
        "size_bytes": doc.size_bytes,
        "publicly_visible": doc.publicly_visible,
        "derived_from": list(doc.derived_from),
        "triggered_by": list(doc.triggered_by),
        "attachment_names": list(doc.attachment_names),
        "attachment_descriptions": list(doc.attachment_descriptions),
        "attachment_sha256s": list(doc.attachment_sha256s),
        "attachment_mime_types": list(doc.attachment_mime_types),
        "attachment_sizes": list(doc.attachment_sizes),
        "summary": doc.summary,
        "metadata_json": doc.metadata_json,
        "created_at": _serialize_datetime(doc.created_at),
        "version": doc.version,
    }


def _deserialize_document(data: dict[str, Any]) -> DocumentRecord:
    """Deserialize a dict to DocumentRecord."""
    producing = data.get("producing_node_id")
    created_at_raw = data.get("created_at")
    created_at = datetime.fromisoformat(created_at_raw) if created_at_raw else datetime.now(UTC)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)

    return DocumentRecord(
        document_sha256=data["document_sha256"],
        content_sha256=data["content_sha256"],
        deployment_id=UUID(data["deployment_id"]),
        producing_node_id=UUID(producing) if producing else None,
        document_type=data["document_type"],
        name=data["name"],
        run_scope=RunScope(data.get("run_scope", "")),
        description=data.get("description", ""),
        mime_type=data.get("mime_type", ""),
        size_bytes=data.get("size_bytes", 0),
        publicly_visible=data.get("publicly_visible", False),
        derived_from=tuple(data.get("derived_from", ())),
        triggered_by=tuple(data.get("triggered_by", ())),
        attachment_names=tuple(data.get("attachment_names", ())),
        attachment_descriptions=tuple(data.get("attachment_descriptions", ())),
        attachment_sha256s=tuple(data.get("attachment_sha256s", ())),
        attachment_mime_types=tuple(data.get("attachment_mime_types", ())),
        attachment_sizes=tuple(data.get("attachment_sizes", ())),
        summary=data.get("summary", ""),
        metadata_json=data.get("metadata_json", "{}"),
        created_at=created_at,
        version=data.get("version", 1),
    )


def _serialize_log(log: ExecutionLog) -> dict[str, Any]:
    """Serialize ExecutionLog to a JSON-compatible dict."""
    return {
        "node_id": str(log.node_id),
        "deployment_id": str(log.deployment_id),
        "root_deployment_id": str(log.root_deployment_id),
        "flow_id": _serialize_uuid(log.flow_id),
        "task_id": _serialize_uuid(log.task_id),
        "timestamp": _serialize_datetime(log.timestamp),
        "sequence_no": log.sequence_no,
        "level": log.level,
        "category": log.category,
        "logger_name": log.logger_name,
        "message": log.message,
        "event_type": log.event_type,
        "fields": log.fields,
        "exception_text": log.exception_text,
    }


def _deserialize_log(data: dict[str, Any]) -> ExecutionLog:
    """Deserialize a dict to ExecutionLog."""

    def _parse_uuid(v: str | None) -> UUID | None:
        return UUID(v) if v is not None else None

    timestamp_raw = data["timestamp"]
    timestamp = datetime.fromisoformat(timestamp_raw)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)

    return ExecutionLog(
        node_id=UUID(data["node_id"]),
        deployment_id=UUID(data["deployment_id"]),
        root_deployment_id=UUID(data["root_deployment_id"]),
        flow_id=_parse_uuid(data.get("flow_id")),
        task_id=_parse_uuid(data.get("task_id")),
        timestamp=timestamp,
        sequence_no=int(data["sequence_no"]),
        level=data["level"],
        category=data["category"],
        logger_name=data["logger_name"],
        message=data["message"],
        event_type=data.get("event_type", ""),
        fields=data.get("fields", "{}"),
        exception_text=data.get("exception_text", ""),
    )


def _atomic_write_text(path: Path, data: str) -> None:
    """Write text atomically via tempfile + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes atomically via tempfile + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


@contextlib.contextmanager
def _locked_log_file(path: Path, mode: str) -> Any:
    """Open ``logs.jsonl`` under a sidecar filesystem lock shared by all instances.

    Yields:
        Text IO handle opened in the requested mode while the lock is held.
    """
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            with path.open(mode, encoding="utf-8") as handle:
                yield handle
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


class FilesystemDatabase:
    """Filesystem backend implementing DatabaseWriter and DatabaseReader.

    Stores nodes as JSON files in a hierarchical directory structure.
    Maintains in-memory indexes for query support.
    """

    supports_remote = False

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path
        self._base_path.mkdir(parents=True, exist_ok=True)

        # In-memory indexes (built on init, updated on writes)
        self._node_index: dict[UUID, Path] = {}  # node_id -> json file path
        self._children_index: dict[UUID, list[UUID]] = {}  # parent_node_id -> [child_ids]
        self._cache_index: dict[str, list[UUID]] = {}  # cache_key -> node_ids
        self._doc_index: dict[str, Path] = {}  # document_sha256 -> json file path
        self._run_dirs: dict[str, Path] = {}  # run_id -> run directory

        self._rebuild_indexes()

    @property
    def base_path(self) -> Path:
        """Return the root directory for all database files."""
        return self._base_path

    @property
    def _logs_path(self) -> Path:
        """Return the append-only JSONL file for execution logs."""
        return self._base_path / "logs.jsonl"

    # --- DatabaseWriter ---

    async def insert_node(self, node: ExecutionNode) -> None:
        """Insert a new execution node."""
        path = self._node_path(node)
        await asyncio.to_thread(self._insert_node_sync, node, path)
        self._index_node(node, path)

    async def update_node(self, node_id: UUID, **updates: Any) -> None:
        """Update fields on an existing execution node."""
        path = self._node_index.get(node_id)
        if path is None:
            msg = f"Node {node_id} not found in filesystem index"
            raise KeyError(msg)

        if "updated_at" not in updates:
            updates["updated_at"] = datetime.now(UTC)

        old_node, updated_node = await asyncio.to_thread(self._update_node_sync, path, updates)

        # Update cache index: remove stale key, add new key
        if old_node.cache_key and (old_node.cache_key != updated_node.cache_key or updated_node.status != NodeStatus.COMPLETED):
            self._remove_cached_node(old_node.cache_key, node_id)
        if updated_node.cache_key and updated_node.status == NodeStatus.COMPLETED:
            self._add_cached_node(updated_node.cache_key, node_id)

    async def save_document(self, record: DocumentRecord) -> None:
        """Persist a single document record."""
        path = self._document_path(record)
        await asyncio.to_thread(self._save_document_sync, path, record)
        self._doc_index[record.document_sha256] = path

    async def save_document_batch(self, records: list[DocumentRecord]) -> None:
        """Persist multiple document records in one operation."""
        if not records:
            return
        path_records = [(self._document_path(record), record) for record in records]
        await asyncio.to_thread(self._save_document_batch_sync, path_records)
        for path, record in path_records:
            self._doc_index[record.document_sha256] = path

    async def save_blob(self, blob: BlobRecord) -> None:
        """Persist a single binary blob."""
        path = self._blob_path(blob.content_sha256)
        await asyncio.to_thread(self._save_blob_sync, path, blob)

    async def save_blob_batch(self, blobs: list[BlobRecord]) -> None:
        """Persist multiple binary blobs in one operation."""
        if not blobs:
            return
        path_blobs = [(self._blob_path(blob.content_sha256), blob) for blob in blobs]
        await asyncio.to_thread(self._save_blob_batch_sync, path_blobs)

    async def save_logs_batch(self, logs: list[ExecutionLog]) -> None:
        """Persist multiple execution logs in append-only JSONL format."""
        if not logs:
            return
        await asyncio.to_thread(self._save_logs_batch_sync, self._logs_path, logs)

    async def update_document_summary(self, document_sha256: str, summary: str) -> None:
        """Update the summary field of an existing document."""
        path = self._doc_index.get(document_sha256)
        if path is None:
            return
        await asyncio.to_thread(self._update_document_summary_sync, path, summary)

    async def flush(self) -> None:
        """Flush any buffered writes to storage."""

    async def shutdown(self) -> None:
        """Release resources and close connections."""

    # --- DatabaseReader ---

    async def get_node(self, node_id: UUID) -> ExecutionNode | None:
        """Retrieve an execution node by its ID."""
        path = self._node_index.get(node_id)
        if path is None:
            return None
        return await asyncio.to_thread(self._read_node_sync, path)

    async def get_children(self, parent_node_id: UUID) -> list[ExecutionNode]:
        """Retrieve all direct child nodes of a parent node."""
        child_ids = self._children_index.get(parent_node_id, [])
        children: list[ExecutionNode] = []
        for cid in child_ids:
            node = await self.get_node(cid)
            if node is not None:
                children.append(node)
        return sorted(children, key=lambda n: (n.sequence_no, n.node_id))

    async def get_deployment_tree(self, deployment_id: UUID) -> list[ExecutionNode]:
        """Retrieve all nodes belonging to a deployment."""
        return await asyncio.to_thread(self._get_deployment_tree_sync, deployment_id)

    async def get_deployment_by_run_id(self, run_id: str) -> ExecutionNode | None:
        """Find the deployment node for a given run ID."""
        return await asyncio.to_thread(self._get_deployment_by_run_id_sync, run_id)

    async def get_deployment_by_run_scope(self, run_scope: str) -> ExecutionNode | None:
        """Find the deployment node for a given run scope."""
        return await asyncio.to_thread(self._get_deployment_by_run_scope_sync, run_scope)

    async def get_document(self, document_sha256: str) -> DocumentRecord | None:
        """Retrieve a document record by its SHA256."""
        path = self._doc_index.get(document_sha256)
        if path is None:
            return None
        return await asyncio.to_thread(self._read_document_sync, path)

    async def find_document_by_name(self, name: str) -> DocumentRecord | None:
        """Find the newest document with an exact name match."""
        return await asyncio.to_thread(self._find_document_by_name_sync, name)

    async def get_documents_batch(self, sha256s: list[DocumentSha256]) -> dict[DocumentSha256, DocumentRecord]:
        """Retrieve multiple document records by their SHA256s."""
        path_shas = [(sha, self._doc_index[sha]) for sha in sha256s if sha in self._doc_index]
        return await asyncio.to_thread(self._read_documents_batch_sync, path_shas)

    async def get_blob(self, content_sha256: str) -> BlobRecord | None:
        """Retrieve a binary blob by its content SHA256."""
        path = self._blob_path(content_sha256)
        return await asyncio.to_thread(self._read_blob_sync, path, content_sha256)

    async def get_blobs_batch(self, content_sha256s: list[str]) -> dict[str, BlobRecord]:
        """Retrieve multiple binary blobs by their content SHA256s."""
        path_shas = [(sha, self._blob_path(sha)) for sha in content_sha256s]
        return await asyncio.to_thread(self._read_blobs_batch_sync, path_shas)

    async def get_documents_by_deployment(self, deployment_id: UUID) -> list[DocumentRecord]:
        """Retrieve all documents belonging to a deployment chain."""
        return await asyncio.to_thread(self._get_documents_by_deployment_sync, deployment_id)

    async def get_documents_by_node(self, node_id: UUID) -> list[DocumentRecord]:
        """Retrieve all documents produced by a specific node."""
        return await asyncio.to_thread(self._get_documents_by_node_sync, node_id)

    async def get_all_document_shas_for_deployment(self, deployment_id: UUID) -> set[str]:
        """Retrieve all document SHA256s referenced by a deployment's nodes."""
        return await asyncio.to_thread(self._get_all_document_shas_for_deployment_sync, deployment_id)

    async def check_existing_documents(self, sha256s: list[DocumentSha256]) -> set[DocumentSha256]:
        """Return the subset of SHA256s that already exist in storage."""
        return {sha for sha in sha256s if sha in self._doc_index}

    async def find_documents_by_source(self, source_sha256: str) -> list[DocumentRecord]:
        """Find documents derived from a given source SHA256."""
        return await asyncio.to_thread(self._find_documents_by_source_sync, source_sha256)

    async def get_document_ancestry(self, sha256: DocumentSha256) -> dict[str, DocumentRecord]:
        """Return all ancestor documents reachable from derived_from and triggered_by."""
        return await asyncio.to_thread(self._get_document_ancestry_sync, sha256)

    async def find_documents_by_origin(self, sha256: DocumentSha256) -> list[DocumentRecord]:
        """Find documents referencing a SHA256 in derived_from or triggered_by."""
        return await asyncio.to_thread(self._find_documents_by_origin_sync, sha256)

    async def list_run_scopes(self, limit: int) -> list[RunScopeInfo]:
        """List non-empty document run scopes ordered by latest activity."""
        return await asyncio.to_thread(self._list_run_scopes_sync, limit)

    async def search_documents(
        self,
        name: str | None,
        document_type: str | None,
        run_scope: str | None,
        limit: int,
        offset: int,
    ) -> list[DocumentRecord]:
        """Search documents by metadata with pagination."""
        return await asyncio.to_thread(self._search_documents_sync, name, document_type, run_scope, limit, offset)

    async def get_deployment_cost_totals(self, deployment_id: UUID) -> tuple[float, int]:
        """Return total conversation-turn cost and total tokens for a deployment."""
        return await asyncio.to_thread(self._get_deployment_cost_totals_sync, deployment_id)

    async def get_documents_by_run_scope(self, run_scope: str) -> list[DocumentRecord]:
        """Retrieve all documents for a run scope."""
        return await asyncio.to_thread(self._get_documents_by_run_scope_sync, run_scope)

    async def list_deployments(self, limit: int, status: str | None) -> list[ExecutionNode]:
        """List deployment nodes ordered by newest start time first."""
        return await asyncio.to_thread(self._list_deployments_sync, limit, status)

    async def get_cached_completion(self, cache_key: str, max_age: timedelta | None = None) -> ExecutionNode | None:
        """Find a completed node matching the cache key within the max age."""
        candidates: list[ExecutionNode] = []
        for node_id in self._cache_index.get(cache_key, []):
            node = await self.get_node(node_id)
            if node is None or node.status != NodeStatus.COMPLETED or node.cache_key != cache_key:
                continue
            if max_age is not None and (node.ended_at is None or datetime.now(UTC) - node.ended_at > max_age):
                continue
            candidates.append(node)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda node: (
                node.ended_at is not None,
                node.ended_at or datetime.min.replace(tzinfo=UTC),
                node.updated_at,
                node.node_id.hex,
            ),
        )

    async def get_node_logs(
        self,
        node_id: UUID,
        *,
        level: str | None = None,
        category: str | None = None,
    ) -> list[ExecutionLog]:
        """Retrieve execution logs for a specific node."""
        return await asyncio.to_thread(self._read_logs_sync, self._logs_path, node_id=node_id, deployment_id=None, level=level, category=category)

    async def get_deployment_logs(
        self,
        deployment_id: UUID,
        *,
        level: str | None = None,
        category: str | None = None,
    ) -> list[ExecutionLog]:
        """Retrieve execution logs for an entire deployment."""
        return await asyncio.to_thread(
            self._read_logs_sync,
            self._logs_path,
            node_id=None,
            deployment_id=deployment_id,
            level=level,
            category=category,
        )

    # --- Sync I/O helpers (called via asyncio.to_thread) ---

    def _insert_node_sync(self, node: ExecutionNode, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, json.dumps(_serialize_node(node), indent=2, default=str))

    def _update_node_sync(self, path: Path, updates: dict[str, Any]) -> tuple[ExecutionNode, ExecutionNode]:
        """Read node, apply updates, write back. Returns (old_node, updated_node)."""
        data = json.loads(path.read_text(encoding="utf-8"))
        old_node = _deserialize_node(data)
        updates = dict(updates)
        updates.setdefault("version", old_node.version + 1)

        # Convert enums to values for serialization
        serialized_updates: dict[str, Any] = {}
        for k, v in updates.items():
            if isinstance(v, (NodeStatus, NodeKind)):
                serialized_updates[k] = v.value
            elif isinstance(v, UUID):
                serialized_updates[k] = str(v)
            elif isinstance(v, datetime):
                serialized_updates[k] = v.isoformat()
            elif isinstance(v, (tuple, list)):
                serialized_updates[k] = _as_json_list(v)
            else:
                serialized_updates[k] = v

        data.update(serialized_updates)
        _atomic_write_text(path, json.dumps(data, indent=2, default=str))

        updated_node = replace(old_node, **updates)
        return old_node, updated_node

    def _save_document_sync(self, path: Path, record: DocumentRecord) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, json.dumps(_serialize_document(record), indent=2, default=str))

    def _save_document_batch_sync(self, path_records: list[tuple[Path, DocumentRecord]]) -> None:
        for path, record in path_records:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(path, json.dumps(_serialize_document(record), indent=2, default=str))

    def _save_blob_sync(self, path: Path, blob: BlobRecord) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(path, blob.content)

    def _save_blob_batch_sync(self, path_blobs: list[tuple[Path, BlobRecord]]) -> None:
        for path, blob in path_blobs:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_bytes(path, blob.content)

    def _save_logs_batch_sync(self, path: Path, logs: list[ExecutionLog]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _locked_log_file(path, "a") as handle:
            for log in logs:
                handle.write(json.dumps(_serialize_log(log), default=str))
                handle.write("\n")

    def _update_document_summary_sync(self, path: Path, summary: str) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        data["summary"] = summary
        data["version"] = data.get("version", 1) + 1
        _atomic_write_text(path, json.dumps(data, indent=2, default=str))

    def _read_node_sync(self, path: Path) -> ExecutionNode | None:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return _deserialize_node(data)

    def _read_document_sync(self, path: Path) -> DocumentRecord | None:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return _deserialize_document(data)

    def _read_documents_batch_sync(self, path_shas: list[tuple[DocumentSha256, Path]]) -> dict[DocumentSha256, DocumentRecord]:
        result: dict[DocumentSha256, DocumentRecord] = {}
        for sha, path in path_shas:
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            result[sha] = _deserialize_document(data)
        return result

    def _read_logs_sync(
        self,
        path: Path,
        *,
        node_id: UUID | None,
        deployment_id: UUID | None,
        level: str | None,
        category: str | None,
    ) -> list[ExecutionLog]:
        if not path.exists():
            return []

        logs: list[ExecutionLog] = []
        with _locked_log_file(path, "r") as handle:
            for line in handle:
                if not line.strip():
                    continue
                log = _deserialize_log(json.loads(line))
                if node_id is not None and log.node_id != node_id:
                    continue
                if deployment_id is not None and log.deployment_id != deployment_id:
                    continue
                if level is not None and log.level != level:
                    continue
                if category is not None and log.category != category:
                    continue
                logs.append(log)

        if node_id is not None:
            return sorted(logs, key=lambda log: (log.sequence_no, log.timestamp))
        return sorted(logs, key=lambda log: (log.timestamp, log.sequence_no, str(log.node_id)))

    def _read_blob_sync(self, path: Path, content_sha256: str) -> BlobRecord | None:
        if not path.exists():
            return None
        content = path.read_bytes()
        return BlobRecord(content_sha256=content_sha256, content=content, size_bytes=len(content))

    def _read_blobs_batch_sync(self, path_shas: list[tuple[str, Path]]) -> dict[str, BlobRecord]:
        result: dict[str, BlobRecord] = {}
        for sha, path in path_shas:
            if not path.exists():
                continue
            content = path.read_bytes()
            result[sha] = BlobRecord(content_sha256=sha, content=content, size_bytes=len(content))
        return result

    def _get_deployment_tree_sync(self, deployment_id: UUID) -> list[ExecutionNode]:
        nodes: list[ExecutionNode] = []
        for path in self._node_index.values():
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if UUID(data["deployment_id"]) == deployment_id:
                nodes.append(_deserialize_node(data))
        return sorted(nodes, key=lambda n: (n.sequence_no, n.node_id))

    def _get_deployment_by_run_id_sync(self, run_id: str) -> ExecutionNode | None:
        for path in self._node_index.values():
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("run_id") == run_id and data.get("node_kind") == NodeKind.DEPLOYMENT.value:
                return _deserialize_node(data)
        return None

    def _get_deployment_by_run_scope_sync(self, run_scope: str) -> ExecutionNode | None:
        for path in self._node_index.values():
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("run_scope") == run_scope and data.get("node_kind") == NodeKind.DEPLOYMENT.value:
                return _deserialize_node(data)
        return None

    def _get_documents_by_deployment_sync(self, deployment_id: UUID) -> list[DocumentRecord]:
        deployment_ids = self._deployment_chain_ids_sync(deployment_id)
        results: list[DocumentRecord] = []
        for path in self._doc_index.values():
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if UUID(data["deployment_id"]) in deployment_ids:
                results.append(_deserialize_document(data))
        return results

    def _deployment_chain_ids_sync(self, deployment_id: UUID) -> set[UUID]:
        deployment_path = self._node_index.get(deployment_id)
        if deployment_path is None or not deployment_path.exists():
            return {deployment_id}

        data = json.loads(deployment_path.read_text(encoding="utf-8"))
        if data.get("node_kind") != NodeKind.DEPLOYMENT.value:
            return {deployment_id}

        root_deployment_id = UUID(data["root_deployment_id"])
        deployment_ids: set[UUID] = set()
        for path in self._node_index.values():
            if not path.exists():
                continue
            node_data = json.loads(path.read_text(encoding="utf-8"))
            if node_data.get("node_kind") != NodeKind.DEPLOYMENT.value:
                continue
            if UUID(node_data["root_deployment_id"]) == root_deployment_id:
                deployment_ids.add(UUID(node_data["deployment_id"]))
        return deployment_ids or {deployment_id}

    def _get_documents_by_node_sync(self, node_id: UUID) -> list[DocumentRecord]:
        results: list[DocumentRecord] = []
        for path in self._doc_index.values():
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            producing = data.get("producing_node_id")
            if producing is not None and UUID(producing) == node_id:
                results.append(_deserialize_document(data))

        # For deployment nodes, include root input documents
        node_path = self._node_index.get(node_id)
        if node_path is not None and node_path.exists():
            node_data = json.loads(node_path.read_text(encoding="utf-8"))
            if node_data.get("node_kind") == NodeKind.DEPLOYMENT.value:
                deploy_id = UUID(node_data["deployment_id"])
                for path in self._doc_index.values():
                    if not path.exists():
                        continue
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if data.get("producing_node_id") is None and UUID(data["deployment_id"]) == deploy_id:
                        results.append(_deserialize_document(data))
        return results

    def _get_all_document_shas_for_deployment_sync(self, deployment_id: UUID) -> set[str]:
        shas: set[str] = set()
        for path in self._node_index.values():
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if UUID(data["deployment_id"]) == deployment_id:
                shas.update(data.get("input_document_shas", []))
                shas.update(data.get("output_document_shas", []))
                shas.update(data.get("context_document_shas", []))
        return shas

    def _find_documents_by_source_sync(self, source_sha256: str) -> list[DocumentRecord]:
        results: list[DocumentRecord] = []
        for doc in self._read_all_documents_sync():
            if source_sha256 in doc.derived_from:
                results.append(doc)
        return results

    def _find_document_by_name_sync(self, name: str) -> DocumentRecord | None:
        matches = [doc for doc in self._read_all_documents_sync() if doc.name == name]
        if not matches:
            return None
        return max(matches, key=lambda doc: (doc.created_at, doc.document_sha256))

    def _get_document_ancestry_sync(self, sha256: DocumentSha256) -> dict[str, DocumentRecord]:
        all_docs: dict[str, DocumentRecord] = {doc.document_sha256: doc for doc in self._read_all_documents_sync()}
        target = all_docs.get(sha256)
        if target is None:
            return {}

        ancestors: dict[str, DocumentRecord] = {}
        pending = list(target.derived_from) + list(target.triggered_by)
        seen = set(pending)

        while pending:
            current_sha = pending.pop(0)
            current = all_docs.get(current_sha)
            if current is None:
                continue
            ancestors[current_sha] = current
            for parent_sha in (*current.derived_from, *current.triggered_by):
                if parent_sha in seen:
                    continue
                seen.add(parent_sha)
                pending.append(parent_sha)

        return ancestors

    def _find_documents_by_origin_sync(self, sha256: DocumentSha256) -> list[DocumentRecord]:
        matches = [doc for doc in self._read_all_documents_sync() if sha256 in doc.derived_from or sha256 in doc.triggered_by]
        return sorted(matches, key=lambda doc: (doc.created_at, doc.document_sha256), reverse=True)

    def _list_run_scopes_sync(self, limit: int) -> list[RunScopeInfo]:
        grouped: dict[RunScope, RunScopeInfo] = {}
        for doc in self._read_all_documents_sync():
            if not doc.run_scope:
                continue
            current = grouped.get(doc.run_scope)
            if current is None:
                grouped[doc.run_scope] = RunScopeInfo(
                    run_scope=doc.run_scope,
                    document_count=1,
                    latest_created_at=doc.created_at,
                )
                continue
            grouped[doc.run_scope] = RunScopeInfo(
                run_scope=doc.run_scope,
                document_count=current.document_count + 1,
                latest_created_at=max(current.latest_created_at, doc.created_at),
            )

        ordered = sorted(grouped.values(), key=lambda info: (info.latest_created_at, info.run_scope), reverse=True)
        return ordered[:limit]

    def _search_documents_sync(
        self,
        name: str | None,
        document_type: str | None,
        run_scope: str | None,
        limit: int,
        offset: int,
    ) -> list[DocumentRecord]:
        normalized_name = name.lower() if name is not None else None
        matches: list[DocumentRecord] = []
        for doc in self._read_all_documents_sync():
            if normalized_name is not None and normalized_name not in doc.name.lower():
                continue
            if document_type is not None and doc.document_type != document_type:
                continue
            if run_scope is not None and doc.run_scope != run_scope:
                continue
            matches.append(doc)
        ordered = sorted(matches, key=lambda doc: (doc.created_at, doc.document_sha256), reverse=True)
        return ordered[offset : offset + limit]

    def _get_deployment_cost_totals_sync(self, deployment_id: UUID) -> tuple[float, int]:
        total_cost = 0.0
        total_tokens = 0
        for node in self._read_all_nodes_sync():
            if node.deployment_id != deployment_id or node.node_kind != NodeKind.CONVERSATION_TURN:
                continue
            total_cost += node.cost_usd
            total_tokens += node.tokens_input + node.tokens_output
        return total_cost, total_tokens

    def _get_documents_by_run_scope_sync(self, run_scope: str) -> list[DocumentRecord]:
        matches = [doc for doc in self._read_all_documents_sync() if doc.run_scope == run_scope]
        return sorted(matches, key=lambda doc: (doc.created_at, doc.document_sha256), reverse=True)

    def _list_deployments_sync(self, limit: int, status: str | None) -> list[ExecutionNode]:
        matches = [node for node in self._read_all_nodes_sync() if node.node_kind == NodeKind.DEPLOYMENT and (status is None or node.status.value == status)]
        ordered = sorted(matches, key=lambda node: (node.started_at, node.node_id), reverse=True)
        return ordered[:limit]

    def _read_all_documents_sync(self) -> list[DocumentRecord]:
        docs: list[DocumentRecord] = []
        for path in self._doc_index.values():
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            docs.append(_deserialize_document(data))
        return docs

    def _read_all_nodes_sync(self) -> list[ExecutionNode]:
        nodes: list[ExecutionNode] = []
        for path in self._node_index.values():
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            nodes.append(_deserialize_node(data))
        return nodes

    # --- Path computation ---

    def _run_dir_for_node(self, node: ExecutionNode) -> Path:
        """Get or create the run directory for a deployment."""
        if node.run_id in self._run_dirs:
            return self._run_dirs[node.run_id]

        date_str = node.started_at.strftime("%Y%m%d")
        safe_name = _sanitize_dir_name(node.deployment_name)
        run_short = node.run_id[:_RUN_ID_SHORT_LENGTH]
        dir_name = f"{date_str}_{safe_name}_{run_short}"
        run_dir = self._base_path / "runs" / dir_name
        self._run_dirs[node.run_id] = run_dir
        return run_dir

    def _node_path(self, node: ExecutionNode) -> Path:
        """Compute the file path for a node based on its kind and hierarchy."""
        run_dir = self._run_dir_for_node(node)
        id_short = node.node_id.hex[:_NODE_ID_SHORT_LENGTH]

        if node.node_kind == NodeKind.DEPLOYMENT:
            return run_dir / "deployment.json"

        if node.node_kind == NodeKind.FLOW:
            seq = f"{node.sequence_no:02d}"
            safe = _sanitize_dir_name(node.name)
            return run_dir / "flows" / f"{seq}_{safe}_{id_short}" / "flow.json"

        if node.node_kind == NodeKind.TASK:
            parent_dir = self._find_parent_dir(node.parent_node_id, run_dir)
            seq = f"{node.sequence_no:02d}"
            safe = _sanitize_dir_name(node.name)
            return parent_dir / "tasks" / f"{seq}_{safe}_{id_short}" / "task.json"

        if node.node_kind == NodeKind.CONVERSATION:
            parent_dir = self._find_parent_dir(node.parent_node_id, run_dir)
            seq = f"{node.sequence_no:02d}"
            safe = _sanitize_dir_name(node.name)
            return parent_dir / "conversations" / f"{seq}_{safe}_{id_short}" / "conversation.json"

        if node.node_kind == NodeKind.CONVERSATION_TURN:
            parent_dir = self._find_parent_dir(node.parent_node_id, run_dir)
            seq = f"{node.sequence_no:02d}"
            return parent_dir / "turns" / f"{seq}.json"

        return run_dir / f"{node.node_id}.json"

    def _find_parent_dir(self, parent_node_id: UUID, run_dir: Path) -> Path:
        """Find the directory of a parent node. Falls back to run_dir."""
        if parent_node_id == NULL_PARENT:
            return run_dir
        parent_path = self._node_index.get(parent_node_id)
        if parent_path is not None:
            return parent_path.parent
        return run_dir

    def _document_path(self, record: DocumentRecord) -> Path:
        """Compute path for a document record."""
        sha_short = record.document_sha256[:6]
        safe_type = _sanitize_dir_name(record.document_type)
        filename = f"{safe_type}_{sha_short}.json"

        if record.producing_node_id is not None:
            # Task-produced document: store under task's documents/
            node_path = self._node_index.get(record.producing_node_id)
            if node_path is not None:
                return node_path.parent / "documents" / filename

        # Root input or orphan: use run-level documents/ via deployment node
        for path in self._node_index.values():
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if UUID(data["deployment_id"]) == record.deployment_id and data.get("node_kind") == NodeKind.DEPLOYMENT.value:
                    return path.parent / "documents" / filename
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

        # Fallback: top-level documents directory
        return self._base_path / "documents" / filename

    def _blob_path(self, content_sha256: str) -> Path:
        """Content-addressed blob path: blobs/{prefix2}/{sha256}."""
        prefix = content_sha256[:2]
        return self._base_path / "blobs" / prefix / content_sha256

    # --- Index management ---

    def _index_node(self, node: ExecutionNode, path: Path) -> None:
        """Add a node to in-memory indexes."""
        self._node_index[node.node_id] = path

        if node.parent_node_id != NULL_PARENT:
            self._children_index.setdefault(node.parent_node_id, [])
            if node.node_id not in self._children_index[node.parent_node_id]:
                self._children_index[node.parent_node_id].append(node.node_id)

        if node.cache_key and node.status == NodeStatus.COMPLETED:
            self._add_cached_node(node.cache_key, node.node_id)

        if node.node_kind == NodeKind.DEPLOYMENT:
            self._run_dirs[node.run_id] = path.parent

    def _rebuild_indexes(self) -> None:
        """Rebuild all in-memory indexes by scanning the runs/ directory."""
        self._node_index.clear()
        self._children_index.clear()
        self._cache_index.clear()
        self._doc_index.clear()
        self._run_dirs.clear()

        runs_dir = self._base_path / "runs"
        if not runs_dir.exists():
            return

        # Scan all JSON files for nodes
        for json_path in runs_dir.rglob("*.json"):
            if json_path.name.startswith("."):
                continue
            # Skip document files (in documents/ subdirs)
            if json_path.parent.name == "documents":
                self._index_document_file(json_path)
                continue
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                if "node_id" in data and "node_kind" in data:
                    node = _deserialize_node(data)
                    self._index_node(node, json_path)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.debug("Skipping malformed node file %s: %s", json_path, e)

        top_level_documents_dir = self._base_path / "documents"
        if not top_level_documents_dir.exists():
            return
        for json_path in top_level_documents_dir.rglob("*.json"):
            if json_path.name.startswith("."):
                continue
            self._index_document_file(json_path)

    def _index_document_file(self, path: Path) -> None:
        """Index a document JSON file."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sha = data.get("document_sha256")
            if sha:
                self._doc_index[sha] = path
        except (json.JSONDecodeError, KeyError) as exc:
            logger.debug("Skipping malformed document file %s: %s", path, exc)

    def _add_cached_node(self, cache_key: str, node_id: UUID) -> None:
        """Add a node id to the cache-key index without duplicating it."""
        node_ids = self._cache_index.setdefault(cache_key, [])
        if node_id not in node_ids:
            node_ids.append(node_id)

    def _remove_cached_node(self, cache_key: str, node_id: UUID) -> None:
        """Remove a node id from the cache-key index and delete empty keys."""
        node_ids = self._cache_index.get(cache_key)
        if not node_ids:
            return
        filtered_node_ids = [existing_id for existing_id in node_ids if existing_id != node_id]
        if filtered_node_ids:
            self._cache_index[cache_key] = filtered_node_ids
            return
        del self._cache_index[cache_key]
