"""In-memory database backend for testing.

Dict-based storage implementing both DatabaseWriter and DatabaseReader protocols.
All data is lost when the process exits.
"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from ai_pipeline_core.database._types import BlobRecord, DocumentRecord, ExecutionLog, ExecutionNode, NodeKind, NodeStatus, RunScopeInfo
from ai_pipeline_core.documents._context import DocumentSha256, RunScope

__all__ = [
    "MemoryDatabase",
]


def _document_sort_key(document: DocumentRecord) -> tuple[datetime, str]:
    """Sort documents by creation time, then SHA, newest first when reversed."""
    return document.created_at, document.document_sha256


class MemoryDatabase:
    """Dict-based database backend for unit tests.

    Implements both DatabaseWriter and DatabaseReader protocols.
    """

    supports_remote = False

    def __init__(self) -> None:
        self._nodes: dict[UUID, ExecutionNode] = {}
        self._documents: dict[str, DocumentRecord] = {}
        self._blobs: dict[str, BlobRecord] = {}
        self._logs: list[ExecutionLog] = []

    # --- DatabaseWriter ---

    async def insert_node(self, node: ExecutionNode) -> None:
        """Insert a new execution node."""
        self._nodes[node.node_id] = node

    async def update_node(self, node_id: UUID, **updates: Any) -> None:
        """Update fields on an existing execution node."""
        existing = self._nodes.get(node_id)
        if existing is None:
            msg = f"Node {node_id} not found"
            raise KeyError(msg)
        updates = dict(updates)
        if "updated_at" not in updates:
            updates["updated_at"] = datetime.now(UTC)
        if "version" not in updates:
            updates["version"] = existing.version + 1
        self._nodes[node_id] = replace(existing, **updates)

    async def save_document(self, record: DocumentRecord) -> None:
        """Persist a single document record."""
        self._documents[record.document_sha256] = record

    async def save_document_batch(self, records: list[DocumentRecord]) -> None:
        """Persist multiple document records in one operation."""
        for record in records:
            self._documents[record.document_sha256] = record

    async def save_blob(self, blob: BlobRecord) -> None:
        """Persist a single binary blob."""
        self._blobs[blob.content_sha256] = blob

    async def save_blob_batch(self, blobs: list[BlobRecord]) -> None:
        """Persist multiple binary blobs in one operation."""
        for blob in blobs:
            self._blobs[blob.content_sha256] = blob

    async def save_logs_batch(self, logs: list[ExecutionLog]) -> None:
        """Persist multiple execution logs in one operation."""
        self._logs.extend(logs)

    async def update_document_summary(self, document_sha256: str, summary: str) -> None:
        """Update the summary field of an existing document."""
        existing = self._documents.get(document_sha256)
        if existing is None:
            return
        self._documents[document_sha256] = replace(existing, summary=summary, version=existing.version + 1)

    async def flush(self) -> None:
        """Flush any buffered writes to storage."""

    async def shutdown(self) -> None:
        """Release resources and close connections."""

    # --- DatabaseReader ---

    async def get_node(self, node_id: UUID) -> ExecutionNode | None:
        """Retrieve an execution node by its ID."""
        return self._nodes.get(node_id)

    async def get_children(self, parent_node_id: UUID) -> list[ExecutionNode]:
        """Retrieve all direct child nodes of a parent node."""
        return sorted(
            (n for n in self._nodes.values() if n.parent_node_id == parent_node_id),
            key=lambda n: (n.sequence_no, n.node_id),
        )

    async def get_deployment_tree(self, deployment_id: UUID) -> list[ExecutionNode]:
        """Retrieve all nodes belonging to a deployment."""
        return sorted(
            (n for n in self._nodes.values() if n.deployment_id == deployment_id),
            key=lambda n: (n.sequence_no, n.node_id),
        )

    async def get_deployment_by_run_id(self, run_id: str) -> ExecutionNode | None:
        """Find the deployment node for a given run ID."""
        for node in self._nodes.values():
            if node.node_kind == NodeKind.DEPLOYMENT and node.run_id == run_id:
                return node
        return None

    async def get_deployment_by_run_scope(self, run_scope: str) -> ExecutionNode | None:
        """Find the deployment node for a given run scope."""
        for node in self._nodes.values():
            if node.node_kind == NodeKind.DEPLOYMENT and node.run_scope == run_scope:
                return node
        return None

    async def get_document(self, document_sha256: str) -> DocumentRecord | None:
        """Retrieve a document record by its SHA256."""
        return self._documents.get(document_sha256)

    async def find_document_by_name(self, name: str) -> DocumentRecord | None:
        """Find the newest document with an exact name match."""
        matches = [doc for doc in self._documents.values() if doc.name == name]
        if not matches:
            return None
        return max(matches, key=lambda doc: (doc.created_at, doc.document_sha256))

    async def get_documents_batch(self, sha256s: list[DocumentSha256]) -> dict[DocumentSha256, DocumentRecord]:
        """Retrieve multiple document records by their SHA256s."""
        return {sha: self._documents[sha] for sha in sha256s if sha in self._documents}

    async def get_blob(self, content_sha256: str) -> BlobRecord | None:
        """Retrieve a binary blob by its content SHA256."""
        return self._blobs.get(content_sha256)

    async def get_blobs_batch(self, content_sha256s: list[str]) -> dict[str, BlobRecord]:
        """Retrieve multiple binary blobs by their content SHA256s."""
        return {sha: self._blobs[sha] for sha in content_sha256s if sha in self._blobs}

    async def get_documents_by_deployment(self, deployment_id: UUID) -> list[DocumentRecord]:
        """Retrieve all documents belonging to a deployment chain."""
        deployment_ids = self._deployment_chain_ids(deployment_id)
        return [doc for doc in self._documents.values() if doc.deployment_id in deployment_ids]

    async def get_documents_by_node(self, node_id: UUID) -> list[DocumentRecord]:
        """Retrieve all documents produced by a specific node."""
        result: list[DocumentRecord] = []
        for doc in self._documents.values():
            if doc.producing_node_id == node_id:
                result.append(doc)
        # For deployment nodes, also return root input documents (producing_node_id is None)
        node = self._nodes.get(node_id)
        if node is not None and node.node_kind == NodeKind.DEPLOYMENT:
            for doc in self._documents.values():
                if doc.producing_node_id is None and doc.deployment_id == node.deployment_id:
                    result.append(doc)
        return result

    async def get_all_document_shas_for_deployment(self, deployment_id: UUID) -> set[str]:
        """Retrieve all document SHA256s referenced by a deployment's nodes."""
        shas: set[str] = set()
        for node in self._nodes.values():
            if node.deployment_id == deployment_id:
                shas.update(node.input_document_shas)
                shas.update(node.output_document_shas)
                shas.update(node.context_document_shas)
        return shas

    async def check_existing_documents(self, sha256s: list[DocumentSha256]) -> set[DocumentSha256]:
        """Return the subset of SHA256s that already exist in storage."""
        return {sha for sha in sha256s if sha in self._documents}

    async def find_documents_by_source(self, source_sha256: str) -> list[DocumentRecord]:
        """Find documents derived from a given source SHA256."""
        return [doc for doc in self._documents.values() if source_sha256 in doc.derived_from]

    async def get_document_ancestry(self, sha256: DocumentSha256) -> dict[str, DocumentRecord]:
        """Return all ancestor documents reachable from derived_from and triggered_by."""
        target = self._documents.get(sha256)
        if target is None:
            return {}

        ancestors: dict[str, DocumentRecord] = {}
        pending = list(target.derived_from) + list(target.triggered_by)
        seen = set(pending)

        while pending:
            current_sha = pending.pop(0)
            current = self._documents.get(current_sha)
            if current is None:
                continue
            ancestors[current_sha] = current
            for parent_sha in (*current.derived_from, *current.triggered_by):
                if parent_sha in seen:
                    continue
                seen.add(parent_sha)
                pending.append(parent_sha)

        return ancestors

    def _deployment_chain_ids(self, deployment_id: UUID) -> set[UUID]:
        deployment_node = self._nodes.get(deployment_id)
        if deployment_node is None or deployment_node.node_kind != NodeKind.DEPLOYMENT:
            return {deployment_id}

        root_deployment_id = deployment_node.root_deployment_id
        deployment_nodes = (node for node in self._nodes.values() if node.node_kind == NodeKind.DEPLOYMENT)
        chain_ids = {node.deployment_id for node in deployment_nodes if node.root_deployment_id == root_deployment_id}
        return chain_ids or {deployment_id}

    async def find_documents_by_origin(self, sha256: DocumentSha256) -> list[DocumentRecord]:
        """Find documents referencing a SHA256 in derived_from or triggered_by."""
        return sorted(
            [doc for doc in self._documents.values() if sha256 in doc.derived_from or sha256 in doc.triggered_by],
            key=lambda doc: (doc.created_at, doc.document_sha256),
            reverse=True,
        )

    async def list_run_scopes(self, limit: int) -> list[RunScopeInfo]:
        """List non-empty document run scopes ordered by latest activity."""
        grouped: dict[RunScope, RunScopeInfo] = {}
        for doc in self._documents.values():
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

        ordered = sorted(
            grouped.values(),
            key=lambda info: (info.latest_created_at, info.run_scope),
            reverse=True,
        )
        return ordered[:limit]

    async def search_documents(
        self,
        name: str | None,
        document_type: str | None,
        run_scope: str | None,
        limit: int,
        offset: int,
    ) -> list[DocumentRecord]:
        """Search documents by metadata with pagination."""
        matches: list[DocumentRecord] = []
        normalized_name = name.lower() if name is not None else None
        for doc in self._documents.values():
            if normalized_name is not None and normalized_name not in doc.name.lower():
                continue
            if document_type is not None and doc.document_type != document_type:
                continue
            if run_scope is not None and doc.run_scope != run_scope:
                continue
            matches.append(doc)

        ordered = sorted(matches, key=_document_sort_key, reverse=True)
        return ordered[offset : offset + limit]

    async def get_deployment_cost_totals(self, deployment_id: UUID) -> tuple[float, int]:
        """Return total conversation-turn cost and total tokens for a deployment."""
        total_cost = 0.0
        total_tokens = 0
        for node in self._nodes.values():
            if node.deployment_id != deployment_id or node.node_kind != NodeKind.CONVERSATION_TURN:
                continue
            total_cost += node.cost_usd
            total_tokens += node.tokens_input + node.tokens_output
        return total_cost, total_tokens

    async def get_documents_by_run_scope(self, run_scope: str) -> list[DocumentRecord]:
        """Retrieve all documents for a run scope."""
        return sorted(
            [doc for doc in self._documents.values() if doc.run_scope == run_scope],
            key=lambda doc: (doc.created_at, doc.document_sha256),
            reverse=True,
        )

    async def list_deployments(self, limit: int, status: str | None) -> list[ExecutionNode]:
        """List deployment nodes ordered by newest start time first."""
        matches = [node for node in self._nodes.values() if node.node_kind == NodeKind.DEPLOYMENT and (status is None or node.status.value == status)]
        ordered = sorted(matches, key=lambda node: (node.started_at, node.node_id), reverse=True)
        return ordered[:limit]

    async def get_cached_completion(self, cache_key: str, max_age: timedelta | None = None) -> ExecutionNode | None:
        """Find a completed node matching the cache key within the max age."""
        now = datetime.now(UTC)
        best: ExecutionNode | None = None
        for node in self._nodes.values():
            if node.cache_key != cache_key:
                continue
            if node.status != NodeStatus.COMPLETED:
                continue
            if max_age is not None:
                if node.ended_at is None:
                    continue
                if now - node.ended_at > max_age:
                    continue
            if best is None or (node.ended_at is not None and (best.ended_at is None or node.ended_at > best.ended_at)):
                best = node
        return best

    async def get_node_logs(
        self,
        node_id: UUID,
        *,
        level: str | None = None,
        category: str | None = None,
    ) -> list[ExecutionLog]:
        """Retrieve execution logs for a specific node."""
        return sorted(
            (log for log in self._logs if log.node_id == node_id and (level is None or log.level == level) and (category is None or log.category == category)),
            key=lambda log: (log.sequence_no, log.timestamp),
        )

    async def get_deployment_logs(
        self,
        deployment_id: UUID,
        *,
        level: str | None = None,
        category: str | None = None,
    ) -> list[ExecutionLog]:
        """Retrieve execution logs for an entire deployment."""
        return sorted(
            (
                log
                for log in self._logs
                if log.deployment_id == deployment_id and (level is None or log.level == level) and (category is None or log.category == category)
            ),
            key=lambda log: (log.timestamp, log.sequence_no, str(log.node_id)),
        )
