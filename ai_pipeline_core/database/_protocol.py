"""Database read/write protocols for execution DAG and document storage."""

from datetime import timedelta
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from ai_pipeline_core.database._types import BlobRecord, DocumentRecord, ExecutionLog, ExecutionNode, RunScopeInfo
from ai_pipeline_core.documents._context import DocumentSha256, RunScope

__all__ = [
    "DatabaseReader",
    "DatabaseWriter",
]


@runtime_checkable
class DatabaseWriter(Protocol):
    """Write protocol for framework-internal use.

    Lifecycle-oriented methods for persisting execution nodes, documents, and blobs.
    """

    @property
    def supports_remote(self) -> bool:
        """Whether this backend supports Prefect-based remote deployment execution."""
        ...

    async def insert_node(self, node: ExecutionNode) -> None:
        """Insert a new execution node."""
        ...

    async def update_node(self, node_id: UUID, **updates: Any) -> None:
        """Update fields on an existing execution node."""
        ...

    async def save_document(self, record: DocumentRecord) -> None:
        """Persist a single document record."""
        ...

    async def save_document_batch(self, records: list[DocumentRecord]) -> None:
        """Persist multiple document records in one operation."""
        ...

    async def save_blob(self, blob: BlobRecord) -> None:
        """Persist a single binary blob."""
        ...

    async def save_blob_batch(self, blobs: list[BlobRecord]) -> None:
        """Persist multiple binary blobs in one operation."""
        ...

    async def save_logs_batch(self, logs: list[ExecutionLog]) -> None:
        """Persist multiple execution logs in one operation."""
        ...

    async def update_document_summary(self, document_sha256: DocumentSha256, summary: str) -> None:
        """Update the summary field of an existing document."""
        ...

    async def flush(self) -> None:
        """Flush any buffered writes to storage."""
        ...

    async def shutdown(self) -> None:
        """Release resources and close connections."""
        ...


class _ExecutionNodeReader(Protocol):
    """Read protocol for execution nodes and deployment metadata."""

    async def get_node(self, node_id: UUID) -> ExecutionNode | None:
        """Retrieve an execution node by its ID."""
        ...

    async def get_children(self, parent_node_id: UUID) -> list[ExecutionNode]:
        """Retrieve all direct child nodes of a parent node."""
        ...

    async def get_deployment_tree(self, deployment_id: UUID) -> list[ExecutionNode]:
        """Retrieve all nodes belonging to a deployment."""
        ...

    async def get_deployment_by_run_id(self, run_id: str) -> ExecutionNode | None:
        """Find the deployment node for a given run ID."""
        ...

    async def get_deployment_by_run_scope(self, run_scope: RunScope) -> ExecutionNode | None:
        """Find the deployment node for a given run scope."""
        ...

    async def get_cached_completion(self, cache_key: str, max_age: timedelta | None = None) -> ExecutionNode | None:
        """Find a completed node matching the cache key within the max age."""
        ...

    async def list_deployments(self, limit: int, status: str | None) -> list[ExecutionNode]:
        """List deployment nodes ordered by newest start time first."""
        ...

    async def get_deployment_cost_totals(self, deployment_id: UUID) -> tuple[float, int]:
        """Return total conversation-turn cost and total tokens for a deployment."""
        ...


class _DocumentBlobReader(Protocol):
    """Read protocol for document and blob lookup operations."""

    async def get_document(self, document_sha256: DocumentSha256) -> DocumentRecord | None:
        """Retrieve a document record by its SHA256."""
        ...

    async def find_document_by_name(self, name: str) -> DocumentRecord | None:
        """Find the newest document with an exact name match."""
        ...

    async def get_documents_batch(self, sha256s: list[DocumentSha256]) -> dict[DocumentSha256, DocumentRecord]:
        """Retrieve multiple document records by their SHA256s."""
        ...

    async def get_blob(self, content_sha256: str) -> BlobRecord | None:
        """Retrieve a binary blob by its content SHA256."""
        ...

    async def get_blobs_batch(self, content_sha256s: list[str]) -> dict[str, BlobRecord]:
        """Retrieve multiple binary blobs by their content SHA256s."""
        ...

    async def get_documents_by_deployment(self, deployment_id: UUID) -> list[DocumentRecord]:
        """Retrieve all documents belonging to a deployment chain."""
        ...

    async def get_documents_by_node(self, node_id: UUID) -> list[DocumentRecord]:
        """Retrieve all documents produced by a specific node."""
        ...

    async def get_all_document_shas_for_deployment(self, deployment_id: UUID) -> set[str]:
        """Retrieve all document SHA256s referenced by a deployment's nodes."""
        ...

    async def check_existing_documents(self, sha256s: list[DocumentSha256]) -> set[DocumentSha256]:
        """Return the subset of SHA256s that already exist in storage."""
        ...

    async def find_documents_by_source(self, source_sha256: DocumentSha256) -> list[DocumentRecord]:
        """Find documents derived from a given source SHA256."""
        ...

    async def get_document_ancestry(self, sha256: DocumentSha256) -> dict[str, DocumentRecord]:
        """Return all ancestor documents reachable from derived_from and triggered_by."""
        ...

    async def find_documents_by_origin(self, sha256: DocumentSha256) -> list[DocumentRecord]:
        """Find documents that reference a SHA256 in derived_from or triggered_by."""
        ...

    async def list_run_scopes(self, limit: int) -> list[RunScopeInfo]:
        """List non-empty document run scopes ordered by latest activity."""
        ...

    async def search_documents(
        self,
        name: str | None,
        document_type: str | None,
        run_scope: str | None,
        limit: int,
        offset: int,
    ) -> list[DocumentRecord]:
        """Search documents by metadata with pagination."""
        ...

    async def get_documents_by_run_scope(self, run_scope: str) -> list[DocumentRecord]:
        """Retrieve all documents for a run scope."""
        ...


class _ExecutionLogReader(Protocol):
    """Read protocol for execution logs."""

    async def get_node_logs(
        self,
        node_id: UUID,
        *,
        level: str | None = None,
        category: str | None = None,
    ) -> list[ExecutionLog]:
        """Retrieve execution logs for a specific node."""
        ...

    async def get_deployment_logs(
        self,
        deployment_id: UUID,
        *,
        level: str | None = None,
        category: str | None = None,
    ) -> list[ExecutionLog]:
        """Retrieve execution logs for an entire deployment."""
        ...


@runtime_checkable
class DatabaseReader(
    _ExecutionNodeReader,
    _DocumentBlobReader,
    _ExecutionLogReader,
    Protocol,
):
    """Combined read protocol for execution nodes, documents, blobs, and logs."""
