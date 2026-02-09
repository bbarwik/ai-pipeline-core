"""Document store protocol and singleton management.

Defines the DocumentStore protocol that all storage backends must implement,
along with get/set helpers for the process-global singleton.
"""

from typing import Protocol, TypeVar, runtime_checkable

from ai_pipeline_core.document_store._models import DocumentNode
from ai_pipeline_core.documents._types import DocumentSha256, RunScope
from ai_pipeline_core.documents.document import Document

_D = TypeVar("_D", bound=Document)


@runtime_checkable
class DocumentStore(Protocol):
    """Protocol for document storage backends.

    Implementations: ClickHouseDocumentStore (production), LocalDocumentStore (CLI/debug),
    MemoryDocumentStore (testing).
    """

    async def save(self, document: Document, run_scope: RunScope) -> None:
        """Save a single document to the store. Idempotent — same SHA256 is a no-op."""
        ...

    async def save_batch(self, documents: list[Document], run_scope: RunScope) -> None:
        """Save multiple documents. Dependencies must be sorted (caller's responsibility)."""
        ...

    async def load(self, run_scope: RunScope, document_types: list[type[Document]]) -> list[Document]:
        """Load all documents of the given types from a run scope."""
        ...

    async def has_documents(self, run_scope: RunScope, document_type: type[Document]) -> bool:
        """Check if any documents of this type exist in the run scope."""
        ...

    async def check_existing(self, sha256s: list[DocumentSha256]) -> set[DocumentSha256]:
        """Return the subset of sha256s that already exist in the store."""
        ...

    async def update_summary(self, document_sha256: DocumentSha256, summary: str) -> None:
        """Update summary for a stored document. No-op if document doesn't exist."""
        ...

    async def load_summaries(self, document_sha256s: list[DocumentSha256]) -> dict[DocumentSha256, str]:
        """Load summaries by SHA256. Returns {sha256: summary} for docs that have summaries."""
        ...

    async def load_by_sha256s(self, sha256s: list[DocumentSha256], document_type: type[_D], run_scope: RunScope | None = None) -> dict[DocumentSha256, _D]:
        """Batch-load full documents by SHA256.

        document_type is used for construction only — class_name is not enforced as a filter.
        When run_scope is provided, only returns documents belonging to that scope.
        When run_scope is None, searches across all scopes (cross-pipeline lookups).
        Returns {sha256: document} for found documents. Missing SHA256s are omitted.
        """
        ...

    async def load_nodes_by_sha256s(self, sha256s: list[DocumentSha256]) -> dict[DocumentSha256, DocumentNode]:
        """Batch-load lightweight metadata for documents by SHA256, searching all scopes.

        Returns {sha256: DocumentNode} for found documents. Missing SHA256s are omitted.
        No content or attachments loaded. No document type required.
        """
        ...

    async def load_scope_metadata(self, run_scope: RunScope) -> list[DocumentNode]:
        """Load lightweight metadata for ALL documents in a run scope.

        No content or attachments loaded.
        """
        ...

    def flush(self) -> None:
        """Block until all pending background work (summaries) is processed."""
        ...

    def shutdown(self) -> None:
        """Flush pending work and stop background workers."""
        ...


_document_store: DocumentStore | None = None


def get_document_store() -> DocumentStore | None:
    """Get the process-global document store singleton."""
    return _document_store


def set_document_store(store: DocumentStore | None) -> None:
    """Set the process-global document store singleton."""
    global _document_store
    _document_store = store
