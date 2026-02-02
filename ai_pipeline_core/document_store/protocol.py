"""Document store protocol and singleton management.

Defines the DocumentStore protocol that all storage backends must implement,
along with get/set helpers for the process-global singleton.
"""

from typing import Protocol, runtime_checkable

from ai_pipeline_core.documents.document import Document


@runtime_checkable
class DocumentStore(Protocol):
    """Protocol for document storage backends.

    Implementations: ClickHouseDocumentStore (production), LocalDocumentStore (CLI/debug),
    MemoryDocumentStore (testing).
    """

    async def save(self, document: Document, run_scope: str) -> None:
        """Save a single document to the store. Idempotent — same SHA256 is a no-op."""
        ...

    async def save_batch(self, documents: list[Document], run_scope: str) -> None:
        """Save multiple documents. Dependencies must be sorted (caller's responsibility)."""
        ...

    async def load(self, run_scope: str, document_types: list[type[Document]]) -> list[Document]:
        """Load all documents of the given types from a run scope."""
        ...

    async def has_documents(self, run_scope: str, document_type: type[Document]) -> bool:
        """Check if any documents of this type exist in the run scope."""
        ...

    async def check_existing(self, sha256s: list[str]) -> set[str]:
        """Return the subset of sha256s that already exist in the store."""
        ...

    async def update_summary(self, run_scope: str, document_sha256: str, summary: str) -> None:
        """Update summary for a stored document. No-op if document doesn't exist."""
        ...

    async def load_summaries(self, run_scope: str, document_sha256s: list[str]) -> dict[str, str]:
        """Load summaries by SHA256. Returns {sha256: summary} for docs that have summaries."""
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
