"""@internal Full read+write document store protocol and singleton management.

DocumentStore extends DocumentReader with write operations. Framework-internal only.
Application code should use DocumentReader from protocol.py.
"""

from typing import Protocol, runtime_checkable

from ai_pipeline_core.document_store._singleton import get_store, set_store
from ai_pipeline_core.document_store.protocol import DocumentReader
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.documents.types import DocumentSha256, RunScope

__all__ = [
    "DocumentStore",
    "get_document_store",
    "set_document_store",
]


@runtime_checkable
class DocumentStore(DocumentReader, Protocol):
    """Full read+write protocol for framework-internal use.

    Implementations: ClickHouseDocumentStore (production), LocalDocumentStore (CLI/debug),
    MemoryDocumentStore (testing).

    Application code should use DocumentReader for read-only access.
    Write operations (save, save_batch, etc.) are called by the framework only.
    """

    async def save(self, document: Document, run_scope: RunScope) -> None:
        """Save a single document to the store. Idempotent — same SHA256 is a no-op."""
        ...

    async def save_batch(self, documents: list[Document], run_scope: RunScope) -> None:
        """Save multiple documents. Dependencies must be sorted (caller's responsibility)."""
        ...

    async def update_summary(self, document_sha256: DocumentSha256, summary: str) -> None:
        """Update summary for a stored document. No-op if document doesn't exist."""
        ...

    async def save_flow_completion(
        self,
        run_scope: RunScope,
        flow_name: str,
        input_sha256s: tuple[str, ...],
        output_sha256s: tuple[str, ...],
    ) -> None:
        """Record that a flow completed successfully."""
        ...

    def flush(self) -> None:
        """Block until all pending background work (summaries) is processed."""
        ...

    def shutdown(self) -> None:
        """Flush pending work and stop background workers."""
        ...


def get_document_store() -> DocumentStore | None:
    """Get the process-global document store singleton (framework-internal, returns full DocumentStore)."""
    return get_store()


def set_document_store(store: DocumentStore | None) -> None:
    """Set the process-global document store singleton."""
    set_store(store)
