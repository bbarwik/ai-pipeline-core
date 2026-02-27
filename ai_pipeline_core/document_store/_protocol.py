"""Document store protocols, singleton management, and factory.

DocumentReader — public read-only protocol for application code.
DocumentStore — internal read+write protocol extending DocumentReader.
Singleton get/set — process-global document store instance.
Factory — creates store instances from settings.
"""

from datetime import timedelta
from typing import Any, Protocol, TypeVar, runtime_checkable

from ai_pipeline_core.document_store._models import DocumentNode, FlowCompletion
from ai_pipeline_core.document_store._summary_worker import SummaryGenerator
from ai_pipeline_core.documents._context import DocumentSha256, RunScope
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.settings import Settings

_D = TypeVar("_D", bound=Document)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class DocumentReader(Protocol):
    """Read-only protocol for application code that consumes documents.

    Backend varies by PipelineDeployment execution mode: MemoryDocumentStore
    (run_local), DualDocumentStore or LocalDocumentStore (run_cli),
    auto-configured from settings (as_prefect_flow).

    Users should depend on this protocol when they only need to read documents.
    """

    async def load(self, run_scope: RunScope, document_types: list[type[Document]]) -> list[Document]:
        """Load all documents of the given types from a run scope."""
        ...

    async def has_documents(self, run_scope: RunScope, document_type: type[Document], *, max_age: timedelta | None = None) -> bool:
        """Check if any documents of this type exist in the run scope. max_age filters on ``stored_at`` timestamp."""
        ...

    async def check_existing(self, sha256s: list[DocumentSha256]) -> set[DocumentSha256]:
        """Return the subset of sha256s that already exist in the store."""
        ...

    async def load_by_sha256s(self, sha256s: list[DocumentSha256], document_type: type[_D], run_scope: RunScope | None = None) -> dict[DocumentSha256, _D]:
        """Batch-load full documents by SHA256."""
        ...

    async def load_nodes_by_sha256s(self, sha256s: list[DocumentSha256]) -> dict[DocumentSha256, DocumentNode]:
        """Batch-load lightweight metadata for documents by SHA256."""
        ...

    async def load_scope_metadata(self, run_scope: RunScope) -> list[DocumentNode]:
        """Load lightweight metadata for ALL documents in a run scope."""
        ...

    async def load_summaries(self, document_sha256s: list[DocumentSha256]) -> dict[DocumentSha256, str]:
        """Load summaries by SHA256."""
        ...

    async def find_by_source(
        self,
        source_values: list[str],
        document_type: type[Document],
        *,
        max_age: timedelta | None = None,
    ) -> dict[str, Document]:
        """Find most recent document per source value, matched against ``derived_from`` entries. max_age filters on ``stored_at`` timestamp."""
        ...

    async def get_flow_completion(
        self,
        run_scope: RunScope,
        flow_name: str,
        *,
        max_age: timedelta | None = None,
    ) -> FlowCompletion | None:
        """Get the completion record for a flow, or None if not found / expired. max_age filters on ``stored_at`` timestamp."""
        ...


@runtime_checkable
class DocumentStore(DocumentReader, Protocol):
    """Full read+write protocol for framework-internal use.

    Implementations: ClickHouseDocumentStore (production), LocalDocumentStore (CLI/debug),
    MemoryDocumentStore (testing).

    Application code should use DocumentReader for read-only access.
    Write operations (save, save_batch, etc.) are called by the framework only.
    """

    async def save(self, document: Document, run_scope: RunScope) -> None:
        """Save a single document to the store. Idempotent -- same SHA256 is a no-op."""
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


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_document_store: Any = None


def get_store() -> Any:
    """Get the process-global document store singleton (untyped to avoid circular imports)."""
    return _document_store


def set_store(store: Any) -> None:
    """Set the process-global document store singleton."""
    global _document_store
    _document_store = store


def get_document_store() -> DocumentStore | None:
    """Get the process-global document store singleton (framework-internal, returns full DocumentStore)."""
    return get_store()


def set_document_store(store: DocumentStore | None) -> None:
    """Set the process-global document store singleton."""
    set_store(store)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_document_store(
    settings: Settings,
    *,
    summary_generator: SummaryGenerator | None = None,
) -> DocumentStore:
    """Create a DocumentStore based on settings.

    Selects ClickHouseDocumentStore when clickhouse_host is configured,
    otherwise falls back to LocalDocumentStore.

    Backends are imported lazily to avoid circular imports.
    """
    if settings.clickhouse_host:
        from ai_pipeline_core.document_store._clickhouse import ClickHouseDocumentStore

        return ClickHouseDocumentStore(
            host=settings.clickhouse_host,
            port=settings.clickhouse_port,
            database=settings.clickhouse_database,
            username=settings.clickhouse_user,
            password=settings.clickhouse_password,
            secure=settings.clickhouse_secure,
            connect_timeout=settings.clickhouse_connect_timeout,
            send_receive_timeout=settings.clickhouse_send_receive_timeout,
            summary_generator=summary_generator,
        )

    from ai_pipeline_core.document_store._local import LocalDocumentStore

    return LocalDocumentStore(summary_generator=summary_generator)
