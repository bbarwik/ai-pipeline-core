"""Read-only document store protocol for application code.

Defines the DocumentReader protocol — the only document store interface
that application code should depend on. Write operations are framework-internal.
"""

from datetime import timedelta
from typing import Protocol, TypeVar, runtime_checkable

from ai_pipeline_core.document_store._models import DocumentNode, FlowCompletion
from ai_pipeline_core.document_store._singleton import get_store
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.documents.types import DocumentSha256, RunScope

__all__ = [
    "DocumentNode",
    "DocumentReader",
    "FlowCompletion",
    "get_document_store",
]

_D = TypeVar("_D", bound=Document)


@runtime_checkable
class DocumentReader(Protocol):
    """Read-only protocol for application code that consumes documents.

    Users should depend on this protocol when they only need to read documents.
    """

    async def load(self, run_scope: RunScope, document_types: list[type[Document]]) -> list[Document]:
        """Load all documents of the given types from a run scope."""
        ...

    async def has_documents(self, run_scope: RunScope, document_type: type[Document], *, max_age: timedelta | None = None) -> bool:
        """Check if any documents of this type exist in the run scope."""
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
        """Find the most recent document per source value."""
        ...

    async def get_flow_completion(
        self,
        run_scope: RunScope,
        flow_name: str,
        *,
        max_age: timedelta | None = None,
    ) -> FlowCompletion | None:
        """Get the completion record for a flow, or None if not found / expired."""
        ...


def get_document_store() -> DocumentReader | None:
    """Get the process-global document store for read-only access."""
    return get_store()
