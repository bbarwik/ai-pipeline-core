"""Dual document store — fans out saves to primary + secondary.

All read operations (load, has_documents, check_existing, load_summaries)
delegate to the primary store only. Save operations fan out to both stores,
with secondary failures logged as warnings.

Summary generation is owned by this store (not the primary), so that
update_summary fans out to both stores via this class's method.
"""

from typing import TypeVar

from ai_pipeline_core.document_store._models import DocumentNode
from ai_pipeline_core.document_store._summary import SummaryGenerator
from ai_pipeline_core.document_store._summary_worker import SummaryWorker
from ai_pipeline_core.document_store.protocol import DocumentStore
from ai_pipeline_core.documents._types import DocumentSha256, RunScope
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.logging import get_pipeline_logger

_D = TypeVar("_D", bound=Document)

logger = get_pipeline_logger(__name__)


class DualDocumentStore:
    """Saves to both stores; reads from primary only. Secondary failures are best-effort.

    Owns the SummaryWorker so that generated summaries are written to both stores
    via update_summary fan-out.
    """

    def __init__(
        self,
        primary: DocumentStore,
        secondary: DocumentStore,
        *,
        summary_generator: SummaryGenerator | None = None,
    ) -> None:
        self._primary = primary
        self._secondary = secondary
        self._summary_worker: SummaryWorker | None = None
        if summary_generator:
            self._summary_worker = SummaryWorker(
                generator=summary_generator,
                update_fn=self.update_summary,
            )
            self._summary_worker.start()

    async def save(self, document: Document, run_scope: RunScope) -> None:
        """Save document to both stores; schedule summary generation."""
        await self._primary.save(document, run_scope)
        try:
            await self._secondary.save(document, run_scope)
        except Exception:
            logger.warning("Secondary store save failed for '%s'", document.name, exc_info=True)
        if self._summary_worker:
            self._summary_worker.schedule(document)

    async def save_batch(self, documents: list[Document], run_scope: RunScope) -> None:
        """Save documents to both stores; schedule summary generation for each."""
        await self._primary.save_batch(documents, run_scope)
        try:
            await self._secondary.save_batch(documents, run_scope)
        except Exception:
            logger.warning("Secondary store save_batch failed", exc_info=True)
        if self._summary_worker:
            for doc in documents:
                self._summary_worker.schedule(doc)

    async def load(self, run_scope: RunScope, document_types: list[type[Document]]) -> list[Document]:
        """Load documents from primary store only."""
        return await self._primary.load(run_scope, document_types)

    async def has_documents(self, run_scope: RunScope, document_type: type[Document]) -> bool:
        """Check primary store for documents of given type."""
        return await self._primary.has_documents(run_scope, document_type)

    async def check_existing(self, sha256s: list[DocumentSha256]) -> set[DocumentSha256]:
        """Check primary store for existing document SHA256s."""
        return await self._primary.check_existing(sha256s)

    async def update_summary(self, document_sha256: DocumentSha256, summary: str) -> None:
        """Update summary in both stores (fan-out). Primary failure re-raised after secondary attempt."""
        try:
            await self._primary.update_summary(document_sha256, summary)
        finally:
            try:
                await self._secondary.update_summary(document_sha256, summary)
            except Exception:
                logger.warning("Secondary store update_summary failed", exc_info=True)

    async def load_summaries(self, document_sha256s: list[DocumentSha256]) -> dict[DocumentSha256, str]:
        """Load summaries from primary store only."""
        return await self._primary.load_summaries(document_sha256s)

    async def load_by_sha256s(self, sha256s: list[DocumentSha256], document_type: type[_D], run_scope: RunScope | None = None) -> dict[DocumentSha256, _D]:
        """Delegate to primary store."""
        return await self._primary.load_by_sha256s(sha256s, document_type, run_scope)

    async def load_nodes_by_sha256s(self, sha256s: list[DocumentSha256]) -> dict[DocumentSha256, DocumentNode]:
        """Delegate to primary store."""
        return await self._primary.load_nodes_by_sha256s(sha256s)

    async def load_scope_metadata(self, run_scope: RunScope) -> list[DocumentNode]:
        """Delegate to primary store."""
        return await self._primary.load_scope_metadata(run_scope)

    def flush(self) -> None:
        """Flush both stores; secondary failures are best-effort."""
        if self._summary_worker:
            self._summary_worker.flush()
        try:
            self._primary.flush()
        finally:
            try:
                self._secondary.flush()
            except Exception:
                logger.warning("Secondary store flush failed", exc_info=True)

    def shutdown(self) -> None:
        """Shut down summary worker and both stores; secondary failures are best-effort."""
        if self._summary_worker:
            self._summary_worker.shutdown()
        try:
            self._primary.shutdown()
        finally:
            try:
                self._secondary.shutdown()
            except Exception:
                logger.warning("Secondary store shutdown failed", exc_info=True)
