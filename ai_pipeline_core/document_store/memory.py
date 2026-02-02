"""In-memory document store for testing.

Simple dict-based storage implementing the full DocumentStore protocol.
Not for production use — all data is lost when the process exits.
"""

from ai_pipeline_core.document_store._summary import SummaryGenerator
from ai_pipeline_core.document_store._summary_worker import SummaryWorker
from ai_pipeline_core.documents.document import Document


class MemoryDocumentStore:
    """Dict-based document store for unit tests.

    Storage layout: dict[run_scope, dict[document_sha256, Document]].
    """

    def __init__(
        self,
        *,
        summary_generator: SummaryGenerator | None = None,
    ) -> None:
        self._data: dict[str, dict[str, Document]] = {}
        self._summaries: dict[str, dict[str, str]] = {}  # run_scope -> sha256 -> summary
        self._summary_worker: SummaryWorker | None = None
        if summary_generator:
            self._summary_worker = SummaryWorker(
                generator=summary_generator,
                update_fn=self.update_summary,
            )
            self._summary_worker.start()

    async def save(self, document: Document, run_scope: str) -> None:
        """Store document in memory, keyed by SHA256."""
        scope = self._data.setdefault(run_scope, {})
        if document.sha256 in scope:
            return  # Idempotent — same document already saved
        scope[document.sha256] = document
        if self._summary_worker:
            self._summary_worker.schedule(run_scope, document)

    async def save_batch(self, documents: list[Document], run_scope: str) -> None:
        """Save multiple documents sequentially."""
        for doc in documents:
            await self.save(doc, run_scope)

    async def load(self, run_scope: str, document_types: list[type[Document]]) -> list[Document]:
        """Return all documents matching the given types from a run scope."""
        scope = self._data.get(run_scope, {})
        type_tuple = tuple(document_types)
        return [doc for doc in scope.values() if isinstance(doc, type_tuple)]

    async def has_documents(self, run_scope: str, document_type: type[Document]) -> bool:
        """Check if any documents of this type exist in the run scope."""
        scope = self._data.get(run_scope, {})
        return any(isinstance(doc, document_type) for doc in scope.values())

    async def check_existing(self, sha256s: list[str]) -> set[str]:
        """Return the subset of sha256s that exist across all scopes."""
        all_hashes: set[str] = set()
        for scope in self._data.values():
            all_hashes.update(scope.keys())
        return all_hashes & set(sha256s)

    async def update_summary(self, run_scope: str, document_sha256: str, summary: str) -> None:
        """Update summary for a stored document. No-op if document doesn't exist."""
        scope = self._data.get(run_scope, {})
        if document_sha256 not in scope:
            return
        self._summaries.setdefault(run_scope, {})[document_sha256] = summary

    async def load_summaries(self, run_scope: str, document_sha256s: list[str]) -> dict[str, str]:
        """Load summaries by SHA256."""
        scope_summaries = self._summaries.get(run_scope, {})
        return {sha: scope_summaries[sha] for sha in document_sha256s if sha in scope_summaries}

    def flush(self) -> None:
        """Block until all pending document summaries are processed."""
        if self._summary_worker:
            self._summary_worker.flush()

    def shutdown(self) -> None:
        """Flush pending summaries and stop the summary worker."""
        if self._summary_worker:
            self._summary_worker.shutdown()
