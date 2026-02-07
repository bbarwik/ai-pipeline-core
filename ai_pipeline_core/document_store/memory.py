"""In-memory document store for testing.

Simple dict-based storage implementing the full DocumentStore protocol.
Not for production use — all data is lost when the process exits.
"""

from typing import TypeVar

from ai_pipeline_core.document_store._models import DocumentNode
from ai_pipeline_core.document_store._summary import SummaryGenerator
from ai_pipeline_core.document_store._summary_worker import SummaryWorker
from ai_pipeline_core.documents.document import Document

_D = TypeVar("_D", bound=Document)


class MemoryDocumentStore:
    """Dict-based document store for unit tests.

    Storage layout: global documents dict + per-run membership sets + global summaries.
    """

    def __init__(
        self,
        *,
        summary_generator: SummaryGenerator | None = None,
    ) -> None:
        self._documents: dict[str, Document] = {}  # sha256 -> Document (global)
        self._run_docs: dict[str, set[str]] = {}  # run_scope -> set of sha256s
        self._summaries: dict[str, str] = {}  # sha256 -> summary (global)
        self._summary_worker: SummaryWorker | None = None
        if summary_generator:
            self._summary_worker = SummaryWorker(
                generator=summary_generator,
                update_fn=self.update_summary,
            )
            self._summary_worker.start()

    async def save(self, document: Document, run_scope: str) -> None:
        """Store document in memory, keyed by SHA256."""
        is_new = document.sha256 not in self._documents
        if is_new:
            self._documents[document.sha256] = document
        self._run_docs.setdefault(run_scope, set()).add(document.sha256)
        if is_new and self._summary_worker:
            self._summary_worker.schedule(document)

    async def save_batch(self, documents: list[Document], run_scope: str) -> None:
        """Save multiple documents sequentially."""
        for doc in documents:
            await self.save(doc, run_scope)

    async def load(self, run_scope: str, document_types: list[type[Document]]) -> list[Document]:
        """Return all documents matching the given types from a run scope."""
        sha256s = self._run_docs.get(run_scope, set())
        type_tuple = tuple(document_types)
        return [self._documents[sha] for sha in sha256s if sha in self._documents and isinstance(self._documents[sha], type_tuple)]

    async def has_documents(self, run_scope: str, document_type: type[Document]) -> bool:
        """Check if any documents of this type exist in the run scope."""
        sha256s = self._run_docs.get(run_scope, set())
        return any(sha in self._documents and isinstance(self._documents[sha], document_type) for sha in sha256s)

    async def check_existing(self, sha256s: list[str]) -> set[str]:
        """Return the subset of sha256s that exist in global documents."""
        return self._documents.keys() & set(sha256s)

    async def update_summary(self, document_sha256: str, summary: str) -> None:
        """Update summary for a stored document. No-op if document doesn't exist."""
        if document_sha256 not in self._documents:
            return
        self._summaries[document_sha256] = summary

    async def load_summaries(self, document_sha256s: list[str]) -> dict[str, str]:
        """Load summaries by SHA256."""
        return {sha: self._summaries[sha] for sha in document_sha256s if sha in self._summaries}

    async def load_by_sha256s(self, sha256s: list[str], document_type: type[_D], run_scope: str | None = None) -> dict[str, _D]:
        """Batch-load documents by SHA256. document_type is for construction hint only, not enforced."""
        if not sha256s:
            return {}
        scope_members = self._run_docs.get(run_scope, set()) if run_scope is not None else None
        result: dict[str, _D] = {}
        for sha256 in sha256s:
            if scope_members is not None and sha256 not in scope_members:
                continue
            doc = self._documents.get(sha256)
            if doc is not None:
                result[sha256] = doc  # type: ignore[assignment]
        return result

    async def load_nodes_by_sha256s(self, sha256s: list[str]) -> dict[str, DocumentNode]:
        """Batch-load lightweight metadata by SHA256 from global documents."""
        if not sha256s:
            return {}
        result: dict[str, DocumentNode] = {}
        for sha256 in sha256s:
            doc = self._documents.get(sha256)
            if doc is not None:
                result[sha256] = DocumentNode(
                    sha256=doc.sha256,
                    class_name=doc.__class__.__name__,
                    name=doc.name,
                    description=doc.description or "",
                    sources=doc.sources,
                    origins=doc.origins,
                    summary=self._summaries.get(sha256, ""),
                )
        return result

    async def load_scope_metadata(self, run_scope: str) -> list[DocumentNode]:
        """Load lightweight metadata for all documents in a run scope."""
        sha256s = self._run_docs.get(run_scope, set())
        return [
            DocumentNode(
                sha256=doc.sha256,
                class_name=doc.__class__.__name__,
                name=doc.name,
                description=doc.description or "",
                sources=doc.sources,
                origins=doc.origins,
                summary=self._summaries.get(doc.sha256, ""),
            )
            for sha in sha256s
            if (doc := self._documents.get(sha)) is not None
        ]

    def flush(self) -> None:
        """Block until all pending document summaries are processed."""
        if self._summary_worker:
            self._summary_worker.flush()

    def shutdown(self) -> None:
        """Flush pending summaries and stop the summary worker."""
        if self._summary_worker:
            self._summary_worker.shutdown()
