"""In-memory document store for testing.

Simple dict-based storage implementing the full DocumentStore protocol.
Not for production use — all data is lost when the process exits.
"""

from datetime import UTC, datetime, timedelta
from typing import TypeVar

from ai_pipeline_core.document_store._models import DocumentNode, FlowCompletion
from ai_pipeline_core.document_store._summary_worker import SummaryGenerator, SummaryWorker
from ai_pipeline_core.documents._context import DocumentSha256, RunScope
from ai_pipeline_core.documents.document import Document

__all__ = [
    "MemoryDocumentStore",
]

_D = TypeVar("_D", bound=Document)

# Composite key for run-scoped membership: (sha256, class_name)
_RunKey = tuple[str, str]


class MemoryDocumentStore:
    """Dict-based document store for unit tests.

    Storage layout:
    - Global documents dict supports multiple class variants per SHA256
    - Per-run membership tracks (sha256, class_name) pairs for scope isolation
    - Global summaries keyed by SHA256
    """

    def __init__(
        self,
        *,
        summary_generator: SummaryGenerator | None = None,
    ) -> None:
        # sha256 -> {class_name: Document} — supports same SHA with different types
        self._documents: dict[str, dict[str, Document]] = {}
        # run_scope -> set of (sha256, class_name) — scope-isolated membership
        self._run_docs: dict[str, set[_RunKey]] = {}
        self._summaries: dict[str, str] = {}  # sha256 -> summary (global)
        # (run_scope, flow_name) -> FlowCompletion
        self._flow_completions: dict[tuple[str, str], FlowCompletion] = {}
        self._summary_worker: SummaryWorker | None = None
        if summary_generator:
            self._summary_worker = SummaryWorker(
                generator=summary_generator,
                update_fn=self.update_summary,
            )
            self._summary_worker.start()

    async def save(self, document: Document, run_scope: RunScope) -> None:
        """Store document in memory, keyed by (SHA256, class_name)."""
        sha256 = document.sha256
        class_name = document.__class__.__name__
        variants = self._documents.setdefault(sha256, {})
        is_new = class_name not in variants
        variants[class_name] = document
        self._run_docs.setdefault(run_scope, set()).add((sha256, class_name))
        if is_new and self._summary_worker:
            self._summary_worker.schedule(document)

    async def save_batch(self, documents: list[Document], run_scope: RunScope) -> None:
        """Save multiple documents sequentially."""
        for doc in documents:
            await self.save(doc, run_scope)

    def _iter_scope_documents(self, run_scope: RunScope) -> list[Document]:
        """Return documents for a scope, respecting per-class membership."""
        members = self._run_docs.get(run_scope, set())
        result: list[Document] = []
        for sha256, class_name in members:
            variants = self._documents.get(sha256)
            if variants:
                doc = variants.get(class_name)
                if doc is not None:
                    result.append(doc)
        return result

    async def load(self, run_scope: RunScope, document_types: list[type[Document]]) -> list[Document]:
        """Return all documents matching the given types from a run scope."""
        type_tuple = tuple(document_types)
        return [doc for doc in self._iter_scope_documents(run_scope) if isinstance(doc, type_tuple)]

    async def has_documents(self, run_scope: RunScope, document_type: type[Document], *, max_age: timedelta | None = None) -> bool:
        """Check if documents of this type exist in the run scope. Ignores max_age.

        When the document type has a FILES enum, verifies all expected filenames
        are present — not just any document of the type.
        """
        matching = [doc for doc in self._iter_scope_documents(run_scope) if isinstance(doc, document_type)]

        expected_files = document_type.get_expected_files()
        if expected_files is None:
            return len(matching) > 0

        found_names = {doc.name for doc in matching}
        return all(f in found_names for f in expected_files)

    async def check_existing(self, sha256s: list[DocumentSha256]) -> set[DocumentSha256]:
        """Return the subset of sha256s that exist in global documents."""
        return {sha for sha in sha256s if sha in self._documents}

    async def update_summary(self, document_sha256: DocumentSha256, summary: str) -> None:
        """Update summary for a stored document. No-op if document doesn't exist."""
        if document_sha256 not in self._documents:
            return
        self._summaries[document_sha256] = summary

    async def load_summaries(self, document_sha256s: list[DocumentSha256]) -> dict[DocumentSha256, str]:
        """Load summaries by SHA256."""
        return {sha: self._summaries[sha] for sha in document_sha256s if sha in self._summaries}

    async def load_by_sha256s(self, sha256s: list[DocumentSha256], document_type: type[_D], run_scope: RunScope | None = None) -> dict[DocumentSha256, _D]:
        """Batch-load documents by SHA256. document_type is for construction hint only, not enforced."""
        if not sha256s:
            return {}
        scope_shas: set[str] | None = None
        if run_scope is not None:
            scope_shas = {sha for sha, _ in self._run_docs.get(run_scope, set())}
        result: dict[DocumentSha256, _D] = {}
        for sha256 in sha256s:
            if scope_shas is not None and sha256 not in scope_shas:
                continue
            variants = self._documents.get(sha256)
            if variants:
                # Prefer the variant matching the requested type, fall back to first
                doc = variants.get(document_type.__name__) or next(iter(variants.values()))
                result[sha256] = doc  # type: ignore[assignment]
        return result

    async def load_nodes_by_sha256s(self, sha256s: list[DocumentSha256]) -> dict[DocumentSha256, DocumentNode]:
        """Batch-load lightweight metadata by SHA256 from global documents."""
        if not sha256s:
            return {}
        result: dict[DocumentSha256, DocumentNode] = {}
        for sha256 in sha256s:
            variants = self._documents.get(sha256)
            if variants:
                # Use first variant for metadata (all variants share same content)
                doc = next(iter(variants.values()))
                result[sha256] = DocumentNode(
                    sha256=sha256,
                    class_name=doc.__class__.__name__,
                    name=doc.name,
                    description=doc.description or "",
                    derived_from=doc.derived_from,
                    triggered_by=doc.triggered_by,
                    summary=self._summaries.get(sha256, ""),
                )
        return result

    async def load_scope_metadata(self, run_scope: RunScope) -> list[DocumentNode]:
        """Load lightweight metadata for all documents in a run scope."""
        return [
            DocumentNode(
                sha256=DocumentSha256(doc.sha256),
                class_name=doc.__class__.__name__,
                name=doc.name,
                description=doc.description or "",
                derived_from=doc.derived_from,
                triggered_by=doc.triggered_by,
                summary=self._summaries.get(doc.sha256, ""),
            )
            for doc in self._iter_scope_documents(run_scope)
        ]

    async def find_by_source(
        self,
        source_values: list[str],
        document_type: type[Document],
        *,
        max_age: timedelta | None = None,
    ) -> dict[str, Document]:
        """Find documents by source value. Ignores max_age (no timestamps in memory store)."""
        if not source_values:
            return {}
        source_set = set(source_values)
        result: dict[str, Document] = {}
        for variants in self._documents.values():
            for doc in variants.values():
                if not isinstance(doc, document_type):
                    continue
                for src in doc.derived_from:
                    if src in source_set and src not in result:
                        result[src] = doc
        return result

    async def save_flow_completion(
        self,
        run_scope: RunScope,
        flow_name: str,
        input_sha256s: tuple[str, ...],
        output_sha256s: tuple[str, ...],
    ) -> None:
        """Record flow completion in memory."""
        self._flow_completions[run_scope, flow_name] = FlowCompletion(
            flow_name=flow_name,
            input_sha256s=input_sha256s,
            output_sha256s=output_sha256s,
            stored_at=datetime.now(UTC),
        )

    async def get_flow_completion(
        self,
        run_scope: RunScope,
        flow_name: str,
        *,
        max_age: timedelta | None = None,
    ) -> FlowCompletion | None:
        """Get flow completion record. Ignores max_age (no expiry in memory store)."""
        return self._flow_completions.get((run_scope, flow_name))

    def flush(self) -> None:
        """Block until all pending document summaries are processed."""
        if self._summary_worker:
            self._summary_worker.flush()

    def shutdown(self) -> None:
        """Flush pending summaries and stop the summary worker."""
        if self._summary_worker:
            self._summary_worker.shutdown()
