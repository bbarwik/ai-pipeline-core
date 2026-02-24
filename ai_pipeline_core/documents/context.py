"""Run context and task document context for document lifecycle management.

RunContext tracks the current run scope via ContextVar.
TaskDocumentContext tracks document creation within a pipeline task/flow,
providing provenance validation, orphan detection, and deduplication.
"""

from dataclasses import dataclass, field

from ai_pipeline_core.documents._context_vars import (
    RunContext,
    get_run_context,
    reset_run_context,
    set_run_context,
)
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.documents.types import DocumentSha256
from ai_pipeline_core.documents.utils import is_document_sha256

__all__ = [
    "RunContext",
    "TaskDocumentContext",
    "get_run_context",
    "reset_run_context",
    "set_run_context",
]


@dataclass
class TaskDocumentContext:
    """Tracks documents created within a single pipeline task or flow execution.

    Used by @pipeline_task and @pipeline_flow decorators to:
    - Validate that all derived_from/triggered_by SHA256 references point to pre-existing documents
    - Detect same-task interdependencies (doc B referencing doc A created in the same task)
    - Warn about documents with no provenance (no derived_from and no triggered_by)
    - Detect documents created but not returned (orphaned)
    - Deduplicate returned documents by SHA256
    """

    created: set[DocumentSha256] = field(default_factory=set)

    def register_created(self, doc: Document) -> None:
        """Register a document as created in this task/flow context."""
        self.created.add(doc.sha256)

    def validate_provenance(
        self,
        documents: list[Document],
        existing_sha256s: set[DocumentSha256],
    ) -> list[str]:
        """Validate provenance (derived_from and triggered_by) for returned documents.

        Checks:
        1. All SHA256 derived_from references exist in the store (existing_sha256s).
        2. All triggered_by references exist in the store (existing_sha256s).
        3. No same-task interdependencies: a returned document must not reference
           (via derived_from or triggered_by SHA256) another document created in this same context.
        4. Documents with no derived_from AND no triggered_by get a warning (no provenance).

        Only SHA256-formatted entries in derived_from are validated; URLs and other reference
        strings are skipped. Initial pipeline inputs (documents with no provenance)
        are acceptable and warned about for awareness.

        Returns a list of warning messages (empty if everything is valid).
        """
        warnings: list[str] = []

        for doc in documents:
            # Check derived_from
            for src in doc.derived_from:
                if not is_document_sha256(src):
                    continue
                if src in self.created:
                    warnings.append(f"Document '{doc.name}' references derived_from {src[:12]}... created in the same task (same-task interdependency)")
                elif src not in existing_sha256s:
                    warnings.append(f"Document '{doc.name}' references derived_from {src[:12]}... which does not exist in the store")

            # Check triggered_by
            for trigger in doc.triggered_by:
                if trigger in self.created:
                    warnings.append(f"Document '{doc.name}' references triggered_by {trigger[:12]}... created in the same task (same-task interdependency)")
                elif trigger not in existing_sha256s:
                    warnings.append(f"Document '{doc.name}' references triggered_by {trigger[:12]}... which does not exist in the store")

            # Warn about no provenance
            if not doc.derived_from and not doc.triggered_by:
                warnings.append(f"Document '{doc.name}' has no derived_from and no triggered_by (no provenance)")

        return warnings

    def finalize(self, returned_docs: list[Document]) -> list[DocumentSha256]:
        """Detect orphaned documents — created but not returned.

        Returns list of orphaned document SHA256 hashes.
        """
        returned_sha256s = {doc.sha256 for doc in returned_docs}
        return sorted(self.created - returned_sha256s)

    @staticmethod
    def deduplicate(documents: list[Document]) -> list[Document]:
        """Deduplicate documents by SHA256, preserving first occurrence order."""
        seen: dict[DocumentSha256, Document] = {}
        for doc in documents:
            if doc.sha256 not in seen:
                seen[doc.sha256] = doc
        return list(seen.values())
