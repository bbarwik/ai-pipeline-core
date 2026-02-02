"""Run context and task document context for document lifecycle management.

RunContext tracks the current run scope via ContextVar.
TaskDocumentContext tracks document creation within a pipeline task/flow,
providing provenance validation, finalize checks, and deduplication.
"""

from dataclasses import dataclass, field

from ai_pipeline_core.documents._context_vars import (
    RunContext,
    get_run_context,
    get_task_context,
    is_registration_suppressed,
    reset_run_context,
    reset_task_context,
    set_run_context,
    set_task_context,
    suppress_registration,
)
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.documents.utils import is_document_sha256

# Re-export everything from _context_vars so existing imports from context.py keep working
__all__ = [
    "RunContext",
    "TaskDocumentContext",
    "get_run_context",
    "get_task_context",
    "is_registration_suppressed",
    "reset_run_context",
    "reset_task_context",
    "set_run_context",
    "set_task_context",
    "suppress_registration",
]


@dataclass
class TaskDocumentContext:
    """Tracks documents created within a single pipeline task or flow execution.

    Used by @pipeline_task and @pipeline_flow decorators to:
    - Validate that all source/origin SHA256 references point to pre-existing documents
    - Detect same-task interdependencies (doc B referencing doc A created in the same task)
    - Warn about documents with no provenance (no sources and no origins)
    - Detect documents created but not returned (orphaned)
    - Deduplicate returned documents by SHA256
    """

    created: set[str] = field(default_factory=set)

    def register_created(self, doc: Document) -> None:
        """Register a document as created in this task/flow context."""
        self.created.add(doc.sha256)

    def validate_provenance(
        self,
        documents: list[Document],
        existing_sha256s: set[str],
        *,
        check_created: bool = False,
    ) -> list[str]:
        """Validate provenance (sources and origins) for returned documents.

        Checks:
        1. All SHA256 source references exist in the store (existing_sha256s).
        2. All origin references exist in the store (existing_sha256s).
        3. No same-task interdependencies: a returned document must not reference
           (via source or origin SHA256) another document created in this same context.
        4. Documents with no sources AND no origins get a warning (no provenance).
        5. (When check_created=True) Returned documents must have been created in
           this context. Only applicable for @pipeline_task — flows delegate creation
           to nested tasks whose documents register in the task's own context.

        Only SHA256-formatted sources are validated; URLs and other reference strings
        in sources are skipped. Initial pipeline inputs (documents with no provenance)
        are acceptable and warned about for awareness.

        Returns a list of warning messages (empty if everything is valid).
        """
        warnings: list[str] = []

        for doc in documents:
            # Check that returned doc was created in this context (task-only)
            if check_created and doc.sha256 not in self.created:
                warnings.append(f"Document '{doc.name}' was not created in this task — only newly created documents should be returned")

            # Check sources
            for src in doc.sources:
                if not is_document_sha256(src):
                    continue
                if src in self.created:
                    warnings.append(f"Document '{doc.name}' references source {src[:12]}... created in the same task (same-task interdependency)")
                elif src not in existing_sha256s:
                    warnings.append(f"Document '{doc.name}' references source {src[:12]}... which does not exist in the store")

            # Check origins
            for origin in doc.origins:
                if origin in self.created:
                    warnings.append(f"Document '{doc.name}' references origin {origin[:12]}... created in the same task (same-task interdependency)")
                elif origin not in existing_sha256s:
                    warnings.append(f"Document '{doc.name}' references origin {origin[:12]}... which does not exist in the store")

            # Warn about no provenance
            if not doc.sources and not doc.origins:
                warnings.append(f"Document '{doc.name}' has no sources and no origins (no provenance)")

        return warnings

    def finalize(self, returned_docs: list[Document]) -> list[str]:
        """Check for documents created but not returned from the task/flow.

        Returns a list of warning messages for orphaned documents — those registered
        via Document.__init__ but not present in the returned result.
        """
        returned_sha256s = {doc.sha256 for doc in returned_docs}
        orphaned = self.created - returned_sha256s
        return [f"Document {sha[:12]}... was created but not returned" for sha in sorted(orphaned)]

    @staticmethod
    def deduplicate(documents: list[Document]) -> list[Document]:
        """Deduplicate documents by SHA256, preserving first occurrence order."""
        seen: dict[str, Document] = {}
        for doc in documents:
            if doc.sha256 not in seen:
                seen[doc.sha256] = doc
        return list(seen.values())
