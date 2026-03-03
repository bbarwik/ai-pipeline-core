"""Domain types, context variables, and document lifecycle tracking.

Provides DocumentSha256/RunScope types, run/task context via ContextVars,
and TaskDocumentContext for provenance validation and orphan detection.

The ordering of definitions in this module is load-order sensitive:
DocumentSha256, RunScope, and ContextVar helpers are defined first (no Document
dependency) so that document.py can import them without circular import issues.
TaskDocumentContext follows, importing Document after document.py has finished loading.
"""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import NewType
from uuid import UUID

__all__ = [
    "DocumentSha256",
    "RunContext",
    "RunScope",
    "TaskContext",
    "_TaskDocumentContext",
    "_get_run_context",
    "_reset_run_context",
    "_set_run_context",
    "_suppress_document_registration",
    "get_task_context",
    "is_registration_suppressed",
    "reset_task_context",
    "set_task_context",
]

DocumentSha256 = NewType("DocumentSha256", str)
"""BASE32-encoded SHA256 identity hash of a Document (name + content + derived_from + triggered_by + attachments)."""

RunScope = NewType("RunScope", str)
"""Scoping identifier for a pipeline run, used to partition documents in the store."""


# --- Run-level context ---


@dataclass(frozen=True, slots=True)
class RunContext:
    """Immutable context for a pipeline run, carried via ContextVar."""

    run_scope: RunScope
    execution_id: UUID | None = None


_run_context: ContextVar[RunContext | None] = ContextVar("_run_context", default=None)


def _get_run_context() -> RunContext | None:
    """Get the current run context, or None if not set."""
    return _run_context.get()


def _set_run_context(ctx: RunContext) -> Token[RunContext | None]:
    """Set the run context. Returns a token for restoring the previous value."""
    return _run_context.set(ctx)


def _reset_run_context(token: Token[RunContext | None]) -> None:
    """Reset the run context to its previous value using a token from _set_run_context."""
    _run_context.reset(token)


# --- Task-level document lifecycle tracking ---


@dataclass
class TaskContext:
    """Mutable set of document SHA256s created within the current task/flow."""

    created: set[DocumentSha256] = field(default_factory=set)
    scope_kind: str = "task"


_task_context: ContextVar[TaskContext | None] = ContextVar("_task_context", default=None)
_suppress_registration: ContextVar[bool] = ContextVar("_suppress_registration", default=False)


def get_task_context() -> TaskContext | None:
    """Get the current task context, or None if not inside a pipeline task/flow."""
    return _task_context.get()


def set_task_context(ctx: TaskContext) -> Token[TaskContext | None]:
    """Set the task context. Returns a token for restoring the previous value."""
    return _task_context.set(ctx)


def reset_task_context(token: Token[TaskContext | None]) -> None:
    """Reset the task context to its previous value."""
    _task_context.reset(token)


@contextmanager
def _suppress_document_registration() -> Generator[None, None, None]:
    """Suppress document registration during deserialization (store loads, from_dict, etc.)."""
    token = _suppress_registration.set(True)
    try:
        yield
    finally:
        _suppress_registration.reset(token)


def is_registration_suppressed() -> bool:
    """Check if document registration is currently suppressed."""
    return _suppress_registration.get()


# ---------------------------------------------------------------------------
# TaskDocumentContext — depends on Document, so it must be defined AFTER all
# symbols that document.py imports from this module (above).
# ---------------------------------------------------------------------------

from ai_pipeline_core.documents.document import Document  # noqa: E402
from ai_pipeline_core.documents.utils import is_document_sha256  # noqa: E402


@dataclass
class _TaskDocumentContext:
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
        """Detect orphaned documents -- created but not returned.

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
