"""Shared document-ancestry traversal helpers for database backends."""

from collections.abc import Callable

from ai_pipeline_core.database._types import DocumentRecord


def collect_document_ancestry(
    *,
    target: DocumentRecord,
    docs_by_sha: dict[str, DocumentRecord],
    load_extra_documents: Callable[[list[str]], dict[str, DocumentRecord]],
) -> dict[str, DocumentRecord]:
    """Traverse ``derived_from`` and ``triggered_by`` links across deployment and cross-chain documents."""
    ancestors: dict[str, DocumentRecord] = {}
    pending = list(target.derived_from) + list(target.triggered_by)
    seen = set(pending)

    while pending:
        next_pending: list[str] = []
        unresolved_batch: list[str] = []

        for current_sha in pending:
            current = docs_by_sha.get(current_sha)
            if current is None:
                unresolved_batch.append(current_sha)
                continue
            ancestors[current_sha] = current
            for parent_sha in (*current.derived_from, *current.triggered_by):
                if parent_sha in seen:
                    continue
                seen.add(parent_sha)
                next_pending.append(parent_sha)

        if unresolved_batch:
            extra_docs = load_extra_documents(list(dict.fromkeys(unresolved_batch)))
            for extra_sha, extra_doc in extra_docs.items():
                docs_by_sha[extra_sha] = extra_doc
                ancestors[extra_sha] = extra_doc
                for parent_sha in (*extra_doc.derived_from, *extra_doc.triggered_by):
                    if parent_sha in seen:
                        continue
                    seen.add(parent_sha)
                    next_pending.append(parent_sha)

        pending = next_pending

    return ancestors
