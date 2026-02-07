"""Lightweight document metadata and provenance graph traversal."""

from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ai_pipeline_core.documents.utils import is_document_sha256

__all__ = [
    "MAX_PROVENANCE_GRAPH_NODES",
    "DocumentNode",
    "build_provenance_graph",
    "walk_provenance",
]

MAX_PROVENANCE_GRAPH_NODES = 5000


@dataclass(frozen=True, slots=True)
class DocumentNode:
    """Lightweight document metadata without content or attachments."""

    sha256: str
    class_name: str
    name: str
    description: str = ""
    sources: tuple[str, ...] = ()
    origins: tuple[str, ...] = ()
    summary: str = ""


def build_provenance_graph(
    root_sha256: str,
    nodes: list[DocumentNode],
) -> dict[str, DocumentNode]:
    """BFS from root following sources + origins. Returns reachable ancestors keyed by SHA256.

    Traverses up to MAX_PROVENANCE_GRAPH_NODES nodes. Uses is_document_sha256() to
    distinguish document SHA256 references from URL strings in sources.
    """
    index: dict[str, DocumentNode] = {node.sha256: node for node in nodes}

    root = index.get(root_sha256)
    if root is None:
        return {}

    visited: dict[str, DocumentNode] = {}
    enqueued: set[str] = {root_sha256}
    queue: deque[str] = deque([root_sha256])

    while queue and len(visited) < MAX_PROVENANCE_GRAPH_NODES:
        sha256 = queue.popleft()

        node = index.get(sha256)
        if node is None:
            continue

        visited[sha256] = node

        for ref in (*node.sources, *node.origins):
            if ref not in enqueued and is_document_sha256(ref):
                enqueued.add(ref)
                queue.append(ref)

    return visited


async def walk_provenance(
    root_sha256: str,
    load_nodes: Callable[[list[str]], Awaitable[dict[str, DocumentNode]]],
) -> dict[str, DocumentNode]:
    """BFS walk of provenance chain using batch lookups per level.

    Loads root node, then iteratively loads all referenced sources/origins in batches
    until the full reachable graph is resolved (up to MAX_PROVENANCE_GRAPH_NODES).
    Requires no knowledge of run_scope — uses cross-scope batch lookup.

    Args:
        root_sha256: SHA256 of the root document to start from.
        load_nodes: Async callable that batch-loads DocumentNodes by SHA256 list.
            Typically ``store.load_nodes_by_sha256s``.
    """
    visited: dict[str, DocumentNode] = {}
    # dict preserves insertion order for deterministic BFS traversal
    pending: dict[str, None] = {root_sha256: None}

    while pending and len(visited) < MAX_PROVENANCE_GRAPH_NODES:
        batch = list(pending)[: MAX_PROVENANCE_GRAPH_NODES - len(visited)]
        nodes = await load_nodes(batch)

        # Remove only the items we actually requested, preserving unprocessed ones
        for sha in batch:
            pending.pop(sha, None)

        for sha256, node in nodes.items():
            if sha256 in visited:
                continue
            visited[sha256] = node
            for ref in (*node.sources, *node.origins):
                if ref not in visited and ref not in pending and is_document_sha256(ref):
                    pending[ref] = None

        # If none of the batch was found, stop (all remaining are dangling)
        if not nodes:
            break

    return visited
