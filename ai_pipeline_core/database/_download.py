"""Export deployment data as a portable FilesystemDatabase snapshot."""

import asyncio
from collections import defaultdict, deque
from pathlib import Path
from typing import cast
from uuid import UUID

from ai_pipeline_core.database._filesystem import FilesystemDatabase
from ai_pipeline_core.database._protocol import DatabaseReader
from ai_pipeline_core.database._summary import generate_costs, generate_summary
from ai_pipeline_core.database._types import ExecutionNode, NodeKind
from ai_pipeline_core.documents._context import DocumentSha256

__all__ = ["download_deployment"]

_NODE_KIND_PRIORITY: dict[NodeKind, int] = {
    NodeKind.DEPLOYMENT: 0,
    NodeKind.FLOW: 1,
    NodeKind.TASK: 2,
    NodeKind.CONVERSATION: 3,
    NodeKind.CONVERSATION_TURN: 4,
}
_DOCUMENT_REF_KEY = "$doc_ref"


def _node_sort_key(node: ExecutionNode) -> tuple[int, int, str]:
    """Sort siblings deterministically while letting traversal enforce parent-first order."""
    return (_NODE_KIND_PRIORITY[node.node_kind], node.sequence_no, str(node.node_id))


def _sort_nodes_parent_first(nodes: list[ExecutionNode]) -> list[ExecutionNode]:
    """Sort nodes so every parent is inserted before any descendant."""
    nodes_by_id = {node.node_id: node for node in nodes}
    children_by_parent: dict[UUID, list[ExecutionNode]] = defaultdict(list)
    roots: list[ExecutionNode] = []

    for node in nodes:
        if node.parent_node_id in nodes_by_id:
            children_by_parent[node.parent_node_id].append(node)
            continue
        roots.append(node)

    for children in children_by_parent.values():
        children.sort(key=_node_sort_key)

    ordered: list[ExecutionNode] = []
    visited: set[UUID] = set()
    queue = deque(sorted(roots, key=_node_sort_key))

    while queue:
        node = queue.popleft()
        if node.node_id in visited:
            continue
        visited.add(node.node_id)
        ordered.append(node)
        for child in children_by_parent.get(node.node_id, ()):
            if child.node_id not in visited:
                queue.append(child)

    ordered.extend(sorted([item for item in nodes if item.node_id not in visited], key=_node_sort_key))

    return ordered


def _collect_replay_payload_document_shas(value: object, document_shas: set[DocumentSha256]) -> None:
    """Recursively harvest document refs embedded inside replay payload JSON."""
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        doc_ref = mapping.get(_DOCUMENT_REF_KEY)
        if isinstance(doc_ref, str):
            document_shas.add(DocumentSha256(doc_ref))
        for nested_value in mapping.values():
            _collect_replay_payload_document_shas(nested_value, document_shas)
        return

    if isinstance(value, list | tuple):
        for item in cast("list[object] | tuple[object, ...]", value):
            _collect_replay_payload_document_shas(item, document_shas)


async def download_deployment(
    source: DatabaseReader,
    deployment_id: UUID,
    output_path: Path,
) -> None:
    """Export a deployment as a FilesystemDatabase snapshot."""
    tree = await source.get_deployment_tree(deployment_id)
    target = await asyncio.to_thread(FilesystemDatabase, output_path)

    for node in _sort_nodes_parent_first(tree):
        await target.insert_node(node)

    document_shas: set[DocumentSha256] = set()
    for node in tree:
        document_shas.update(DocumentSha256(sha) for sha in node.context_document_shas)
        document_shas.update(DocumentSha256(sha) for sha in node.input_document_shas)
        document_shas.update(DocumentSha256(sha) for sha in node.output_document_shas)
        _collect_replay_payload_document_shas(node.payload.get("replay_payload"), document_shas)

    for document in await source.get_documents_by_deployment(deployment_id):
        document_shas.add(document.document_sha256)

    documents = await source.get_documents_batch(list(document_shas))
    if documents:
        await target.save_document_batch(list(documents.values()))

    blob_shas: set[str] = set()
    for document in documents.values():
        blob_shas.add(document.content_sha256)
        blob_shas.update(document.attachment_sha256s)

    blobs = await source.get_blobs_batch(list(blob_shas))
    if blobs:
        await target.save_blob_batch(list(blobs.values()))

    logs = await source.get_deployment_logs(deployment_id)
    if logs:
        await target.save_logs_batch(logs)
    else:
        await asyncio.to_thread((output_path / "logs.jsonl").write_text, "", encoding="utf-8")

    summary = await generate_summary(target, deployment_id)
    costs = await generate_costs(target, deployment_id)
    await asyncio.to_thread((output_path / "summary.md").write_text, summary, encoding="utf-8")
    await asyncio.to_thread((output_path / "costs.md").write_text, costs, encoding="utf-8")
