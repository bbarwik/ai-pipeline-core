"""Tests for DocumentNode, build_provenance_graph, and walk_provenance."""

import base64
import hashlib

import pytest

from ai_pipeline_core.document_store._models import (
    MAX_PROVENANCE_GRAPH_NODES,
    DocumentNode,
    build_provenance_graph,
    walk_provenance,
)
from ai_pipeline_core.documents._types import DocumentSha256

# Valid base32-encoded SHA256 hashes (A-Z, 2-7 only, 52 chars, high entropy)
SHA_A = DocumentSha256("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567ABCDEFGHIJKLMNOPQRST")
SHA_B = DocumentSha256("BCDEFGHIJKLMNOPQRSTUVWXYZ234567ABCDEFGHIJKLMNOPQRSTU")
SHA_C = DocumentSha256("CDEFGHIJKLMNOPQRSTUVWXYZ234567ABCDEFGHIJKLMNOPQRSTUV")
SHA_D = DocumentSha256("DEFGHIJKLMNOPQRSTUVWXYZ234567ABCDEFGHIJKLMNOPQRSTUVW")
SHA_E = DocumentSha256("EFGHIJKLMNOPQRSTUVWXYZ234567ABCDEFGHIJKLMNOPQRSTUVWX")


def _node(sha256: str, sources: tuple[str, ...] = (), origins: tuple[str, ...] = ()) -> DocumentNode:
    return DocumentNode(sha256=sha256, class_name="TestDoc", name=f"doc-{sha256[:6]}", sources=sources, origins=origins)


class TestDocumentNode:
    def test_frozen_and_slots(self):
        node = _node(SHA_A)
        assert node.sha256 == SHA_A
        assert node.class_name == "TestDoc"
        assert node.sources == ()
        assert node.origins == ()
        assert node.summary == ""

    def test_defaults(self):
        node = DocumentNode(sha256=SHA_A, class_name="X", name="x")
        assert node.description == ""
        assert node.sources == ()
        assert node.origins == ()
        assert node.summary == ""


class TestBuildProvenanceGraph:
    def test_linear_chain(self):
        """A -> B -> C: all three found."""
        nodes = [
            _node(SHA_A, sources=(SHA_B,)),
            _node(SHA_B, sources=(SHA_C,)),
            _node(SHA_C),
        ]
        graph = build_provenance_graph(SHA_A, nodes)
        assert set(graph.keys()) == {SHA_A, SHA_B, SHA_C}

    def test_diamond(self):
        """A -> B, A -> C, B -> D, C -> D: no duplicates."""
        nodes = [
            _node(SHA_A, sources=(SHA_B, SHA_C)),
            _node(SHA_B, sources=(SHA_D,)),
            _node(SHA_C, sources=(SHA_D,)),
            _node(SHA_D),
        ]
        graph = build_provenance_graph(SHA_A, nodes)
        assert set(graph.keys()) == {SHA_A, SHA_B, SHA_C, SHA_D}

    def test_root_not_found(self):
        nodes = [_node(SHA_A)]
        graph = build_provenance_graph(SHA_B, nodes)
        assert graph == {}

    def test_empty_nodes_list(self):
        graph = build_provenance_graph(SHA_A, [])
        assert graph == {}

    def test_url_sources_skipped(self):
        """URL strings in sources are not followed as graph edges."""
        nodes = [
            _node(SHA_A, sources=("https://example.com", SHA_B)),
            _node(SHA_B),
        ]
        graph = build_provenance_graph(SHA_A, nodes)
        assert set(graph.keys()) == {SHA_A, SHA_B}
        assert "https://example.com" not in graph

    def test_dangling_sha256_skipped(self):
        """References to SHA256s not in the nodes list are silently skipped."""
        nodes = [
            _node(SHA_A, sources=(SHA_B,)),
            # SHA_B not in nodes list
        ]
        graph = build_provenance_graph(SHA_A, nodes)
        assert set(graph.keys()) == {SHA_A}

    def test_cycle_handling(self):
        """Cycles in provenance don't cause infinite loops."""
        nodes = [
            _node(SHA_A, sources=(SHA_B,)),
            _node(SHA_B, sources=(SHA_A,)),
        ]
        graph = build_provenance_graph(SHA_A, nodes)
        assert set(graph.keys()) == {SHA_A, SHA_B}

    def test_mixed_sources_and_origins(self):
        """Both sources and origins edges are followed."""
        nodes = [
            _node(SHA_A, sources=(SHA_B,), origins=(SHA_C,)),
            _node(SHA_B),
            _node(SHA_C, origins=(SHA_D,)),
            _node(SHA_D),
        ]
        graph = build_provenance_graph(SHA_A, nodes)
        assert set(graph.keys()) == {SHA_A, SHA_B, SHA_C, SHA_D}

    def test_single_root_only(self):
        """Root with no references returns just the root."""
        nodes = [_node(SHA_A)]
        graph = build_provenance_graph(SHA_A, nodes)
        assert set(graph.keys()) == {SHA_A}

    def test_max_nodes_limit_constant(self):
        """MAX_PROVENANCE_GRAPH_NODES constant exists with expected value."""
        assert MAX_PROVENANCE_GRAPH_NODES == 5000

    def test_traversal_stops_at_max_nodes(self):
        """BFS stops at MAX_PROVENANCE_GRAPH_NODES, returning partial graph."""
        count = MAX_PROVENANCE_GRAPH_NODES + 500

        # Generate valid base32-encoded SHA256 hashes by hashing incrementing integers
        shas = [base64.b32encode(hashlib.sha256(str(i).encode()).digest()).decode()[:52] for i in range(count)]

        # Build linear chain: sha[0] -> sha[1] -> ... -> sha[count-1]
        nodes = [_node(shas[i], sources=(shas[i + 1],) if i < count - 1 else ()) for i in range(count)]
        graph = build_provenance_graph(shas[0], nodes)
        assert len(graph) == MAX_PROVENANCE_GRAPH_NODES

    def test_duplicate_nodes_in_input(self):
        """If same SHA256 appears multiple times in input, last one wins in index, but graph is correct."""
        nodes = [
            _node(SHA_A, sources=(SHA_B,)),
            _node(SHA_B),
            DocumentNode(sha256=SHA_A, class_name="Other", name="duplicate", sources=(SHA_B,)),
        ]
        graph = build_provenance_graph(SHA_A, nodes)
        assert SHA_A in graph
        assert SHA_B in graph


class TestWalkProvenance:
    """Tests for the async walk_provenance BFS helper."""

    @pytest.mark.asyncio
    async def test_linear_chain(self):
        """Walks A -> B -> C via batch lookups."""
        all_nodes = {
            SHA_A: _node(SHA_A, sources=(SHA_B,)),
            SHA_B: _node(SHA_B, sources=(SHA_C,)),
            SHA_C: _node(SHA_C),
        }

        async def loader(sha256s: list[str]) -> dict[str, DocumentNode]:
            return {s: all_nodes[s] for s in sha256s if s in all_nodes}

        graph = await walk_provenance(SHA_A, loader)
        assert set(graph.keys()) == {SHA_A, SHA_B, SHA_C}

    @pytest.mark.asyncio
    async def test_unknown_root_returns_empty(self):
        async def loader(sha256s: list[str]) -> dict[str, DocumentNode]:
            return {}

        graph = await walk_provenance(SHA_A, loader)
        assert graph == {}

    @pytest.mark.asyncio
    async def test_url_sources_not_followed(self):
        all_nodes = {
            SHA_A: _node(SHA_A, sources=("https://example.com", SHA_B)),
            SHA_B: _node(SHA_B),
        }

        async def loader(sha256s: list[str]) -> dict[str, DocumentNode]:
            return {s: all_nodes[s] for s in sha256s if s in all_nodes}

        graph = await walk_provenance(SHA_A, loader)
        assert set(graph.keys()) == {SHA_A, SHA_B}

    @pytest.mark.asyncio
    async def test_diamond_pattern(self):
        all_nodes = {
            SHA_A: _node(SHA_A, sources=(SHA_B, SHA_C)),
            SHA_B: _node(SHA_B, sources=(SHA_D,)),
            SHA_C: _node(SHA_C, sources=(SHA_D,)),
            SHA_D: _node(SHA_D),
        }

        async def loader(sha256s: list[str]) -> dict[str, DocumentNode]:
            return {s: all_nodes[s] for s in sha256s if s in all_nodes}

        graph = await walk_provenance(SHA_A, loader)
        assert set(graph.keys()) == {SHA_A, SHA_B, SHA_C, SHA_D}

    @pytest.mark.asyncio
    async def test_dangling_refs_stop_gracefully(self):
        """References to nonexistent nodes don't cause errors."""
        all_nodes = {
            SHA_A: _node(SHA_A, sources=(SHA_B,)),
            # SHA_B not available
        }

        async def loader(sha256s: list[str]) -> dict[str, DocumentNode]:
            return {s: all_nodes[s] for s in sha256s if s in all_nodes}

        graph = await walk_provenance(SHA_A, loader)
        assert set(graph.keys()) == {SHA_A}

    @pytest.mark.asyncio
    async def test_cycle_handling(self):
        all_nodes = {
            SHA_A: _node(SHA_A, sources=(SHA_B,)),
            SHA_B: _node(SHA_B, sources=(SHA_A,)),
        }

        async def loader(sha256s: list[str]) -> dict[str, DocumentNode]:
            return {s: all_nodes[s] for s in sha256s if s in all_nodes}

        graph = await walk_provenance(SHA_A, loader)
        assert set(graph.keys()) == {SHA_A, SHA_B}

    @pytest.mark.asyncio
    async def test_batch_efficiency(self):
        """Verifies batching: each BFS level is loaded in one call, not per-node."""
        call_count = 0
        all_nodes = {
            SHA_A: _node(SHA_A, sources=(SHA_B, SHA_C)),
            SHA_B: _node(SHA_B, sources=(SHA_D,)),
            SHA_C: _node(SHA_C, sources=(SHA_D,)),
            SHA_D: _node(SHA_D),
        }

        async def loader(sha256s: list[str]) -> dict[str, DocumentNode]:
            nonlocal call_count
            call_count += 1
            return {s: all_nodes[s] for s in sha256s if s in all_nodes}

        graph = await walk_provenance(SHA_A, loader)
        assert set(graph.keys()) == {SHA_A, SHA_B, SHA_C, SHA_D}
        # Level 0: [A], Level 1: [B, C], Level 2: [D] = 3 calls
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_mixed_sources_and_origins(self):
        """Both sources and origins edges are followed."""
        all_nodes = {
            SHA_A: _node(SHA_A, sources=(SHA_B,), origins=(SHA_C,)),
            SHA_B: _node(SHA_B),
            SHA_C: _node(SHA_C, origins=(SHA_D,)),
            SHA_D: _node(SHA_D),
        }

        async def loader(sha256s: list[str]) -> dict[str, DocumentNode]:
            return {s: all_nodes[s] for s in sha256s if s in all_nodes}

        graph = await walk_provenance(SHA_A, loader)
        assert set(graph.keys()) == {SHA_A, SHA_B, SHA_C, SHA_D}

    @pytest.mark.asyncio
    async def test_traversal_stops_at_max_nodes(self):
        """BFS stops at MAX_PROVENANCE_GRAPH_NODES."""
        count = MAX_PROVENANCE_GRAPH_NODES + 500
        shas = [base64.b32encode(hashlib.sha256(str(i).encode()).digest()).decode()[:52] for i in range(count)]
        all_nodes = {shas[i]: _node(shas[i], sources=(shas[i + 1],) if i < count - 1 else ()) for i in range(count)}

        async def loader(sha256s: list[str]) -> dict[str, DocumentNode]:
            return {s: all_nodes[s] for s in sha256s if s in all_nodes}

        graph = await walk_provenance(shas[0], loader)
        assert len(graph) == MAX_PROVENANCE_GRAPH_NODES

    @pytest.mark.asyncio
    async def test_pending_not_dropped_when_batch_capped(self):
        """When batch is capped, remaining pending items are preserved for next iteration."""
        # Create a wide graph: root -> 10 children, each child -> grandchild
        shas = [base64.b32encode(hashlib.sha256(str(i).encode()).digest()).decode()[:52] for i in range(12)]
        root = shas[0]
        children = shas[1:11]
        grandchild = shas[11]

        all_nodes: dict[str, DocumentNode] = {root: _node(root, sources=tuple(children))}
        for child in children:
            all_nodes[child] = _node(child, sources=(grandchild,))
        all_nodes[grandchild] = _node(grandchild)

        async def loader(sha256s: list[str]) -> dict[str, DocumentNode]:
            return {s: all_nodes[s] for s in sha256s if s in all_nodes}

        graph = await walk_provenance(root, loader)
        assert len(graph) == 12  # root + 10 children + 1 grandchild
