"""Tests for Document sources field and methods."""

import pytest

from ai_pipeline_core.documents import FlowDocument, TemporaryDocument


class SampleFlowDoc(FlowDocument):
    """Sample flow document for testing."""


class TestDocumentSources:
    """Test Document sources functionality."""

    def test_create_document_with_sources(self):
        """Test creating document with sources."""
        sources = [
            "https://example.com",
            "P3AEMA2PSYILKFYVBUALJLMIYWVZIS2QDI3S5VTMD2X7SOODF2YQ",  # Valid SHA256
        ]

        doc = SampleFlowDoc.create(name="test.txt", content="test content", sources=sources)

        assert len(doc.sources) == 2
        assert doc.sources[0] == "https://example.com"
        assert doc.sources[1] == "P3AEMA2PSYILKFYVBUALJLMIYWVZIS2QDI3S5VTMD2X7SOODF2YQ"

    def test_default_empty_sources(self):
        """Test that sources defaults to empty list."""
        doc = SampleFlowDoc.create(name="test.txt", content="test")
        assert doc.sources == []

    def test_with_source_document(self):
        """Test adding a document as source."""
        doc1 = SampleFlowDoc.create(name="source.txt", content="source data")

        # Create doc2 with doc1 as source
        doc2 = SampleFlowDoc.create(
            name="derived.txt", content="derived data", sources=[doc1.sha256]
        )

        assert len(doc2.sources) == 1
        assert doc2.sources[0] == doc1.sha256

    def test_create_with_string_source(self):
        """Test creating document with a string reference as source."""
        doc = SampleFlowDoc.create(
            name="test.txt", content="test", sources=["https://data.source.com/file.csv"]
        )

        assert len(doc.sources) == 1
        assert doc.sources[0] == "https://data.source.com/file.csv"

    def test_create_with_mixed_sources(self):
        """Test creating document with mixed source types."""
        doc = SampleFlowDoc.create(
            name="test.txt",
            content="test",
            sources=[
                "manual input",
                "P3AEMA2PSYILKFYVBUALJLMIYWVZIS2QDI3S5VTMD2X7SOODF2YQ",
            ],
        )

        assert len(doc.sources) == 2
        assert doc.sources[0] == "manual input"
        assert doc.sources[1] == "P3AEMA2PSYILKFYVBUALJLMIYWVZIS2QDI3S5VTMD2X7SOODF2YQ"

    def test_create_with_duplicate_sources(self):
        """Test creating document with duplicate sources."""
        doc = SampleFlowDoc.create(name="test.txt", content="test", sources=["ref1", "ref1"])

        # Both sources are kept (no deduplication at creation)
        assert len(doc.sources) == 2

    def test_get_source_documents(self):
        """Test getting document hash sources."""
        sources = [
            "P3AEMA2PSYILKFYVBUALJLMIYWVZIS2QDI3S5VTMD2X7SOODF2YQ",
            "https://example.com",
            "DSITTXMIGUJ5CHKJEVTW3IOQFYJ3LHOXZFWZBN7FH7AR3DGWTAXA",
        ]

        doc = SampleFlowDoc.create(name="test.txt", content="test", sources=sources)

        hashes = doc.get_source_documents()
        assert len(hashes) == 2
        assert "P3AEMA2PSYILKFYVBUALJLMIYWVZIS2QDI3S5VTMD2X7SOODF2YQ" in hashes
        assert "DSITTXMIGUJ5CHKJEVTW3IOQFYJ3LHOXZFWZBN7FH7AR3DGWTAXA" in hashes

    def test_get_source_references(self):
        """Test getting reference sources."""
        sources = [
            "P3AEMA2PSYILKFYVBUALJLMIYWVZIS2QDI3S5VTMD2X7SOODF2YQ",
            "ref1",
            "ref2",
        ]

        doc = SampleFlowDoc.create(name="test.txt", content="test", sources=sources)

        refs = doc.get_source_references()
        assert len(refs) == 2
        assert "ref1" in refs
        assert "ref2" in refs

    def test_has_source(self):
        """Test checking for specific sources."""
        doc1 = SampleFlowDoc.create(name="source.txt", content="source")
        doc2 = SampleFlowDoc.create(name="other.txt", content="other")

        doc = SampleFlowDoc.create(
            name="test.txt",
            content="test",
            sources=[
                doc1.sha256,
                "ref1",
            ],
        )

        # Check by document
        assert doc.has_source(doc1)
        assert not doc.has_source(doc2)

        # Check by string reference
        assert doc.has_source("ref1")
        assert not doc.has_source("ref2")

        # Check by SHA256 string
        assert doc.has_source(doc1.sha256)
        assert not doc.has_source(doc2.sha256)

    def test_serialize_with_sources(self):
        """Test serialization includes sources."""
        sources = [
            "ref1",
            "P3AEMA2PSYILKFYVBUALJLMIYWVZIS2QDI3S5VTMD2X7SOODF2YQ",
        ]

        doc = SampleFlowDoc.create(name="test.txt", content="test", sources=sources)

        data = doc.serialize_model()
        assert "sources" in data
        assert len(data["sources"]) == 2
        assert data["sources"][0] == "ref1"
        assert data["sources"][1] == "P3AEMA2PSYILKFYVBUALJLMIYWVZIS2QDI3S5VTMD2X7SOODF2YQ"

    def test_from_dict_with_sources(self):
        """Test deserialization includes sources."""
        data = {
            "name": "test.txt",
            "content": "test content",
            "sources": [
                "ref1",
                "P3AEMA2PSYILKFYVBUALJLMIYWVZIS2QDI3S5VTMD2X7SOODF2YQ",
            ],
        }

        doc = SampleFlowDoc.from_dict(data)
        assert len(doc.sources) == 2
        assert doc.sources[0] == "ref1"
        assert doc.sources[1] == "P3AEMA2PSYILKFYVBUALJLMIYWVZIS2QDI3S5VTMD2X7SOODF2YQ"

    def test_sources_immutable(self):
        """Test that sources list itself is immutable."""
        from pydantic import ValidationError

        doc = SampleFlowDoc.create(name="test.txt", content="test", sources=["ref1"])

        # Document is frozen, so we can't modify sources directly
        with pytest.raises(ValidationError, match="frozen"):
            doc.sources = []

    def test_temporary_document_with_sources(self):
        """Test that TemporaryDocument also supports sources."""
        sources = ["temp source"]
        doc = TemporaryDocument.create(name="temp.txt", content="temporary", sources=sources)

        assert len(doc.sources) == 1
        assert doc.sources[0] == "temp source"

    def test_low_entropy_hash_not_counted_as_document(self):
        """Test that low-entropy strings are not counted as document hashes."""
        sources = [
            "A" * 52,  # Low entropy - not a real hash
            "https://example.com",
            "P3AEMA2PSYILKFYVBUALJLMIYWVZIS2QDI3S5VTMD2X7SOODF2YQ",  # Real hash
        ]

        doc = SampleFlowDoc.create(name="test.txt", content="test", sources=sources)

        # get_source_documents should only return the real hash
        hashes = doc.get_source_documents()
        assert len(hashes) == 1
        assert hashes[0] == "P3AEMA2PSYILKFYVBUALJLMIYWVZIS2QDI3S5VTMD2X7SOODF2YQ"

        # get_source_references should return the low-entropy string and URL
        refs = doc.get_source_references()
        assert len(refs) == 2
        assert "A" * 52 in refs
        assert "https://example.com" in refs

    def test_has_source_invalid_type(self):
        """Test that has_source raises TypeError for invalid types."""
        doc = SampleFlowDoc.create(name="test.txt", content="test", sources=["ref1"])

        # Invalid type should raise TypeError
        from typing import Any, cast

        with pytest.raises(TypeError, match="Invalid source type"):
            doc.has_source(cast(Any, 123))  # Invalid: int

        with pytest.raises(TypeError, match="Invalid source type"):
            doc.has_source(cast(Any, ["list"]))  # Invalid: list

        with pytest.raises(TypeError, match="Invalid source type"):
            doc.has_source(cast(Any, {"dict": "value"}))  # Invalid: dict

    def test_empty_sources_methods(self):
        """Test methods behavior with empty sources."""
        doc = SampleFlowDoc.create(
            name="test.txt",
            content="test",
            # No sources parameter - defaults to empty list
        )

        # All methods should work with empty sources
        assert doc.sources == []
        assert doc.get_source_documents() == []
        assert doc.get_source_references() == []
        assert not doc.has_source("anything")
        assert not doc.has_source(doc)  # Even checking self returns False

    def test_sources_in_direct_constructor(self):
        """Test sources parameter in direct __init__ constructor."""
        sources = ["ref1", "ref2"]
        doc = SampleFlowDoc(name="test.txt", content=b"test bytes", sources=sources)

        assert doc.sources == sources
        assert len(doc.get_source_references()) == 2

    def test_multiple_identical_hashes(self):
        """Test handling of multiple identical document hashes."""
        doc1 = SampleFlowDoc.create(name="source.txt", content="source")

        # Create doc with duplicate hash sources
        doc2 = SampleFlowDoc.create(
            name="derived.txt", content="derived", sources=[doc1.sha256, doc1.sha256, "other"]
        )

        # All duplicates are kept
        assert len(doc2.sources) == 3
        hashes = doc2.get_source_documents()
        assert len(hashes) == 2  # Both copies of the hash
        assert all(h == doc1.sha256 for h in hashes)

        # has_source still works
        assert doc2.has_source(doc1)
        assert doc2.has_source(doc1.sha256)

    def test_self_referential_source(self):
        """Test document referencing itself as source."""
        doc = SampleFlowDoc.create(name="test.txt", content="test")

        # Create another doc that references itself (edge case)
        doc2 = SampleFlowDoc.create(
            name="self.txt",
            content="self-ref",
            sources=[doc.sha256],  # Reference to first doc
        )

        # Add its own hash (would be unusual but should work)
        doc3 = SampleFlowDoc.create(
            name="self2.txt", content="self-ref2", sources=[doc2.sha256, "self-reference-note"]
        )

        assert doc3.has_source(doc2)
        assert len(doc3.get_source_documents()) == 1

    def test_serialize_deserialize_preserves_sources(self):
        """Test full roundtrip serialization preserves sources."""
        sources = [
            "https://example.com/data",
            "P3AEMA2PSYILKFYVBUALJLMIYWVZIS2QDI3S5VTMD2X7SOODF2YQ",
            "local-file.csv",
        ]

        original = SampleFlowDoc.create(
            name="test.json", content={"key": "value"}, description="Test doc", sources=sources
        )

        # Serialize and deserialize
        serialized = original.serialize_model()
        restored = SampleFlowDoc.from_dict(serialized)

        # Everything should match
        assert restored.name == original.name
        assert restored.content == original.content
        assert restored.description == original.description
        assert restored.sources == original.sources
        assert restored.sha256 == original.sha256  # Content-based, should match

    def test_sources_with_special_characters(self):
        """Test sources containing special characters."""
        sources = [
            "file with spaces.txt",
            "path/to/file.csv",
            "https://example.com/data?param=value&other=123",
            "unicode-źćąęł.txt",
            "tab\tseparated",
            "newline\nincluded",
        ]

        doc = SampleFlowDoc.create(name="test.txt", content="test", sources=sources)

        # All sources should be preserved exactly
        assert doc.sources == sources
        refs = doc.get_source_references()
        assert len(refs) == len(sources)
        for src in sources:
            assert src in refs
