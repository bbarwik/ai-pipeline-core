"""Tests for DocumentList container."""

import pytest

from ai_pipeline_core.documents import DocumentList, FlowDocument


class SampleDoc(FlowDocument):
    """Test document class."""


class AnotherDoc(FlowDocument):
    """Another test document class."""


class SubDoc(SampleDoc):
    """Subclass of SampleDoc."""


class TestDocumentList:
    """Test DocumentList functionality."""

    def test_filter_by_name(self) -> None:
        """Test filtering by document name."""
        doc1 = SampleDoc(name="file1.txt", content=b"content1")
        doc2 = SampleDoc(name="file2.txt", content=b"content2")
        doc3 = AnotherDoc(name="file1.txt", content=b"content3")

        docs = DocumentList([doc1, doc2, doc3])

        # Filter by name
        filtered = docs.filter_by("file1.txt")
        assert len(filtered) == 2
        assert doc1 in filtered
        assert doc3 in filtered
        assert doc2 not in filtered

    def test_filter_by_type(self) -> None:
        """Test filtering by document type."""
        doc1 = SampleDoc(name="file1.txt", content=b"content1")
        doc2 = AnotherDoc(name="file2.txt", content=b"content2")
        doc3 = SubDoc(name="file3.txt", content=b"content3")

        docs = DocumentList([doc1, doc2, doc3])

        # Filter by single type
        filtered = docs.filter_by(SampleDoc)
        assert len(filtered) == 2  # Includes SubDoc (subclass)
        assert doc1 in filtered
        assert doc3 in filtered
        assert doc2 not in filtered

        # Filter by exact type
        filtered = docs.filter_by(SubDoc)
        assert len(filtered) == 1
        assert doc3 in filtered

    def test_filter_by_multiple_types(self) -> None:
        """Test filtering by multiple document types."""
        doc1 = SampleDoc(name="file1.txt", content=b"content1")
        doc2 = AnotherDoc(name="file2.txt", content=b"content2")
        doc3 = SubDoc(name="file3.txt", content=b"content3")

        docs = DocumentList([doc1, doc2, doc3])

        # Filter by multiple types
        filtered = docs.filter_by([AnotherDoc, SubDoc])
        assert len(filtered) == 2
        assert doc2 in filtered
        assert doc3 in filtered
        assert doc1 not in filtered

    def test_filter_by_invalid_arg(self) -> None:
        """Test filter_by with invalid argument type."""
        docs = DocumentList()

        with pytest.raises(TypeError, match="Invalid argument type"):
            docs.filter_by(123)  # type: ignore

    def test_get_by_name(self) -> None:
        """Test getting document by name."""
        doc1 = SampleDoc(name="file1.txt", content=b"content1")
        doc2 = SampleDoc(name="file2.txt", content=b"content2")

        docs = DocumentList([doc1, doc2])

        # Get existing document
        result = docs.get_by("file1.txt")
        assert result == doc1

        # Get non-existing document (required=True, default)
        with pytest.raises(ValueError, match="Document with name 'file3.txt' not found"):
            docs.get_by("file3.txt")

        # Get non-existing document (required=False)
        result = docs.get_by("file3.txt", required=False)
        assert result is None

    def test_get_by_type(self) -> None:
        """Test getting document by type."""
        doc1 = SampleDoc(name="file1.txt", content=b"content1")
        doc2 = AnotherDoc(name="file2.txt", content=b"content2")
        doc3 = SubDoc(name="file3.txt", content=b"content3")

        docs = DocumentList([doc1, doc2, doc3])

        # Get by type (gets first match)
        result = docs.get_by(SampleDoc)
        assert result == doc1  # First SampleDoc (or subclass)

        result = docs.get_by(SubDoc)
        assert result == doc3

        # Get non-existing type (required=True, default)
        class NonExistentDoc(FlowDocument):
            pass

        with pytest.raises(ValueError, match="Document of type 'NonExistentDoc' not found"):
            docs.get_by(NonExistentDoc)

        # Get non-existing type (required=False)
        result = docs.get_by(NonExistentDoc, required=False)
        assert result is None

    def test_get_by_invalid_arg(self) -> None:
        """Test get_by with invalid argument type."""
        docs = DocumentList()

        with pytest.raises(TypeError, match="Invalid argument type"):
            docs.get_by(123)  # type: ignore

    def test_backwards_compatibility(self) -> None:
        """Test that new methods provide same functionality as old ones."""
        doc1 = SampleDoc(name="file1.txt", content=b"content1")
        doc2 = AnotherDoc(name="file2.txt", content=b"content2")
        doc3 = SubDoc(name="file3.txt", content=b"content3")

        docs = DocumentList([doc1, doc2, doc3])

        # Old filter_by_type behavior
        filtered = docs.filter_by(SampleDoc)
        assert len(filtered) == 2  # Includes subclasses

        # Old filter_by_types behavior
        filtered = docs.filter_by([SampleDoc, AnotherDoc])
        assert len(filtered) == 3  # All documents

        # Old get_by_name behavior
        result = docs.get_by("file1.txt", required=False)
        assert result == doc1

        result = docs.get_by("nonexistent.txt", required=False)
        assert result is None
