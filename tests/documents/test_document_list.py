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

        # When there's only one of a type
        docs_single = DocumentList([doc2, doc3])
        result = docs_single.get_by(AnotherDoc)
        assert result == doc2

        result = docs_single.get_by(SubDoc)
        assert result == doc3

        # Test that get_by returns first match when there's only one
        docs_with_sample = DocumentList([doc1, doc2])
        result = docs_with_sample.get_by(SampleDoc)
        assert result == doc1  # Returns first SampleDoc

        # Get non-existing type (required=True, default)
        class NonExistentDoc(FlowDocument):
            pass

        with pytest.raises(ValueError, match="Document of type 'NonExistentDoc' not found"):
            docs_single.get_by(NonExistentDoc)

        # Get non-existing type (required=False)
        result = docs_single.get_by(NonExistentDoc, required=False)
        assert result is None

    def test_get_by_invalid_arg(self) -> None:
        """Test get_by with invalid argument type."""
        docs = DocumentList()

        with pytest.raises(TypeError, match="Invalid argument type"):
            docs.get_by(123)  # type: ignore

    def test_get_by_multiple_matches_by_name(self) -> None:
        """Test that get_by raises error when multiple documents have same name."""
        # This shouldn't normally happen unless validate_duplicates=False
        doc1 = SampleDoc(name="config.yaml", content=b"content1")
        doc2 = AnotherDoc(name="config.yaml", content=b"content2")

        docs = DocumentList([doc1, doc2], validate_duplicates=False)

        # Should raise error for multiple matches
        with pytest.raises(ValueError, match="Multiple documents found with name 'config.yaml'"):
            docs.get_by("config.yaml")

        # The error message should suggest using filter_by
        with pytest.raises(ValueError, match="Use filter_by\\(\\) to get all matches"):
            docs.get_by("config.yaml")

    def test_get_by_multiple_matches_by_type(self) -> None:
        """Test that get_by raises error when multiple documents of same type exist."""
        doc1 = SampleDoc(name="file1.txt", content=b"content1")
        doc2 = SampleDoc(name="file2.txt", content=b"content2")
        doc3 = SubDoc(name="file3.txt", content=b"content3")  # SubDoc is also a SampleDoc

        docs = DocumentList([doc1, doc2, doc3])

        # Should raise error for multiple SampleDoc matches
        with pytest.raises(ValueError, match="Multiple documents found of type 'SampleDoc'"):
            docs.get_by(SampleDoc)

        # The error should mention the count
        with pytest.raises(ValueError, match="Found 3 matches"):
            docs.get_by(SampleDoc)

        # Should work fine when there's only one AnotherDoc
        doc4 = AnotherDoc(name="file4.txt", content=b"content4")
        docs2 = DocumentList([doc1, doc4])
        result = docs2.get_by(AnotherDoc)
        assert result == doc4  # Works when unique

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
