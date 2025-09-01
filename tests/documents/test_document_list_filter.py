"""Tests for DocumentList filter_by method with all overloads."""

import pytest

from ai_pipeline_core.documents import Document, DocumentList, FlowDocument


class SampleDoc(FlowDocument):
    """Test document class."""


class AnotherDoc(FlowDocument):
    """Another test document class."""


class SubDoc(SampleDoc):
    """Subclass of SampleDoc."""


class TestDocumentListFilter:
    """Test DocumentList filter_by functionality with all overloads."""

    def test_filter_by_single_name(self) -> None:
        """Test filtering by single document name (str overload)."""
        doc1 = SampleDoc(name="file1.txt", content=b"content1")
        doc2 = SampleDoc(name="file2.txt", content=b"content2")
        doc3 = AnotherDoc(name="file1.txt", content=b"content3")

        docs = DocumentList([doc1, doc2, doc3])

        # Filter by single name
        filtered = docs.filter_by("file1.txt")
        assert len(filtered) == 2
        assert doc1 in filtered
        assert doc3 in filtered
        assert doc2 not in filtered

        # Filter by non-existing name
        filtered = docs.filter_by("nonexistent.txt")
        assert len(filtered) == 0

    def test_filter_by_single_type(self) -> None:
        """Test filtering by single document type (type[Document] overload)."""
        doc1 = SampleDoc(name="file1.txt", content=b"content1")
        doc2 = AnotherDoc(name="file2.txt", content=b"content2")
        doc3 = SubDoc(name="file3.txt", content=b"content3")

        docs = DocumentList([doc1, doc2, doc3])

        # Filter by single type (includes subclasses)
        filtered = docs.filter_by(SampleDoc)
        assert len(filtered) == 2  # Includes SubDoc
        assert doc1 in filtered
        assert doc3 in filtered
        assert doc2 not in filtered

        # Filter by exact subclass type
        filtered = docs.filter_by(SubDoc)
        assert len(filtered) == 1
        assert doc3 in filtered

        # Filter by AnotherDoc
        filtered = docs.filter_by(AnotherDoc)
        assert len(filtered) == 1
        assert doc2 in filtered

    def test_filter_by_iterator_of_types(self) -> None:
        """Test filtering by iterator of document types (Iterator[type[Document]] overload)."""
        doc1 = SampleDoc(name="file1.txt", content=b"content1")
        doc2 = AnotherDoc(name="file2.txt", content=b"content2")
        doc3 = SubDoc(name="file3.txt", content=b"content3")

        docs = DocumentList([doc1, doc2, doc3])

        # Filter by multiple types using iterator
        filtered = docs.filter_by(iter([AnotherDoc, SubDoc]))
        assert len(filtered) == 2
        assert doc2 in filtered
        assert doc3 in filtered
        assert doc1 not in filtered

        # Filter by all types
        filtered = docs.filter_by(iter([SampleDoc, AnotherDoc]))
        assert len(filtered) == 3  # All documents (SubDoc is included via SampleDoc)

        # Filter by empty iterator of types
        filtered = docs.filter_by(iter([]))
        assert len(filtered) == 0

        # Using generator expression
        types_to_filter = (t for t in [SubDoc, AnotherDoc])
        filtered = docs.filter_by(types_to_filter)
        assert len(filtered) == 2
        assert doc2 in filtered
        assert doc3 in filtered

    def test_filter_by_iterator_of_names(self) -> None:
        """Test filtering by iterator of document names (Iterator[str] overload)."""
        doc1 = SampleDoc(name="file1.txt", content=b"content1")
        doc2 = SampleDoc(name="file2.txt", content=b"content2")
        doc3 = AnotherDoc(name="file3.txt", content=b"content3")
        doc4 = SubDoc(name="file1.txt", content=b"content4")  # Same name as doc1

        docs = DocumentList([doc1, doc2, doc3, doc4])

        # Filter by multiple names using iterator
        filtered = docs.filter_by(iter(["file1.txt", "file3.txt"]))
        assert len(filtered) == 3  # doc1, doc3, doc4
        assert doc1 in filtered
        assert doc3 in filtered
        assert doc4 in filtered
        assert doc2 not in filtered

        # Filter by single name in iterator
        filtered = docs.filter_by(iter(["file2.txt"]))
        assert len(filtered) == 1
        assert doc2 in filtered

        # Filter by empty iterator of names
        filtered = docs.filter_by(iter([]))
        assert len(filtered) == 0

        # Filter with non-existing names
        filtered = docs.filter_by(iter(["nonexistent1.txt", "nonexistent2.txt"]))
        assert len(filtered) == 0

        # Mix of existing and non-existing names
        filtered = docs.filter_by(iter(["file1.txt", "nonexistent.txt"]))
        assert len(filtered) == 2  # Only doc1 and doc4
        assert doc1 in filtered
        assert doc4 in filtered

        # Using generator expression
        names_to_filter = (name for name in ["file2.txt", "file3.txt"])
        filtered = docs.filter_by(names_to_filter)
        assert len(filtered) == 2
        assert doc2 in filtered
        assert doc3 in filtered

    def test_filter_by_list_as_iterator(self) -> None:
        """Test that lists work as iterators (backward compatibility)."""
        doc1 = SampleDoc(name="file1.txt", content=b"content1")
        doc2 = AnotherDoc(name="file2.txt", content=b"content2")
        doc3 = SubDoc(name="file3.txt", content=b"content3")

        docs = DocumentList([doc1, doc2, doc3])

        # Lists should work as iterators of types
        filtered = docs.filter_by([AnotherDoc, SubDoc])  # type: ignore[arg-type]
        assert len(filtered) == 2
        assert doc2 in filtered
        assert doc3 in filtered

        # Lists should work as iterators of names
        filtered = docs.filter_by(["file1.txt", "file2.txt"])  # type: ignore[arg-type]
        assert len(filtered) == 2
        assert doc1 in filtered
        assert doc2 in filtered

    def test_filter_by_invalid_arguments(self) -> None:
        """Test filter_by with invalid argument types."""
        doc1 = SampleDoc(name="file1.txt", content=b"content1")
        docs = DocumentList([doc1])

        # Invalid type (not str, type, or iterator)
        with pytest.raises(TypeError, match="Invalid argument type"):
            docs.filter_by(123)  # type: ignore[arg-type]

        # Invalid iterator content (not str or type[Document])
        with pytest.raises(TypeError, match="Iterable must contain strings or Document types"):
            docs.filter_by(iter([123, 456]))  # type: ignore[arg-type]

        # Mixed iterator content
        with pytest.raises(
            TypeError, match="Iterable must contain only strings or only Document types"
        ):
            docs.filter_by(iter(["file1.txt", SampleDoc]))  # type: ignore[list-item]

    def test_filter_by_with_empty_document_list(self) -> None:
        """Test filter_by on empty DocumentList."""
        docs = DocumentList()

        # Filter by name on empty list
        filtered = docs.filter_by("file1.txt")
        assert len(filtered) == 0

        # Filter by type on empty list
        filtered = docs.filter_by(SampleDoc)
        assert len(filtered) == 0

        # Filter by iterator of types on empty list
        filtered = docs.filter_by(iter([SampleDoc, AnotherDoc]))
        assert len(filtered) == 0

        # Filter by iterator of names on empty list
        filtered = docs.filter_by(iter(["file1.txt", "file2.txt"]))
        assert len(filtered) == 0

    def test_filter_by_duplicates_in_iterator(self) -> None:
        """Test that duplicates in iterator don't cause duplicate results."""
        doc1 = SampleDoc(name="file1.txt", content=b"content1")
        doc2 = AnotherDoc(name="file2.txt", content=b"content2")

        docs = DocumentList([doc1, doc2])

        # Duplicate types in iterator
        filtered = docs.filter_by(iter([SampleDoc, SampleDoc]))
        assert len(filtered) == 1
        assert doc1 in filtered

        # Duplicate names in iterator
        filtered = docs.filter_by(iter(["file1.txt", "file1.txt", "file1.txt"]))
        assert len(filtered) == 1
        assert doc1 in filtered

    def test_filter_by_complex_scenarios(self) -> None:
        """Test complex filtering scenarios."""
        # Create documents with various combinations
        doc1 = SampleDoc(name="report.pdf", content=b"pdf1")
        doc2 = SampleDoc(name="data.csv", content=b"csv1")
        doc3 = AnotherDoc(name="report.pdf", content=b"pdf2")  # Same name, different type
        doc4 = SubDoc(name="analysis.txt", content=b"txt1")
        doc5 = AnotherDoc(name="summary.doc", content=b"doc1")

        docs = DocumentList([doc1, doc2, doc3, doc4, doc5])

        # Filter by name that appears in multiple types
        filtered = docs.filter_by("report.pdf")
        assert len(filtered) == 2
        assert doc1 in filtered
        assert doc3 in filtered

        # Filter by base type that has subclasses
        filtered = docs.filter_by(SampleDoc)
        assert len(filtered) == 3  # doc1, doc2, doc4 (SubDoc)
        assert doc1 in filtered
        assert doc2 in filtered
        assert doc4 in filtered

        # Filter by multiple names with overlapping results
        filtered = docs.filter_by(iter(["report.pdf", "data.csv", "nonexistent.txt"]))
        assert len(filtered) == 3  # doc1, doc2, doc3
        assert doc1 in filtered
        assert doc2 in filtered
        assert doc3 in filtered

        # Filter by multiple types with inheritance
        filtered = docs.filter_by(iter([SampleDoc, AnotherDoc]))
        assert len(filtered) == 5  # All documents

    def test_filter_by_sets(self) -> None:
        """Test filtering with sets as iterables."""
        doc1 = SampleDoc(name="file1.txt", content=b"content1")
        doc2 = AnotherDoc(name="file2.txt", content=b"content2")
        doc3 = SubDoc(name="file3.txt", content=b"content3")
        doc4 = SampleDoc(name="file4.txt", content=b"content4")

        docs = DocumentList([doc1, doc2, doc3, doc4])

        # Filter by set of types
        filtered = docs.filter_by({AnotherDoc, SubDoc})  # type: ignore[arg-type]
        assert len(filtered) == 2
        assert doc2 in filtered
        assert doc3 in filtered

        # Filter by set of names
        filtered = docs.filter_by({"file1.txt", "file3.txt", "file4.txt"})  # type: ignore[arg-type]
        assert len(filtered) == 3
        assert doc1 in filtered
        assert doc3 in filtered
        assert doc4 in filtered

        # Empty set
        filtered = docs.filter_by(set())  # type: ignore[arg-type]
        assert len(filtered) == 0

    def test_filter_by_tuples(self) -> None:
        """Test filtering with tuples as iterables."""
        doc1 = SampleDoc(name="file1.txt", content=b"content1")
        doc2 = AnotherDoc(name="file2.txt", content=b"content2")
        doc3 = SubDoc(name="file3.txt", content=b"content3")

        docs = DocumentList([doc1, doc2, doc3])

        # Filter by tuple of types
        filtered = docs.filter_by((SampleDoc, AnotherDoc))  # type: ignore[arg-type]
        assert len(filtered) == 3  # All documents (SubDoc is included via SampleDoc)

        # Filter by tuple of names
        filtered = docs.filter_by(("file2.txt", "file3.txt"))  # type: ignore[arg-type]
        assert len(filtered) == 2
        assert doc2 in filtered
        assert doc3 in filtered

        # Single-element tuple
        filtered = docs.filter_by(("file1.txt",))  # type: ignore[arg-type]
        assert len(filtered) == 1
        assert doc1 in filtered

        # Empty tuple
        filtered = docs.filter_by(())  # type: ignore[arg-type]
        assert len(filtered) == 0

    def test_filter_by_custom_iterable(self) -> None:
        """Test filtering with custom iterable objects."""
        doc1 = SampleDoc(name="file1.txt", content=b"content1")
        doc2 = AnotherDoc(name="file2.txt", content=b"content2")
        doc3 = SubDoc(name="file3.txt", content=b"content3")

        docs = DocumentList([doc1, doc2, doc3])

        # Using range() as an iterable (though unusual, should handle gracefully)
        with pytest.raises(TypeError, match="Iterable must contain strings or Document types"):
            docs.filter_by(range(3))  # type: ignore[arg-type]

        # Using dict keys as iterable (which are strings)
        name_dict = {"file1.txt": 1, "file3.txt": 2}
        filtered = docs.filter_by(name_dict.keys())  # type: ignore[arg-type]
        assert len(filtered) == 2
        assert doc1 in filtered
        assert doc3 in filtered

        # Using dict values (should fail with appropriate error)
        type_dict = {1: SampleDoc, 2: AnotherDoc}
        filtered = docs.filter_by(type_dict.values())  # type: ignore[arg-type]
        assert len(filtered) == 3  # All documents match

    def test_filter_by_none_and_invalid_iterables(self) -> None:
        """Test edge cases with None and non-iterable types."""
        doc1 = SampleDoc(name="file1.txt", content=b"content1")
        docs = DocumentList([doc1])

        # None should fail
        with pytest.raises(TypeError, match="Invalid argument type"):
            docs.filter_by(None)  # type: ignore[arg-type]

        # Integer (non-iterable)
        with pytest.raises(TypeError, match="Invalid argument type"):
            docs.filter_by(42)  # type: ignore[arg-type]

        # Float (non-iterable)
        with pytest.raises(TypeError, match="Invalid argument type"):
            docs.filter_by(3.14)  # type: ignore[arg-type]

        # Boolean (technically iterable in Python but not meaningful here)
        with pytest.raises(TypeError, match="Invalid argument type"):
            docs.filter_by(True)  # type: ignore[arg-type]

    def test_filter_by_performance_with_large_lists(self) -> None:
        """Test that set conversion provides efficient lookup for large iterables."""
        # Create many documents
        sample_docs: list[Document] = [
            SampleDoc(name=f"sample_{i}.txt", content=f"content_{i}".encode()) for i in range(50)
        ]
        another_docs: list[Document] = [
            AnotherDoc(name=f"another_{i}.txt", content=f"content_{i}".encode()) for i in range(50)
        ]

        docs = DocumentList(sample_docs + another_docs)

        # Filter by large list of names (should use set internally for efficiency)
        names_to_filter = [f"sample_{i}.txt" for i in range(0, 50, 2)]  # Every other sample
        filtered = docs.filter_by(names_to_filter)  # type: ignore[arg-type]
        assert len(filtered) == 25

        # Filter by large list of types
        filtered = docs.filter_by([SampleDoc] * 10)  # type: ignore[arg-type]  # Duplicates should be handled
        assert len(filtered) == 50  # All SampleDoc instances

    def test_filter_by_with_special_characters_in_names(self) -> None:
        """Test filtering with special characters in document names."""
        doc1 = SampleDoc(name="file with spaces.txt", content=b"content1")
        doc2 = SampleDoc(name="file-with-dashes.txt", content=b"content2")
        doc3 = AnotherDoc(name="file.with.dots.txt", content=b"content3")
        doc4 = SubDoc(name="file_with_underscores.txt", content=b"content4")

        docs = DocumentList([doc1, doc2, doc3, doc4])

        # Filter by names with special characters
        filtered = docs.filter_by(["file with spaces.txt", "file.with.dots.txt"])  # type: ignore[arg-type]
        assert len(filtered) == 2
        assert doc1 in filtered
        assert doc3 in filtered

        # Single name with special characters
        filtered = docs.filter_by("file-with-dashes.txt")
        assert len(filtered) == 1
        assert doc2 in filtered
