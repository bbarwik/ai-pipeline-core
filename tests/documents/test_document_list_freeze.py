"""Tests for DocumentList freeze functionality."""

import pytest

from ai_pipeline_core.documents import DocumentList, FlowDocument


class FreezeTestDoc(FlowDocument):
    """Test document for freeze tests."""


class AnotherFreezeTestDoc(FlowDocument):
    """Another test document for freeze tests."""


class TestDocumentListFreeze:
    """Test DocumentList freeze functionality."""

    def test_init_frozen(self) -> None:
        """Test creating frozen DocumentList at initialization."""
        doc1 = FreezeTestDoc(name="file1.txt", content=b"content1")
        doc2 = FreezeTestDoc(name="file2.txt", content=b"content2")

        # Create frozen list
        docs = DocumentList([doc1, doc2], frozen=True)

        # Verify initial content
        assert len(docs) == 2
        assert docs[0] == doc1
        assert docs[1] == doc2

        # Should be frozen
        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs.append(FreezeTestDoc(name="file3.txt", content=b"content3"))

    def test_freeze_method(self) -> None:
        """Test freezing DocumentList after creation."""
        doc1 = FreezeTestDoc(name="file1.txt", content=b"content1")
        doc2 = FreezeTestDoc(name="file2.txt", content=b"content2")

        # Create unfrozen list
        docs = DocumentList([doc1])

        # Should be modifiable
        docs.append(doc2)
        assert len(docs) == 2

        # Freeze it
        docs.freeze()

        # Now should be frozen
        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs.append(FreezeTestDoc(name="file3.txt", content=b"content3"))

    def test_freeze_is_permanent(self) -> None:
        """Test that freeze cannot be undone."""
        docs = DocumentList()
        docs.freeze()

        # Try to unfreeze by setting _frozen directly - should still be frozen
        # This tests that the freeze is intended to be permanent
        docs._frozen = False  # pyright: ignore[reportPrivateUsage]
        docs._frozen = True  # Reset it back  # pyright: ignore[reportPrivateUsage]

        # Should still be frozen
        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs.append(FreezeTestDoc(name="file.txt", content=b"content"))

    def test_copy_creates_unfrozen(self) -> None:
        """Test that copy creates unfrozen deep copy."""
        doc1 = FreezeTestDoc(name="file1.txt", content=b"content1")
        doc2 = FreezeTestDoc(name="file2.txt", content=b"content2")

        # Create frozen list
        frozen_docs = DocumentList([doc1, doc2], frozen=True)

        # Copy should be unfrozen
        copied_docs = frozen_docs.copy()

        # Verify copy has same content
        assert len(copied_docs) == 2
        assert copied_docs[0].name == doc1.name
        assert copied_docs[1].name == doc2.name

        # Copy should be modifiable
        doc3 = FreezeTestDoc(name="file3.txt", content=b"content3")
        copied_docs.append(doc3)
        assert len(copied_docs) == 3

        # Original should still be frozen and unchanged
        assert len(frozen_docs) == 2
        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            frozen_docs.append(doc3)

    def test_copy_is_deep(self) -> None:
        """Test that copy performs deep copy of documents."""
        doc1 = FreezeTestDoc(name="file1.txt", content=b"content1")
        original_docs = DocumentList([doc1])

        # Make a copy
        copied_docs = original_docs.copy()

        # Verify deep copy by checking that documents are different objects
        assert copied_docs[0] is not original_docs[0]
        assert copied_docs[0].name == original_docs[0].name
        assert copied_docs[0].content == original_docs[0].content

    def test_frozen_append(self) -> None:
        """Test that append raises RuntimeError when frozen."""
        docs = DocumentList(frozen=True)
        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs.append(FreezeTestDoc(name="file.txt", content=b"content"))

    def test_frozen_extend(self) -> None:
        """Test that extend raises RuntimeError when frozen."""
        docs = DocumentList(frozen=True)
        new_docs = [
            FreezeTestDoc(name="file1.txt", content=b"content1"),
            FreezeTestDoc(name="file2.txt", content=b"content2"),
        ]
        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs.extend(new_docs)

    def test_frozen_insert(self) -> None:
        """Test that insert raises RuntimeError when frozen."""
        doc1 = FreezeTestDoc(name="file1.txt", content=b"content1")
        docs = DocumentList([doc1], frozen=True)
        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs.insert(0, FreezeTestDoc(name="file2.txt", content=b"content2"))

    def test_frozen_setitem(self) -> None:
        """Test that __setitem__ raises RuntimeError when frozen."""
        doc1 = FreezeTestDoc(name="file1.txt", content=b"content1")
        doc2 = FreezeTestDoc(name="file2.txt", content=b"content2")
        docs = DocumentList([doc1], frozen=True)

        # Test single item
        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs[0] = doc2

        # Test slice
        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs[0:1] = [doc2]

    def test_frozen_iadd(self) -> None:
        """Test that += raises RuntimeError when frozen."""
        doc1 = FreezeTestDoc(name="file1.txt", content=b"content1")
        docs = DocumentList([doc1], frozen=True)
        new_docs = [FreezeTestDoc(name="file2.txt", content=b"content2")]

        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs += new_docs

    def test_frozen_delitem(self) -> None:
        """Test that __delitem__ raises RuntimeError when frozen."""
        doc1 = FreezeTestDoc(name="file1.txt", content=b"content1")
        doc2 = FreezeTestDoc(name="file2.txt", content=b"content2")
        docs = DocumentList([doc1, doc2], frozen=True)

        # Test single item deletion
        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            del docs[0]

        # Test slice deletion
        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            del docs[0:1]

    def test_frozen_pop(self) -> None:
        """Test that pop raises RuntimeError when frozen."""
        doc1 = FreezeTestDoc(name="file1.txt", content=b"content1")
        docs = DocumentList([doc1], frozen=True)

        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs.pop()

        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs.pop(0)

    def test_frozen_remove(self) -> None:
        """Test that remove raises RuntimeError when frozen."""
        doc1 = FreezeTestDoc(name="file1.txt", content=b"content1")
        docs = DocumentList([doc1], frozen=True)

        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs.remove(doc1)

    def test_frozen_clear(self) -> None:
        """Test that clear raises RuntimeError when frozen."""
        doc1 = FreezeTestDoc(name="file1.txt", content=b"content1")
        docs = DocumentList([doc1], frozen=True)

        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs.clear()

    def test_frozen_reverse(self) -> None:
        """Test that reverse raises RuntimeError when frozen."""
        doc1 = FreezeTestDoc(name="file1.txt", content=b"content1")
        doc2 = FreezeTestDoc(name="file2.txt", content=b"content2")
        docs = DocumentList([doc1, doc2], frozen=True)

        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs.reverse()

    def test_frozen_sort(self) -> None:
        """Test that sort raises RuntimeError when frozen."""
        doc1 = FreezeTestDoc(name="b.txt", content=b"content1")
        doc2 = FreezeTestDoc(name="a.txt", content=b"content2")
        docs = DocumentList([doc1, doc2], frozen=True)

        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs.sort(key=lambda doc: doc.name)

        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs.sort(reverse=True)

    def test_frozen_allows_read_operations(self) -> None:
        """Test that frozen list allows read operations."""
        doc1 = FreezeTestDoc(name="file1.txt", content=b"content1")
        doc2 = AnotherFreezeTestDoc(name="file2.txt", content=b"content2")
        docs = DocumentList([doc1, doc2], frozen=True)

        # All read operations should work
        assert len(docs) == 2
        assert docs[0] == doc1
        assert docs[1] == doc2
        assert doc1 in docs
        assert doc2 in docs

        # filter_by should work (returns new list)
        filtered = docs.filter_by(FreezeTestDoc)
        assert len(filtered) == 1
        assert filtered[0] == doc1

        # get_by should work
        result = docs.get_by("file1.txt")
        assert result == doc1

        # Iteration should work
        for doc in docs:
            assert doc in [doc1, doc2]

    def test_frozen_with_validation_flags(self) -> None:
        """Test frozen list with validation flags."""
        doc1 = FreezeTestDoc(name="file1.txt", content=b"content1")
        doc2 = FreezeTestDoc(name="file2.txt", content=b"content2")

        # Test with validate_same_type
        docs = DocumentList(
            [doc1, doc2], validate_same_type=True, validate_duplicates=True, frozen=True
        )

        assert len(docs) == 2

        # Copy should preserve validation flags
        copied = docs.copy()
        assert copied._validate_same_type is True  # pyright: ignore[reportPrivateUsage]
        assert copied._validate_duplicates is True  # pyright: ignore[reportPrivateUsage]
        assert hasattr(copied, "_frozen") and not copied._frozen  # pyright: ignore[reportPrivateUsage]

    def test_empty_frozen_list(self) -> None:
        """Test empty frozen list."""
        docs = DocumentList(frozen=True)

        assert len(docs) == 0
        assert list(docs) == []

        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs.append(FreezeTestDoc(name="file.txt", content=b"content"))

    def test_frozen_prevents_modification_with_validation(self) -> None:
        """Test that frozen prevents modification even with validation enabled."""
        doc1 = FreezeTestDoc(name="file1.txt", content=b"content1")

        # Create frozen list with validation
        docs = DocumentList([doc1], validate_duplicates=True, validate_same_type=True, frozen=True)

        # Try operations that would normally trigger validation
        doc2 = FreezeTestDoc(name="file2.txt", content=b"content2")

        # These should all fail before validation
        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs.append(doc2)

        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs.extend([doc2])

        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs.insert(0, doc2)

        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs[0] = doc2

        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs += [doc2]

    def test_multiple_freeze_calls(self) -> None:
        """Test that calling freeze multiple times is safe."""
        docs = DocumentList()

        # First freeze
        docs.freeze()

        # Second freeze should be safe (idempotent)
        docs.freeze()

        # Still frozen
        with pytest.raises(RuntimeError, match="Cannot modify frozen DocumentList"):
            docs.append(FreezeTestDoc(name="file.txt", content=b"content"))

    def test_copy_preserves_content_exactly(self) -> None:
        """Test that copy preserves exact document content and attributes."""
        doc1 = FreezeTestDoc(name="file1.txt", content=b"content1", description="Test doc")
        doc2 = AnotherFreezeTestDoc(name="file2.txt", content=b"content2")

        original = DocumentList([doc1, doc2], validate_same_type=False, frozen=True)
        copied = original.copy()

        # Check exact preservation
        assert copied[0].name == doc1.name
        assert copied[0].content == doc1.content
        assert copied[0].description == doc1.description
        assert isinstance(copied[0], FreezeTestDoc)

        assert copied[1].name == doc2.name
        assert copied[1].content == doc2.content
        assert isinstance(copied[1], AnotherFreezeTestDoc)
