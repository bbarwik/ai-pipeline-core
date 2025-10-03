"""Tests to detect mutable default parameter issues in Documents module."""

from ai_pipeline_core.documents import (
    DocumentList,
    FlowDocument,
    TaskDocument,
    TemporaryDocument,
)


class SampleFlowDoc(FlowDocument):
    """Concrete FlowDocument for testing."""

    pass


class SampleTaskDoc(TaskDocument):
    """Concrete TaskDocument for testing."""

    pass


class TestFlowDocumentMutableDefaults:
    """Test that FlowDocument doesn't use mutable defaults for sources parameter."""

    def test_multiple_instances_without_sources_have_independent_lists(self):
        """Verify that creating multiple FlowDocuments without sources doesn't share lists."""
        # Create first document without sources (using default)
        doc1 = SampleFlowDoc(name="doc1.txt", content=b"content1")

        # Create second document without sources (using default)
        doc2 = SampleFlowDoc(name="doc2.txt", content=b"content2")

        # Both should have empty sources lists
        assert doc1.sources == []
        assert doc2.sources == []

        # But they should NOT be the same list instance
        assert doc1.sources is not doc2.sources, (
            "FlowDocument instances are sharing the same sources list! "
            "This indicates a mutable default parameter bug."
        )

    def test_sources_default_is_not_shared_across_calls(self):
        """Verify sources default doesn't persist across function calls."""
        # Create document without sources
        doc1 = SampleFlowDoc(name="doc1.txt", content=b"content1")
        assert len(doc1.sources) == 0

        # Try to append to sources (this should fail with frozen model, but test the default)
        # Instead, check that the default list is fresh
        doc2 = SampleFlowDoc(name="doc2.txt", content=b"content2")

        # If default is mutable and shared, doc2 might see doc1's list
        assert doc1.sources is not doc2.sources, (
            "sources list is shared between instances created with default parameter"
        )

    def test_explicit_sources_list_not_affected_by_default(self):
        """Verify that explicitly passing sources works correctly."""
        # Create doc with explicit sources
        doc1 = SampleFlowDoc(name="doc1.txt", content=b"content1", sources=["source1", "source2"])

        # Create doc without sources (default)
        doc2 = SampleFlowDoc(name="doc2.txt", content=b"content2")

        # Verify they are independent
        assert len(doc1.sources) == 2
        assert len(doc2.sources) == 0
        assert doc1.sources is not doc2.sources


class TestTaskDocumentMutableDefaults:
    """Test that TaskDocument doesn't use mutable defaults for sources parameter."""

    def test_multiple_instances_without_sources_have_independent_lists(self):
        """Verify that creating multiple TaskDocuments without sources doesn't share lists."""
        # Create first document without sources (using default)
        doc1 = SampleTaskDoc(name="doc1.txt", content=b"content1")

        # Create second document without sources (using default)
        doc2 = SampleTaskDoc(name="doc2.txt", content=b"content2")

        # Both should have empty sources lists
        assert doc1.sources == []
        assert doc2.sources == []

        # But they should NOT be the same list instance
        assert doc1.sources is not doc2.sources, (
            "TaskDocument instances are sharing the same sources list! "
            "This indicates a mutable default parameter bug."
        )

    def test_sources_default_is_not_shared_across_calls(self):
        """Verify sources default doesn't persist across function calls."""
        doc1 = SampleTaskDoc(name="doc1.txt", content=b"content1")
        assert len(doc1.sources) == 0

        doc2 = SampleTaskDoc(name="doc2.txt", content=b"content2")

        # If default is mutable and shared, they might share the same list
        assert doc1.sources is not doc2.sources, (
            "sources list is shared between instances created with default parameter"
        )

    def test_explicit_sources_list_not_affected_by_default(self):
        """Verify that explicitly passing sources works correctly."""
        doc1 = SampleTaskDoc(name="doc1.txt", content=b"content1", sources=["source1", "source2"])

        doc2 = SampleTaskDoc(name="doc2.txt", content=b"content2")

        assert len(doc1.sources) == 2
        assert len(doc2.sources) == 0
        assert doc1.sources is not doc2.sources


class TestTemporaryDocumentMutableDefaults:
    """Test that TemporaryDocument (base Document) doesn't use mutable defaults."""

    def test_multiple_instances_without_sources_have_independent_lists(self):
        """Verify that creating multiple Documents without sources doesn't share lists."""
        doc1 = TemporaryDocument(name="doc1.txt", content=b"content1")
        doc2 = TemporaryDocument(name="doc2.txt", content=b"content2")

        assert doc1.sources == []
        assert doc2.sources == []

        # Critical: They must NOT be the same list instance
        assert doc1.sources is not doc2.sources, (
            "TemporaryDocument instances are sharing the same sources list! "
            "This indicates a mutable default parameter bug."
        )

    def test_document_create_method_with_default_sources(self):
        """Test the Document.create() class method with default sources."""
        # TemporaryDocument inherits from Document, test create method
        doc1 = TemporaryDocument.create(name="doc1.txt", content=b"content1")
        doc2 = TemporaryDocument.create(name="doc2.txt", content=b"content2")

        assert doc1.sources == []
        assert doc2.sources == []
        assert doc1.sources is not doc2.sources, (
            "Document.create() is sharing the same sources list across instances"
        )

    def test_mixing_default_and_explicit_sources(self):
        """Test mixing documents with default sources and explicit sources."""
        docs = []

        # Create multiple docs with defaults
        docs.append(TemporaryDocument(name="doc1.txt", content=b"content1"))
        docs.append(TemporaryDocument(name="doc2.txt", content=b"content2"))

        # Create doc with explicit sources
        docs.append(
            TemporaryDocument(name="doc3.txt", content=b"content3", sources=["hash1", "hash2"])
        )

        # Create another with default
        docs.append(TemporaryDocument(name="doc4.txt", content=b"content4"))

        # Verify all sources lists are independent
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                assert docs[i].sources is not docs[j].sources, (
                    f"Documents {i} and {j} share the same sources list"
                )

        # Verify the explicit one has correct content
        assert len(docs[2].sources) == 2
        assert "hash1" in docs[2].sources


class TestDocumentListWithDefaultSources:
    """Test DocumentList operations with documents that have default sources."""

    def test_document_list_of_docs_with_default_sources(self):
        """Verify DocumentList can hold multiple docs with default sources."""
        doc1 = TemporaryDocument(name="doc1.txt", content=b"content1")
        doc2 = TemporaryDocument(name="doc2.txt", content=b"content2")
        doc3 = TemporaryDocument(name="doc3.txt", content=b"content3")

        doc_list = DocumentList([doc1, doc2, doc3])

        # All docs should have independent sources lists
        sources_ids = [id(doc.sources) for doc in doc_list]
        assert len(set(sources_ids)) == 3, (
            "Some documents in DocumentList share the same sources list"
        )

    def test_document_list_filter_with_default_sources(self):
        """Test filtering documents with default sources."""
        docs: list[TemporaryDocument] = [
            TemporaryDocument(name=f"doc{i}.txt", content=f"content{i}".encode()) for i in range(5)
        ]

        doc_list = DocumentList(docs)  # type: ignore[arg-type]
        # Use get_by with filename as argument
        filtered_doc = doc_list.get_by("doc2.txt")

        assert filtered_doc is not None
        assert filtered_doc.name == "doc2.txt"

        # Verify sources is still independent
        assert filtered_doc.sources is not docs[0].sources


class TestDocumentConstructorSignatures:
    """Test various constructor signatures to ensure mutable defaults are handled correctly."""

    def test_document_init_with_keyword_only_args(self):
        """Verify documents require keyword arguments and handle sources correctly."""
        # All document constructors should use keyword-only arguments
        doc = TemporaryDocument(name="test.txt", content=b"test")
        assert doc.sources == []

        # Create another to verify independence
        doc2 = TemporaryDocument(name="test2.txt", content=b"test2")
        assert doc.sources is not doc2.sources

    def test_document_with_description_and_default_sources(self):
        """Test documents with description but default sources."""
        doc1 = TemporaryDocument(name="doc1.txt", content=b"content1", description="First document")

        doc2 = TemporaryDocument(
            name="doc2.txt", content=b"content2", description="Second document"
        )

        assert doc1.description == "First document"
        assert doc2.description == "Second document"
        assert doc1.sources == []
        assert doc2.sources == []
        assert doc1.sources is not doc2.sources

    def test_document_with_all_parameters(self):
        """Test documents with all parameters explicitly set."""
        doc1 = TemporaryDocument(
            name="doc1.txt",
            content=b"content1",
            description="First",
            sources=["hash1"],
        )

        doc2 = TemporaryDocument(
            name="doc2.txt",
            content=b"content2",
            description="Second",
            sources=["hash2"],
        )

        assert len(doc1.sources) == 1
        assert len(doc2.sources) == 1
        assert doc1.sources is not doc2.sources
        assert doc1.sources[0] == "hash1"
        assert doc2.sources[0] == "hash2"
