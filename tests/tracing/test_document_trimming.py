"""Tests for document trimming in trace decorator."""

from unittest.mock import Mock, patch

from ai_pipeline_core.documents import FlowDocument, TaskDocument
from ai_pipeline_core.documents.document_list import DocumentList
from ai_pipeline_core.tracing import trace


class TracingTestFlowDoc(FlowDocument):
    """Flow document for testing."""


class TracingTestTaskDoc(TaskDocument):
    """Task document for testing."""


class TestDocumentTrimming:
    """Test document trimming functionality in trace decorator."""

    def test_trim_document_content_text_non_flow(self):
        """Test trimming text content for non-flow documents."""
        from ai_pipeline_core.tracing import (
            _trim_document_content,  # pyright: ignore[reportPrivateUsage]
        )

        # Long text content for task document
        long_text = "a" * 300
        doc_dict = {
            "base_type": "task",
            "content": long_text,
            "content_encoding": "utf-8",
        }

        result = _trim_document_content(doc_dict)

        # Should be trimmed to first 100 + last 100 chars
        assert len(result["content"]) < len(long_text)
        assert result["content"].startswith("a" * 100)
        assert result["content"].endswith("a" * 100)
        assert " ... [trimmed 100 chars] ... " in result["content"]

    def test_trim_document_content_text_flow(self):
        """Test that flow document text content is NOT trimmed."""
        from ai_pipeline_core.tracing import (
            _trim_document_content,  # pyright: ignore[reportPrivateUsage]
        )

        # Long text content for flow document
        long_text = "b" * 300
        doc_dict = {
            "base_type": "flow",
            "content": long_text,
            "content_encoding": "utf-8",
        }

        result = _trim_document_content(doc_dict)

        # Should NOT be trimmed for flow documents
        assert result["content"] == long_text

    def test_trim_document_content_short_text(self):
        """Test that short text content is not trimmed."""
        from ai_pipeline_core.tracing import (
            _trim_document_content,  # pyright: ignore[reportPrivateUsage]
        )

        # Short text content (< 250 chars)
        short_text = "Hello World"
        doc_dict = {
            "base_type": "task",
            "content": short_text,
            "content_encoding": "utf-8",
        }

        result = _trim_document_content(doc_dict)

        # Should NOT be trimmed when under 250 chars
        assert result["content"] == short_text

    def test_trim_document_content_binary(self):
        """Test that binary content is removed."""
        from ai_pipeline_core.tracing import (
            _trim_document_content,  # pyright: ignore[reportPrivateUsage]
        )

        # Binary content (base64 encoded)
        doc_dict = {
            "base_type": "task",
            "content": "SGVsbG8gV29ybGQ=",  # base64
            "content_encoding": "base64",
        }

        result = _trim_document_content(doc_dict)

        # Binary content should be removed
        assert result["content"] == "[binary content removed]"

    def test_trim_document_content_binary_flow(self):
        """Test that flow document binary content is also removed."""
        from ai_pipeline_core.tracing import (
            _trim_document_content,  # pyright: ignore[reportPrivateUsage]
        )

        # Binary content for flow document
        doc_dict = {
            "base_type": "flow",
            "content": "SGVsbG8gV29ybGQ=",  # base64
            "content_encoding": "base64",
        }

        result = _trim_document_content(doc_dict)

        # Binary content should be removed even for flow documents
        assert result["content"] == "[binary content removed]"

    def test_trim_documents_in_data_nested(self):
        """Test trimming documents in nested data structures."""
        from ai_pipeline_core.tracing import (
            _trim_documents_in_data,  # pyright: ignore[reportPrivateUsage]
        )

        # Create nested structure with documents
        long_text = "x" * 300
        data = {
            "docs": [
                {
                    "base_type": "task",
                    "content": long_text,
                    "content_encoding": "utf-8",
                },
                {
                    "base_type": "flow",
                    "content": long_text,
                    "content_encoding": "utf-8",
                },
            ],
            "nested": {
                "doc": {
                    "base_type": "temporary",
                    "content": "SGVsbG8=",  # base64
                    "content_encoding": "base64",
                }
            },
        }

        result = _trim_documents_in_data(data)

        # First doc (task) should be trimmed
        assert "[trimmed" in result["docs"][0]["content"]
        assert "chars]" in result["docs"][0]["content"]

        # Second doc (flow) should NOT be trimmed
        assert result["docs"][1]["content"] == long_text

        # Nested binary doc should have content removed
        assert result["nested"]["doc"]["content"] == "[binary content removed]"

    @patch("ai_pipeline_core.tracing.observe")
    def test_trace_with_trim_documents_true(self, mock_observe):
        """Test trace decorator with trim_documents=True (default)."""
        # Create a mock that captures the formatters
        mock_decorator = Mock(return_value=lambda f: f)
        mock_observe.return_value = mock_decorator

        # Create a traced function with document inputs
        @trace(trim_documents=True)
        def process_doc(doc):
            return doc

        # Create a long document
        long_content = "content" * 100  # 700 chars
        doc = TracingTestTaskDoc(name="test.txt", content=long_content.encode())

        # Call the function
        process_doc(doc)

        # Verify observe was called
        mock_decorator.assert_called_once()

        # Get the observe parameters
        observe_kwargs = mock_observe.call_args[1]

        # Should have input and output formatters
        assert "input_formatter" in observe_kwargs
        assert "output_formatter" in observe_kwargs

    @patch("ai_pipeline_core.tracing.observe")
    def test_trace_with_trim_documents_false(self, mock_observe):
        """Test trace decorator with trim_documents=False."""
        # Create a mock that captures the formatters
        mock_decorator = Mock(return_value=lambda f: f)
        mock_observe.return_value = mock_decorator

        # Create a traced function without document trimming
        @trace(trim_documents=False)
        def process_doc(doc):
            return doc

        # Create a document
        doc = TracingTestTaskDoc(name="test.txt", content=b"content")

        # Call the function
        process_doc(doc)

        # Verify observe was called
        mock_decorator.assert_called_once()

        # Get the observe parameters
        observe_kwargs = mock_observe.call_args[1]

        # Should not have trimming formatters (unless custom ones provided)
        # Since we didn't provide custom formatters, they should be absent
        assert "input_formatter" not in observe_kwargs or observe_kwargs["input_formatter"] is None
        assert (
            "output_formatter" not in observe_kwargs or observe_kwargs["output_formatter"] is None
        )

    def test_trimming_formatter_with_document_list(self):
        """Test that DocumentList is properly handled by trimming formatters."""
        from ai_pipeline_core.tracing import (
            _trim_documents_in_data,  # pyright: ignore[reportPrivateUsage]
        )

        # Create documents
        doc1 = TracingTestTaskDoc(name="file1.txt", content=b"a" * 300)
        doc2 = TracingTestFlowDoc(name="file2.txt", content=b"b" * 300)

        # Create DocumentList
        doc_list = DocumentList([doc1, doc2])

        # Convert to serializable format (as the formatter would)
        serialized = [doc.serialize_model() for doc in doc_list]

        # Trim the documents
        result = _trim_documents_in_data(serialized)

        # First doc (task) should be trimmed
        assert "[trimmed" in result[0]["content"]
        assert "chars]" in result[0]["content"]

        # Second doc (flow) should NOT be trimmed
        assert result[1]["content"] == "b" * 300

    def test_trimming_formatter_with_ai_messages(self):
        """Test that AIMessages with documents are properly handled."""
        from ai_pipeline_core.tracing import (
            _trim_documents_in_data,  # pyright: ignore[reportPrivateUsage]
        )

        # Create a document with long content
        long_content = "message" * 100
        doc = TracingTestTaskDoc(name="msg.txt", content=long_content.encode())

        # Simulate serialized AIMessages with document
        messages_data = [
            "User question",
            doc.serialize_model(),
            "AI response",
        ]

        # Trim the documents in messages
        result = _trim_documents_in_data(messages_data)

        # String messages should remain unchanged
        assert result[0] == "User question"
        assert result[2] == "AI response"

        # Document should be trimmed
        assert "[trimmed" in result[1]["content"]
        assert "chars]" in result[1]["content"]

    def test_trimming_preserves_document_metadata(self):
        """Test that trimming preserves all document metadata except content."""
        from ai_pipeline_core.tracing import (
            _trim_document_content,  # pyright: ignore[reportPrivateUsage]
        )

        doc_dict = {
            "name": "test.txt",
            "base_type": "task",
            "content": "a" * 300,
            "content_encoding": "utf-8",
            "sha256": "ABC123",
            "mime_type": "text/plain",
            "size": 300,
            "id": "ABC",
        }

        result = _trim_document_content(doc_dict)

        # All metadata should be preserved
        assert result["name"] == doc_dict["name"]
        assert result["base_type"] == doc_dict["base_type"]
        assert result["sha256"] == doc_dict["sha256"]
        assert result["mime_type"] == doc_dict["mime_type"]
        assert result["size"] == doc_dict["size"]
        assert result["id"] == doc_dict["id"]

        # Only content should be modified
        assert result["content"] != doc_dict["content"]
        assert "[trimmed" in result["content"]
        assert "chars]" in result["content"]

    def test_trimming_with_custom_formatter(self):
        """Test that custom formatters work with trimming."""
        from ai_pipeline_core.tracing import (
            _trim_documents_in_data,  # pyright: ignore[reportPrivateUsage]
        )

        # Mock custom formatter that returns document data
        custom_formatter = Mock(
            return_value={
                "doc": {"base_type": "task", "content": "a" * 300, "content_encoding": "utf-8"}
            }
        )

        # Simulate what the trimming formatter does internally:
        # 1. Call the custom formatter
        # 2. Trim the result
        result = custom_formatter("arg1", "arg2")
        trimmed = _trim_documents_in_data(result)

        # Document should be trimmed
        assert "[trimmed" in trimmed["doc"]["content"]
        assert "chars]" in trimmed["doc"]["content"]

    def test_temporary_document_trimming(self):
        """Test that TemporaryDocument content is also trimmed."""
        from ai_pipeline_core.tracing import (
            _trim_document_content,  # pyright: ignore[reportPrivateUsage]
        )

        # Long text content for temporary document
        long_text = "t" * 300
        doc_dict = {
            "base_type": "temporary",
            "content": long_text,
            "content_encoding": "utf-8",
        }

        result = _trim_document_content(doc_dict)

        # Should be trimmed like task documents
        assert len(result["content"]) < len(long_text)
        assert "[trimmed" in result["content"]
        assert "chars]" in result["content"]

    def test_edge_case_exactly_250_chars(self):
        """Test edge case with exactly 250 characters."""
        from ai_pipeline_core.tracing import (
            _trim_document_content,  # pyright: ignore[reportPrivateUsage]
        )

        # Exactly 250 characters
        text_250 = "x" * 250
        doc_dict = {
            "base_type": "task",
            "content": text_250,
            "content_encoding": "utf-8",
        }

        result = _trim_document_content(doc_dict)

        # Should NOT be trimmed at exactly 250 chars
        assert result["content"] == text_250

    def test_edge_case_251_chars(self):
        """Test edge case with 251 characters (just over threshold)."""
        from ai_pipeline_core.tracing import (
            _trim_document_content,  # pyright: ignore[reportPrivateUsage]
        )

        # 251 characters (just over the threshold)
        text_251 = "x" * 251
        doc_dict = {
            "base_type": "task",
            "content": text_251,
            "content_encoding": "utf-8",
        }

        result = _trim_document_content(doc_dict)

        # Should be trimmed at 251 chars
        assert len(result["content"]) < len(text_251)
        assert " ... [trimmed 51 chars] ... " in result["content"]
