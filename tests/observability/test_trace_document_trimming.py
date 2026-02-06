"""Tests for document trimming in trace decorator."""

from unittest.mock import Mock, patch

from ai_pipeline_core.documents import Document
from ai_pipeline_core.observability.tracing import trace


class TracingTestFlowDoc(Document):
    """Flow document for testing."""


class TracingTestTaskDoc(Document):
    """Task document for testing."""


class TestDocumentTrimming:
    """Test document trimming functionality in trace decorator."""

    def test_trim_document_content_text(self):
        """Test trimming text content for documents."""
        from ai_pipeline_core.observability.tracing import (
            _trim_document_content,
        )

        long_text = "a" * 300
        doc_dict = {
            "class_name": "TracingTestTaskDoc",
            "content": {"v": long_text, "e": "utf-8"},
        }

        result = _trim_document_content(doc_dict)

        # Should be trimmed to first 100 + last 100 chars - output is {v, e} format
        content = result["content"]
        assert isinstance(content, dict)
        assert len(content["v"]) < len(long_text)
        assert content["v"].startswith("a" * 100)
        assert content["v"].endswith("a" * 100)
        assert " ... [trimmed 100 chars] ... " in content["v"]

    def test_trim_document_content_all_documents_trimmed_equally(self):
        """Test that all documents are trimmed equally (no flow/task distinction)."""
        from ai_pipeline_core.observability.tracing import (
            _trim_document_content,
        )

        long_text = "b" * 300
        doc_dict = {
            "class_name": "TracingTestFlowDoc",
            "content": {"v": long_text, "e": "utf-8"},
        }

        result = _trim_document_content(doc_dict)

        # All documents are trimmed equally - output is {v, e} format
        content = result["content"]
        assert isinstance(content, dict)
        assert len(content["v"]) < len(long_text)
        assert " ... [trimmed 100 chars] ... " in content["v"]

    def test_trim_document_content_short_text(self):
        """Test that short text content is not trimmed."""
        from ai_pipeline_core.observability.tracing import (
            _trim_document_content,
        )

        short_text = "Hello World"
        doc_dict = {
            "class_name": "TracingTestTaskDoc",
            "content": {"v": short_text, "e": "utf-8"},
        }

        result = _trim_document_content(doc_dict)

        # Should NOT be trimmed when under 250 chars - preserves {v, e} format
        assert result["content"] == {"v": short_text, "e": "utf-8"}

    def test_trim_document_content_binary(self):
        """Test that binary content is removed."""
        from ai_pipeline_core.observability.tracing import (
            _trim_document_content,
        )

        doc_dict = {
            "class_name": "TracingTestTaskDoc",
            "content": {"v": "SGVsbG8gV29ybGQ=", "e": "base64"},
        }

        result = _trim_document_content(doc_dict)

        assert result["content"] == {"v": "[binary content removed]", "e": "utf-8"}

    def test_trim_document_content_binary_all_types(self):
        """Test that binary content is removed for all document types."""
        from ai_pipeline_core.observability.tracing import (
            _trim_document_content,
        )

        doc_dict = {
            "class_name": "TracingTestFlowDoc",
            "content": {"v": "SGVsbG8gV29ybGQ=", "e": "base64"},
        }

        result = _trim_document_content(doc_dict)

        assert result["content"] == {"v": "[binary content removed]", "e": "utf-8"}

    def test_trim_documents_in_data_nested(self):
        """Test trimming documents in nested data structures."""
        from ai_pipeline_core.observability.tracing import (
            _trim_documents_in_data,
        )

        long_text = "x" * 300
        data = {
            "docs": [
                {
                    "class_name": "TracingTestTaskDoc",
                    "content": {"v": long_text, "e": "utf-8"},
                },
                {
                    "class_name": "TracingTestFlowDoc",
                    "content": {"v": long_text, "e": "utf-8"},
                },
            ],
            "nested": {
                "doc": {
                    "class_name": "SomeDocument",
                    "content": {"v": "SGVsbG8=", "e": "base64"},
                }
            },
        }

        result = _trim_documents_in_data(data)

        # Both docs should be trimmed (all documents trimmed equally) - output is {v, e} format
        assert "[trimmed" in result["docs"][0]["content"]["v"]
        assert "chars]" in result["docs"][0]["content"]["v"]
        assert "[trimmed" in result["docs"][1]["content"]["v"]
        assert "chars]" in result["docs"][1]["content"]["v"]

        # Nested binary doc should have content removed
        assert result["nested"]["doc"]["content"] == {"v": "[binary content removed]", "e": "utf-8"}

    @patch("ai_pipeline_core.observability.tracing.observe")
    def test_trace_with_trim_documents_true(self, mock_observe):
        """Test trace decorator with trim_documents=True (default)."""
        mock_decorator = Mock(return_value=lambda f: f)
        mock_observe.return_value = mock_decorator

        @trace(trim_documents=True)
        def process_doc(doc):
            return doc

        long_content = "content" * 100  # 700 chars
        doc = TracingTestTaskDoc(name="test.txt", content=long_content.encode())

        process_doc(doc)

        mock_decorator.assert_called_once()

        observe_kwargs = mock_observe.call_args[1]

        assert "input_formatter" in observe_kwargs
        assert "output_formatter" in observe_kwargs

    @patch("ai_pipeline_core.observability.tracing.observe")
    def test_trace_with_trim_documents_false(self, mock_observe):
        """Test trace decorator with trim_documents=False."""
        mock_decorator = Mock(return_value=lambda f: f)
        mock_observe.return_value = mock_decorator

        @trace(trim_documents=False)
        def process_doc(doc):
            return doc

        doc = TracingTestTaskDoc(name="test.txt", content=b"content")

        process_doc(doc)

        mock_decorator.assert_called_once()

        observe_kwargs = mock_observe.call_args[1]

        assert "input_formatter" not in observe_kwargs or observe_kwargs["input_formatter"] is None
        assert "output_formatter" not in observe_kwargs or observe_kwargs["output_formatter"] is None

    def test_trimming_formatter_with_document_list(self):
        """Test that list[Document] is properly handled by trimming formatters."""
        from ai_pipeline_core.observability.tracing import (
            _trim_documents_in_data,
        )

        doc1 = TracingTestTaskDoc(name="file1.txt", content=b"a" * 300)
        doc2 = TracingTestFlowDoc(name="file2.txt", content=b"b" * 300)

        doc_list = list([doc1, doc2])

        serialized = [doc.serialize_model() for doc in doc_list]

        result = _trim_documents_in_data(serialized)

        # Both documents should be trimmed (all trimmed equally) - output is {v, e} format
        assert "[trimmed" in result[0]["content"]["v"]
        assert "chars]" in result[0]["content"]["v"]
        assert "[trimmed" in result[1]["content"]["v"]
        assert "chars]" in result[1]["content"]["v"]

    def test_trimming_formatter_with_ai_messages(self):
        """Test that AIMessages with documents are properly handled."""
        from ai_pipeline_core.observability.tracing import (
            _trim_documents_in_data,
        )

        long_content = "message" * 100
        doc = TracingTestTaskDoc(name="msg.txt", content=long_content.encode())

        messages_data = [
            "User question",
            doc.serialize_model(),
            "AI response",
        ]

        result = _trim_documents_in_data(messages_data)

        assert result[0] == "User question"
        assert result[2] == "AI response"

        # Output is {v, e} format
        assert "[trimmed" in result[1]["content"]["v"]
        assert "chars]" in result[1]["content"]["v"]

    def test_trimming_preserves_document_metadata(self):
        """Test that trimming preserves all document metadata except content."""
        from ai_pipeline_core.observability.tracing import (
            _trim_document_content,
        )

        doc_dict = {
            "name": "test.txt",
            "class_name": "TracingTestTaskDoc",
            "content": {"v": "a" * 300, "e": "utf-8"},
            "sha256": "ABC123",
            "mime_type": "text/plain",
            "size": 300,
            "id": "ABC",
        }

        result = _trim_document_content(doc_dict)

        # All metadata should be preserved
        assert result["name"] == doc_dict["name"]
        assert result["class_name"] == doc_dict["class_name"]
        assert result["sha256"] == doc_dict["sha256"]
        assert result["mime_type"] == doc_dict["mime_type"]
        assert result["size"] == doc_dict["size"]
        assert result["id"] == doc_dict["id"]

        # Only content should be modified - output is {v, e} format
        assert result["content"] != doc_dict["content"]
        assert "[trimmed" in result["content"]["v"]
        assert "chars]" in result["content"]["v"]

    def test_trimming_with_custom_formatter(self):
        """Test that custom formatters work with trimming."""
        from ai_pipeline_core.observability.tracing import (
            _trim_documents_in_data,
        )

        custom_formatter = Mock(return_value={"doc": {"class_name": "SomeDoc", "content": {"v": "a" * 300, "e": "utf-8"}}})

        result = custom_formatter("arg1", "arg2")
        trimmed = _trim_documents_in_data(result)

        # Output is {v, e} format
        assert "[trimmed" in trimmed["doc"]["content"]["v"]
        assert "chars]" in trimmed["doc"]["content"]["v"]

    def test_document_trimming_any_subclass(self):
        """Test that any Document subclass content is trimmed."""
        from ai_pipeline_core.observability.tracing import (
            _trim_document_content,
        )

        long_text = "t" * 300
        doc_dict = {
            "class_name": "SomeCustomDocument",
            "content": {"v": long_text, "e": "utf-8"},
        }

        result = _trim_document_content(doc_dict)

        # Output is {v, e} format
        content = result["content"]
        assert isinstance(content, dict)
        assert len(content["v"]) < len(long_text)
        assert "[trimmed" in content["v"]
        assert "chars]" in content["v"]

    def test_edge_case_exactly_250_chars(self):
        """Test edge case with exactly 250 characters."""
        from ai_pipeline_core.observability.tracing import (
            _trim_document_content,
        )

        text_250 = "x" * 250
        doc_dict = {
            "class_name": "SomeDocument",
            "content": {"v": text_250, "e": "utf-8"},
        }

        result = _trim_document_content(doc_dict)

        # Should NOT be trimmed at exactly 250 chars - preserves {v, e} format
        assert result["content"] == {"v": text_250, "e": "utf-8"}

    def test_edge_case_251_chars(self):
        """Test edge case with 251 characters (just over threshold)."""
        from ai_pipeline_core.observability.tracing import (
            _trim_document_content,
        )

        text_251 = "x" * 251
        doc_dict = {
            "class_name": "SomeDocument",
            "content": {"v": text_251, "e": "utf-8"},
        }

        result = _trim_document_content(doc_dict)

        # Output is {v, e} format
        content = result["content"]
        assert isinstance(content, dict)
        assert len(content["v"]) < len(text_251)
        assert " ... [trimmed 51 chars] ... " in content["v"]
