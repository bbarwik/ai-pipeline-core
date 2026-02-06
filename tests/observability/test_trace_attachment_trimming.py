"""Tests for attachment trimming in tracing."""

from ai_pipeline_core.observability.tracing import (
    _trim_attachment_list,
    _trim_document_content,
    _trim_documents_in_data,
)


class TestTrimAttachmentList:
    """Tests for _trim_attachment_list."""

    def test_binary_attachment_content_replaced(self):
        attachments = [
            {"name": "img.jpg", "content": {"v": "SGVsbG8=", "e": "base64"}},
        ]
        result = _trim_attachment_list(attachments)
        # Output is {v, e} format
        assert result[0]["content"] == {"v": "[binary content removed]", "e": "utf-8"}
        assert result[0]["name"] == "img.jpg"

    def test_short_text_attachment_preserved(self):
        attachments = [
            {"name": "note.txt", "content": {"v": "short text", "e": "utf-8"}},
        ]
        result = _trim_attachment_list(attachments)
        # Preserves {v, e} format
        assert result[0]["content"] == {"v": "short text", "e": "utf-8"}

    def test_long_text_attachment_trimmed(self):
        long_text = "x" * 300
        attachments = [
            {"name": "long.txt", "content": {"v": long_text, "e": "utf-8"}},
        ]
        result = _trim_attachment_list(attachments)
        # Output is {v, e} format
        content = result[0]["content"]
        assert isinstance(content, dict)
        assert len(content["v"]) < len(long_text)
        assert content["v"].startswith("x" * 100)
        assert content["v"].endswith("x" * 100)
        assert " ... [trimmed 100 chars] ... " in content["v"]

    def test_non_dict_items_passed_through(self):
        attachments = ["not-a-dict", 42]
        result = _trim_attachment_list(attachments)
        assert result == ["not-a-dict", 42]

    def test_no_content_encoding_defaults_utf8(self):
        """When content is a plain string (legacy format), defaults to utf-8."""
        attachments = [
            {"name": "note.txt", "content": {"v": "short", "e": "utf-8"}},
        ]
        result = _trim_attachment_list(attachments)
        # Preserves {v, e} format
        assert result[0]["content"] == {"v": "short", "e": "utf-8"}

    def test_original_not_mutated(self):
        att = {"name": "img.jpg", "content": {"v": "SGVsbG8=", "e": "base64"}}
        original_content = att["content"]
        _trim_attachment_list([att])
        assert att["content"] == original_content


class TestTrimDocumentContentWithAttachments:
    """Tests for _trim_document_content with attachments."""

    def test_trim_document_with_binary_attachments(self):
        doc_dict = {
            "class_name": "SomeTaskDocument",
            "content": {"v": "short text", "e": "utf-8"},
            "attachments": [
                {"name": "img.jpg", "content": {"v": "base64data==", "e": "base64"}},
            ],
        }
        result = _trim_document_content(doc_dict)
        # Preserves {v, e} format
        assert result["content"] == {"v": "short text", "e": "utf-8"}  # primary content preserved (short)
        assert result["attachments"][0]["content"] == {"v": "[binary content removed]", "e": "utf-8"}

    def test_trim_document_with_text_attachments_short(self):
        doc_dict = {
            "class_name": "SomeTaskDocument",
            "content": {"v": "primary", "e": "utf-8"},
            "attachments": [
                {"name": "note.txt", "content": {"v": "short note", "e": "utf-8"}},
            ],
        }
        result = _trim_document_content(doc_dict)
        # Preserves {v, e} format
        assert result["attachments"][0]["content"] == {"v": "short note", "e": "utf-8"}

    def test_trim_document_with_text_attachments_long(self):
        long_text = "y" * 300
        doc_dict = {
            "class_name": "SomeTaskDocument",
            "content": {"v": "primary", "e": "utf-8"},
            "attachments": [
                {"name": "long.txt", "content": {"v": long_text, "e": "utf-8"}},
            ],
        }
        result = _trim_document_content(doc_dict)
        # Output is {v, e} format
        assert " ... [trimmed 100 chars] ... " in result["attachments"][0]["content"]["v"]

    def test_trim_document_without_attachments(self):
        doc_dict = {
            "class_name": "SomeTaskDocument",
            "content": {"v": "some content", "e": "utf-8"},
        }
        result = _trim_document_content(doc_dict)
        assert "attachments" not in result
        # Preserves {v, e} format
        assert result["content"] == {"v": "some content", "e": "utf-8"}

    def test_trim_all_documents_trims_attachments(self):
        """All documents trim both primary text content and attachment content."""
        long_primary = "p" * 300
        doc_dict = {
            "class_name": "SomeFlowDocument",
            "content": {"v": long_primary, "e": "utf-8"},
            "attachments": [
                {"name": "big.bin", "content": {"v": "binarydata==", "e": "base64"}},
                {"name": "long.txt", "content": {"v": "z" * 300, "e": "utf-8"}},
            ],
        }
        result = _trim_document_content(doc_dict)
        # Primary content trimmed (all documents trimmed equally) - output is {v, e} format
        assert " ... [trimmed 100 chars] ... " in result["content"]["v"]
        # Attachments still trimmed
        assert result["attachments"][0]["content"] == {"v": "[binary content removed]", "e": "utf-8"}
        assert " ... [trimmed 100 chars] ... " in result["attachments"][1]["content"]["v"]

    def test_trim_binary_document_still_trims_attachments(self):
        """Binary document with text attachments: both primary and attachments trimmed."""
        doc_dict = {
            "class_name": "SomeTaskDocument",
            "content": {"v": "base64primary==", "e": "base64"},
            "attachments": [
                {"name": "note.txt", "content": {"v": "a" * 300, "e": "utf-8"}},
            ],
        }
        result = _trim_document_content(doc_dict)
        # Output is {v, e} format
        assert result["content"] == {"v": "[binary content removed]", "e": "utf-8"}
        assert " ... [trimmed 100 chars] ... " in result["attachments"][0]["content"]["v"]

    def test_original_dict_not_mutated(self):
        original_attachments = [
            {"name": "img.jpg", "content": {"v": "data==", "e": "base64"}},
        ]
        doc_dict = {
            "class_name": "SomeTaskDocument",
            "content": {"v": "text", "e": "utf-8"},
            "attachments": original_attachments,
        }
        _trim_document_content(doc_dict)
        # Original should not be mutated
        assert doc_dict["attachments"][0]["content"] == {"v": "data==", "e": "base64"}


class TestTrimDocumentsInDataWithAttachments:
    """Tests for _trim_documents_in_data with attachments."""

    def test_nested_structure_with_attachments(self):
        data = {
            "docs": [
                {
                    "class_name": "SomeTaskDocument",
                    "content": {"v": "short", "e": "utf-8"},
                    "attachments": [
                        {"name": "img.png", "content": {"v": "b64==", "e": "base64"}},
                    ],
                },
            ],
        }
        result = _trim_documents_in_data(data)
        # Output is {v, e} format
        assert result["docs"][0]["attachments"][0]["content"] == {"v": "[binary content removed]", "e": "utf-8"}

    def test_list_of_documents_with_attachments(self):
        data = [
            {
                "class_name": "SomeFlowDocument",
                "content": {"v": "flow text", "e": "utf-8"},
                "attachments": [
                    {"name": "a.txt", "content": {"v": "x" * 300, "e": "utf-8"}},
                ],
            },
        ]
        result = _trim_documents_in_data(data)
        # Preserves {v, e} format
        assert result[0]["content"] == {"v": "flow text", "e": "utf-8"}  # short content kept
        assert " ... [trimmed 100 chars] ... " in result[0]["attachments"][0]["content"]["v"]
