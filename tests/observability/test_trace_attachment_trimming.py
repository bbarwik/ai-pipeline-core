"""Tests for attachment trimming in tracing."""

from ai_pipeline_core.observability._trimming import (
    _trim_attachment_list,
    _trim_document_content,
    _trim_documents_in_data,
)


class TestTrimAttachmentList:
    """Tests for _trim_attachment_list."""

    def test_binary_attachment_content_replaced(self):
        attachments = [
            {"name": "img.jpg", "content": "data:image/jpeg;base64,SGVsbG8="},
        ]
        result = _trim_attachment_list(attachments)
        assert result[0]["content"] == "[binary content removed]"
        assert result[0]["name"] == "img.jpg"

    def test_short_text_attachment_preserved(self):
        attachments = [
            {"name": "note.txt", "content": "short text"},
        ]
        result = _trim_attachment_list(attachments)
        assert result[0]["content"] == "short text"

    def test_long_text_attachment_trimmed(self):
        long_text = "x" * 300
        attachments = [
            {"name": "long.txt", "content": long_text},
        ]
        result = _trim_attachment_list(attachments)
        content = result[0]["content"]
        assert isinstance(content, str)
        assert len(content) < len(long_text)
        assert content.startswith("x" * 100)
        assert content.endswith("x" * 100)
        assert " ... [trimmed 100 chars] ... " in content

    def test_non_dict_items_passed_through(self):
        attachments = ["not-a-dict", 42]
        result = _trim_attachment_list(attachments)
        assert result == ["not-a-dict", 42]

    def test_plain_string_text_preserved(self):
        """Plain string content is treated as text."""
        attachments = [
            {"name": "note.txt", "content": "short"},
        ]
        result = _trim_attachment_list(attachments)
        assert result[0]["content"] == "short"

    def test_original_not_mutated(self):
        att = {"name": "img.jpg", "content": "data:image/jpeg;base64,SGVsbG8="}
        original_content = att["content"]
        _trim_attachment_list([att])
        assert att["content"] == original_content


class TestTrimDocumentContentWithAttachments:
    """Tests for _trim_document_content with attachments."""

    def test_trim_document_with_binary_attachments(self):
        doc_dict = {
            "class_name": "SomeTaskDocument",
            "content": "short text",
            "attachments": [
                {"name": "img.jpg", "content": "data:image/jpeg;base64,base64data=="},
            ],
        }
        result = _trim_document_content(doc_dict)
        assert result["content"] == "short text"  # primary content preserved (short)
        assert result["attachments"][0]["content"] == "[binary content removed]"

    def test_trim_document_with_text_attachments_short(self):
        doc_dict = {
            "class_name": "SomeTaskDocument",
            "content": "primary",
            "attachments": [
                {"name": "note.txt", "content": "short note"},
            ],
        }
        result = _trim_document_content(doc_dict)
        assert result["attachments"][0]["content"] == "short note"

    def test_trim_document_with_text_attachments_long(self):
        long_text = "y" * 300
        doc_dict = {
            "class_name": "SomeTaskDocument",
            "content": "primary",
            "attachments": [
                {"name": "long.txt", "content": long_text},
            ],
        }
        result = _trim_document_content(doc_dict)
        assert " ... [trimmed 100 chars] ... " in result["attachments"][0]["content"]

    def test_trim_document_without_attachments(self):
        doc_dict = {
            "class_name": "SomeTaskDocument",
            "content": "some content",
        }
        result = _trim_document_content(doc_dict)
        assert "attachments" not in result
        assert result["content"] == "some content"

    def test_trim_all_documents_trims_attachments(self):
        """All documents trim both primary text content and attachment content."""
        long_primary = "p" * 300
        doc_dict = {
            "class_name": "SomeFlowDocument",
            "content": long_primary,
            "attachments": [
                {"name": "big.bin", "content": "data:application/octet-stream;base64,binarydata=="},
                {"name": "long.txt", "content": "z" * 300},
            ],
        }
        result = _trim_document_content(doc_dict)
        # Primary content trimmed
        assert " ... [trimmed 100 chars] ... " in result["content"]
        # Attachments still trimmed
        assert result["attachments"][0]["content"] == "[binary content removed]"
        assert " ... [trimmed 100 chars] ... " in result["attachments"][1]["content"]

    def test_trim_binary_document_still_trims_attachments(self):
        """Binary document with text attachments: both primary and attachments trimmed."""
        doc_dict = {
            "class_name": "SomeTaskDocument",
            "content": "data:application/octet-stream;base64,base64primary==",
            "attachments": [
                {"name": "note.txt", "content": "a" * 300},
            ],
        }
        result = _trim_document_content(doc_dict)
        assert result["content"] == "[binary content removed]"
        assert " ... [trimmed 100 chars] ... " in result["attachments"][0]["content"]

    def test_original_dict_not_mutated(self):
        original_attachments = [
            {"name": "img.jpg", "content": "data:image/jpeg;base64,data=="},
        ]
        doc_dict = {
            "class_name": "SomeTaskDocument",
            "content": "text",
            "attachments": original_attachments,
        }
        _trim_document_content(doc_dict)
        # Original should not be mutated
        assert doc_dict["attachments"][0]["content"] == "data:image/jpeg;base64,data=="


class TestTrimDocumentsInDataWithAttachments:
    """Tests for _trim_documents_in_data with attachments."""

    def test_nested_structure_with_attachments(self):
        data = {
            "docs": [
                {
                    "class_name": "SomeTaskDocument",
                    "content": "short",
                    "attachments": [
                        {"name": "img.png", "content": "data:image/png;base64,b64=="},
                    ],
                },
            ],
        }
        result = _trim_documents_in_data(data)
        assert result["docs"][0]["attachments"][0]["content"] == "[binary content removed]"

    def test_list_of_documents_with_attachments(self):
        data = [
            {
                "class_name": "SomeFlowDocument",
                "content": "flow text",
                "attachments": [
                    {"name": "a.txt", "content": "x" * 300},
                ],
            },
        ]
        result = _trim_documents_in_data(data)
        assert result[0]["content"] == "flow text"  # short content kept
        assert " ... [trimmed 100 chars] ... " in result[0]["attachments"][0]["content"]
