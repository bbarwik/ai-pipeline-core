"""Core Document class tests."""

import base64
from enum import StrEnum

import pytest

from ai_pipeline_core.documents import Document, FlowDocument, TaskDocument
from ai_pipeline_core.exceptions import DocumentNameError, DocumentSizeError


class ConcreteTestFlowDocument(FlowDocument):
    """Concrete FlowDocument for testing."""

    def get_type(self) -> str:
        return "test_flow"


class ConcreteTestTaskDoc(TaskDocument):
    """Concrete TaskDocument for testing."""

    def get_type(self) -> str:
        return "test_task"


class AllowedDocumentNames(StrEnum):
    ALLOWED_FILE_1 = "allowed1.txt"
    ALLOWED_FILE_2 = "allowed2.json"


class RestrictedDocument(FlowDocument):
    """Document with restricted file names."""

    FILES = AllowedDocumentNames

    def get_type(self) -> str:
        return "restricted"


class SmallDocument(FlowDocument):
    """Document with small size limit."""

    MAX_CONTENT_SIZE = 10

    def get_type(self) -> str:
        return "small"


class TestDocumentValidation:
    """Test document validation logic."""

    def test_name_validation_with_files_enum(self):
        """Test name validation against FILES enum."""
        # Valid names should work
        doc = RestrictedDocument(
            name="allowed1.txt",
            content=b"test",
        )
        assert doc.name == "allowed1.txt"

        # Invalid name should raise
        with pytest.raises(DocumentNameError) as exc_info:
            RestrictedDocument(
                name="not_allowed.txt",
                content=b"test",
            )
        assert "Invalid filename" in str(exc_info.value)
        assert "allowed1.txt" in str(exc_info.value)

    def test_content_size_limit(self):
        """Test content size limit enforcement."""
        # Within limit should work
        doc = SmallDocument(
            name="test.txt",
            content=b"123456789",  # 9 bytes
        )
        assert doc.size == 9

        # Exceeding limit should raise
        with pytest.raises(DocumentSizeError) as exc_info:
            SmallDocument(
                name="test.txt",
                content=b"12345678901",  # 11 bytes
            )
        assert "exceeds maximum allowed size" in str(exc_info.value)

    def test_name_security_validation(self):
        """Test security validation for document names."""
        # Path traversal attempts should fail
        with pytest.raises(DocumentNameError):
            ConcreteTestFlowDocument(name="../etc/passwd", content=b"test")

        with pytest.raises(DocumentNameError):
            ConcreteTestFlowDocument(name="..\\windows\\system32", content=b"test")

        with pytest.raises(DocumentNameError):
            ConcreteTestFlowDocument(name="/etc/passwd", content=b"test")

        # Description extension should be rejected
        with pytest.raises(DocumentNameError) as exc_info:
            ConcreteTestFlowDocument(name="test.description.md", content=b"test")
        assert "cannot end with .description.md" in str(exc_info.value)

        # Empty or whitespace names should fail
        with pytest.raises(DocumentNameError):
            ConcreteTestFlowDocument(name="", content=b"test")

        with pytest.raises(DocumentNameError):
            ConcreteTestFlowDocument(name=" ", content=b"test")

        with pytest.raises(DocumentNameError):
            ConcreteTestFlowDocument(name="test ", content=b"test")


class TestAbstractInstantiation:
    """Test that abstract document classes cannot be instantiated directly."""

    def test_cannot_instantiate_document(self):
        """Test that Document cannot be instantiated directly."""
        # Document is abstract and cannot be instantiated even if we try
        # Python will raise TypeError about abstract methods before our __init__ check
        with pytest.raises(TypeError) as exc_info:
            Document(name="test.txt", content=b"test")  # type: ignore[abstract]
        # The error message will be about abstract methods, not our custom message
        assert "abstract" in str(exc_info.value).lower()

    def test_cannot_instantiate_flow_document(self):
        """Test that FlowDocument cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            FlowDocument(name="test.txt", content=b"test")
        assert "Cannot instantiate abstract FlowDocument class directly" in str(exc_info.value)

    def test_cannot_instantiate_task_document(self):
        """Test that TaskDocument cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            TaskDocument(name="test.txt", content=b"test")
        assert "Cannot instantiate abstract TaskDocument class directly" in str(exc_info.value)

    def test_can_instantiate_concrete_subclasses(self):
        """Test that concrete subclasses can be instantiated."""

        # Concrete FlowDocument subclass
        class ConcreteFlowDoc(FlowDocument):
            def get_type(self) -> str:
                return "concrete_flow"

        doc = ConcreteFlowDoc(name="test.txt", content=b"test")
        assert doc.name == "test.txt"
        assert doc.content == b"test"

        # Concrete TaskDocument subclass
        class ConcreteTaskDoc(TaskDocument):
            def get_type(self) -> str:
                return "concrete_task"

        doc = ConcreteTaskDoc(name="test.txt", content=b"test")
        assert doc.name == "test.txt"
        assert doc.content == b"test"


class TestDocumentProperties:
    """Test document properties and methods."""

    def test_id_determinism(self):
        """Test that document ID is deterministic based on content."""
        content1 = b"Hello, world!"
        content2 = b"Different content"

        # Same content should produce same ID
        doc1 = ConcreteTestFlowDocument(name="test1.txt", content=content1)
        doc2 = ConcreteTestFlowDocument(name="test2.txt", content=content1)
        assert doc1.id == doc2.id
        assert doc1.sha256 == doc2.sha256

        # Different content should produce different ID
        doc3 = ConcreteTestFlowDocument(name="test3.txt", content=content2)
        assert doc3.id != doc1.id
        assert doc3.sha256 != doc1.sha256

        # ID should be first 6 chars of SHA256
        assert len(doc1.id) == 6
        assert doc1.id == doc1.sha256[:6]

    def test_mime_detection_text(self):
        """Test MIME type detection for text documents."""
        # Plain text
        doc = ConcreteTestFlowDocument(name="test.txt", content=b"Hello, world!")
        assert doc.is_text is True
        assert doc.is_image is False
        assert doc.is_pdf is False
        assert "text" in doc.mime_type

        # Markdown
        doc_md = ConcreteTestFlowDocument(name="test.md", content=b"# Header\n\nContent")
        assert doc_md.is_text is True
        assert doc_md.mime_type == "text/markdown"

        # JSON
        doc_json = ConcreteTestFlowDocument(name="test.json", content=b'{"key": "value"}')
        assert doc_json.is_text is True

    def test_mime_detection_binary(self):
        """Test MIME type detection for binary documents."""
        # Simple PNG header (minimal valid PNG)
        png_header = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        )
        doc_png = ConcreteTestFlowDocument(name="test.png", content=png_header)
        assert doc_png.is_image is True
        assert doc_png.is_text is False
        assert doc_png.is_pdf is False
        assert "image" in doc_png.mime_type

        # PDF header
        pdf_header = b"%PDF-1.4\n%\xd3\xeb\xe9\xe1\n1 0 obj\n<</Type/Catalog>>\nendobj"
        doc_pdf = ConcreteTestFlowDocument(name="test.pdf", content=pdf_header)
        assert doc_pdf.is_pdf is True
        assert doc_pdf.is_text is False
        assert doc_pdf.is_image is False
        assert "pdf" in doc_pdf.mime_type

    def test_as_text_method(self):
        """Test as_text() method for text documents."""
        text_content = "Hello, 世界! 🌍"
        doc = ConcreteTestFlowDocument(name="test.txt", content=text_content.encode("utf-8"))

        assert doc.as_text() == text_content

        # Binary document should raise
        doc_binary = ConcreteTestFlowDocument(name="test.bin", content=b"\x00\x01\x02\x03")
        with pytest.raises(ValueError) as exc_info:
            doc_binary.as_text()
        assert "not text" in str(exc_info.value)

    def test_as_json_method(self):
        """Test as_json() method."""
        json_data = {"key": "value", "number": 123}
        import json

        doc = ConcreteTestFlowDocument(name="test.json", content=json.dumps(json_data).encode())

        assert doc.as_json() == json_data

        # Invalid JSON should raise
        doc_invalid = ConcreteTestFlowDocument(name="test.txt", content=b"not json")
        with pytest.raises(json.JSONDecodeError):
            doc_invalid.as_json()

    def test_as_yaml_method(self):
        """Test as_yaml() method."""
        yaml_content = "key: value\nnumber: 123\n"
        doc = ConcreteTestFlowDocument(name="test.yaml", content=yaml_content.encode())

        result = doc.as_yaml()
        assert result["key"] == "value"
        assert result["number"] == 123

    def test_serialize_model(self):
        """Test serialize_model method."""
        doc = ConcreteTestFlowDocument(
            name="test.txt", content=b"Hello, world!", description="Test document"
        )

        serialized = doc.serialize_model()

        assert serialized["name"] == "test.txt"
        assert serialized["description"] == "Test document"
        assert serialized["base_type"] == "flow"
        assert serialized["size"] == 13
        assert serialized["id"] == doc.id
        assert serialized["sha256"] == doc.sha256
        assert "mime_type" in serialized
        assert serialized["content"] == "Hello, world!"
        assert serialized["content_encoding"] == "utf-8"

        # Binary content should be base64 encoded
        doc_binary = ConcreteTestFlowDocument(name="test.bin", content=b"\x00\x01\x02\x03")
        serialized_binary = doc_binary.serialize_model()
        assert serialized_binary["content_encoding"] == "base64"
        assert serialized_binary["content"] == base64.b64encode(b"\x00\x01\x02\x03").decode()

    def test_from_dict(self):
        """Test from_dict deserialization."""
        # UTF-8 content
        data = {
            "name": "test.txt",
            "content": "Hello, world!",
            "content_encoding": "utf-8",
            "description": "Test doc",
        }
        doc = ConcreteTestFlowDocument.from_dict(data)
        assert doc.name == "test.txt"
        assert doc.content == b"Hello, world!"
        assert doc.description == "Test doc"

        # Base64 content
        data_base64 = {
            "name": "test.bin",
            "content": base64.b64encode(b"\x00\x01\x02\x03").decode(),
            "content_encoding": "base64",
        }
        doc_binary = ConcreteTestFlowDocument.from_dict(data_base64)
        assert doc_binary.content == b"\x00\x01\x02\x03"

    def test_base_type_properties(self):
        """Test base_type related properties."""
        flow_doc = ConcreteTestFlowDocument(name="test.txt", content=b"test")
        assert flow_doc.base_type == "flow"
        assert flow_doc.is_flow is True
        assert flow_doc.is_task is False
        assert flow_doc.get_base_type() == "flow"

        task_doc = ConcreteTestTaskDoc(name="test.txt", content=b"test")
        assert task_doc.base_type == "task"
        assert task_doc.is_task is True
        assert task_doc.is_flow is False
        assert task_doc.get_base_type() == "task"

    def test_canonical_name(self):
        """Test canonical_name method."""

        # Simple case - removes Document suffix
        class TestDocument(FlowDocument):
            def get_type(self) -> str:
                return "test"

        assert TestDocument.canonical_name() == "test"

        # Removes FlowDocument suffix
        class MyFlowDocument(FlowDocument):
            def get_type(self) -> str:
                return "my"

        assert MyFlowDocument.canonical_name() == "my"

        # Complex name with multiple words
        class FinalReportDocument(FlowDocument):
            def get_type(self) -> str:
                return "final_report"

        assert FinalReportDocument.canonical_name() == "final_report"


class TestMarkdownListHelpers:
    """Tests for create_as_markdown_list and as_markdown_list helpers."""

    def test_create_as_markdown_list_basic_join(self):
        items = [
            "First item line 1\nFirst item line 2",
            "Second item",
            "Third item with --- inside a word (should stay) like pre---post",
        ]
        doc = ConcreteTestFlowDocument.create_as_markdown_list(
            name="list.md",
            description="basic join",
            items=items,
        )
        # Content should be items joined by the canonical separator
        expected_content = (
            items[0]
            + Document.MARKDOWN_LIST_SEPARATOR
            + items[1]
            + Document.MARKDOWN_LIST_SEPARATOR
            + items[2]
        ).encode("utf-8")
        assert isinstance(doc, ConcreteTestFlowDocument)
        assert doc.name == "list.md"
        assert doc.content == expected_content
        assert doc.is_text is True

        # as_markdown_list should split back to the same items
        assert doc.as_markdown_list() == items

    def test_create_as_markdown_list_removes_separator_lines(self):
        sep_line = Document.MARKDOWN_LIST_SEPARATOR.strip()  # '---'
        items = [
            # Plain separator line alone should be removed
            f"Line A\n{sep_line}\nLine B",
            # Separator with surrounding whitespace should be removed
            f"Start\n   {sep_line}   \nEnd",
            # Tabs and mixed whitespace
            f"One\n\t{sep_line}\t\nTwo",
            # Separator at end of string without newline
            f"X\n{sep_line}",
            # Multiple consecutive separator-only lines
            f"P\n{sep_line}\n{sep_line}\nQ",
            # Lines containing separator but with other text should remain
            f"keep {sep_line} here",
            f"prefix-{sep_line}-suffix",
        ]

        doc = ConcreteTestFlowDocument.create_as_markdown_list(
            name="list.md",
            description=None,
            items=items,
        )

        # Verify expected per-item cleaning effects when splitting back
        result_items = doc.as_markdown_list()
        # Ensure no separator-only lines remain within items
        for item in result_items:
            for line in item.splitlines():
                assert line.strip() != sep_line, (
                    f"Found stray separator-only line within item: {line!r}"
                )
        assert result_items[0] == "Line A\nLine B"
        assert result_items[1] == "Start\nEnd"
        assert result_items[2] == "One\nTwo"
        assert result_items[3] == "X\n"  # trailing newline preserved; separator-only line removed
        assert result_items[4] == "P\nQ"  # consecutive separator-only lines removed
        # Inline/embedded separators should remain unchanged
        assert result_items[5] == "keep --- here"
        assert result_items[6] == "prefix-----suffix"

    def test_create_as_markdown_list_handles_crlf(self):
        sep_line = Document.MARKDOWN_LIST_SEPARATOR.strip()
        items = [
            f"A\r\n{sep_line}\r\nB\r\n",
            f"C\r\n   {sep_line}\r\n   D",
            f"E\r\n\t{sep_line}\r\nF",
            f"G\r\n{sep_line}",  # separator as last line
        ]
        doc = ConcreteTestFlowDocument.create_as_markdown_list(
            name="list_crlf.md",
            description="crlf",
            items=items,
        )
        result_items = doc.as_markdown_list()
        assert (
            result_items[0] == "A\nB\n"
        )  # trailing newline preserved from original item text apart from removed lines
        assert result_items[1] == "C\n   D"
        assert result_items[2] == "E\nF"
        assert (
            result_items[3] == "G\n"
        )  # trailing newline from original line preserved; separator-only line removed

    def test_as_markdown_list_is_inverse_of_create_as_markdown_list(self):
        # Items that include various whitespace and embedded dashes that should not be removed
        items = [
            "alpha\n---embedded---\nbeta",
            "no separators here",
            "spaces- --- -around",
        ]
        doc = ConcreteTestFlowDocument.create_as_markdown_list(
            name="roundtrip.md",
            description="roundtrip",
            items=items,
        )
        assert doc.as_markdown_list() == items


class TestCreateFactory:
    """Tests for the generic create() classmethod."""

    def test_create_with_str_content(self):
        text = "Hello ✨"
        doc = ConcreteTestFlowDocument.create(
            name="greeting.txt",
            description="str content",
            content=text,
        )
        assert isinstance(doc, ConcreteTestFlowDocument)
        assert doc.name == "greeting.txt"
        assert doc.description == "str content"
        assert doc.content == text.encode("utf-8")
        assert doc.size == len(text.encode("utf-8"))
        # Deterministic hashing
        assert doc.id == doc.sha256[:6]

    def test_create_with_bytes_content(self):
        raw = b"\x00\xffdata"
        doc = ConcreteTestFlowDocument.create(
            name="raw.bin",
            description=None,
            content=raw,
        )
        assert doc.content == raw
        assert doc.size == len(raw)
        assert not doc.is_text

    def test_create_validation_applies(self):
        # Name validation still applies through create()
        with pytest.raises(DocumentNameError):
            ConcreteTestFlowDocument.create(name="../hack.txt", description=None, content="x")

        # Size validation applies too (using SmallDocument)
        doc_ok = SmallDocument.create(name="ok.txt", description=None, content="12345")
        assert doc_ok.size == 5
        with pytest.raises(DocumentSizeError):
            SmallDocument.create(name="big.txt", description=None, content="12345678901")
