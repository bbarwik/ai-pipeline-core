"""Core Document class tests."""

import base64
from enum import StrEnum
from typing import ClassVar

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

    FILES: ClassVar[type[AllowedDocumentNames]] = AllowedDocumentNames

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

    def test_text_property(self):
        """Test text property for text documents."""
        text_content = "Hello, 世界! 🌍"
        doc = ConcreteTestFlowDocument(name="test.txt", content=text_content.encode("utf-8"))

        assert doc.text == text_content

        # Binary document should raise
        doc_binary = ConcreteTestFlowDocument(name="test.bin", content=b"\x00\x01\x02\x03")
        with pytest.raises(ValueError) as exc_info:
            doc_binary.text
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
        class SampleDocument(FlowDocument):
            def get_type(self) -> str:
                return "sample"

        assert SampleDocument.canonical_name() == "sample"

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
    """Tests for as_markdown_list helper."""

    def test_as_markdown_list_basic_split(self):
        items = [
            "First item line 1\nFirst item line 2",
            "Second item",
            "Third item with --- inside a word (should stay) like pre---post",
        ]
        # Join items with separator
        content = Document.MARKDOWN_LIST_SEPARATOR.join(items)
        doc = ConcreteTestFlowDocument(
            name="list.md",
            description="basic join",
            content=content,
        )
        assert isinstance(doc, ConcreteTestFlowDocument)
        assert doc.name == "list.md"
        assert doc.is_text is True

        # as_markdown_list should split back to the same items
        assert doc.as_markdown_list() == items

    def test_as_markdown_list_with_separators(self):
        # Test that as_markdown_list correctly splits content
        items = [
            "Line A\nLine B",
            "Start\nEnd",
            "One\nTwo",
            "Item with text",
            "keep --- here",  # separator in text should remain
        ]
        content = Document.MARKDOWN_LIST_SEPARATOR.join(items)
        doc = ConcreteTestFlowDocument(
            name="list.md",
            description=None,
            content=content,
        )
        result_items = doc.as_markdown_list()
        assert result_items == items

    def test_as_markdown_list_roundtrip(self):
        # Test that joining and splitting with separator works correctly
        items = [
            "alpha\nbeta",
            "no separators here",
            "text with --- embedded",
        ]
        content = Document.MARKDOWN_LIST_SEPARATOR.join(items)
        doc = ConcreteTestFlowDocument(
            name="roundtrip.md",
            description="roundtrip",
            content=content,
        )
        assert doc.as_markdown_list() == items


class TestDocumentConstructor:
    """Tests for the Document constructor."""

    def test_constructor_with_str_content(self):
        text = "Hello ✨"
        doc = ConcreteTestFlowDocument(
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

    def test_constructor_with_bytes_content(self):
        raw = b"\x00\xffdata"
        doc = ConcreteTestFlowDocument(
            name="raw.bin",
            description=None,
            content=raw,
        )
        assert doc.content == raw
        assert doc.size == len(raw)
        assert not doc.is_text

    def test_constructor_validation_applies(self):
        # Name validation still applies through constructor
        with pytest.raises(DocumentNameError):
            ConcreteTestFlowDocument(name="../hack.txt", description=None, content="x")

        # Size validation applies too (using SmallDocument)
        doc_ok = SmallDocument(name="ok.txt", description=None, content="12345")
        assert doc_ok.size == 5
        with pytest.raises(DocumentSizeError):
            SmallDocument(name="big.txt", description=None, content="12345678901")


class TestNewDocumentMethods:
    """Tests for Document methods like as_pydantic_model."""

    def test_as_pydantic_model_from_json(self):
        """Test parsing JSON document as Pydantic model."""
        from pydantic import BaseModel, Field

        class TestModel(BaseModel):
            name: str
            age: int = Field(ge=0)
            tags: list[str] = []

        # Create a JSON document
        json_content = '{"name": "Alice", "age": 30, "tags": ["python", "ai"]}'
        doc = ConcreteTestFlowDocument(name="data.json", content=json_content.encode())

        # Parse as Pydantic model
        model = doc.as_pydantic_model(TestModel)
        assert isinstance(model, TestModel)
        assert model.name == "Alice"
        assert model.age == 30
        assert model.tags == ["python", "ai"]

        # Test with missing optional field
        json_content2 = '{"name": "Bob", "age": 25}'
        doc2 = ConcreteTestFlowDocument(name="data2.json", content=json_content2.encode())
        model2 = doc2.as_pydantic_model(TestModel)
        assert model2.name == "Bob"
        assert model2.age == 25
        assert model2.tags == []

        # Test validation error
        json_invalid = '{"name": "Charlie", "age": -5}'
        doc_invalid = ConcreteTestFlowDocument(name="invalid.json", content=json_invalid.encode())
        with pytest.raises(ValueError):  # Pydantic validation error
            doc_invalid.as_pydantic_model(TestModel)

    def test_as_pydantic_model_from_yaml(self):
        """Test parsing YAML document as Pydantic model."""
        from pydantic import BaseModel

        class ConfigModel(BaseModel):
            host: str
            port: int
            debug: bool = False

        # Create a YAML document
        yaml_content = """host: localhost
port: 8080
debug: true
"""
        doc = ConcreteTestFlowDocument(name="config.yaml", content=yaml_content.encode())

        # Parse as Pydantic model
        model = doc.as_pydantic_model(ConfigModel)
        assert isinstance(model, ConfigModel)
        assert model.host == "localhost"
        assert model.port == 8080
        assert model.debug is True

        # Test with .yml extension
        doc_yml = ConcreteTestFlowDocument(name="config.yml", content=yaml_content.encode())
        model_yml = doc_yml.as_pydantic_model(ConfigModel)
        assert model_yml.host == "localhost"

    def test_create_json_with_dict(self):
        """Test creating JSON document with dictionary data."""
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        import json

        doc = ConcreteTestFlowDocument(
            name="test.json", description="JSON from dict", content=json.dumps(data, indent=2)
        )

        assert doc.name == "test.json"
        assert doc.description == "JSON from dict"
        assert doc.is_text is True
        assert "json" in doc.mime_type

        # Verify content can be parsed back
        parsed = doc.as_json()
        assert parsed == data

        # Verify formatting (should be indented)
        text = doc.text
        assert "  " in text  # Check for indentation

    def test_create_json_with_pydantic_model(self):
        """Test creating JSON document with Pydantic model."""
        import json

        from pydantic import BaseModel

        class UserModel(BaseModel):
            username: str
            email: str
            active: bool = True

        user = UserModel(username="alice", email="alice@example.com")
        doc = ConcreteTestFlowDocument(
            name="user.json",
            description="User data",
            content=json.dumps(user.model_dump(mode="json"), indent=2),
        )

        assert doc.name == "user.json"
        parsed = doc.as_json()
        assert parsed["username"] == "alice"
        assert parsed["email"] == "alice@example.com"
        assert parsed["active"] is True

    def test_json_document_creation(self):
        """Test creating JSON documents."""
        import json

        # Create JSON document directly
        data = {"key": "value"}
        doc = ConcreteTestFlowDocument(name="test.json", description=None, content=json.dumps(data))
        assert doc.as_json() == data

    def test_create_yaml_with_dict(self):
        """Test creating YAML document with dictionary data."""
        from io import BytesIO

        from ruamel.yaml import YAML

        data = {"database": {"host": "localhost", "port": 5432}, "cache": {"ttl": 300}}
        yaml = YAML()
        stream = BytesIO()
        yaml.dump(data, stream)
        doc = ConcreteTestFlowDocument(
            name="config.yaml", description="YAML config", content=stream.getvalue()
        )

        assert doc.name == "config.yaml"
        assert doc.description == "YAML config"
        assert doc.is_text is True
        assert "yaml" in doc.mime_type

        # Verify content can be parsed back
        parsed = doc.as_yaml()
        assert parsed == data

        # Verify YAML formatting
        text = doc.text
        assert "database:" in text
        assert "  host:" in text  # Check for indentation

    def test_create_yaml_with_yml_extension(self):
        """Test creating YAML document with .yml extension."""
        from io import BytesIO

        from ruamel.yaml import YAML

        data = {"test": "value"}
        yaml = YAML()
        stream = BytesIO()
        yaml.dump(data, stream)
        doc = ConcreteTestFlowDocument(
            name="config.yml", description=None, content=stream.getvalue()
        )
        assert doc.name == "config.yml"
        assert doc.as_yaml() == data

    def test_create_yaml_with_pydantic_model(self):
        """Test creating YAML document with Pydantic model."""
        from io import BytesIO

        from pydantic import BaseModel
        from ruamel.yaml import YAML

        class ServerConfig(BaseModel):
            name: str
            workers: int
            ssl_enabled: bool

        config = ServerConfig(name="api-server", workers=4, ssl_enabled=True)
        yaml = YAML()
        stream = BytesIO()
        yaml.dump(config.model_dump(mode="json"), stream)
        doc = ConcreteTestFlowDocument(
            name="server.yaml", description="Server config", content=stream.getvalue()
        )

        assert doc.name == "server.yaml"
        parsed = doc.as_yaml()
        assert parsed["name"] == "api-server"
        assert parsed["workers"] == 4
        assert parsed["ssl_enabled"] is True

    def test_json_yaml_roundtrip(self):
        """Test that JSON/YAML parsing roundtrips correctly."""
        import json
        from io import BytesIO

        from pydantic import BaseModel
        from ruamel.yaml import YAML

        class DataModel(BaseModel):
            items: list[str]
            metadata: dict[str, int]

        original = DataModel(items=["a", "b", "c"], metadata={"count": 3, "version": 1})

        # JSON roundtrip
        json_doc = ConcreteTestFlowDocument(
            name="data.json",
            description=None,
            content=json.dumps(original.model_dump(mode="json"), indent=2),
        )
        json_model = json_doc.as_pydantic_model(DataModel)
        assert json_model == original

        # YAML roundtrip
        yaml = YAML()
        stream = BytesIO()
        yaml.dump(original.model_dump(mode="json"), stream)
        yaml_doc = ConcreteTestFlowDocument(
            name="data.yaml", description=None, content=stream.getvalue()
        )
        yaml_model = yaml_doc.as_pydantic_model(DataModel)
        assert yaml_model == original


class TestParsedMethod:
    """Tests for the parsed() method that reverses document creation."""

    def test_parsed_str_roundtrip(self):
        """Test that str content roundtrips correctly."""
        original = "Hello, World! 🌍"
        doc = ConcreteTestFlowDocument(name="test.txt", content=original)
        assert doc.parsed(str) == original

    def test_parsed_bytes_roundtrip(self):
        """Test that bytes content roundtrips correctly."""
        original = b"Binary \x00\xff\xfe data"
        doc = ConcreteTestFlowDocument(name="data.bin", content=original)
        assert doc.parsed(bytes) == original

    def test_parsed_json_dict_roundtrip(self):
        """Test that dict → JSON → dict roundtrips correctly."""
        import json

        original = {"name": "test", "values": [1, 2, 3], "nested": {"key": "value"}}
        doc = ConcreteTestFlowDocument(name="data.json", content=json.dumps(original, indent=2))
        assert doc.parsed(dict) == original

    def test_parsed_json_list_roundtrip(self):
        """Test that list → JSON → list roundtrips correctly."""
        import json

        original = [1, 2, {"key": "value"}, "text"]
        doc = ConcreteTestFlowDocument(name="array.json", content=json.dumps(original))
        assert doc.parsed(list) == original

    def test_parsed_yaml_dict_roundtrip(self):
        """Test that dict → YAML → dict roundtrips correctly."""
        from io import BytesIO

        from ruamel.yaml import YAML

        original = {"database": {"host": "localhost", "port": 5432}}
        yaml = YAML()
        stream = BytesIO()
        yaml.dump(original, stream)

        doc = ConcreteTestFlowDocument(name="config.yaml", content=stream.getvalue())
        assert doc.parsed(dict) == original

    def test_parsed_yaml_with_yml_extension(self):
        """Test that .yml extension works with parsed()."""
        from io import BytesIO

        from ruamel.yaml import YAML

        original = {"test": "value"}
        yaml = YAML()
        stream = BytesIO()
        yaml.dump(original, stream)

        doc = ConcreteTestFlowDocument(name="config.yml", content=stream.getvalue())
        assert doc.parsed(dict) == original

    def test_parsed_markdown_list_roundtrip(self):
        """Test that list → markdown → list roundtrips correctly."""
        original = ["First item\nwith lines", "Second item", "Third item"]
        content = Document.MARKDOWN_LIST_SEPARATOR.join(original)
        doc = ConcreteTestFlowDocument(name="items.md", content=content)
        assert doc.parsed(list) == original

    def test_parsed_pydantic_model_json(self):
        """Test parsing JSON to Pydantic model."""
        import json

        from pydantic import BaseModel

        class UserModel(BaseModel):
            name: str
            age: int
            active: bool = True

        original = UserModel(name="Alice", age=30)
        doc = ConcreteTestFlowDocument(name="user.json", content=json.dumps(original.model_dump()))
        parsed = doc.parsed(UserModel)
        assert parsed == original
        assert isinstance(parsed, UserModel)

    def test_parsed_pydantic_model_yaml(self):
        """Test parsing YAML to Pydantic model."""
        from io import BytesIO

        from pydantic import BaseModel
        from ruamel.yaml import YAML

        class ConfigModel(BaseModel):
            host: str
            port: int
            ssl: bool

        original = ConfigModel(host="localhost", port=8080, ssl=True)
        yaml = YAML()
        stream = BytesIO()
        yaml.dump(original.model_dump(), stream)

        doc = ConcreteTestFlowDocument(name="config.yaml", content=stream.getvalue())
        parsed = doc.parsed(ConfigModel)
        assert parsed == original
        assert isinstance(parsed, ConfigModel)

    def test_parsed_unsupported_type(self):
        """Test that unsupported types raise ValueError."""
        doc = ConcreteTestFlowDocument(name="test.txt", content="text")

        with pytest.raises(ValueError, match="Unsupported type"):
            doc.parsed(int)

        with pytest.raises(ValueError, match="Unsupported type"):
            doc.parsed(float)

    def test_parsed_wrong_extension_for_type(self):
        """Test that wrong extension for requested type raises error."""
        doc = ConcreteTestFlowDocument(name="test.txt", content="not json")

        # .txt file cannot be parsed as dict without being JSON/YAML
        with pytest.raises(ValueError):
            doc.parsed(dict)

    def test_parsed_binary_as_str(self):
        """Test that binary content can be decoded to str."""
        doc = ConcreteTestFlowDocument(name="test.bin", content=b"Hello UTF-8")
        assert doc.parsed(str) == "Hello UTF-8"

    def test_parsed_edge_cases(self):
        """Test edge cases for parsed method."""
        # Empty string
        doc1 = ConcreteTestFlowDocument(name="empty.txt", content="")
        assert doc1.parsed(str) == ""
        assert doc1.parsed(bytes) == b""

        # Empty JSON array
        import json

        doc2 = ConcreteTestFlowDocument(name="empty.json", content=json.dumps([]))
        assert doc2.parsed(list) == []

        # Empty JSON object
        doc3 = ConcreteTestFlowDocument(name="empty.json", content=json.dumps({}))
        assert doc3.parsed(dict) == {}

        # Single item markdown list
        doc4 = ConcreteTestFlowDocument(name="single.md", content="Only item")
        assert doc4.parsed(list) == ["Only item"]
