"""Tests for Document.create classmethod with various content types."""

from io import BytesIO
from typing import Any

import pytest
from pydantic import BaseModel
from ruamel.yaml import YAML

from ai_pipeline_core.documents import Document, FlowDocument, TemporaryDocument
from ai_pipeline_core.exceptions import DocumentNameError, DocumentSizeError


class ConcreteTestDocument(FlowDocument):
    """Concrete document class for testing."""

    def get_type(self) -> str:
        return "test"


class SmallDocument(FlowDocument):
    """Document with small size limit for testing."""

    MAX_CONTENT_SIZE = 10

    def get_type(self) -> str:
        return "small"


class TestCreateMethod:
    """Tests for the Document.create classmethod."""

    def test_create_with_str_content(self):
        """Test create with string content."""
        text = "Hello ✨"
        doc = ConcreteTestDocument.create(
            name="greeting.txt",
            description="str content",
            content=text,
        )
        assert doc.content == text.encode("utf-8")
        assert doc.text == text
        assert doc.is_text

    def test_create_with_bytes_content(self):
        """Test create with bytes content."""
        raw = b"\x00\xffdata"
        doc = ConcreteTestDocument.create(
            name="raw.bin",
            description=None,
            content=raw,
        )
        assert doc.content == raw
        assert doc.size == len(raw)
        assert not doc.is_text

    def test_create_validation_applies(self):
        """Test that validation still applies through create()."""
        # Name validation still applies through create()
        with pytest.raises(DocumentNameError):
            ConcreteTestDocument.create(name="../hack.txt", description=None, content="x")

        # Size validation applies too (using SmallDocument)
        doc_ok = SmallDocument.create(name="ok.txt", description=None, content="12345")
        assert doc_ok.size == 5
        with pytest.raises(DocumentSizeError):
            SmallDocument.create(name="big.txt", description=None, content="12345678901")

    def test_create_with_dict_json(self):
        """Test create with dictionary for JSON file."""
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        doc = ConcreteTestDocument.create(
            name="test.json", description="JSON from dict", content=data
        )

        assert doc.is_text
        assert doc.mime_type == "application/json"

        # Should be valid JSON
        parsed = doc.as_json()
        assert parsed == data

    def test_create_with_dict_yaml(self):
        """Test create with dictionary for YAML file."""
        data = {"database": {"host": "localhost", "port": 5432}, "cache": {"ttl": 300}}
        doc = ConcreteTestDocument.create(
            name="config.yaml", description="YAML config", content=data
        )

        assert doc.is_text
        assert doc.mime_type == "application/yaml"

        # Should be valid YAML
        yaml = YAML()
        parsed = yaml.load(BytesIO(doc.content))
        assert parsed == data

    def test_create_with_pydantic_model_json(self):
        """Test create with Pydantic model for JSON file."""

        class UserModel(BaseModel):
            username: str
            email: str
            active: bool = True

        user = UserModel(username="alice", email="alice@example.com")
        doc = ConcreteTestDocument.create(name="user.json", description="User data", content=user)

        assert doc.is_text
        assert doc.mime_type == "application/json"

        # Should be valid JSON with model data
        parsed = doc.as_json()
        assert parsed["username"] == "alice"
        assert parsed["email"] == "alice@example.com"
        assert parsed["active"] is True

        # Should roundtrip
        restored = UserModel(**parsed)
        assert restored == user

    def test_create_with_pydantic_model_yaml(self):
        """Test create with Pydantic model for YAML file."""

        class ServerConfig(BaseModel):
            name: str
            workers: int
            ssl_enabled: bool

        config = ServerConfig(name="api-server", workers=4, ssl_enabled=True)
        doc = ConcreteTestDocument.create(
            name="server.yaml", description="Server config", content=config
        )

        assert doc.is_text
        assert doc.mime_type == "application/yaml"

        # Should be valid YAML with model data
        yaml = YAML()
        parsed = yaml.load(BytesIO(doc.content))
        assert parsed["name"] == "api-server"
        assert parsed["workers"] == 4
        assert parsed["ssl_enabled"] is True

        # Should roundtrip
        restored = ServerConfig(**parsed)
        assert restored == config

    def test_create_with_list_json(self):
        """Test create with list for JSON file."""
        data = ["item1", "item2", "item3"]
        doc = ConcreteTestDocument.create(name="list.json", description="JSON list", content=data)

        assert doc.is_text
        assert doc.mime_type == "application/json"

        # Should be valid JSON
        parsed = doc.as_json()
        assert parsed == data

    def test_create_with_list_yaml(self):
        """Test create with list for YAML file."""
        data = ["item1", "item2", "item3"]
        doc = ConcreteTestDocument.create(name="list.yaml", description="YAML list", content=data)

        assert doc.is_text
        assert doc.mime_type == "application/yaml"

        # Should be valid YAML
        yaml = YAML()
        parsed = yaml.load(BytesIO(doc.content))
        assert parsed == data

    def test_create_with_list_markdown(self):
        """Test create with list for Markdown file."""
        items = ["# Section 1", "# Section 2", "# Section 3"]
        doc = ConcreteTestDocument.create(
            name="sections.md", description="Markdown sections", content=items
        )

        assert doc.is_text
        # List should be joined with markdown separator
        assert Document.MARKDOWN_LIST_SEPARATOR.join(items).encode() == doc.content

        # Should parse back correctly
        parsed_items = doc.as_markdown_list()
        assert parsed_items == items

    def test_create_with_list_of_models_json(self):
        """Test create with list of Pydantic models for JSON file."""

        class SampleModel(BaseModel):
            name: str
            value: int

        models = [
            SampleModel(name="first", value=1),
            SampleModel(name="second", value=2),
        ]

        doc = ConcreteTestDocument.create(
            name="models.json", description="List of models", content=models
        )

        assert doc.is_text
        assert doc.mime_type == "application/json"

        # Should be valid JSON with list of model data
        parsed = doc.as_json()
        assert len(parsed) == 2
        assert parsed[0]["name"] == "first"
        assert parsed[1]["value"] == 2

        # Should roundtrip
        restored = [SampleModel(**item) for item in parsed]
        assert restored == models

    def test_create_with_list_of_models_yaml(self):
        """Test create with list of Pydantic models for YAML file."""

        class SampleModel(BaseModel):
            name: str
            value: int

        models = [
            SampleModel(name="first", value=1),
            SampleModel(name="second", value=2),
        ]

        doc = ConcreteTestDocument.create(
            name="models.yaml", description="List of models", content=models
        )

        assert doc.is_text
        assert doc.mime_type == "application/yaml"

        # Should be valid YAML with list of model data
        yaml = YAML()
        parsed = yaml.load(BytesIO(doc.content))
        assert len(parsed) == 2
        assert parsed[0]["name"] == "first"
        assert parsed[1]["value"] == 2

        # Should roundtrip
        restored = [SampleModel(**item) for item in parsed]
        assert restored == models

    def test_create_temporary_document(self):
        """Test creating TemporaryDocument with various content types."""
        # With string
        doc1 = TemporaryDocument.create(
            name="temp.txt", content="Temporary content", description="Test temp doc"
        )
        assert doc1.is_temporary
        assert doc1.text == "Temporary content"

        # With dict
        doc2 = TemporaryDocument.create(name="temp.json", content={"temp": "data"})
        assert doc2.is_temporary
        assert doc2.as_json() == {"temp": "data"}

        # With Pydantic model
        class TempModel(BaseModel):
            value: str

        model = TempModel(value="test")
        doc3 = TemporaryDocument.create(name="temp.json", content=model)
        assert doc3.is_temporary
        assert doc3.as_json()["value"] == "test"

    # Removed test_create_with_numeric_content and test_create_with_boolean_content
    # as int/float/bool are not supported content types for create method

    def test_create_with_yml_extension(self):
        """Test create accepts .yml extension for YAML."""
        data = {"test": "value"}
        doc = ConcreteTestDocument.create(name="config.yml", content=data)
        assert doc.mime_type == "application/yaml"

    def test_json_yaml_roundtrip(self):
        """Test that JSON/YAML creation and parsing roundtrips correctly."""

        class DataModel(BaseModel):
            items: list[str]
            metadata: dict[str, Any]

        original = DataModel(items=["a", "b", "c"], metadata={"count": 3, "version": 1})

        # JSON roundtrip
        json_doc = ConcreteTestDocument.create(name="data.json", content=original)
        json_model = DataModel(**json_doc.as_json())
        assert json_model == original

        # YAML roundtrip
        yaml_doc = ConcreteTestDocument.create(name="data.yaml", content=original)
        yaml = YAML()
        yaml_data = yaml.load(BytesIO(yaml_doc.content))
        yaml_model = DataModel(**yaml_data)
        assert yaml_model == original

    def test_create_with_unsupported_content_type(self):
        """Test that unsupported content types raise an error."""

        # Custom class
        class CustomClass:
            pass

        obj = CustomClass()
        with pytest.raises(ValueError) as exc_info:
            ConcreteTestDocument.create(
                name="test.txt",
                content=obj,  # type: ignore[arg-type]
            )
        assert "Unsupported content type" in str(exc_info.value)

    def test_create_with_dict_non_json_yaml(self):
        """Test that dict content for non-JSON/YAML files raises error."""
        with pytest.raises(ValueError) as exc_info:
            ConcreteTestDocument.create(name="test.txt", content={"key": "value"})
        assert "Unsupported content type: <class 'dict'> for file test.txt" in str(exc_info.value)

    def test_create_with_list_non_json_yaml_md(self):
        """Test that list content for non-JSON/YAML/MD files raises error."""
        with pytest.raises(ValueError) as exc_info:
            ConcreteTestDocument.create(name="test.txt", content=["item1", "item2"])
        assert "Unsupported content type" in str(exc_info.value)

    def test_create_with_mixed_list_markdown(self):
        """Test that mixed-type lists are rejected for markdown."""
        from typing import cast

        mixed_list = ["string", 123, "another string"]
        with pytest.raises(ValueError) as exc_info:
            ConcreteTestDocument.create(name="test.md", content=cast(list[str], mixed_list))
        assert "mixed-type list for markdown" in str(exc_info.value)
