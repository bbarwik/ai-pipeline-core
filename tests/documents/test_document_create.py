"""Comprehensive tests for Document constructor with various content types."""

from typing import Any

import pytest
from pydantic import BaseModel, Field

from ai_pipeline_core.documents import FlowDocument


class ConcreteTestDocument(FlowDocument):
    """Concrete test document for testing."""

    def get_type(self) -> str:
        return "test"


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    age: int = Field(ge=0)
    tags: list[str] = []
    config: dict[str, Any] = {}


class TestConstructorWithBytes:
    """Test constructor with bytes content."""

    def test_constructor_with_bytes_content(self):
        """Test constructing document with raw bytes."""
        content = b"Hello, World!"
        doc = ConcreteTestDocument(
            name="test.txt",
            description="Test document",
            content=content,
        )
        assert doc.name == "test.txt"
        assert doc.description == "Test document"
        assert doc.content == content
        assert doc.size == len(content)

    def test_constructor_with_empty_bytes(self):
        """Test constructing document with empty bytes."""
        doc = ConcreteTestDocument(
            name="empty.txt",
            description=None,
            content=b"",
        )
        assert doc.content == b""
        assert doc.size == 0

    def test_constructor_with_binary_bytes(self):
        """Test constructing document with binary content."""
        content = b"\x00\x01\x02\x03\xff\xfe"
        doc = ConcreteTestDocument(
            name="binary.dat",
            description="Binary data",
            content=content,
        )
        assert doc.content == content
        assert not doc.is_text


class TestConstructorWithString:
    """Test constructor method with string content."""

    def test_constructor_with_string_content(self):
        """Test constructing document with string content."""
        content = "Hello, World! 你好世界 🌍"
        doc = ConcreteTestDocument(
            name="test.txt",
            description="Unicode text",
            content=content,
        )
        assert doc.content == content.encode("utf-8")
        assert doc.text == content
        assert doc.is_text

    def test_constructor_with_empty_string(self):
        """Test constructing document with empty string."""
        doc = ConcreteTestDocument(
            name="empty.txt",
            description=None,
            content="",
        )
        assert doc.content == b""
        # Empty content has mime type application/x-empty which is not text
        assert doc.mime_type == "application/x-empty"

    def test_constructor_with_multiline_string(self):
        """Test constructing document with multiline string."""
        content = """Line 1
Line 2
Line 3"""
        doc = ConcreteTestDocument(
            name="multiline.txt",
            description="Multiline content",
            content=content,
        )
        assert doc.text == content


class TestConstructorWithListForMarkdown:
    """Test constructor method with list[str] for markdown files."""

    def test_constructor_markdown_with_string_list(self):
        """Test constructing markdown document with list of strings."""
        items = [
            "# Section 1\nContent for section 1",
            "# Section 2\nContent for section 2",
            "# Section 3\nContent for section 3",
        ]
        doc = ConcreteTestDocument(
            name="sections.md",
            description="Markdown sections",
            content=items,
        )
        assert doc.name == "sections.md"
        assert doc.is_text
        assert "markdown" in doc.mime_type

        # Should use markdown list separator
        parsed_items = doc.as_markdown_list()
        assert len(parsed_items) == 3
        assert parsed_items[0] == items[0]
        assert parsed_items[1] == items[1]
        assert parsed_items[2] == items[2]

    def test_constructor_markdown_with_empty_list(self):
        """Test constructing markdown document with empty list."""
        doc = ConcreteTestDocument(
            name="empty.md",
            description="Empty markdown",
            content=[],
        )
        assert doc.content == b""
        # Empty content has mime type application/x-empty which is not text
        assert doc.mime_type == "application/x-empty"

    def test_constructor_markdown_with_single_item_list(self):
        """Test constructing markdown document with single item list."""
        items = ["Single item content"]
        doc = ConcreteTestDocument(
            name="single.md",
            description=None,
            content=items,
        )
        assert doc.as_markdown_list() == items

    def test_non_markdown_file_with_list_raises_error(self):
        """Test that non-markdown files with list content raise an error."""
        items = ["item1", "item2"]
        with pytest.raises(ValueError) as exc_info:
            ConcreteTestDocument(
                name="test.txt",  # Not .md extension
                description=None,
                content=items,
            )
        assert "Unsupported content type" in str(exc_info.value)
        assert "list" in str(exc_info.value)


class TestConstructorWithDictForJSON:
    """Test constructor method with dict/BaseModel for JSON files."""

    def test_constructor_json_with_dict(self):
        """Test constructing JSON document with dictionary."""
        data = {
            "name": "Alice",
            "age": 30,
            "tags": ["python", "ai"],
            "active": True,
        }
        doc = ConcreteTestDocument(
            name="data.json",
            description="JSON data",
            content=data,
        )
        assert doc.name == "data.json"
        assert doc.is_text
        assert "json" in doc.mime_type

        # Verify content is valid JSON
        parsed = doc.as_json()
        assert parsed == data

        # Check formatting (should be indented)
        text = doc.text
        assert "  " in text  # Has indentation

    def test_constructor_json_with_pydantic_model(self):
        """Test constructing JSON document with Pydantic model."""
        model = SampleModel(
            name="Bob",
            age=25,
            tags=["developer", "python"],
            config={"debug": True, "timeout": 30},
        )
        doc = ConcreteTestDocument(
            name="model.json",
            description="Model data",
            content=model,
        )

        # Verify content
        parsed = doc.as_json()
        assert parsed["name"] == "Bob"
        assert parsed["age"] == 25
        assert parsed["tags"] == ["developer", "python"]
        assert parsed["config"]["debug"] is True

    def test_constructor_json_with_nested_dict(self):
        """Test constructing JSON document with nested dictionary."""
        data = {"level1": {"level2": {"level3": {"value": "deep"}}}}
        doc = ConcreteTestDocument(
            name="nested.json",
            description=None,
            content=data,
        )
        parsed = doc.as_json()
        assert parsed["level1"]["level2"]["level3"]["value"] == "deep"

    def test_constructor_json_with_list(self):
        """Test constructing JSON document with list (not string list)."""
        data = [1, 2, 3, {"key": "value"}]
        doc = ConcreteTestDocument(
            name="array.json",
            description="JSON array",
            content=data,
        )
        parsed = doc.as_json()
        assert parsed == data


class TestConstructorWithDictForYAML:
    """Test constructor method with dict/BaseModel for YAML files."""

    def test_constructor_yaml_with_dict(self):
        """Test constructing YAML document with dictionary."""
        data = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {
                    "username": "admin",
                    "password": "secret",
                },
            },
            "cache": {
                "enabled": True,
                "ttl": 300,
            },
        }
        doc = ConcreteTestDocument(
            name="config.yaml",
            description="YAML config",
            content=data,
        )
        assert doc.name == "config.yaml"
        assert doc.is_text
        assert "yaml" in doc.mime_type

        # Verify content is valid YAML
        parsed = doc.as_yaml()
        assert parsed == data

        # Check YAML formatting
        text = doc.text
        assert "database:" in text
        assert "  host:" in text  # Has indentation

    def test_constructor_yml_extension(self):
        """Test constructing YAML document with .yml extension."""
        data = {"test": "value", "number": 42}
        doc = ConcreteTestDocument(
            name="config.yml",
            description=None,
            content=data,
        )
        assert doc.name == "config.yml"
        assert doc.as_yaml() == data

    def test_constructor_yaml_with_pydantic_model(self):
        """Test constructing YAML document with Pydantic model."""
        model = SampleModel(
            name="Charlie",
            age=35,
            tags=["yaml", "config"],
            config={"environment": "production"},
        )
        doc = ConcreteTestDocument(
            name="model.yaml",
            description="Model as YAML",
            content=model,
        )

        # Verify content
        parsed = doc.as_yaml()
        assert parsed["name"] == "Charlie"
        assert parsed["age"] == 35
        assert parsed["tags"] == ["yaml", "config"]

    def test_constructor_yaml_with_list(self):
        """Test constructing YAML document with list."""
        data = ["item1", "item2", "item3"]
        doc = ConcreteTestDocument(
            name="list.yaml",
            description="YAML list",
            content=data,
        )
        parsed = doc.as_yaml()
        assert parsed == data


class TestConstructorWithListOfPydanticModels:
    """Test constructor method with list[BaseModel] for JSON/YAML files."""

    def test_constructor_json_with_list_of_models(self):
        """Test constructing JSON document with list of Pydantic models."""
        models = [
            SampleModel(name="Alice", age=25, tags=["dev"]),
            SampleModel(name="Bob", age=30, tags=["qa", "test"]),
            SampleModel(name="Charlie", age=35, tags=[]),
        ]
        doc = ConcreteTestDocument(
            name="users.json",
            description="List of users",
            content=models,
        )
        assert doc.name == "users.json"
        assert doc.is_text
        assert "json" in doc.mime_type

        # Verify content is valid JSON array
        parsed = doc.as_json()
        assert isinstance(parsed, list)
        assert len(parsed) == 3
        assert parsed[0]["name"] == "Alice"
        assert parsed[1]["name"] == "Bob"
        assert parsed[2]["name"] == "Charlie"

        # Parse back to list of models
        parsed_models = doc.as_pydantic_model(list[SampleModel])
        assert len(parsed_models) == 3
        assert all(isinstance(m, SampleModel) for m in parsed_models)
        assert parsed_models[0].name == "Alice"
        assert parsed_models[1].age == 30
        assert parsed_models[2].tags == []

    def test_constructor_yaml_with_list_of_models(self):
        """Test constructing YAML document with list of Pydantic models."""
        models = [
            SampleModel(name="Server1", age=1, tags=["prod"], config={"port": 8080}),
            SampleModel(name="Server2", age=2, tags=["dev"], config={"port": 8081}),
        ]
        doc = ConcreteTestDocument(
            name="servers.yaml",
            description="Server configurations",
            content=models,
        )
        assert doc.name == "servers.yaml"
        assert doc.is_text
        assert "yaml" in doc.mime_type

        # Verify content is valid YAML array
        parsed = doc.as_yaml()
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "Server1"
        assert parsed[1]["config"]["port"] == 8081

        # Parse back to list of models
        parsed_models = doc.as_pydantic_model(list[SampleModel])
        assert len(parsed_models) == 2
        assert all(isinstance(m, SampleModel) for m in parsed_models)
        assert parsed_models[0].config["port"] == 8080

    def test_constructor_empty_list_of_models_json(self):
        """Test constructing JSON document with empty list of models."""
        models: list[SampleModel] = []
        doc = ConcreteTestDocument(
            name="empty.json",
            description=None,
            content=models,
        )
        parsed = doc.as_json()
        assert parsed == []

        # Parse back to empty list of models
        parsed_models = doc.as_pydantic_model(list[SampleModel])
        assert parsed_models == []
        assert isinstance(parsed_models, list)

    def test_constructor_list_of_models_without_extension_fails(self):
        """Test that list[BaseModel] without proper extension raises error."""
        models = [
            SampleModel(name="Test", age=20),
        ]
        with pytest.raises(ValueError) as exc_info:
            ConcreteTestDocument(
                name="test.txt",  # Wrong extension
                description=None,
                content=models,
            )
        assert "list[BaseModel] requires .json or .yaml extension" in str(exc_info.value)

    def test_as_pydantic_model_with_wrong_list_type_fails(self):
        """Test that as_pydantic_model fails with mismatched list data."""
        # Create a document with a dict (not a list)
        doc = ConcreteTestDocument(
            name="single.json",
            description=None,
            content={"name": "Alice", "age": 25, "tags": []},
        )

        # Try to parse as list[SampleModel] - should fail
        with pytest.raises(ValueError) as exc_info:
            doc.as_pydantic_model(list[SampleModel])
        assert "Expected list data" in str(exc_info.value)

    def test_mixed_pydantic_and_non_pydantic_list_json(self):
        """Test that mixed list types are handled correctly."""
        # Create a list with mixed types - the constructor should handle appropriately
        data = [{"name": "Alice", "age": 25, "tags": []}]  # Raw dict, not models
        doc = ConcreteTestDocument(
            name="data.json",
            description=None,
            content=data,
        )
        # Should still work as regular JSON
        parsed = doc.as_json()
        assert parsed == data

        # And can be parsed as models
        parsed_models = doc.as_pydantic_model(list[SampleModel])
        assert len(parsed_models) == 1
        assert parsed_models[0].name == "Alice"


class TestConstructorErrorCases:
    """Test error cases and edge cases for constructor."""

    def test_unsupported_content_type_without_extension(self):
        """Test that unsupported content types raise an error."""
        # Dict without .json or .yaml extension
        with pytest.raises(ValueError) as exc_info:
            ConcreteTestDocument(
                name="test.txt",
                description=None,
                content={"key": "value"},
            )
        assert "Unsupported content type" in str(exc_info.value)
        assert "dict" in str(exc_info.value)

    def test_unsupported_object_type(self):
        """Test that arbitrary objects raise an error."""

        class CustomClass:
            pass

        obj = CustomClass()
        with pytest.raises(ValueError) as exc_info:
            ConcreteTestDocument(
                name="test.txt",
                description=None,
                content=obj,
            )
        assert "Unsupported content type" in str(exc_info.value)

    def test_none_content_raises_error(self):
        """Test that None content raises an error."""
        with pytest.raises(ValueError) as exc_info:
            ConcreteTestDocument(
                name="test.txt",
                description=None,
                content=None,
            )
        assert "Unsupported content type" in str(exc_info.value)
        assert "NoneType" in str(exc_info.value)

    def test_mixed_type_list_for_markdown(self):
        """Test that mixed-type lists are rejected for markdown."""
        mixed_list = ["string", 123, "another string"]
        with pytest.raises(ValueError) as exc_info:
            ConcreteTestDocument(
                name="test.md",
                description=None,
                content=mixed_list,
            )
        assert "Unsupported content type" in str(exc_info.value)

    def test_numeric_content_for_json(self):
        """Test that numeric content works for JSON files."""
        # Single number should work for JSON
        doc = ConcreteTestDocument(
            name="number.json",
            description=None,
            content=42,
        )
        assert doc.as_json() == 42

        # Float should also work
        doc2 = ConcreteTestDocument(
            name="float.json",
            description=None,
            content=3.14159,
        )
        assert doc2.as_json() == 3.14159

    def test_boolean_content_for_json(self):
        """Test that boolean content works for JSON files."""
        doc = ConcreteTestDocument(
            name="bool.json",
            description=None,
            content=True,
        )
        assert doc.as_json() is True


class TestConstructorPriorityOrder:
    """Test the priority order of content type detection."""

    def test_bytes_takes_precedence(self):
        """Test that bytes content is used directly regardless of extension."""
        # Even with .json extension, bytes should be used directly
        content = b"not valid json"
        doc = ConcreteTestDocument(
            name="test.json",
            description=None,
            content=content,
        )
        assert doc.content == content

    def test_string_converts_to_bytes(self):
        """Test that string content is converted to bytes regardless of extension."""
        # Even with .json extension, string should be converted to bytes
        content = "plain text, not JSON"
        doc = ConcreteTestDocument(
            name="test.json",
            description=None,
            content=content,
        )
        assert doc.content == content.encode("utf-8")

    def test_list_markdown_before_yaml_json(self):
        """Test that list[str] with .md extension uses markdown list."""
        items = ["item1", "item2"]
        doc = ConcreteTestDocument(
            name="test.md",
            description=None,
            content=items,
        )
        # Should construct markdown list, not try to serialize as YAML/JSON
        assert doc.as_markdown_list() == items

    def test_yaml_extension_with_dict(self):
        """Test that .yaml extension with dict uses YAML serialization."""
        data = {"key": "value"}
        doc = ConcreteTestDocument(
            name="test.yaml",
            description=None,
            content=data,
        )
        assert doc.as_yaml() == data
        # YAML formatting should be applied
        text = doc.text
        assert "key: value" in text

    def test_json_extension_with_dict(self):
        """Test that .json extension with dict uses JSON serialization."""
        data = {"key": "value"}
        doc = ConcreteTestDocument(
            name="test.json",
            description=None,
            content=data,
        )
        assert doc.as_json() == data
        # JSON formatting should be applied
        text = doc.text
        assert '"key"' in text  # JSON uses double quotes


class TestConstructorIntegration:
    """Integration tests for constructor."""

    def test_roundtrip_json_document(self):
        """Test constructing and reading back a JSON document."""
        original_data = {
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ],
            "metadata": {
                "version": "1.0",
                "created": "2024-01-01",
            },
        }

        # Create document
        doc = ConcreteTestDocument(
            name="data.json",
            description="Test data",
            content=original_data,
        )

        # Read back
        parsed = doc.as_json()
        assert parsed == original_data

        # Parse as Pydantic model
        class User(BaseModel):
            id: int
            name: str

        class DataModel(BaseModel):
            users: list[User]
            metadata: dict[str, str]

        model = doc.as_pydantic_model(DataModel)
        assert len(model.users) == 2
        assert model.users[0].name == "Alice"
        assert model.metadata["version"] == "1.0"

    def test_roundtrip_yaml_document(self):
        """Test constructing and reading back a YAML document."""
        original_data = {
            "server": {
                "host": "0.0.0.0",
                "port": 8080,
                "ssl": True,
            },
            "features": ["auth", "logging", "monitoring"],
        }

        # Create document
        doc = ConcreteTestDocument(
            name="config.yaml",
            description="Server config",
            content=original_data,
        )

        # Read back
        parsed = doc.as_yaml()
        assert parsed == original_data

    def test_roundtrip_markdown_list(self):
        """Test constructing and reading back a markdown list document."""
        original_items = [
            "# Chapter 1\n\nIntroduction to the topic.",
            "# Chapter 2\n\nDeep dive into details.",
            "# Chapter 3\n\nConclusion and summary.",
        ]

        # Create document
        doc = ConcreteTestDocument(
            name="chapters.md",
            description="Book chapters",
            content=original_items,
        )

        # Read back
        parsed_items = doc.as_markdown_list()
        assert parsed_items == original_items
