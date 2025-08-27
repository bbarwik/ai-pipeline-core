import base64
import hashlib
import json
import re
from abc import ABC, abstractmethod
from base64 import b32encode
from enum import StrEnum
from functools import cached_property
from io import BytesIO
from typing import (
    Any,
    ClassVar,
    Literal,
    Self,
    TypeGuard,
    TypeVar,
    cast,
    final,
    get_args,
    get_origin,
    overload,
)

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from ruamel.yaml import YAML

from ai_pipeline_core.documents.utils import canonical_name_key
from ai_pipeline_core.exceptions import DocumentNameError, DocumentSizeError

from .mime_type import (
    detect_mime_type,
    is_image_mime_type,
    is_pdf_mime_type,
    is_text_mime_type,
    is_yaml_mime_type,
)

TModel = TypeVar("TModel", bound=BaseModel)
ContentInput = bytes | str | BaseModel | list[str] | Any


def _is_str_list(content: Any) -> TypeGuard[list[str]]:
    """Type guard to check if content is list[str]."""
    if not isinstance(content, list):
        return False
    return all(isinstance(item, str) for item in content)  # type: ignore[misc]


def _is_basemodel_list(content: Any) -> TypeGuard[list[BaseModel]]:
    """Type guard to check if content is list[BaseModel]."""
    if not isinstance(content, list):
        return False
    return all(isinstance(item, BaseModel) for item in content)  # type: ignore[misc]


class Document(BaseModel, ABC):
    """Abstract base class for all documents.

    Warning: Document subclasses should NOT start with 'Test' prefix as this
    causes conflicts with pytest test discovery. Classes with 'Test' prefix
    will be rejected at definition time.
    """

    MAX_CONTENT_SIZE: ClassVar[int] = 25 * 1024 * 1024  # 25MB default
    DESCRIPTION_EXTENSION: ClassVar[str] = ".description.md"
    MARKDOWN_LIST_SEPARATOR: ClassVar[str] = "\n\n---\n\n"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate subclass names to prevent pytest conflicts."""
        super().__init_subclass__(**kwargs)
        if cls.__name__.startswith("Test"):
            raise TypeError(
                f"Document subclass '{cls.__name__}' cannot start with 'Test' prefix. "
                "This causes conflicts with pytest test discovery. "
                "Please use a different name (e.g., 'SampleDocument', 'ExampleDocument')."
            )
        if hasattr(cls, "FILES"):
            files = getattr(cls, "FILES")
            if not issubclass(files, StrEnum):
                raise TypeError(
                    f"Document subclass '{cls.__name__}'.FILES must be an Enum of string values"
                )
        # Check that the Document's model_fields only contain the allowed fields
        # It prevents AI models from adding additional fields to documents
        allowed = {"name", "description", "content"}
        current = set(getattr(cls, "model_fields", {}).keys())
        extras = current - allowed
        if extras:
            raise TypeError(
                f"Document subclass '{cls.__name__}' cannot declare additional fields: "
                f"{', '.join(sorted(extras))}. Only {', '.join(sorted(allowed))} are allowed."
            )

    def __init__(self, **data: Any) -> None:
        """Prevent direct instantiation of abstract Document class."""
        if type(self) is Document:
            raise TypeError("Cannot instantiate abstract Document class directly")
        super().__init__(**data)

    name: str
    description: str | None = None
    content: bytes

    # Pydantic configuration
    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @abstractmethod
    def get_base_type(self) -> Literal["flow", "task", "temporary"]:
        """Get the type of the document - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")

    @final
    @property
    def base_type(self) -> Literal["flow", "task", "temporary"]:
        """Alias for document_type for backward compatibility"""
        return self.get_base_type()

    @final
    @property
    def is_flow(self) -> bool:
        """Check if document is a flow document"""
        return self.get_base_type() == "flow"

    @final
    @property
    def is_task(self) -> bool:
        """Check if document is a task document"""
        return self.get_base_type() == "task"

    @final
    @property
    def is_temporary(self) -> bool:
        """Check if document is a temporary document"""
        return self.get_base_type() == "temporary"

    @final
    @classmethod
    def get_expected_files(cls) -> list[str] | None:
        """
        Return the list of allowed file names for this document class, or None if unrestricted.
        """
        if not hasattr(cls, "FILES"):
            return None
        files = getattr(cls, "FILES")
        if not files:
            return None
        assert issubclass(files, StrEnum)
        try:
            values = [member.value for member in files]
        except TypeError:
            raise DocumentNameError(f"{cls.__name__}.FILES must be an Enum of string values")
        if len(values) == 0:
            return None
        return values

    @classmethod
    def validate_file_name(cls, name: str) -> None:
        """
        Optional file-name validation hook.

        Default behavior:
        - If `FILES` enum is defined on the subclass, ensure the **basename** of `name`
          equals one of the enum values (exact string match).
        - If `FILES` is None, do nothing.

        Override this method in subclasses for custom conventions (regex, prefixes, etc.).
        Raise DocumentNameError when invalid.
        """
        allowed = cls.get_expected_files()
        if not allowed:
            return

        if len(allowed) > 0 and name not in allowed:
            allowed_str = ", ".join(sorted(allowed))
            raise DocumentNameError(f"Invalid filename '{name}'. Allowed names: {allowed_str}")

    @field_validator("name")
    def validate_name(cls, v: str) -> str:
        """Validate document name matches expected patterns and is secure"""
        if v.endswith(cls.DESCRIPTION_EXTENSION):
            raise DocumentNameError(
                f"Document names cannot end with {cls.DESCRIPTION_EXTENSION}: {v}"
            )

        if ".." in v or "\\" in v or "/" in v:
            raise DocumentNameError(f"Invalid filename - contains path traversal characters: {v}")

        if not v or v.startswith(" ") or v.endswith(" "):
            raise DocumentNameError(f"Invalid filename format: {v}")

        cls.validate_file_name(v)

        return v

    @field_validator("content")
    def validate_content(cls, v: bytes) -> bytes:
        """Validate content size"""
        # Check content size limit
        max_size = getattr(cls, "MAX_CONTENT_SIZE", 100 * 1024 * 1024)
        if len(v) > max_size:
            raise DocumentSizeError(
                f"Document size ({len(v)} bytes) exceeds maximum allowed size ({max_size} bytes)"
            )

        return v

    @field_serializer("content")
    def serialize_content(self, v: bytes) -> str:
        """Serialize bytes content to string for JSON serialization"""
        try:
            return v.decode("utf-8")
        except UnicodeDecodeError:
            # Fall back to base64 for binary content
            return base64.b64encode(v).decode("ascii")

    @final
    @property
    def id(self) -> str:
        """Return the first 6 characters of the SHA256 hash of the content, encoded in base32"""
        return self.sha256[:6]

    @final
    @cached_property
    def sha256(self) -> str:
        """Full SHA256 hash of content, encoded in base32"""
        return b32encode(hashlib.sha256(self.content).digest()).decode("ascii").upper()

    @final
    @property
    def size(self) -> int:
        """Size of content in bytes"""
        return len(self.content)

    @cached_property
    def detected_mime_type(self) -> str:
        """Detect MIME type from content using python-magic"""
        return detect_mime_type(self.content, self.name)

    @property
    def mime_type(self) -> str:
        """Get MIME type - uses content detection with fallback to extension"""
        return self.detected_mime_type

    @property
    def is_text(self) -> bool:
        """Check if document is text based on MIME type"""
        return is_text_mime_type(self.mime_type)

    @property
    def is_pdf(self) -> bool:
        """Check if document is PDF"""
        return is_pdf_mime_type(self.mime_type)

    @property
    def is_image(self) -> bool:
        """Check if document is an image"""
        return is_image_mime_type(self.mime_type)

    @classmethod
    def canonical_name(cls) -> str:
        """Get the canonical name of the document"""
        return canonical_name_key(cls)

    def as_text(self) -> str:
        """Parse document as text"""
        if not self.is_text:
            raise ValueError(f"Document is not text: {self.name}")
        return self.content.decode("utf-8")

    def as_yaml(self) -> Any:
        """Parse document as YAML"""
        yaml = YAML()
        return yaml.load(self.as_text())  # type: ignore[no-untyped-call, no-any-return]

    def as_json(self) -> Any:
        """Parse document as JSON"""
        return json.loads(self.as_text())

    @overload
    def as_pydantic_model(self, model_type: type[TModel]) -> TModel: ...

    @overload
    def as_pydantic_model(self, model_type: type[list[TModel]]) -> list[TModel]: ...

    def as_pydantic_model(
        self, model_type: type[TModel] | type[list[TModel]]
    ) -> TModel | list[TModel]:
        """Parse document as a pydantic model and return the validated instance"""
        data = self.as_yaml() if is_yaml_mime_type(self.mime_type) else self.as_json()

        if get_origin(model_type) is list:
            if not isinstance(data, list):
                raise ValueError(f"Expected list data for {model_type}, got {type(data)}")
            item_type = get_args(model_type)[0]
            # Type guard for list case
            result_list = [item_type.model_validate(item) for item in data]  # type: ignore[attr-defined]
            return cast(list[TModel], result_list)

        # At this point model_type must be type[TModel], not type[list[TModel]]
        single_model = cast(type[TModel], model_type)
        return single_model.model_validate(data)

    def as_markdown_list(self) -> list[str]:
        """Parse document as a markdown list"""
        return self.as_text().split(self.MARKDOWN_LIST_SEPARATOR)

    @overload
    @classmethod
    def create(cls, name: str, content: ContentInput, /) -> Self: ...
    @overload
    @classmethod
    def create(cls, name: str, *, content: ContentInput) -> Self: ...
    @overload
    @classmethod
    def create(cls, name: str, description: str | None, content: ContentInput, /) -> Self: ...
    @overload
    @classmethod
    def create(cls, name: str, description: str | None, *, content: ContentInput) -> Self: ...

    @classmethod
    def create(
        cls,
        name: str,
        description: ContentInput = None,
        content: ContentInput = None,
    ) -> Self:
        """Create a document from a name, description, and content"""
        if content is None:
            if description is None:
                raise ValueError(f"Unsupported content type: {type(content)} for {name}")
            content = description
            description = None
        else:
            assert description is None or isinstance(description, str)

        is_yaml_extension = name.endswith(".yaml") or name.endswith(".yml")
        is_json_extension = name.endswith(".json")
        is_markdown_extension = name.endswith(".md")

        if isinstance(content, bytes):
            pass
        elif isinstance(content, str):
            content = content.encode("utf-8")
        elif _is_str_list(content) and is_markdown_extension:
            return cls.create_as_markdown_list(name, description, content)
        elif _is_basemodel_list(content):
            # Handle list[BaseModel] for JSON/YAML files
            if is_yaml_extension:
                return cls.create_as_yaml(name, description, content)
            elif is_json_extension:
                return cls.create_as_json(name, description, content)
            else:
                raise ValueError(f"list[BaseModel] requires .json or .yaml extension, got {name}")
        elif is_yaml_extension:
            return cls.create_as_yaml(name, description, content)
        elif is_json_extension:
            return cls.create_as_json(name, description, content)
        else:
            raise ValueError(f"Unsupported content type: {type(content)} for {name}")

        return cls(name=name, description=description, content=content)

    @final
    @classmethod
    def create_as_markdown_list(cls, name: str, description: str | None, items: list[str]) -> Self:
        """Create a document from a name, description, and list of strings"""
        # remove other list separators (lines that are only the separator + whitespace)
        separator = Document.MARKDOWN_LIST_SEPARATOR.strip()
        pattern = re.compile(rf"^[ \t]*{re.escape(separator)}[ \t]*(?:\r?\n|$)", flags=re.MULTILINE)
        # Normalize CRLF/CR to LF before cleaning to ensure consistent behavior
        normalized_items = [re.sub(r"\r\n?", "\n", item) for item in items]
        cleaned_items = [pattern.sub("", item) for item in normalized_items]
        content = Document.MARKDOWN_LIST_SEPARATOR.join(cleaned_items)
        return cls.create(name, description, content)

    @final
    @classmethod
    def create_as_json(cls, name: str, description: str | None, data: Any) -> Self:
        """Create a document from a name, description, and JSON data"""
        assert name.endswith(".json"), f"Document name must end with .json: {name}"
        if isinstance(data, BaseModel):
            data = data.model_dump(mode="json")
        elif _is_basemodel_list(data):
            # Now data is typed as list[BaseModel]
            data = [item.model_dump(mode="json") for item in data]
        content = json.dumps(data, indent=2).encode("utf-8")
        return cls.create(name, description, content)

    @final
    @classmethod
    def create_as_yaml(cls, name: str, description: str | None, data: Any) -> Self:
        """Create a document from a name, description, and YAML data"""
        assert name.endswith(".yaml") or name.endswith(".yml"), (
            f"Document name must end with .yaml or .yml: {name}"
        )
        if isinstance(data, BaseModel):
            data = data.model_dump(mode="json")
        elif _is_basemodel_list(data):
            # Now data is typed as list[BaseModel]
            data = [item.model_dump(mode="json") for item in data]
        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)  # type: ignore[no-untyped-call]

        stream = BytesIO()
        yaml.dump(data, stream)  # type: ignore[no-untyped-call]
        content = stream.getvalue()
        return cls.create(name, description, content)

    @final
    def serialize_model(self) -> dict[str, Any]:
        """Serialize document to a dictionary with proper encoding."""
        result = {
            "name": self.name,
            "description": self.description,
            "base_type": self.get_base_type(),
            "size": self.size,
            "id": self.id,
            "sha256": self.sha256,
            "mime_type": self.mime_type,
        }

        # Try to encode content as UTF-8, fall back to base64
        if self.is_text or self.mime_type.startswith("text/"):
            try:
                result["content"] = self.content.decode("utf-8")
                result["content_encoding"] = "utf-8"
            except UnicodeDecodeError:
                # For text files with encoding issues, use UTF-8 with replacement
                result["content"] = self.content.decode("utf-8", errors="replace")
                result["content_encoding"] = "utf-8"
        else:
            # Binary content - use base64
            result["content"] = base64.b64encode(self.content).decode("ascii")
            result["content_encoding"] = "base64"

        return result

    @final
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize document from dictionary."""
        # Extract content and encoding
        content_raw = data.get("content", "")
        content_encoding = data.get("content_encoding", "utf-8")

        # Decode content based on encoding
        content: bytes
        if content_encoding == "base64":
            assert isinstance(content_raw, str), "base64 content must be string"
            content = base64.b64decode(content_raw)
        elif isinstance(content_raw, str):
            # Default to UTF-8
            content = content_raw.encode("utf-8")
        elif isinstance(content_raw, bytes):
            content = content_raw
        else:
            raise ValueError(f"Invalid content type: {type(content_raw)}")

        # Create document with the required fields
        return cls(
            name=data["name"],
            content=content,
            description=data.get("description"),
        )
