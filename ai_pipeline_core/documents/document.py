import base64
import hashlib
import json
from abc import ABC, abstractmethod
from base64 import b32encode
from enum import StrEnum
from functools import cached_property
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from ruamel.yaml import YAML

from ai_pipeline_core.exceptions import DocumentNameError, DocumentSizeError

from .mime_type import (
    detect_mime_type,
    is_image_mime_type,
    is_pdf_mime_type,
    is_text_mime_type,
)


class Document(BaseModel, ABC):
    """Abstract base class for all documents"""

    MAX_CONTENT_SIZE: ClassVar[int] = 10 * 1024 * 1024  # 10MB default
    DESCRIPTION_EXTENSION: ClassVar[str] = ".description.md"

    # Optional enum of allowed file names. Subclasses may set this.
    # This is used to validate the document name.
    FILES: ClassVar[type[StrEnum] | None] = None

    name: str
    description: str | None = None
    content: bytes

    # Pydantic configuration
    model_config = ConfigDict(
        frozen=True,  # Make documents immutable
        arbitrary_types_allowed=True,
    )

    @abstractmethod
    def get_base_type(self) -> Literal["flow", "task"]:
        """Get the type of the document - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def base_type(self) -> Literal["flow", "task"]:
        """Alias for document_type for backward compatibility"""
        return self.get_base_type()

    @property
    def is_flow(self) -> bool:
        """Check if document is a flow document"""
        return self.get_base_type() == "flow"

    @property
    def is_task(self) -> bool:
        """Check if document is a task document"""
        return self.get_base_type() == "task"

    @classmethod
    def get_expected_files(cls) -> list[str] | None:
        """
        Return the list of allowed file names for this document class, or None if unrestricted.
        """
        if cls.FILES is None:
            return None
        try:
            values = [member.value for member in cls.FILES]
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
        if cls.FILES is None:
            return

        try:
            allowed = {str(member.value) for member in cls.FILES}  # type: ignore[arg-type]
        except TypeError:
            raise DocumentNameError(f"{cls.__name__}.FILES must be an Enum of string values")

        if name not in allowed:
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

    @property
    def id(self) -> str:
        """Return the first 6 characters of the SHA256 hash of the content, encoded in base32"""
        return self.sha256[:6]

    @cached_property
    def sha256(self) -> str:
        """Full SHA256 hash of content, encoded in base32"""
        return b32encode(hashlib.sha256(self.content).digest()).decode("ascii").upper()

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

    @property
    def should_be_cached(self) -> bool:
        """Check if document should be cached"""
        return False

    def as_text(self) -> str:
        """Parse document as text"""
        if not self.is_text:
            raise ValueError(f"Document is not text: {self.name}")
        return self.content.decode("utf-8")

    def as_yaml(self) -> Any:
        """Parse document as YAML"""
        if not self.is_text:
            raise ValueError(f"Document is not text: {self.name}")
        return YAML().load(self.content.decode("utf-8"))  # type: ignore

    def as_json(self) -> Any:
        """Parse document as JSON"""
        if not self.is_text:
            raise ValueError(f"Document is not text: {self.name}")
        return json.loads(self.content.decode("utf-8"))

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize document from dictionary."""
        # Extract content and encoding
        content_str = data.get("content", "")
        content_encoding = data.get("content_encoding", "utf-8")

        # Decode content based on encoding
        if content_encoding == "base64":
            content = base64.b64decode(content_str)
        else:
            # Default to UTF-8
            content = content_str.encode("utf-8")

        # Create document with the required fields
        return cls(
            name=data["name"],
            content=content,
            description=data.get("description"),
        )
