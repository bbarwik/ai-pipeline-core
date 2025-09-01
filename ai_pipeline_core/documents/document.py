"""Document abstraction layer for AI pipeline flows.

@public

This module provides the core document abstraction for working with various types of data
in AI pipelines. Documents are immutable Pydantic models that wrap binary content with metadata.
"""

from __future__ import annotations

import base64
import hashlib
import json
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
    TypeVar,
    cast,
    final,
    get_args,
    get_origin,
    overload,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
)
from ruamel.yaml import YAML

from ai_pipeline_core.documents.utils import canonical_name_key, is_document_sha256
from ai_pipeline_core.exceptions import DocumentNameError, DocumentSizeError

from .mime_type import (
    detect_mime_type,
    is_image_mime_type,
    is_pdf_mime_type,
    is_text_mime_type,
    is_yaml_mime_type,
)

TModel = TypeVar("TModel", bound=BaseModel)


class Document(BaseModel, ABC):
    r"""Abstract base class for all documents in the AI Pipeline Core system.

    @public

    Document is the fundamental data abstraction for all content flowing through
    pipelines. It provides automatic encoding, MIME type detection, serialization,
    and validation. All documents must be subclassed from FlowDocument or TaskDocument
    based on their persistence requirements. TemporaryDocument is a special concrete
    class that can be instantiated directly (not abstract).

    VALIDATION IS AUTOMATIC - Do not add manual validation!
        Size validation, name validation, and MIME type detection are built-in.
        The framework handles all standard validations internally.

        # WRONG - These checks already happen automatically:
        if document.size > document.MAX_CONTENT_SIZE:
            raise DocumentSizeError(...)  # NO! Already handled
        document.validate_file_name(document.name)  # NO! Automatic

    Best Practices:
        - Use create() classmethod for automatic type conversion (90% of cases)
        - Omit description parameter unless truly needed for metadata
        - When using LLM functions, pass AIMessages or str. Wrap any Document values
          in AIMessages([...]). Do not call .text yourself

    Standard Usage:
        >>> # CORRECT - minimal parameters
        >>> doc = MyDocument.create(name="data.json", content={"key": "value"})

        >>> # AVOID - unnecessary description
        >>> doc = MyDocument.create(
        ...     name="data.json",
        ...     content={"key": "value"},
        ...     description="This is data"  # Usually not needed!
        ... )

    Key features:
    - Immutable by default (frozen Pydantic model)
    - Automatic MIME type detection
    - Content size validation
    - SHA256 hashing for deduplication
    - Support for text, JSON, YAML, PDF, and image formats
    - Conversion utilities between different formats
    - Source provenance tracking via sources field

    Class Variables:
        MAX_CONTENT_SIZE: Maximum allowed content size in bytes (default 25MB)

    Attributes:
        name: Document filename (validated for security)
        description: Optional human-readable description
        content: Raw document content as bytes
        sources: List of source references tracking document provenance

    Creating Documents:
        **Use the `create` classmethod** for most use cases. It accepts various
        content types (str, dict, list, BaseModel) and converts them automatically.
        Only use __init__ directly when you already have bytes content.

        >>> # RECOMMENDED: Use create for automatic conversion
        >>> doc = MyDocument.create(name="data.json", content={"key": "value"})
        >>>
        >>> # Direct constructor: Only for bytes
        >>> doc = MyDocument(name="data.bin", content=b"\x00\x01\x02")

    Warning:
        - Document subclasses should NOT start with 'Test' prefix (pytest conflict)
        - Cannot instantiate Document directly - must subclass FlowDocument or TaskDocument
        - Cannot add custom fields - only name, description, content, sources are allowed
        - Document is an abstract class and cannot be instantiated directly

    Metadata Attachment Patterns:
        Since custom fields are not allowed, use these patterns for metadata:
        1. Use the 'description' field for human-readable metadata
        2. Embed metadata in content (e.g., JSON with data + metadata fields)
        3. Create a separate MetadataDocument type to accompany data documents
        4. Use document naming conventions (e.g., "data_v2_2024.json")
        5. Store metadata in flow_options or pass through TraceInfo

    Example:
        >>> from enum import StrEnum
        >>>
        >>> # Simple document:
        >>> class MyDocument(FlowDocument):
        ...     pass
        >>>
        >>> # Document with file restrictions:
        >>> class ConfigDocument(FlowDocument):
        ...     class FILES(StrEnum):
        ...         CONFIG = "config.yaml"
        ...         SETTINGS = "settings.json"
        >>>
        >>> # RECOMMENDED: Use create for automatic conversion
        >>> doc = MyDocument.create(name="data.json", content={"key": "value"})
        >>> print(doc.is_text)  # True
        >>> data = doc.as_json()  # {'key': 'value'}
        >>>
        >>> # Track document provenance with sources
        >>> source_doc = MyDocument.create(name="input.txt", content="raw data")
        >>> processed = MyDocument.create(
        ...     name="output.txt",
        ...     content="processed data",
        ...     sources=[source_doc.sha256]  # Reference source document
        ... )
        >>> processed.has_source(source_doc)  # True
    """

    MAX_CONTENT_SIZE: ClassVar[int] = 25 * 1024 * 1024
    """Maximum allowed content size in bytes (default 25MB).

    @public
    """

    DESCRIPTION_EXTENSION: ClassVar[str] = ".description.md"
    """File extension for description files."""

    MARKDOWN_LIST_SEPARATOR: ClassVar[str] = "\n\n-----------------\n\n"
    """Separator for markdown list items."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate subclass configuration at definition time.

        Performs several validation checks when a Document subclass is defined:
        1. Prevents class names starting with 'Test' (pytest conflict)
        2. Validates FILES enum if present (must be StrEnum)
        3. Prevents adding custom fields beyond name, description, content

        Args:
            **kwargs: Additional keyword arguments passed to parent __init_subclass__.

        Raises:
            TypeError: If subclass violates naming rules, FILES enum requirements,
                      or attempts to add extra fields.

        Note:
            This validation happens at class definition time, not instantiation,
            providing early error detection during development.
        """
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
        allowed = {"name", "description", "content", "sources"}
        current = set(getattr(cls, "model_fields", {}).keys())
        extras = current - allowed
        if extras:
            raise TypeError(
                f"Document subclass '{cls.__name__}' cannot declare additional fields: "
                f"{', '.join(sorted(extras))}. Only {', '.join(sorted(allowed))} are allowed."
            )

    @overload
    @classmethod
    def create(
        cls,
        *,
        name: str,
        content: bytes,
        description: str | None = None,
        sources: list[str] = [],
    ) -> Self: ...

    @overload
    @classmethod
    def create(
        cls,
        *,
        name: str,
        content: str,
        description: str | None = None,
        sources: list[str] = [],
    ) -> Self: ...

    @overload
    @classmethod
    def create(
        cls,
        *,
        name: str,
        content: dict[str, Any],
        description: str | None = None,
        sources: list[str] = [],
    ) -> Self: ...

    @overload
    @classmethod
    def create(
        cls,
        *,
        name: str,
        content: list[Any],
        description: str | None = None,
        sources: list[str] = [],
    ) -> Self: ...

    @overload
    @classmethod
    def create(
        cls,
        *,
        name: str,
        content: BaseModel,
        description: str | None = None,
        sources: list[str] = [],
    ) -> Self: ...

    @classmethod
    def create(
        cls,
        *,
        name: str,
        content: str | bytes | dict[str, Any] | list[Any] | BaseModel,
        description: str | None = None,
        sources: list[str] = [],
    ) -> Self:
        r"""Create a Document with automatic content type conversion (recommended).

        @public

        This is the **recommended way to create documents**. It accepts various
        content types and automatically converts them to bytes based on the file
        extension. Use the `parse` method to reverse this conversion.

        Best Practice (90% of cases):
            Only provide name and content. The description parameter is RARELY needed.

        Args:
            name: Document filename (required, keyword-only).
                  Extension determines serialization:
                  - .json → JSON serialization
                  - .yaml/.yml → YAML serialization
                  - .md → Markdown list joining (for list[str])
                  - Others → UTF-8 encoding (for str)
            content: Document content in various formats (required, keyword-only):
                - bytes: Used directly without conversion
                - str: Encoded to UTF-8 bytes
                - dict[str, Any]: Serialized to JSON (.json) or YAML (.yaml/.yml)
                - list[str]: Joined with separator for .md (validates no items
                            contain separator), else JSON/YAML
                - list[BaseModel]: Serialized to JSON or YAML based on extension
                - BaseModel: Serialized to JSON or YAML based on extension
            description: Optional description - USUALLY OMIT THIS (defaults to None).
                        Only use when meaningful metadata helps downstream processing
            sources: Optional list of source strings (document SHA256 hashes or references).
                    Used to track what sources contributed to creating this document.
                    Can contain document SHA256 hashes (for referencing other documents)
                    or arbitrary reference strings (URLs, file paths, descriptions).
                    Defaults to empty list

        Returns:
            New Document instance with content converted to bytes

        Raises:
            ValueError: If content type is not supported for the file extension,
                       or if markdown list items contain the separator
            DocumentNameError: If filename violates validation rules
            DocumentSizeError: If content exceeds MAX_CONTENT_SIZE

        Note:
            All conversions are reversible using the `parse` method.
            For example: MyDocument.create(name="data.json", content={"key": "value"}).parse(dict)
            returns the original dictionary {"key": "value"}.

        Example:
            >>> # CORRECT - no description needed (90% of cases)
            >>> doc = MyDocument.create(name="test.txt", content="Hello World")
            >>> doc.content  # b'Hello World'
            >>> doc.parse(str)  # "Hello World"

            >>> # CORRECT - Dictionary to JSON, no description
            >>> doc = MyDocument.create(name="config.json", content={"key": "value"})
            >>> doc.content  # b'{"key": "value", ...}'
            >>> doc.parse(dict)  # {"key": "value"}

            >>> # AVOID unless description adds real value
            >>> doc = MyDocument.create(
            ...     name="config.json",
            ...     content={"key": "value"},
            ...     description="Config file"  # Usually redundant!
            ... )

            >>> # Pydantic model to YAML
            >>> from pydantic import BaseModel
            >>> class Config(BaseModel):
            ...     host: str
            ...     port: int
            >>> config = Config(host="localhost", port=8080)
            >>> doc = MyDocument.create(name="config.yaml", content=config)
            >>> doc.parse(Config)  # Returns Config instance

            >>> # List to Markdown
            >>> items = ["Section 1", "Section 2"]
            >>> doc = MyDocument.create(name="sections.md", content=items)
            >>> doc.parse(list)  # ["Section 1", "Section 2"]

            >>> # Document with sources for provenance tracking
            >>> source_doc = MyDocument.create(name="source.txt", content="original")
            >>> derived = MyDocument.create(
            ...     name="result.txt",
            ...     content="processed",
            ...     sources=[source_doc.sha256, "https://api.example.com/data"]
            ... )
            >>> derived.get_source_documents()  # [source_doc.sha256]
            >>> derived.get_source_references()  # ["https://api.example.com/data"]
        """
        # Use model_validate to leverage the existing validator logic
        temp = cls.model_validate({
            "name": name,
            "content": content,
            "description": description,
            "sources": sources,
        })
        # Now construct with type-checker-friendly call (bytes only)
        return cls(
            name=temp.name,
            content=temp.content,
            description=temp.description,
            sources=temp.sources,
        )

    def __init__(
        self,
        *,
        name: str,
        content: bytes,
        description: str | None = None,
        sources: list[str] = [],
    ) -> None:
        """Initialize a Document instance with raw bytes content.

        @public

        Important:
            **Most users should use the `create` classmethod instead of __init__.**
            The create method provides automatic content conversion for various types
            (str, dict, list, Pydantic models) while __init__ only accepts bytes.

        This constructor accepts only bytes content for type safety. It prevents
        direct instantiation of the abstract Document class.

        Args:
            name: Document filename (required, keyword-only)
            content: Document content as raw bytes (required, keyword-only)
            description: Optional human-readable description (keyword-only)
            sources: Optional list of source strings for provenance tracking.
                    Can contain document SHA256 hashes (for referencing other documents)
                    or arbitrary reference strings (URLs, file paths, descriptions).
                    Defaults to empty list

        Raises:
            TypeError: If attempting to instantiate Document directly.

        Example:
            >>> # Direct constructor - only for bytes content:
            >>> doc = MyDocument(name="test.txt", content=b"Hello World")
            >>> doc.content  # b'Hello World'

            >>> # RECOMMENDED: Use create for automatic conversion:
            >>> doc = MyDocument.create(name="text.txt", content="Hello World")
            >>> doc = MyDocument.create(name="data.json", content={"key": "value"})
            >>> doc = MyDocument.create(name="config.yaml", content=my_model)
            >>> doc = MyDocument.create(name="items.md", content=["item1", "item2"])

        See Also:
            create: Recommended factory method with automatic type conversion
            parse: Method to reverse the conversion done by create
        """
        if type(self) is Document:
            raise TypeError("Cannot instantiate abstract Document class directly")

        super().__init__(name=name, content=content, description=description, sources=sources)

    name: str
    description: str | None = None
    content: bytes  # Note: constructor accepts str | bytes, but field stores bytes only
    sources: list[str] = Field(
        default_factory=list,
        description="List of source references for tracking document provenance. "
        "Can contain document SHA256 hashes (for referencing other documents) "
        "or arbitrary reference strings (URLs, file paths, descriptions)",
    )

    # Pydantic configuration
    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @abstractmethod
    def get_base_type(self) -> Literal["flow", "task", "temporary"]:
        """Get the base type of the document.

        Abstract method that must be implemented by all Document subclasses
        to indicate their persistence behavior.

        Returns:
            One of "flow" (persisted across flow runs), "task" (temporary
            within task execution), or "temporary" (never persisted).

        Note:
            This method determines document persistence and lifecycle.
            FlowDocument returns "flow", TaskDocument returns "task",
            TemporaryDocument returns "temporary".
        """
        raise NotImplementedError("Subclasses must implement this method")

    @final
    @property
    def base_type(self) -> Literal["flow", "task", "temporary"]:
        """Get the document's base type.

        Property alias for get_base_type() providing a cleaner API.
        This property cannot be overridden by subclasses.

        Returns:
            The document's base type: "flow", "task", or "temporary".
        """
        return self.get_base_type()

    @final
    @property
    def is_flow(self) -> bool:
        """Check if this is a flow document.

        Flow documents persist across Prefect flow runs and are saved
        to the file system between pipeline steps.

        Returns:
            True if this is a FlowDocument subclass, False otherwise.
        """
        return self.get_base_type() == "flow"

    @final
    @property
    def is_task(self) -> bool:
        """Check if this is a task document.

        Task documents are temporary within Prefect task execution
        and are not persisted between pipeline steps.

        Returns:
            True if this is a TaskDocument subclass, False otherwise.
        """
        return self.get_base_type() == "task"

    @final
    @property
    def is_temporary(self) -> bool:
        """Check if this is a temporary document.

        Temporary documents are never persisted and exist only
        during execution.

        Returns:
            True if this is a TemporaryDocument, False otherwise.
        """
        return self.get_base_type() == "temporary"

    @final
    @classmethod
    def get_expected_files(cls) -> list[str] | None:
        """Get the list of allowed file names for this document class.

        If the document class defines a FILES enum, returns the list of
        valid file names. Used to restrict documents to specific files.

        Returns:
            List of allowed file names if FILES enum is defined,
            None if unrestricted.

        Raises:
            DocumentNameError: If FILES is defined but not a valid StrEnum.

        Example:
            >>> class ConfigDocument(FlowDocument):
            ...     class FILES(StrEnum):
            ...         CONFIG = "config.yaml"
            ...         SETTINGS = "settings.json"
            >>> ConfigDocument.get_expected_files()
            ['config.yaml', 'settings.json']
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
        """Validate that a file name matches allowed patterns.

        @public

        DO NOT OVERRIDE this method if you define a FILES enum!
        The validation is automatic when FILES enum is present.

        # CORRECT - FILES enum provides automatic validation:
        class MyDocument(FlowDocument):
            class FILES(StrEnum):
                CONFIG = "config.yaml"  # Validation happens automatically!

        # WRONG - Unnecessary override:
        class MyDocument(FlowDocument):
            class FILES(StrEnum):
                CONFIG = "config.yaml"

            def validate_file_name(cls, name):  # DON'T DO THIS!
                pass  # Validation already happens via FILES enum

        Only override for custom validation logic BEYOND FILES enum constraints.

        Args:
            name: The file name to validate.

        Raises:
            DocumentNameError: If the name doesn't match allowed patterns.

        Note:
            - If FILES enum is defined, name must exactly match one of the values
            - If FILES is not defined, any name is allowed
            - Override in subclasses ONLY for custom regex patterns or logic
        """
        allowed = cls.get_expected_files()
        if not allowed:
            return

        if len(allowed) > 0 and name not in allowed:
            allowed_str = ", ".join(sorted(allowed))
            raise DocumentNameError(f"Invalid filename '{name}'. Allowed names: {allowed_str}")

    @field_validator("name")
    def validate_name(cls, v: str) -> str:
        r"""Pydantic validator for the document name field.

        Ensures the document name is secure and follows conventions:
        - No path traversal characters (.., \\, /)
        - Cannot end with .description.md
        - No leading/trailing whitespace
        - Must match FILES enum if defined

        Performance:
            Validation is O(n) where n is the length of the name.
            FILES enum check is O(m) where m is the number of allowed files

        Args:
            v: The name value to validate.

        Returns:
            The validated name.

        Raises:
            DocumentNameError: If the name violates any validation rules.

        Note:
            This is called automatically by Pydantic during model construction.
        """
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

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v: Any, info: ValidationInfo) -> bytes:
        """Pydantic validator that converts various content types to bytes.

        This validator is called automatically during model construction and
        handles the intelligent type conversion that powers the `create` method.
        It determines the appropriate serialization based on file extension.

        Conversion Strategy:
            1. bytes → Passthrough (no conversion)
            2. str → UTF-8 encoding
            3. dict/BaseModel + .json → JSON serialization (indented)
            4. dict/BaseModel + .yaml/.yml → YAML serialization
            5. list[str] + .md → Join with markdown separator (validates no items contain separator)
            6. list[Any] + .json/.yaml → JSON/YAML array
            7. int/float/bool + .json → JSON primitive

        Args:
            v: Content to validate (any supported type)
            info: Validation context containing other field values

        Returns:
            Content converted to bytes

        Raises:
            DocumentSizeError: If content exceeds MAX_CONTENT_SIZE
            ValueError: If content type unsupported for file extension

        Note:
            This validator enables create() to accept multiple types while
            ensuring __init__ only receives bytes for type safety.
        """
        # Get the name from validation context if available
        name = ""
        if hasattr(info, "data") and "name" in info.data:
            name = info.data["name"]
        name_lower = name.lower()

        # Convert based on content type
        if isinstance(v, bytes):
            pass  # Already bytes
        elif isinstance(v, str):
            v = v.encode("utf-8")
        elif isinstance(v, dict):
            # Serialize dict based on extension
            if name_lower.endswith((".yaml", ".yml")):
                # Use YAML format for YAML files
                yaml = YAML()
                stream = BytesIO()
                yaml.dump(v, stream)
                v = stream.getvalue()
            elif name_lower.endswith(".json"):
                # Use JSON for JSON files
                v = json.dumps(v, indent=2).encode("utf-8")
            else:
                # Dict not supported for other file types
                raise ValueError(f"Unsupported content type: {type(v)} for file {name}")
        elif isinstance(v, list):
            # Handle lists based on file extension
            if name_lower.endswith(".md"):
                # For markdown files, join with separator
                if all(isinstance(item, str) for item in v):
                    # Check that no string contains the separator
                    for item in v:
                        if cls.MARKDOWN_LIST_SEPARATOR in item:
                            raise ValueError(
                                f"Markdown list item cannot contain the separator "
                                f"'{cls.MARKDOWN_LIST_SEPARATOR}' as it will mess up formatting"
                            )
                    v = cls.MARKDOWN_LIST_SEPARATOR.join(v).encode("utf-8")
                else:
                    raise ValueError(
                        f"Unsupported content type: mixed-type list for markdown file {name}"
                    )
            elif name_lower.endswith((".yaml", ".yml")):
                # Check if it's a list of Pydantic models
                if v and isinstance(v[0], BaseModel):
                    # Convert models to dicts first
                    v = [item.model_dump(mode="json") for item in v]
                # Use YAML format for YAML files
                yaml = YAML()
                stream = BytesIO()
                yaml.dump(v, stream)
                v = stream.getvalue()
            elif name_lower.endswith(".json"):
                # Check if it's a list of Pydantic models
                if v and isinstance(v[0], BaseModel):
                    # Convert models to dicts first
                    v = [item.model_dump(mode="json") for item in v]
                # For JSON files, serialize as JSON
                v = json.dumps(v, indent=2).encode("utf-8")
            else:
                # Check if it's a list of BaseModel
                if v and isinstance(v[0], BaseModel):
                    raise ValueError("list[BaseModel] requires .json or .yaml extension")
                # List content not supported for other file types
                raise ValueError(f"Unsupported content type: {type(v)} for file {name}")
        elif isinstance(v, BaseModel):
            # Serialize Pydantic models
            if name_lower.endswith((".yaml", ".yml")):
                yaml = YAML()
                stream = BytesIO()
                yaml.dump(v.model_dump(mode="json"), stream)
                v = stream.getvalue()
            else:
                v = json.dumps(v.model_dump(mode="json"), indent=2).encode("utf-8")
        elif isinstance(v, (int, float, bool)):
            # Numbers and booleans: JSON-serialize for .json, string for others
            if name_lower.endswith(".json"):
                v = json.dumps(v).encode("utf-8")
            elif name_lower.endswith((".yaml", ".yml")):
                v = str(v).encode("utf-8")
            elif name_lower.endswith(".txt"):
                v = str(v).encode("utf-8")
            else:
                # For other extensions, convert to string
                v = str(v).encode("utf-8")
        elif v is None:
            # Handle None - only supported for JSON/YAML
            if name_lower.endswith((".json", ".yaml", ".yml")):
                if name_lower.endswith((".yaml", ".yml")):
                    v = b"null\n"
                else:
                    v = b"null"
            else:
                raise ValueError(f"Unsupported content type: {type(None)} for file {name}")
        else:
            # Try to see if it has model_dump (duck typing for Pydantic-like)
            if hasattr(v, "model_dump"):
                if name_lower.endswith((".yaml", ".yml")):
                    yaml = YAML()
                    stream = BytesIO()
                    yaml.dump(v.model_dump(mode="json"), stream)  # type: ignore[attr-defined]
                    v = stream.getvalue()
                else:
                    v = json.dumps(v.model_dump(mode="json"), indent=2).encode("utf-8")  # type: ignore[attr-defined]
            else:
                # List non-.json files should raise error
                if name_lower.endswith(".txt") and isinstance(v, list):
                    raise ValueError("List content not supported for text files")
                raise ValueError(f"Unsupported content type: {type(v)}")

        # Check content size limit
        max_size = getattr(cls, "MAX_CONTENT_SIZE", 100 * 1024 * 1024)
        if len(v) > max_size:
            raise DocumentSizeError(
                f"Document size ({len(v)} bytes) exceeds maximum allowed size ({max_size} bytes)"
            )

        return v

    @field_serializer("content")
    def serialize_content(self, v: bytes) -> str:
        """Pydantic serializer for content field.

        Converts bytes content to string for JSON serialization.
        Attempts UTF-8 decoding first, falls back to base64 encoding
        for binary content.

        Args:
            v: The content bytes to serialize.

        Returns:
            UTF-8 decoded string for text content,
            base64-encoded string for binary content.

        Note:
            This is called automatically by Pydantic during
            model serialization to JSON.
        """
        try:
            return v.decode("utf-8")
        except UnicodeDecodeError:
            # Fall back to base64 for binary content
            return base64.b64encode(v).decode("ascii")

    @final
    @property
    def id(self) -> str:
        """Get a short unique identifier for the document.

        @public

        This ID is crucial for LLM interactions. When documents are provided to
        LLMs via generate() or generate_structured(), their IDs are included,
        allowing the LLM to reference documents in prompts by either name or ID.
        The ID is content-based (derived from SHA256 hash of content only),
        so the same content always produces the same ID. Changing the name or
        description does NOT change the ID.

        Returns:
            6-character base32-encoded string (uppercase, e.g., "A7B2C9").
            This is the first 6 chars of the full base32 SHA256, NOT hex.

        Collision Rate:
            With base32 encoding (5 bits per char), 6 chars = 30 bits.
            Expect collisions after ~32K documents (birthday paradox).
            For higher uniqueness requirements, use the full sha256 property.

        Note:
            While shorter than full SHA256, this provides
            reasonable uniqueness for most use cases.
        """
        return self.sha256[:6]

    @final
    @cached_property
    def sha256(self) -> str:
        """Get the full SHA256 hash of the document content.

        @public

        Computes and caches the SHA256 hash of the content,
        encoded in base32 (uppercase). Used for content
        deduplication and integrity verification.

        Returns:
            Full SHA256 hash as base32-encoded uppercase string.

        Why Base32 Instead of Hex:
            - Base32 is case-insensitive, avoiding issues with different file systems
              and AI interactions where casing might be inconsistent
            - More compact than hex (52 chars vs 64 chars for SHA-256)
            - Contains more information per character than hex (5 bits vs 4 bits)
            - Safe for URLs without encoding
            - Compatible with case-insensitive file systems
            - Avoids confusion in AI interactions where models might change casing
            - Not base64 because we want consistent uppercase for all uses

        Note:
            This is computed once and cached for performance.
            The hash is deterministic based on content only.
        """
        return b32encode(hashlib.sha256(self.content).digest()).decode("ascii").upper().rstrip("=")

    @final
    @property
    def size(self) -> int:
        """Get the size of the document content.

        @public

        Returns:
            Size of content in bytes.

        Note:
            Useful for monitoring document sizes and
            ensuring they stay within limits.
        """
        return len(self.content)

    @cached_property
    def detected_mime_type(self) -> str:
        """Detect the MIME type from document content.

        Detection strategy (in order):
        1. Returns 'application/x-empty' for empty content
        2. Extension-based detection for known text formats (preferred)
        3. python-magic content analysis for unknown extensions
        4. Fallback to extension or 'application/octet-stream'

        Returns:
            MIME type string (e.g., "text/plain", "application/json").

        Note:
            This is cached after first access. Extension-based detection
            is preferred for text formats to avoid misidentification.
        """
        return detect_mime_type(self.content, self.name)

    @property
    def mime_type(self) -> str:
        """Get the document's MIME type.

        @public

        Primary property for accessing MIME type information.
        Automatically detects MIME type based on file extension and content.

        Returns:
            MIME type string (e.g., "text/plain", "application/json").

        Note:
            MIME type detection uses extension-based detection for known
            text formats and content analysis for binary formats.
        """
        return self.detected_mime_type

    @property
    def is_text(self) -> bool:
        """Check if document contains text content.

        @public

        Returns:
            True if MIME type indicates text content
            (text/*, application/json, application/x-yaml, text/yaml, etc.),
            False otherwise.

        Note:
            Used to determine if text property can be safely accessed.
        """
        return is_text_mime_type(self.mime_type)

    @property
    def is_pdf(self) -> bool:
        """Check if document is a PDF file.

        @public

        Returns:
            True if MIME type is application/pdf, False otherwise.

        Note:
            PDF documents require special handling and are
            supported by certain LLM models.
        """
        return is_pdf_mime_type(self.mime_type)

    @property
    def is_image(self) -> bool:
        """Check if document is an image file.

        @public

        Returns:
            True if MIME type starts with "image/", False otherwise.

        Note:
            Image documents are automatically encoded for
            vision-capable LLM models.
        """
        return is_image_mime_type(self.mime_type)

    @classmethod
    def canonical_name(cls) -> str:
        """Get the canonical name for this document class.

        Returns a standardized snake_case name derived from the
        class name, used for directory naming and identification.

        Returns:
            Snake_case canonical name.

        Example:
            >>> class UserDataDocument(FlowDocument): ...
            >>> UserDataDocument.canonical_name()
            'user_data'
        """
        return canonical_name_key(cls)

    @property
    def text(self) -> str:
        """Get document content as UTF-8 text string.

        @public

        Decodes the bytes content as UTF-8 text. Only available for
        text-based documents (check is_text property first).

        Returns:
            UTF-8 decoded string.

        Raises:
            ValueError: If document is not text (is_text == False).

        Example:
            >>> doc = MyDocument.create(name="data.txt", content="Hello \u2728")
            >>> if doc.is_text:
            ...     print(doc.text)  # "Hello \u2728"

            >>> # Binary document raises error:
            >>> binary_doc = MyDocument(name="image.png", content=png_bytes)
            >>> binary_doc.text  # Raises ValueError
        """
        if not self.is_text:
            raise ValueError(f"Document is not text: {self.name}")
        return self.content.decode("utf-8")

    def as_yaml(self) -> Any:
        r"""Parse document content as YAML.

        @public

        Parses the document's text content as YAML and returns Python objects.
        Uses ruamel.yaml which is safe by default (no code execution).

        Returns:
            Parsed YAML data: dict, list, str, int, float, bool, or None.

        Raises:
            ValueError: If document is not text-based.
            YAMLError: If content is not valid YAML.

        Example:
            >>> # From dict content
            >>> doc = MyDocument.create(name="config.yaml", content={
            ...     "server": {"host": "localhost", "port": 8080}
            ... })
            >>> doc.as_yaml()  # {'server': {'host': 'localhost', 'port': 8080}}

            >>> # From YAML string
            >>> doc2 = MyDocument(name="simple.yml", content=b"key: value\nitems:\n  - a\n  - b")
            >>> doc2.as_yaml()  # {'key': 'value', 'items': ['a', 'b']}
        """
        yaml = YAML()
        return yaml.load(self.text)  # type: ignore[no-untyped-call, no-any-return]

    def as_json(self) -> Any:
        """Parse document content as JSON.

        @public

        Parses the document's text content as JSON and returns Python objects.
        Document must contain valid JSON text.

        Returns:
            Parsed JSON data: dict, list, str, int, float, bool, or None.

        Raises:
            ValueError: If document is not text-based.
            JSONDecodeError: If content is not valid JSON.

        Example:
            >>> # From dict content
            >>> doc = MyDocument.create(name="data.json", content={"key": "value"})
            >>> doc.as_json()  # {'key': 'value'}

            >>> # From JSON string
            >>> doc2 = MyDocument(name="array.json", content=b'[1, 2, 3]')
            >>> doc2.as_json()  # [1, 2, 3]

            >>> # Invalid JSON
            >>> bad_doc = MyDocument(name="bad.json", content=b"not json")
            >>> bad_doc.as_json()  # Raises JSONDecodeError
        """
        return json.loads(self.text)

    @overload
    def as_pydantic_model(self, model_type: type[TModel]) -> TModel: ...

    @overload
    def as_pydantic_model(self, model_type: type[list[TModel]]) -> list[TModel]: ...

    def as_pydantic_model(
        self, model_type: type[TModel] | type[list[TModel]]
    ) -> TModel | list[TModel]:
        """Parse document content as Pydantic model with validation.

        @public

        Parses JSON or YAML content and validates it against a Pydantic model.
        Automatically detects format based on MIME type. Supports both single
        models and lists of models.

        Args:
            model_type: Pydantic model class to validate against.
                Can be either:
                - type[Model] for single model
                - type[list[Model]] for list of models

        Returns:
            Validated Pydantic model instance or list of instances.

        Raises:
            ValueError: If document is not text or type mismatch.
            ValidationError: If data doesn't match model schema.
            JSONDecodeError/YAMLError: If content parsing fails.

        Example:
            >>> from pydantic import BaseModel
            >>>
            >>> class User(BaseModel):
            ...     name: str
            ...     age: int
            >>>
            >>> # Single model
            >>> doc = MyDocument.create(name="user.json",
            ...     content={"name": "Alice", "age": 30})
            >>> user = doc.as_pydantic_model(User)
            >>> print(user.name)  # "Alice"
            >>>
            >>> # List of models
            >>> doc2 = MyDocument.create(name="users.json",
            ...     content=[{"name": "Bob", "age": 25}, {"name": "Eve", "age": 28}])
            >>> users = doc2.as_pydantic_model(list[User])
            >>> print(len(users))  # 2
        """
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
        r"""Parse document as markdown-separated list of sections.

        @public

        Splits text content using markdown separator ("\n\n-----------------\n\n").
        Designed for markdown documents with multiple sections.

        Returns:
            List of string sections (preserves whitespace within sections).

        Raises:
            ValueError: If document is not text-based.

        Example:
            >>> # Using create with list
            >>> sections = ["# Chapter 1\nIntroduction", "# Chapter 2\nDetails"]
            >>> doc = MyDocument.create(name="book.md", content=sections)
            >>> doc.as_markdown_list()  # Returns original sections

            >>> # Manual creation with separator
            >>> content = "Part 1\n\n-----------------\n\nPart 2\n\n-----------------\n\nPart 3"
            >>> doc2 = MyDocument(name="parts.md", content=content.encode())
            >>> doc2.as_markdown_list()  # ['Part 1', 'Part 2', 'Part 3']
        """
        return self.text.split(self.MARKDOWN_LIST_SEPARATOR)

    def parse(self, type_: type[Any]) -> Any:
        r"""Parse document content to original type (reverses create conversion).

        @public

        This method reverses the automatic conversion performed by the `create`
        classmethod. It intelligently parses the bytes content based on the
        document's file extension and converts to the requested type.

        Designed for roundtrip conversion:
            >>> original = {"key": "value"}
            >>> doc = MyDocument.create(name="data.json", content=original)
            >>> restored = doc.parse(dict)
            >>> assert restored == original  # True

        Args:
            type_: Target type to parse content into. Supported types:
                - bytes: Returns raw content (no conversion)
                - str: Decodes UTF-8 text
                - dict: Parses JSON (.json) or YAML (.yaml/.yml)
                - list: Splits markdown (.md) or parses JSON/YAML
                - BaseModel subclasses: Validates JSON/YAML into model

        Returns:
            Content parsed to the requested type.

        Raises:
            ValueError: If type is unsupported or parsing fails.

        Extension Rules:
            - .json → JSON parsing for dict/list/BaseModel
            - .yaml/.yml → YAML parsing for dict/list/BaseModel
            - .md + list → Split by markdown separator
            - Any + str → UTF-8 decode
            - Any + bytes → Raw content

        Example:
            >>> # String content
            >>> doc = MyDocument(name="test.txt", content=b"Hello")
            >>> doc.parse(str)
            'Hello'

            >>> # JSON content
            >>> doc = MyDocument.create(name="data.json", content={"key": "value"})
            >>> doc.parse(dict)  # Returns {'key': 'value'}

            >>> # Markdown list
            >>> items = ["Item 1", "Item 2"]
            >>> content = "\n\n---\n\n".join(items).encode()
            >>> doc = MyDocument(name="list.md", content=content)
            >>> doc.parse(list)
            ['Item 1', 'Item 2']
        """
        # Handle basic types
        if type_ is bytes:
            return self.content
        elif type_ is str:
            # Handle empty content specially
            if len(self.content) == 0:
                return ""
            return self.text

        # Handle structured data based on extension
        name_lower = self.name.lower()

        # JSON files
        if name_lower.endswith(".json"):
            if type_ is dict or type_ is list:
                result = self.as_json()
                # Ensure the result is the correct type
                if type_ is dict and not isinstance(result, dict):
                    raise ValueError(f"Expected dict but got {type(result).__name__}")
                if type_ is list and not isinstance(result, list):
                    raise ValueError(f"Expected list but got {type(result).__name__}")
                return result
            elif issubclass(type_, BaseModel):
                return self.as_pydantic_model(type_)
            else:
                raise ValueError(f"Cannot parse JSON file to type {type_}")

        # YAML files
        elif name_lower.endswith((".yaml", ".yml")):
            if type_ is dict or type_ is list:
                result = self.as_yaml()
                # Ensure the result is the correct type
                if type_ is dict and not isinstance(result, dict):
                    raise ValueError(f"Expected dict but got {type(result).__name__}")
                if type_ is list and not isinstance(result, list):
                    raise ValueError(f"Expected list but got {type(result).__name__}")
                return result
            elif issubclass(type_, BaseModel):
                return self.as_pydantic_model(type_)
            else:
                raise ValueError(f"Cannot parse YAML file to type {type_}")

        # Markdown files with lists
        elif name_lower.endswith(".md") and type_ is list:
            return self.as_markdown_list()

        # Default: try to return as requested basic type
        elif type_ is dict or type_ is list:
            # Try JSON first, then YAML
            try:
                result = self.as_json()
                # Ensure the result is the correct type
                if type_ is dict and not isinstance(result, dict):
                    raise ValueError(f"Expected dict but got {type(result).__name__}")
                if type_ is list and not isinstance(result, list):
                    raise ValueError(f"Expected list but got {type(result).__name__}")
                return result
            except (json.JSONDecodeError, ValueError):
                try:
                    result = self.as_yaml()
                    # Ensure the result is the correct type
                    if type_ is dict and not isinstance(result, dict):
                        raise ValueError(f"Expected dict but got {type(result).__name__}")
                    if type_ is list and not isinstance(result, list):
                        raise ValueError(f"Expected list but got {type(result).__name__}")
                    return result
                except Exception as e:
                    raise ValueError(f"Cannot parse content to {type_}") from e

        raise ValueError(f"Unsupported type {type_} for file {self.name}")

    def get_source_documents(self) -> list[str]:
        """Get list of document SHA256 hashes referenced as sources.

        Retrieves all document references from this document's sources list,
        filtering for valid SHA256 hashes that reference other documents.
        This is useful for building dependency graphs and tracking document
        lineage in processing pipelines.

        Returns:
            List of SHA256 hashes (base32 encoded) for documents referenced
            as sources. Each hash uniquely identifies another document that
            contributed to creating this one.

        Example:
            >>> # Create a derived document from multiple sources
            >>> source1 = MyDocument.create(name="data1.txt", content="First")
            >>> source2 = MyDocument.create(name="data2.txt", content="Second")
            >>>
            >>> merged = MyDocument.create(
            ...     name="merged.txt",
            ...     content="Combined data",
            ...     sources=[source1.sha256, source2.sha256, "https://api.example.com"]
            ... )
            >>>
            >>> # Get only document references (not URLs)
            >>> doc_refs = merged.get_source_documents()
            >>> print(doc_refs)  # [source1.sha256, source2.sha256]
            >>>
            >>> # Check if specific document is a source
            >>> if source1.sha256 in doc_refs:
            ...     print("Document derived from source1")

        See Also:
            - get_source_references: Get non-document source references (URLs, etc.)
            - has_source: Check if a specific source is tracked
            - Document.create: Add sources when creating documents
        """
        return [src for src in self.sources if is_document_sha256(src)]

    def get_source_references(self) -> list[str]:
        """Get list of arbitrary reference strings from sources.

        Retrieves all non-document references from this document's sources list.
        These are typically URLs, file paths, API endpoints, or descriptive strings
        that indicate where the document's content originated from, but are not
        references to other documents in the pipeline.

        Returns:
            List of reference strings that are not document SHA256 hashes.
            Can include URLs, file paths, API endpoints, dataset names,
            or any other string that provides source context.

        Example:
            >>> # Create document with mixed source types
            >>> doc = MyDocument.create(
            ...     name="report.txt",
            ...     content="Analysis results",
            ...     sources=[
            ...         other_doc.sha256,  # Document reference
            ...         "https://api.example.com/data",  # API URL
            ...         "dataset:customer-2024",  # Dataset identifier
            ...         "/path/to/source.csv",  # File path
            ...     ]
            ... )
            >>>
            >>> # Get only non-document references
            >>> refs = doc.get_source_references()
            >>> print(refs)
            >>> # ["https://api.example.com/data", "dataset:customer-2024", "/path/to/source.csv"]
            >>>
            >>> # Use for attribution or debugging
            >>> for ref in refs:
            ...     print(f"Data sourced from: {ref}")

        See Also:
            - get_source_documents: Get document SHA256 references
            - has_source: Check if a specific source is tracked
            - Document.create: Add sources when creating documents
        """
        return [src for src in self.sources if not is_document_sha256(src)]

    def has_source(self, source: Document | str) -> bool:
        """Check if a specific source is tracked for this document.

        Verifies whether a given source (document or reference string) is
        included in this document's sources list. Useful for dependency
        checking, lineage verification, and conditional processing based
        on document origins.

        Args:
            source: Source to check for. Can be:
                    - Document: Checks if document's SHA256 is in sources
                    - str: Checks if exact string is in sources (hash or reference)

        Returns:
            True if the source is tracked in this document's sources,
            False otherwise.

        Raises:
            TypeError: If source is not a Document or string.

        Example:
            >>> # Check if document was derived from specific source
            >>> source_doc = MyDocument.create(name="original.txt", content="Data")
            >>> api_url = "https://api.example.com/data"
            >>>
            >>> derived = MyDocument.create(
            ...     name="processed.txt",
            ...     content="Processed data",
            ...     sources=[source_doc.sha256, api_url]
            ... )
            >>>
            >>> # Check document source
            >>> if derived.has_source(source_doc):
            ...     print("Derived from source_doc")
            >>>
            >>> # Check string reference
            >>> if derived.has_source(api_url):
            ...     print("Data from API")
            >>>
            >>> # Check by SHA256 directly
            >>> if derived.has_source(source_doc.sha256):
            ...     print("Has specific hash")

        See Also:
            - get_source_documents: Get all document sources
            - get_source_references: Get all reference sources
            - Document.create: Add sources when creating documents
        """
        if isinstance(source, str):
            # Direct string comparison
            return source in self.sources
        elif isinstance(source, Document):  # type: ignore[misc]
            # Check if document's SHA256 is in sources
            return source.sha256 in self.sources
        else:
            raise TypeError(f"Invalid source type: {type(source)}")

    @final
    def serialize_model(self) -> dict[str, Any]:
        """Serialize document to dictionary for storage or transmission.

        Creates a complete JSON-serializable representation of the document
        with all metadata and properly encoded content. Automatically chooses
        the most appropriate encoding (UTF-8 for text, base64 for binary).

        Returns:
            Dictionary with the following keys:
                - name: Document filename (str)
                - description: Optional description (str | None)
                - base_type: Persistence type - "flow", "task", or "temporary" (str)
                - size: Content size in bytes (int)
                - id: Short hash identifier, first 6 chars of SHA256 (str)
                - sha256: Full SHA256 hash in base32 encoding without padding (str)
                - mime_type: Detected MIME type (str)
                - sources: List of source strings (list[dict])
                - content: Encoded content (str)
                - content_encoding: Either "utf-8" or "base64" (str)

        Encoding Strategy:
            - Text files (text/*, application/json, etc.) → UTF-8 string
            - Binary files (images, PDFs, etc.) → Base64 string
            - Invalid UTF-8 in text files → UTF-8 with replacement chars

        Example:
            >>> doc = MyDocument.create(name="data.json", content={"key": "value"})
            >>> serialized = doc.serialize_model()
            >>> serialized["content_encoding"]  # "utf-8"
            >>> serialized["mime_type"]  # "application/json"
        """
        result = {
            "name": self.name,
            "description": self.description,
            "base_type": self.get_base_type(),
            "size": self.size,
            "id": self.id,
            "sha256": self.sha256,
            "mime_type": self.mime_type,
            "sources": self.sources,
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
        r"""Deserialize document from dictionary (inverse of serialize_model).

        Reconstructs a Document instance from the dictionary format produced
        by serialize_model(). Automatically handles content decoding based on
        the content_encoding field.

        Args:
            data: Dictionary containing serialized document. Required keys:
                - name: Document filename (str)
                - content: Encoded content (str or bytes)
                Optional keys:
                - description: Document description (str | None)
                - content_encoding: "utf-8" or "base64" (defaults to "utf-8")
                - sources: List of source strings

        Returns:
            New Document instance with restored content.

        Raises:
            ValueError: If content type is invalid or base64 decoding fails
            KeyError: If required keys are missing from data dictionary

        Note:
            Provides roundtrip guarantee with serialize_model().
            Content and name are preserved exactly.

        Example:
            >>> data = {
            ...     "name": "config.yaml",
            ...     "content": "key: value\n",
            ...     "content_encoding": "utf-8",
            ...     "description": "Config file"
            ... }
            >>> doc = MyDocument.from_dict(data)
        """
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

        return cls(
            name=data["name"],
            content=content,
            description=data.get("description"),
            sources=data.get("sources", []),
        )
