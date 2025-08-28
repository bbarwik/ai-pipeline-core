"""Document abstraction layer for AI pipeline flows.

@public

This module provides the core document abstraction for working with various types of data
in AI pipelines. Documents are immutable Pydantic models that wrap binary content with metadata.
"""

import base64
import hashlib
import json
import warnings
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

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_serializer, field_validator
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
ContentInput = str | bytes | dict[str, Any] | list[Any] | BaseModel


class Document(BaseModel, ABC):
    """Abstract base class for all documents in the AI Pipeline Core system.
    
    @public

    Document is the fundamental data abstraction for all content flowing through
    pipelines. It provides automatic encoding, MIME type detection, serialization,
    and validation. All documents must be subclassed from FlowDocument, TaskDocument,
    or TemporaryDocument based on their persistence requirements.

    Key features:
    - Immutable by default (frozen Pydantic model)
    - Automatic MIME type detection
    - Content size validation
    - SHA256 hashing for deduplication
    - Support for text, JSON, YAML, PDF, and image formats
    - Conversion utilities between different formats

    Class Variables:
        MAX_CONTENT_SIZE: Maximum allowed content size in bytes (default 25MB)
        DESCRIPTION_EXTENSION: File extension for description files (.description.md)
        MARKDOWN_LIST_SEPARATOR: Separator for markdown list items

    Attributes:
        name: Document filename (validated for security)
        description: Optional human-readable description
        content: Raw document content as bytes

    Warning:
        - Document subclasses should NOT start with 'Test' prefix (pytest conflict)
        - Cannot instantiate Document directly - use FlowDocument or TaskDocument
        - Cannot add custom fields - only name, description, content are allowed

    Metadata Attachment Patterns:
        Since custom fields are not allowed, use these patterns for metadata:
        1. Use the 'description' field for human-readable metadata
        2. Embed metadata in content (e.g., JSON with data + metadata fields)
        3. Create a separate MetadataDocument type to accompany data documents
        4. Use document naming conventions (e.g., "data_v2_2024.json")
        5. Store metadata in flow_options or pass through TraceInfo

    Example:
        >>> class MyDocument(FlowDocument):
        ...     def get_type(self) -> str:
        ...         return "my_doc"
        >>> doc = MyDocument(name="data.json", content=b'{"key": "value"}')
        >>> print(doc.is_text)  # True
        >>> data = doc.as_json()  # {'key': 'value'}
    """

    MAX_CONTENT_SIZE: ClassVar[int] = 25 * 1024 * 1024
    """Maximum allowed content size in bytes (default 25MB).
    
    @public
    """
    
    DESCRIPTION_EXTENSION: ClassVar[str] = ".description.md"
    """File extension for description files.
    
    @public
    """
    
    MARKDOWN_LIST_SEPARATOR: ClassVar[str] = "\n\n---\n\n"
    """Separator for markdown list items.
    
    @public
    """

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
        allowed = {"name", "description", "content"}
        current = set(getattr(cls, "model_fields", {}).keys())
        extras = current - allowed
        if extras:
            raise TypeError(
                f"Document subclass '{cls.__name__}' cannot declare additional fields: "
                f"{', '.join(sorted(extras))}. Only {', '.join(sorted(allowed))} are allowed."
            )

    def __init__(self, **data: Any) -> None:
        """Initialize a Document instance.
        
        @public

        Prevents direct instantiation of the abstract Document class.
        Content can be passed as either str or bytes (automatically converted to bytes).
        Handles legacy signature where content was passed as description.

        Args:
            **data: Keyword arguments for fields:
                - name (str): Document filename
                - description (str | None): Optional description (or content in legacy)
                - content (str | bytes | None): Content (automatically converted to bytes)

        Raises:
            TypeError: If attempting to instantiate Document directly.

        Example:
            >>> doc = MyDocument(name="test.txt", content="Hello World")
            >>> doc.content  # b'Hello World'
            >>> # Legacy: content as description
            >>> doc = MyDocument(name="test.txt", description=b"data", content=None)
            >>> doc.content  # b'data'
        """
        if type(self) is Document:
            raise TypeError("Cannot instantiate abstract Document class directly")

        # Handle legacy signature where content was passed as description
        if data.get("content") is None and data.get("description") is not None:
            desc = data["description"]
            if not isinstance(desc, str):
                # If description is not a string, it's actually content in legacy format
                warnings.warn(
                    "Passing content as 'description' parameter is deprecated and will be "
                    "removed in v0.2.0. Use 'content' parameter instead.",
                    DeprecationWarning,
                    stacklevel=2
                )
                data["content"] = desc
                data["description"] = None

        super().__init__(**data)

    name: str
    description: str | None = None
    content: bytes  # Note: constructor accepts str | bytes, but field stores bytes only

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
        
        @public

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
        
        @public

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
        
        @public

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
        
        @public

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

        This method provides a hook for enforcing file naming conventions.
        By default, it checks against the FILES enum if defined.
        Subclasses can override for custom validation logic.

        Args:
            name: The file name to validate.

        Raises:
            DocumentNameError: If the name doesn't match allowed patterns.

        Note:
            - If FILES enum is defined, name must exactly match one of the values
            - If FILES is not defined, any name is allowed
            - Override in subclasses for regex patterns or other conventions

        See Also:
            - get_expected_files: Returns list of allowed file names
            - validate_name: Pydantic validator for security checks
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
        """Pydantic validator for document content.

        Intelligently converts various content types to bytes:
        - str: UTF-8 encoding
        - bytes: passed through
        - dict: JSON or YAML based on extension
        - list: Markdown list for .md, JSON/YAML otherwise
        - Pydantic models: JSON or YAML serialization
        - Numbers/booleans: JSON serialization for .json files

        Ensures content doesn't exceed MAX_CONTENT_SIZE (default 25MB).

        Args:
            v: The content to validate (any supported type).
            info: Pydantic validation info context.

        Returns:
            The validated content as bytes.

        Raises:
            DocumentSizeError: If content exceeds MAX_CONTENT_SIZE.
            ValueError: If content type is not supported.

        Note:
            - Automatically handles type conversion based on file extension
            - Subclasses can customize MAX_CONTENT_SIZE class variable
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

        Returns the first 6 characters of the base32-encoded SHA256 hash,
        providing a compact identifier suitable for logging
        and display purposes.

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
            - Base32 is case-insensitive (safer for filesystems)
            - More compact than hex (52 chars vs 64 chars for SHA-256)
            - Safe for URLs without encoding
            - Compatible with systems that have case-insensitive paths

        Note:
            This is computed once and cached for performance.
            The hash is deterministic based on content only.
        """
        return b32encode(hashlib.sha256(self.content).digest()).decode("ascii").upper()

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
        Currently delegates to detected_mime_type but provides
        a stable API for future enhancements.

        Returns:
            MIME type string.

        Note:
            Prefer this over detected_mime_type for general use.
        """
        return self.detected_mime_type

    @property
    def is_text(self) -> bool:
        """Check if document contains text content.
        
        @public

        Returns:
            True if MIME type indicates text content
            (text/*, application/json, application/yaml, etc.),
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
        """Get document content as text string.
        
        @public

        Decodes the bytes content as UTF-8 text. Only works
        for text-based documents.

        Returns:
            UTF-8 decoded string.

        Raises:
            ValueError: If document is not text-based (check is_text first).

        Example:
            >>> doc = MyDocument(name="data.txt", content=b"Hello")
            >>> doc.text
            'Hello'
        """
        if not self.is_text:
            raise ValueError(f"Document is not text: {self.name}")
        return self.content.decode("utf-8")

    def as_yaml(self) -> Any:
        """Parse document content as YAML.
        
        @public

        Converts text content to Python objects using YAML parser.
        Document must be text-based.

        Returns:
            Parsed YAML data (dict, list, or scalar).

        Note:
            Raises ValueError if document is not text (from text property).
            Uses ruamel.yaml which is safe by default (no arbitrary code execution).

        Example:
            >>> doc = MyDocument(name="config.yaml", content=b"key: value")
            >>> doc.as_yaml()
            {'key': 'value'}
        """
        yaml = YAML()
        return yaml.load(self.text)  # type: ignore[no-untyped-call, no-any-return]

    def as_json(self) -> Any:
        """Parse document content as JSON.
        
        @public

        Converts text content to Python objects using JSON parser.
        Document must be text-based.

        Returns:
            Parsed JSON data (dict, list, or scalar).

        Note:
            Raises ValueError if document is not text (from text property).
            Raises JSONDecodeError if content is not valid JSON (from json.loads()).

        Example:
            >>> doc = MyDocument(name="data.json", content=b'{"key": "value"}')
            >>> doc.as_json()
            {'key': 'value'}
        """
        return json.loads(self.text)

    @overload
    def as_pydantic_model(self, model_type: type[TModel]) -> TModel: ...

    @overload
    def as_pydantic_model(self, model_type: type[list[TModel]]) -> list[TModel]: ...

    def as_pydantic_model(
        self, model_type: type[TModel] | type[list[TModel]]
    ) -> TModel | list[TModel]:
        """Parse document content as a Pydantic model.
        
        @public

        Converts JSON or YAML content to validated Pydantic model instances.
        Supports both single models and lists of models.

        Args:
            model_type: The Pydantic model class or list[ModelClass] to parse into.

        Returns:
            Validated Pydantic model instance or list of instances.

        Raises:
            ValueError: If document is not text or data doesn't match expected type.

        Note:
            May raise ValidationError from Pydantic if data doesn't validate against the model.

        Example:
            >>> class Config(BaseModel):
            ...     key: str
            >>> doc = MyDocument(name="config.json", content=b'{"key": "value"}')
            >>> config = doc.as_pydantic_model(Config)
            >>> config.key
            'value'
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
        r"""Parse document as a markdown-separated list.
        
        @public

        Splits text content by the MARKDOWN_LIST_SEPARATOR
        (default: "\n\n---\n\n") to extract list items.

        Returns:
            List of string items.

        Note:
            Raises ValueError if document is not text (from text property).

        Example:
            >>> content = "Item 1\n\n---\n\nItem 2"
            >>> doc = MyDocument(name="list.md", content=content.encode())
            >>> doc.as_markdown_list()
            ['Item 1', 'Item 2']
        """
        return self.text.split(self.MARKDOWN_LIST_SEPARATOR)

    def parsed(self, type_: type[Any]) -> Any:
        r"""Parse document content based on file extension and requested type.
        
        @public

        Intelligently parses content based on the document's file extension
        and converts to the requested type. This is designed to be the inverse
        of document creation, so that Document(name=n, content=x).parsed(type(x)) == x.

        Args:
            type_: Target type to parse content into. Supported types:
                - bytes: Returns raw content
                - str: Returns decoded text
                - dict: Parses JSON/YAML based on extension
                - list: For .md files, splits by markdown separator
                - BaseModel subclasses: Parses and validates JSON/YAML

        Returns:
            Parsed content in the requested type.

        Raises:
            ValueError: If the type is not supported or content cannot be parsed.

        Example:
            >>> # String content
            >>> doc = MyDocument(name="test.txt", content="Hello")
            >>> doc.parsed(str)
            'Hello'

            >>> # JSON content
            >>> import json
            >>> data = {"key": "value"}
            >>> doc = MyDocument(name="data.json", content=json.dumps(data))
            >>> doc.parsed(dict)
            {'key': 'value'}

            >>> # Markdown list
            >>> items = ["Item 1", "Item 2"]
            >>> content = "\n\n---\n\n".join(items)
            >>> doc = MyDocument(name="list.md", content=content)
            >>> doc.parsed(list)
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

    @final
    def serialize_model(self) -> dict[str, Any]:
        """Serialize document to a dictionary for storage or transmission.

        Creates a complete representation of the document including
        metadata and properly encoded content. Text content uses UTF-8,
        binary content uses base64 encoding.

        Returns:
            Dictionary containing:
            - name: Document filename
            - description: Optional description
            - base_type: "flow", "task", or "temporary"
            - size: Content size in bytes
            - id: Short hash identifier
            - sha256: Full content hash
            - mime_type: Detected MIME type
            - content: Encoded content string
            - content_encoding: "utf-8" or "base64"

        Note:
            Text files with encoding issues use UTF-8 with replacement
            characters rather than failing.
        """
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
        """Deserialize a document from a dictionary representation.

        Reconstructs a document from the dictionary format produced
        by serialize_model(). Handles both UTF-8 and base64 encoded
        content.

        Args:
            data: Dictionary containing serialized document data.
                  Must have 'name' and 'content' keys at minimum.

        Returns:
            New document instance.

        Raises:
            ValueError: If content type is invalid.

        Note:
            May raise KeyError if required keys are missing from the data dictionary.

        Example:
            >>> serialized = doc.serialize_model()
            >>> restored = MyDocument.from_dict(serialized)
            >>> assert restored.content == doc.content
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

        # Create document with the required fields
        return cls(
            name=data["name"],
            content=content,
            description=data.get("description"),
        )
