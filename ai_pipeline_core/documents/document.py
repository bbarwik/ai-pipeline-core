"""Document abstraction layer for AI pipeline flows.

Immutable Pydantic models wrapping binary content with metadata, MIME detection,
SHA256 hashing, and serialization. All documents must be concrete subclasses of Document.
"""

import base64
import functools
import json
from enum import StrEnum
from functools import cached_property
from io import BytesIO
from typing import (
    Any,
    ClassVar,
    Self,
    TypeVar,
    cast,
    final,
    get_args,
    get_origin,
    overload,
)

import tiktoken
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)
from ruamel.yaml import YAML

from ai_pipeline_core.documents._context_vars import get_task_context, is_registration_suppressed
from ai_pipeline_core.documents._hashing import compute_content_sha256, compute_document_sha256
from ai_pipeline_core.documents.utils import DATA_URI_PATTERN, is_document_sha256
from ai_pipeline_core.exceptions import DocumentNameError, DocumentSizeError

from .attachment import Attachment
from .mime_type import (
    detect_mime_type,
    is_image_mime_type,
    is_pdf_mime_type,
    is_text_mime_type,
    is_yaml_mime_type,
)

TModel = TypeVar("TModel", bound=BaseModel)
TDocument = TypeVar("TDocument", bound="Document")

# Registry of class __name__ -> Document subclass for collision detection.
# Only non-test classes are registered. Test modules (tests.*, conftest, etc.) are skipped.
_class_name_registry: dict[str, type["Document"]] = {}  # nosemgrep: no-mutable-module-globals

# Metadata keys added by serialize_model() that should be stripped before validation.
_DOCUMENT_SERIALIZE_METADATA_KEYS: frozenset[str] = frozenset({
    "id",
    "sha256",
    "content_sha256",
    "size",
    "mime_type",
    "class_name",
})


def _is_test_module(cls: type) -> bool:
    """Check if a class is defined in a test module (skip collision detection)."""
    module = getattr(cls, "__module__", "") or ""
    parts = module.split(".")
    return any(p == "tests" or p.startswith("test_") or p == "conftest" for p in parts)


@functools.cache
def get_tiktoken_encoding() -> tiktoken.Encoding:
    """Lazy-cached tiktoken encoding. Deferred to first use, cached forever."""
    return tiktoken.encoding_for_model("gpt-4")


def _serialize_to_json(data: Any) -> bytes:
    """JSON serialize with 2-space indent."""
    return json.dumps(data, indent=2).encode("utf-8")


def _serialize_to_yaml(data: Any) -> bytes:
    """YAML serialize via ruamel."""
    yaml = YAML()
    stream = BytesIO()
    yaml.dump(data, stream)  # pyright: ignore[reportUnknownMemberType]
    return stream.getvalue()


def _serialize_structured(name: str, data: Any) -> bytes:
    """Serialize dict/list to JSON or YAML based on file extension."""
    name_lower = name.lower()
    if name_lower.endswith((".yaml", ".yml")):
        return _serialize_to_yaml(data)
    if name_lower.endswith(".json"):
        return _serialize_to_json(data)
    raise ValueError(f"Structured content ({type(data).__name__}) requires .json or .yaml extension, got: {name}")


def _convert_content(name: str, content: str | bytes | dict[str, Any] | list[Any] | BaseModel) -> bytes:
    """Convert any supported content type to bytes. Dispatch by isinstance."""
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode("utf-8")
    if isinstance(content, dict):
        return _serialize_structured(name, content)
    if isinstance(content, BaseModel):
        return _serialize_structured(name, content.model_dump(mode="json"))
    if isinstance(content, list):  # pyright: ignore[reportUnnecessaryIsInstance]
        data = [item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in content]
        return _serialize_structured(name, data)
    raise ValueError(f"Unsupported content type: {type(content)}")  # pyright: ignore[reportUnreachable]


class Document(BaseModel):
    """Immutable base class for all pipeline documents. Cannot be instantiated directly — must be subclassed.

    Content is stored as bytes. Use `create()` for automatic conversion from str/dict/list/BaseModel.
    Use `parse()` to reverse the conversion. Serialization is extension-driven (.json → JSON, .yaml → YAML).
    """

    MAX_CONTENT_SIZE: ClassVar[int] = 25 * 1024 * 1024
    """Maximum allowed total size in bytes (default 25MB)."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate subclass at definition time. Cannot start with 'Test', cannot add custom fields."""
        super().__init_subclass__(**kwargs)
        if cls.__name__.startswith("Test"):
            raise TypeError(
                f"Document subclass '{cls.__name__}' cannot start with 'Test' prefix. "
                "This causes conflicts with pytest test discovery. "
                "Please use a different name (e.g., 'SampleDocument', 'ExampleDocument')."
            )
        if hasattr(cls, "FILES"):
            files: type[StrEnum] = cls.FILES  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
            if not issubclass(files, StrEnum):
                raise TypeError(f"Document subclass '{cls.__name__}'.FILES must be an Enum of string values")
        # Check that the Document's model_fields only contain the allowed fields
        # It prevents AI models from adding additional fields to documents
        allowed = {"name", "description", "content", "sources", "attachments", "origins"}
        current = set(getattr(cls, "model_fields", {}).keys())
        extras = current - allowed
        if extras:
            raise TypeError(
                f"Document subclass '{cls.__name__}' cannot declare additional fields: "
                f"{', '.join(sorted(extras))}. Only {', '.join(sorted(allowed))} are allowed."
            )

        # Class name collision detection (production classes only)
        if not _is_test_module(cls):
            name = cls.__name__
            existing = _class_name_registry.get(name)
            if existing is not None and existing is not cls:
                raise TypeError(
                    f"Document subclass '{name}' (in {cls.__module__}) collides with "
                    f"existing class in {existing.__module__}. "
                    f"Class names must be unique across the framework."
                )
            _class_name_registry[name] = cls

    @classmethod
    def create(
        cls,
        *,
        name: str,
        content: str | bytes | dict[str, Any] | list[Any] | BaseModel,
        description: str | None = None,
        sources: tuple[str, ...] | None = None,
        origins: tuple[str, ...] | None = None,
        attachments: tuple[Attachment, ...] | None = None,
    ) -> Self:
        """Create a document with automatic content-to-bytes conversion.

        Serialization is extension-driven: .json → JSON, .yaml → YAML, others → UTF-8.
        Reversible via parse(). Cannot be called on Document directly — must use a subclass.
        """
        return cls(
            name=name,
            content=_convert_content(name, content),
            description=description,
            sources=sources,
            origins=origins,
            attachments=attachments,
        )

    def __init__(
        self,
        *,
        name: str,
        content: bytes,
        description: str | None = None,
        sources: tuple[str, ...] | None = None,
        origins: tuple[str, ...] | None = None,
        attachments: tuple[Attachment, ...] | None = None,
    ) -> None:
        """Initialize with raw bytes content. Most users should use `create()` instead."""
        if type(self) is Document:
            raise TypeError("Cannot instantiate Document directly — use a concrete subclass")

        super().__init__(
            name=name,
            content=content,
            description=description,
            sources=sources or (),
            origins=origins or (),
            attachments=attachments or (),
        )

        # Register with task context for document lifecycle tracking
        if not is_registration_suppressed():
            task_ctx = get_task_context()
            if task_ctx is not None:
                task_ctx.register_created(self)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]

    name: str
    description: str | None = None
    content: bytes
    sources: tuple[str, ...] = ()
    """Content provenance: documents and references this document's content was directly
    derived from. Can be document SHA256 hashes (for pipeline documents) or external
    references (URLs, file paths). Answers: 'where did this content come from?'

    An analysis document derived from an input document has
    sources=(input_doc.sha256,). A webpage capture has sources=("https://example.com",)."""

    origins: tuple[str, ...] = ()
    """Causal provenance: documents that caused this document to be created without directly
    contributing to its content. Always document SHA256 hashes, never arbitrary strings.
    Answers: 'why does this document exist?'

    A research plan causes 10 webpages to be captured. Each webpage's source is its
    URL (content provenance), its origin is the research plan (causal provenance — the plan
    caused the capture but didn't contribute to the webpage's content).

    A SHA256 hash must not appear in both sources and origins for the same document.
    Within a pipeline task or flow, all source/origin SHA256 references must point to
    documents that existed before the task/flow started executing."""
    attachments: tuple[Attachment, ...] = ()

    # Pydantic configuration
    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @final
    @classmethod
    def get_expected_files(cls) -> list[str] | None:
        """Return allowed filenames from FILES enum, or None if unrestricted."""
        if not hasattr(cls, "FILES"):
            return None
        files: type[StrEnum] = cls.FILES  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
        if not files:
            return None
        assert issubclass(files, StrEnum)
        try:
            values = [member.value for member in files]
        except TypeError:
            raise DocumentNameError(f"{cls.__name__}.FILES must be an Enum of string values") from None
        if len(values) == 0:
            return None
        return values

    @classmethod
    def validate_file_name(cls, name: str) -> None:
        """Validate filename against FILES enum. Override only for custom validation beyond FILES."""
        allowed = cls.get_expected_files()
        if not allowed:
            return

        if len(allowed) > 0 and name not in allowed:
            allowed_str = ", ".join(sorted(allowed))
            raise DocumentNameError(f"Invalid filename '{name}'. Allowed names: {allowed_str}")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Reject path traversal, whitespace issues, reserved suffixes. Must match FILES enum if defined."""
        if ".." in v or "\\" in v or "/" in v:
            raise DocumentNameError(f"Invalid filename - contains path traversal characters: {v}")

        if not v or v.startswith(" ") or v.endswith(" "):
            raise DocumentNameError(f"Invalid filename format: {v}")

        if v.endswith(".meta.json"):
            raise DocumentNameError(f"Document names cannot end with .meta.json (reserved): {v}")

        cls.validate_file_name(v)

        return v

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v: Any, info: ValidationInfo) -> bytes:
        """Convert content to bytes. Enforces MAX_CONTENT_SIZE.

        Handles:
        1. bytes — passed through directly
        2. str with data URI prefix — base64-decoded to bytes
        3. str (plain text) — UTF-8 encoded to bytes
        4. dict/list/BaseModel — serialized via _convert_content
        """
        if isinstance(v, bytes):
            pass
        elif isinstance(v, str):
            # Data URIs are produced by serialize_content() for binary content only (failed UTF-8 decode).
            # Text content starting with "data:<mime>;base64," would be misinterpreted here, but this is
            # accepted by design — real documents never start with a bare data URI on the first byte.
            if DATA_URI_PATTERN.match(v):
                _, payload = v.split(",", 1)
                v = base64.b64decode(payload)
            else:
                v = v.encode("utf-8")
        else:
            name = info.data.get("name", "") if hasattr(info, "data") else ""
            v = _convert_content(name, v)
        if len(v) > cls.MAX_CONTENT_SIZE:
            raise DocumentSizeError(f"Document size ({len(v)} bytes) exceeds maximum allowed size ({cls.MAX_CONTENT_SIZE} bytes)")
        return v

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        """Sources must be document SHA256 hashes or URLs."""
        for src in v:
            if not is_document_sha256(src) and "://" not in src:
                raise ValueError(f"Source must be a document SHA256 hash or a URL (containing '://'), got: {src!r}")
        return v

    @field_validator("origins")
    @classmethod
    def validate_origins(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        """Origins must be valid document SHA256 hashes."""
        for origin in v:
            if not is_document_sha256(origin):
                raise ValueError(f"Origin must be a document SHA256 hash, got: {origin}")
        return v

    @model_validator(mode="after")
    def validate_no_source_origin_overlap(self) -> Self:
        """Reject documents where the same SHA256 appears in both sources and origins."""
        source_sha256s = {src for src in self.sources if is_document_sha256(src)}
        if source_sha256s:
            overlap = source_sha256s & set(self.origins)
            if overlap:
                sample = next(iter(overlap))
                raise ValueError(
                    f"SHA256 hash {sample[:12]}... appears in both sources and origins. "
                    f"A document reference must be either a source (content provenance) "
                    f"or an origin (causal provenance), not both."
                )
        return self

    @model_validator(mode="after")
    def validate_total_size(self) -> Self:
        """Validate that total document size (content + attachments) is within limits."""
        total = self.size
        if total > self.MAX_CONTENT_SIZE:
            raise DocumentSizeError(f"Total document size ({total} bytes) including attachments exceeds maximum allowed size ({self.MAX_CONTENT_SIZE} bytes)")
        return self

    @field_serializer("content")
    def serialize_content(self, v: bytes) -> str:
        """Serialize content: plain string for text, data URI (RFC 2397) for binary."""
        try:
            return v.decode("utf-8")
        except UnicodeDecodeError:
            b64 = base64.b64encode(v).decode("ascii")
            return f"data:{self.mime_type};base64,{b64}"

    @final
    @property
    def id(self) -> str:
        """First 6 chars of sha256. Used as short document identifier in LLM context."""
        return self.sha256[:6]

    @final
    @cached_property
    def sha256(self) -> str:
        """Full SHA256 identity hash (name + content + sources + origins + attachments). BASE32 encoded, cached."""
        return compute_document_sha256(self)

    @final
    @cached_property
    def content_sha256(self) -> str:
        """SHA256 hash of raw content bytes only. Used for content deduplication."""
        return compute_content_sha256(self.content)

    @final
    @property
    def size(self) -> int:
        """Total size of content + attachments in bytes."""
        return len(self.content) + sum(att.size for att in self.attachments)

    @cached_property
    def mime_type(self) -> str:
        """Detected MIME type. Extension-based for known formats, content analysis for others. Cached."""
        return detect_mime_type(self.content, self.name)

    @property
    def is_text(self) -> bool:
        """True if MIME type indicates text content."""
        return is_text_mime_type(self.mime_type)

    @property
    def is_pdf(self) -> bool:
        """True if MIME type is application/pdf."""
        return is_pdf_mime_type(self.mime_type)

    @property
    def is_image(self) -> bool:
        """True if MIME type starts with image/."""
        return is_image_mime_type(self.mime_type)

    @property
    def text(self) -> str:
        """Content decoded as UTF-8. Raises ValueError if not text."""
        if not self.is_text:
            raise ValueError(f"Document is not text: {self.name}")
        return self.content.decode("utf-8")

    @cached_property
    def approximate_tokens_count(self) -> int:
        """Approximate token count (tiktoken gpt-4 encoding). Images=1080, PDFs/other=1024."""
        enc = get_tiktoken_encoding()
        if self.is_text:
            total = len(enc.encode(self.text))
        elif self.is_image:
            total = 1080
        else:
            total = 1024

        for att in self.attachments:
            if att.is_image:
                total += 1080
            elif att.is_pdf:
                total += 1024
            elif att.is_text:
                total += len(enc.encode(att.text))
            else:
                total += 1024

        return total

    def as_yaml(self) -> Any:
        """Parse content as YAML via ruamel.yaml."""
        yaml = YAML()
        return yaml.load(self.text)  # type: ignore[no-untyped-call, no-any-return]

    def as_json(self) -> Any:
        """Parse content as JSON."""
        return json.loads(self.text)

    @overload
    def as_pydantic_model(self, model_type: type[TModel]) -> TModel: ...

    @overload
    def as_pydantic_model(self, model_type: type[list[TModel]]) -> list[TModel]: ...

    def as_pydantic_model(self, model_type: type[TModel] | type[list[TModel]]) -> TModel | list[TModel]:
        """Parse JSON/YAML content and validate against a Pydantic model. Supports single and list types."""
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

    def _parse_structured(self) -> Any:
        """Parse content as JSON or YAML based on extension. Strict — no guessing."""
        name_lower = self.name.lower()
        if name_lower.endswith(".json"):
            return self.as_json()
        if name_lower.endswith((".yaml", ".yml")):
            return self.as_yaml()
        raise ValueError(f"Cannot parse '{self.name}' as structured data — use .json or .yaml extension")

    def parse(self, type_: type[Any]) -> Any:
        """Parse content to the requested type. Reverses create() conversion. Extension-based dispatch, no guessing."""
        if type_ is bytes:
            return self.content
        if type_ is str:
            return self.text if self.content else ""
        if type_ is dict or type_ is list:
            data = self._parse_structured()
            if not isinstance(data, type_):
                raise ValueError(f"Expected {type_.__name__} but got {type(data).__name__}")
            return data  # pyright: ignore[reportUnknownVariableType]
        if isinstance(type_, type) and issubclass(type_, BaseModel):  # pyright: ignore[reportUnnecessaryIsInstance]
            return self.as_pydantic_model(type_)
        raise ValueError(f"Unsupported parse type: {type_}")

    @property
    def source_documents(self) -> tuple[str, ...]:
        """Document SHA256 hashes from sources (filtered by is_document_sha256)."""
        return tuple(src for src in self.sources if is_document_sha256(src))

    @property
    def source_references(self) -> tuple[str, ...]:
        """Non-hash reference strings from sources (URLs, file paths, etc.)."""
        return tuple(src for src in self.sources if not is_document_sha256(src))

    def has_source(self, source: "Document | str") -> bool:
        """Check if a source (Document or string) is in this document's sources."""
        if isinstance(source, str):
            return source in self.sources
        if not isinstance(source, Document):
            raise TypeError(f"Invalid source type: {type(source)}")  # pyright: ignore[reportUnreachable]
        return source.sha256 in self.sources

    @final
    def serialize_model(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict for storage/transmission. Roundtrips with from_dict().

        Delegates to model_dump() for content serialization (unified format), then adds metadata.
        """
        # Get base serialization from Pydantic (uses field_serializer for content)
        result = self.model_dump(mode="json")

        # Add metadata not present in standard model_dump
        result["id"] = self.id
        result["sha256"] = self.sha256
        result["content_sha256"] = self.content_sha256
        result["size"] = self.size
        result["mime_type"] = self.mime_type
        result["class_name"] = self.__class__.__name__

        # Add metadata to attachments
        for att_dict, att_obj in zip(result.get("attachments", []), self.attachments, strict=False):
            att_dict["mime_type"] = att_obj.mime_type
            att_dict["size"] = att_obj.size

        return result

    @final
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from dict produced by serialize_model(). Roundtrip guarantee.

        Delegates to model_validate() which handles content decoding via field_validator.
        Metadata keys are stripped before validation since custom __init__ receives raw data.
        """
        # Strip metadata keys added by serialize_model() (model_validator mode="before"
        # doesn't work with custom __init__ - Pydantic passes raw data to __init__ first)
        cleaned = {k: v for k, v in data.items() if k not in _DOCUMENT_SERIALIZE_METADATA_KEYS}

        # Strip attachment metadata added by serialize_model()
        if cleaned.get("attachments"):
            cleaned["attachments"] = [{k: v for k, v in att.items() if k not in {"mime_type", "size"}} for att in cleaned["attachments"]]

        return cls.model_validate(cleaned)

    @final
    def model_convert(self, new_type: type[TDocument], *, update: dict[str, Any] | None = None) -> TDocument:
        """Convert to a different Document subclass with optional field overrides."""
        try:
            if not isinstance(new_type, type):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(f"new_type must be a class, got {new_type}")  # pyright: ignore[reportUnreachable]
            if not issubclass(new_type, Document):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(f"new_type must be a subclass of Document, got {new_type}")  # pyright: ignore[reportUnreachable]
        except (TypeError, AttributeError) as err:
            raise TypeError(f"new_type must be a subclass of Document, got {new_type}") from err

        if new_type is Document:
            raise TypeError("Cannot instantiate Document directly — use a concrete subclass")

        data: dict[str, Any] = {  # nosemgrep: mutable-field-on-frozen-pydantic-model
            "name": self.name,
            "content": self.content,
            "description": self.description,
            "sources": self.sources,
            "origins": self.origins,
            "attachments": self.attachments,
        }

        if update:
            data.update(update)

        return new_type(
            name=data["name"],
            content=data["content"],
            description=data.get("description"),
            sources=data.get("sources"),
            origins=data.get("origins"),
            attachments=data.get("attachments"),
        )
