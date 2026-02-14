# MODULE: documents
# CLASSES: Attachment, TaskDocumentContext, Document
# DEPENDS: BaseModel
# PURPOSE: Document system for AI pipeline flows.
# SIZE: ~39KB

# === IMPORTS ===
from ai_pipeline_core import Attachment, Document, DocumentSha256, RunContext, RunScope, TaskDocumentContext, get_run_context, is_document_sha256, reset_run_context, sanitize_url, set_run_context

# === TYPES & CONSTANTS ===

EXTENSION_MIME_MAP = {
    "md": "text/markdown",
    "txt": "text/plain",
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "webp": "image/webp",
    "heic": "image/heic",
    "heif": "image/heif",
    "json": "application/json",
    "yaml": "application/yaml",
    "yml": "application/yaml",
    "xml": "text/xml",
    "html": "text/html",
    "htm": "text/html",
    "py": "text/x-python",
    "css": "text/css",
    "js": "application/javascript",
    "ts": "application/typescript",
    "tsx": "application/typescript",
    "jsx": "application/javascript",
}

DATA_URI_PATTERN = re.compile(r"^data:[a-zA-Z0-9.+/-]+;base64,")

DocumentSha256 = NewType("DocumentSha256", str)

RunScope = NewType("RunScope", str)

# === PUBLIC API ===

class Attachment(BaseModel):
    """Immutable binary attachment for multi-part documents.

Carries binary content (screenshots, PDFs, supplementary files) without full Document machinery.
``mime_type`` is a cached_property — not included in ``model_dump()`` output."""
    model_config = ConfigDict(frozen=True, extra='forbid')
    name: str
    content: bytes
    description: str | None = None

    @property
    def is_image(self) -> bool:
        """True if MIME type starts with image/."""
        return is_image_mime_type(self.mime_type)

    @property
    def is_pdf(self) -> bool:
        """True if MIME type is application/pdf."""
        return is_pdf_mime_type(self.mime_type)

    @property
    def is_text(self) -> bool:
        """True if MIME type indicates text content."""
        return is_text_mime_type(self.mime_type)

    @property
    def size(self) -> int:
        """Content size in bytes."""
        return len(self.content)

    @property
    def text(self) -> str:
        """Content decoded as UTF-8. Raises ValueError if not text."""
        if not self.is_text:
            raise ValueError(f"Attachment is not text: {self.name}")
        return self.content.decode("utf-8")

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v: Any) -> bytes:
        """Convert content to bytes.

        Handles:
        1. bytes — passed through directly
        2. str with data URI prefix — base64-decoded to bytes
        3. str (plain text) — UTF-8 encoded to bytes
        """
        if isinstance(v, bytes):
            return v
        if isinstance(v, str):
            # Data URIs are produced by serialize_content() for binary content only (failed UTF-8 decode).
            # Text starting with "data:<mime>;base64," would be misinterpreted, accepted by design.
            if DATA_URI_PATTERN.match(v):
                _, payload = v.split(",", 1)
                return base64.b64decode(payload, validate=True)
            return v.encode("utf-8")
        raise ValueError(f"Invalid content type: {type(v)}")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Reject path traversal, reserved suffixes, whitespace issues."""
        if v.endswith(".description.md"):
            raise DocumentNameError(f"Attachment names cannot end with .description.md: {v}")
        if v.endswith(".sources.json"):
            raise DocumentNameError(f"Attachment names cannot end with .sources.json: {v}")
        if v.endswith(".attachments.json"):
            raise DocumentNameError(f"Attachment names cannot end with .attachments.json: {v}")
        if ".." in v or "\\" in v or "/" in v:
            raise DocumentNameError(f"Invalid attachment name - contains path traversal characters: {v}")
        if not v or v.startswith(" ") or v.endswith(" "):
            raise DocumentNameError(f"Invalid attachment name format: {v}")
        return v

    @cached_property
    def mime_type(self) -> str:
        """Detected MIME type from content and filename. Cached."""
        return detect_mime_type(self.content, self.name)

    @field_serializer("content")
    def serialize_content(self, v: bytes) -> str:
        """Serialize content: plain string for text, data URI (RFC 2397) for binary."""
        try:
            return v.decode("utf-8")
        except UnicodeDecodeError:
            b64 = base64.b64encode(v).decode("ascii")
            return f"data:{self.mime_type};base64,{b64}"


@dataclass
class TaskDocumentContext:
    """Tracks documents created within a single pipeline task or flow execution.

Used by @pipeline_task and @pipeline_flow decorators to:
- Validate that all source/origin SHA256 references point to pre-existing documents
- Detect same-task interdependencies (doc B referencing doc A created in the same task)
- Warn about documents with no provenance (no sources and no origins)
- Detect documents created but not returned (orphaned)
- Deduplicate returned documents by SHA256"""
    created: set[DocumentSha256] = field(default_factory=set)

    @staticmethod
    def deduplicate(documents: list[Document]) -> list[Document]:
        """Deduplicate documents by SHA256, preserving first occurrence order."""
        seen: dict[DocumentSha256, Document] = {}
        for doc in documents:
            if doc.sha256 not in seen:
                seen[doc.sha256] = doc
        return list(seen.values())

    def finalize(self, returned_docs: list[Document]) -> list[str]:
        """Check for documents created but not returned from the task/flow.

        Returns a list of warning messages for orphaned documents — those registered
        via Document.__init__ but not present in the returned result.
        """
        returned_sha256s = {doc.sha256 for doc in returned_docs}
        orphaned = self.created - returned_sha256s
        return [f"Document {sha[:12]}... was created but not returned" for sha in sorted(orphaned)]

    def register_created(self, doc: Document) -> None:
        """Register a document as created in this task/flow context."""
        self.created.add(doc.sha256)

    def validate_provenance(
        self,
        documents: list[Document],
        existing_sha256s: set[DocumentSha256],
        *,
        check_created: bool = False,
    ) -> list[str]:
        """Validate provenance (sources and origins) for returned documents.

        Checks:
        1. All SHA256 source references exist in the store (existing_sha256s).
        2. All origin references exist in the store (existing_sha256s).
        3. No same-task interdependencies: a returned document must not reference
           (via source or origin SHA256) another document created in this same context.
        4. Documents with no sources AND no origins get a warning (no provenance).
        5. (When check_created=True) Returned documents must have been created in
           this context. Only applicable for @pipeline_task — flows delegate creation
           to nested tasks whose documents register in the task's own context.

        Only SHA256-formatted sources are validated; URLs and other reference strings
        in sources are skipped. Initial pipeline inputs (documents with no provenance)
        are acceptable and warned about for awareness.

        Returns a list of warning messages (empty if everything is valid).
        """
        warnings: list[str] = []

        for doc in documents:
            # Check that returned doc was created in this context (task-only)
            if check_created and doc.sha256 not in self.created:
                warnings.append(f"Document '{doc.name}' was not created in this task — only newly created documents should be returned")

            # Check sources
            for src in doc.sources:
                if not is_document_sha256(src):
                    continue
                if src in self.created:
                    warnings.append(f"Document '{doc.name}' references source {src[:12]}... created in the same task (same-task interdependency)")
                elif src not in existing_sha256s:
                    warnings.append(f"Document '{doc.name}' references source {src[:12]}... which does not exist in the store")

            # Check origins
            for origin in doc.origins:
                if origin in self.created:
                    warnings.append(f"Document '{doc.name}' references origin {origin[:12]}... created in the same task (same-task interdependency)")
                elif origin not in existing_sha256s:
                    warnings.append(f"Document '{doc.name}' references origin {origin[:12]}... which does not exist in the store")

            # Warn about no provenance
            if not doc.sources and not doc.origins:
                warnings.append(f"Document '{doc.name}' has no sources and no origins (no provenance)")

        return warnings


class Document(BaseModel):
    """Immutable base class for all pipeline documents. Cannot be instantiated directly — must be subclassed.

Content is stored as bytes. Use `create()` for automatic conversion from str/dict/list/BaseModel.
Use `parse()` to reverse the conversion. Serialization is extension-driven (.json → JSON, .yaml → YAML)."""
    MAX_CONTENT_SIZE: ClassVar[int] = 25 * 1024 * 1024
    name: str
    description: str | None = None
    content: bytes
    sources: tuple[str, ...] = ()
    origins: tuple[str, ...] = ()
    attachments: tuple[Attachment, ...] = ()
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra='forbid')

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

    @final
    @property
    def id(self) -> str:
        """First 6 chars of sha256. Used as short document identifier in LLM context."""
        return self.sha256[:6]

    @property
    def is_image(self) -> bool:
        """True if MIME type starts with image/."""
        return is_image_mime_type(self.mime_type)

    @property
    def is_pdf(self) -> bool:
        """True if MIME type is application/pdf."""
        return is_pdf_mime_type(self.mime_type)

    @property
    def is_text(self) -> bool:
        """True if MIME type indicates text content."""
        return is_text_mime_type(self.mime_type)

    @final
    @property
    def size(self) -> int:
        """Total size of content + attachments in bytes."""
        return len(self.content) + sum(att.size for att in self.attachments)

    @property
    def source_documents(self) -> tuple[str, ...]:
        """Document SHA256 hashes from sources (filtered by is_document_sha256)."""
        return tuple(src for src in self.sources if is_document_sha256(src))

    @property
    def source_references(self) -> tuple[str, ...]:
        """Non-hash reference strings from sources (URLs, file paths, etc.)."""
        return tuple(src for src in self.sources if not is_document_sha256(src))

    @property
    def text(self) -> str:
        """Content decoded as UTF-8. Raises ValueError if not text."""
        if not self.is_text:
            raise ValueError(f"Document is not text: {self.name}")
        return self.content.decode("utf-8")

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
            cleaned["attachments"] = [{k: v for k, v in att.items() if k not in Attachment._SERIALIZE_METADATA_KEYS} for att in cleaned["attachments"]]

        return cls.model_validate(cleaned)

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
                v = base64.b64decode(payload, validate=True)
            else:
                v = v.encode("utf-8")
        else:
            name = info.data.get("name", "") if hasattr(info, "data") else ""
            v = _convert_content(name, v)
        if len(v) > cls.MAX_CONTENT_SIZE:
            raise DocumentSizeError(f"Document size ({len(v)} bytes) exceeds maximum allowed size ({cls.MAX_CONTENT_SIZE} bytes)")
        return v

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

    @field_validator("origins")
    @classmethod
    def validate_origins(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        """Origins must be valid document SHA256 hashes."""
        for origin in v:
            if not is_document_sha256(origin):
                raise ValueError(f"Origin must be a document SHA256 hash, got: {origin}")
        return v

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        """Sources must be document SHA256 hashes or URLs."""
        for src in v:
            if not is_document_sha256(src) and "://" not in src:
                raise ValueError(f"Source must be a document SHA256 hash or a URL (containing '://'), got: {src!r}")
        return v

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

    def as_json(self) -> Any:
        """Parse content as JSON."""
        return json.loads(self.text)

    @overload
    def as_pydantic_model(self, model_type: type[TModel]) -> TModel: ...

    def as_yaml(self) -> Any:
        """Parse content as YAML via ruamel.yaml."""
        yaml = YAML()
        return yaml.load(self.text)  # type: ignore[no-untyped-call, no-any-return]

    @final
    @cached_property
    def content_sha256(self) -> str:
        """SHA256 hash of raw content bytes only. Used for content deduplication."""
        return compute_content_sha256(self.content)

    def has_source(self, source: "Document | str") -> bool:
        """Check if a source (Document or string) is in this document's sources."""
        if isinstance(source, str):
            return source in self.sources
        if not isinstance(source, Document):
            raise TypeError(f"Invalid source type: {type(source)}")  # pyright: ignore[reportUnreachable]
        return source.sha256 in self.sources

    @cached_property
    def mime_type(self) -> str:
        """Detected MIME type. Extension-based for known formats, content analysis for others. Cached."""
        return detect_mime_type(self.content, self.name)

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

    @field_serializer("content")
    def serialize_content(self, v: bytes) -> str:
        """Serialize content: plain string for text, data URI (RFC 2397) for binary."""
        try:
            return v.decode("utf-8")
        except UnicodeDecodeError:
            b64 = base64.b64encode(v).decode("ascii")
            return f"data:{self.mime_type};base64,{b64}"

    @final
    def serialize_model(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict for storage/transmission. Roundtrips with from_dict().

        Delegates to model_dump() for content serialization (unified format), then adds metadata.
        """
        # Get base serialization from Pydantic (uses field_serializer for content)
        result = self.model_dump(mode="json")

        # Add metadata not present in standard model_dump (keys must match _DOCUMENT_SERIALIZE_METADATA_KEYS, used by from_dict() to strip them)
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
    @cached_property
    def sha256(self) -> DocumentSha256:
        """Full SHA256 identity hash (name + content + sources + origins + attachments). BASE32 encoded, cached."""
        return compute_document_sha256(self)

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


# === FUNCTIONS ===

@functools.cache
def get_tiktoken_encoding() -> tiktoken.Encoding:
    """Lazy-cached tiktoken encoding. Deferred to first use, cached forever."""
    return tiktoken.encoding_for_model("gpt-4")

def detect_mime_type(content: bytes, name: str) -> str:
    r"""Detect MIME type from document content and filename.

    Uses a multi-stage detection strategy for maximum accuracy:
    1. Returns 'text/plain' for empty content
    2. Uses extension-based detection for known formats (most reliable)
    3. Falls back to python-magic content analysis
    4. Final fallback to extension or 'application/octet-stream'

    Args:
        content: Document content as bytes.
        name: Filename with extension.

    Returns:
        MIME type string (e.g., 'text/plain', 'application/json').
        Never returns None or empty string.

    Fallback behavior:
        - Empty content: 'text/plain'
        - Unknown extension with binary content: 'application/octet-stream'
        - Magic library failure: Falls back to extension or 'application/octet-stream'

    Performance:
        Only the first 1024 bytes are analyzed for content detection.
        Extension-based detection is O(1) lookup.

    Extension-based detection is preferred for text formats as
    content analysis can sometimes misidentify structured text.
    """
    # Check for empty content
    if len(content) == 0:
        return "text/plain"

    # Try extension-based detection first for known formats
    # This is more reliable for text formats that magic might misidentify
    ext = name.lower().split(".")[-1] if "." in name else ""
    if ext in EXTENSION_MIME_MAP:
        return EXTENSION_MIME_MAP[ext]

    # Try content-based detection with magic
    try:
        mime = magic.from_buffer(content[:1024], mime=True)
        # If magic returns a valid mime type, use it
        if mime and mime != "application/octet-stream":
            return mime
    except (AttributeError, OSError, magic.MagicException) as e:
        logger.warning(f"MIME detection failed for {name}: {e}")
    except Exception:
        logger.exception(f"Unexpected error in MIME detection for {name}")

    # Final fallback based on extension or default
    return EXTENSION_MIME_MAP.get(ext, "application/octet-stream")

def is_text_mime_type(mime_type: str) -> bool:
    """Check if MIME type represents text-based content.

    Determines if content can be safely decoded as text.
    Includes common text formats and structured text like JSON/YAML.

    Args:
        mime_type: MIME type string to check.

    Returns:
        True if MIME type indicates text content, False otherwise.

    Recognized as text:
        - Any type starting with 'text/'
        - application/json
        - application/xml
        - application/javascript
        - application/yaml
        - application/x-yaml

    """
    text_types = [
        "text/",
        "application/json",
        "application/xml",
        "application/javascript",
        "application/yaml",
        "application/x-yaml",
    ]
    return any(mime_type.startswith(t) for t in text_types)

def is_json_mime_type(mime_type: str) -> bool:
    """Check if MIME type is JSON.

    Args:
        mime_type: MIME type string to check.

    Returns:
        True if MIME type is 'application/json', False otherwise.

    Only matches exact 'application/json', not variants like
    'application/ld+json' or 'application/vnd.api+json'.
    """
    return mime_type == "application/json"

def is_yaml_mime_type(mime_type: str) -> bool:
    """Check if MIME type is YAML.

    Recognizes both standard YAML MIME types.

    Args:
        mime_type: MIME type string to check.

    Returns:
        True if MIME type is YAML, False otherwise.

    Recognized types:
        - application/yaml (standard)
        - application/x-yaml (legacy)

    """
    return mime_type in {"application/yaml", "application/x-yaml"}

def is_pdf_mime_type(mime_type: str) -> bool:
    """Check if MIME type is PDF.

    Args:
        mime_type: MIME type string to check.

    Returns:
        True if MIME type is 'application/pdf', False otherwise.

    PDF documents require special handling in the LLM module
    and are supported by certain vision-capable models.
    """
    return mime_type == "application/pdf"

def is_image_mime_type(mime_type: str) -> bool:
    """Check if MIME type represents an image.

    Args:
        mime_type: MIME type string to check.

    Returns:
        True if MIME type starts with 'image/', False otherwise.

    Recognized formats:
        Any MIME type starting with 'image/' including:
        - image/png
        - image/jpeg
        - image/gif
        - image/webp
        - image/svg+xml

    Image documents are automatically encoded for vision-capable
    LLM models in the AIMessages.document_to_prompt() method.
    """
    return mime_type.startswith("image/")

def is_llm_supported_image(mime_type: str) -> bool:
    """Check if MIME type is an image format directly supported by LLMs.

    Unsupported image formats (gif, bmp, tiff, svg, etc.) need conversion
    to PNG before sending to the LLM.

    Args:
        mime_type: MIME type string to check.

    Returns:
        True if the image format is natively supported by LLMs.
    """
    return mime_type in LLM_SUPPORTED_IMAGE_MIME_TYPES

def sanitize_url(url: str) -> str:
    """Sanitize URL or query string for use in filenames.

    Removes or replaces characters that are invalid in filenames.

    Args:
        url: The URL or query string to sanitize.

    Returns:
        A sanitized string safe for use as a filename.
    """
    # Remove protocol if it's a URL
    if url.startswith(("http://", "https://")):
        parsed = urlparse(url)
        # Use domain + path
        url = parsed.netloc + parsed.path

    # Replace invalid filename characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", url)

    # Replace multiple underscores with single one
    sanitized = re.sub(r"_+", "_", sanitized)

    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip("_.")

    # Limit length to prevent too long filenames
    if len(sanitized) > 100:
        sanitized = sanitized[:100]

    # Ensure we have something
    if not sanitized:
        sanitized = "unnamed"

    return sanitized

def camel_to_snake(name: str) -> str:
    """Convert CamelCase (incl. acronyms) to snake_case.

    Args:
        name: The CamelCase string to convert.

    Returns:
        The converted snake_case string.
    """
    s1 = re.sub(r"(.)([A-Z][a-z0-9]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.replace("__", "_").strip("_").lower()

def is_document_sha256(value: str) -> bool:
    """Check if a string is a valid base32-encoded SHA256 hash with proper entropy.

    This function validates that a string is not just formatted like a SHA256 hash,
    but actually has the entropy characteristics of a real hash. It checks:
    1. Correct length (52 characters without padding)
    2. Valid base32 characters (A-Z, 2-7)
    3. Sufficient entropy (at least 8 unique characters)

    The entropy check prevents false positives like 'AAAAAAA...AAA' from being
    identified as valid document hashes.

    Args:
        value: String to check if it's a document SHA256 hash.

    Returns:
        True if the string appears to be a real base32-encoded SHA256 hash,
        False otherwise.

    Examples:
        >>> # Real SHA256 hash
        >>> is_document_sha256("P3AEMA2PSYILKFYVBUALJLMIYWVZIS2QDI3S5VTMD2X7SOODF2YQ")
        True

        >>> # Too uniform - lacks entropy
        >>> is_document_sha256("A" * 52)
        False

        >>> # Wrong length
        >>> is_document_sha256("ABC123")
        False

        >>> # Invalid characters
        >>> is_document_sha256("a" * 52)  # lowercase
        False
    """
    if not isinstance(value, str) or len(value) != 52:  # pyright: ignore[reportUnnecessaryIsInstance]
        return False

    # Check if all characters are valid base32 (A-Z, 2-7)
    if not re.match(r"^[A-Z2-7]{52}$", value):
        return False

    # Check entropy: real SHA256 hashes have high entropy
    # Require at least 8 unique characters (out of 32 possible in base32)
    # This prevents patterns like "AAAAAAA..." from being identified as real hashes
    unique_chars = len(set(value))
    return unique_chars >= 8

# === EXAMPLES (from tests/) ===

# Example: Attachment mime type
# Source: tests/documents/test_document_core.py:924
def test_attachment_mime_type(self):
    """Attachment.mime_type returns detected MIME type."""
    att = Attachment(name="notes.txt", content=b"Hello")
    assert "text" in att.mime_type

# Example: Attachment no detected mime type
# Source: tests/documents/test_document_core.py:929
def test_attachment_no_detected_mime_type(self):
    """Attachment has no detected_mime_type attribute (renamed to mime_type)."""
    att = Attachment(name="notes.txt", content=b"Hello")
    assert not hasattr(att, "detected_mime_type")

# Example: Attachment order does not affect hash
# Source: tests/documents/test_document_attachments.py:62
def test_attachment_order_does_not_affect_hash(self):
    """Attachments are sorted by name before hashing, so order doesn't matter."""
    att_x = Attachment(name="x.txt", content=b"xxx")
    att_y = Attachment(name="y.txt", content=b"yyy")
    doc_xy = SampleFlowDoc(name="test.txt", content=b"Hello", attachments=(att_x, att_y))
    doc_yx = SampleFlowDoc(name="test.txt", content=b"Hello", attachments=(att_y, att_x))
    assert doc_xy.sha256 == doc_yx.sha256

# Example: Attachment order does not matter
# Source: tests/documents/test_hashing.py:41
def test_attachment_order_does_not_matter(self):
    """Attachments are sorted by name before hashing."""
    att_a = Attachment(name="a.txt", content=b"aaa")
    att_b = Attachment(name="b.txt", content=b"bbb")
    doc1 = HashDoc.create(name="doc.txt", content="content", attachments=(att_a, att_b))
    doc2 = HashDoc.create(name="doc.txt", content="content", attachments=(att_b, att_a))
    assert compute_document_sha256(doc1) == compute_document_sha256(doc2)

# Example: Document mime type
# Source: tests/documents/test_document_core.py:914
def test_document_mime_type(self):
    """Document.mime_type returns detected MIME type."""
    doc = ConcreteTestDocument.create(name="data.json", content={"key": "value"})
    assert doc.mime_type == "application/json"

# === ERROR EXAMPLES (What NOT to Do) ===

# Error: Cannot convert to document
# Source: tests/documents/test_document_model_convert.py:145
def test_cannot_convert_to_document(self):
    """Test that converting to base Document class raises error."""
    doc = SampleTaskDoc.create(name="test.json", content={})

    with pytest.raises(TypeError):
        doc.model_convert(Document)

# Error: Cannot convert to non document
# Source: tests/documents/test_document_model_convert.py:152
def test_cannot_convert_to_non_document(self):
    """Test that converting to non-Document class raises error."""
    doc = SampleTaskDoc.create(name="test.json", content={})

    with pytest.raises(TypeError, match="must be a subclass of Document"):
        doc.model_convert(dict)  # type: ignore

    with pytest.raises(TypeError, match="must be a subclass of Document"):
        doc.model_convert(str)  # type: ignore

# Error: Cannot instantiate document
# Source: tests/documents/test_document_core.py:120
def test_cannot_instantiate_document(self):
    """Test that Document cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Cannot instantiate Document directly"):
        Document(name="test.txt", content=b"test")

# Error: Content plus attachments exceeding limit
# Source: tests/documents/test_document_core.py:1017
def test_content_plus_attachments_exceeding_limit(self):
    """Content + attachments exceeding MAX_CONTENT_SIZE is rejected by model_validator."""
    # Content is 7 bytes (under 10-byte limit), but total with attachment is 12
    with pytest.raises(DocumentSizeError, match="including attachments"):
        SmallDocument(
            name="test.txt",
            content=b"1234567",  # 7 bytes
            attachments=(Attachment(name="a.txt", content=b"12345"),),  # 5 bytes => total 12
        )

# Error: Content plus attachments exceeding limit rejected
# Source: tests/documents/test_document_attachments.py:137
def test_content_plus_attachments_exceeding_limit_rejected(self):
    with pytest.raises(DocumentSizeError, match="including attachments"):
        SmallLimitDoc(
            name="test.txt",
            content=b"A" * 30,  # 30 bytes
            attachments=(Attachment(name="a.txt", content=b"B" * 25),),  # total 55 > 50
        )
