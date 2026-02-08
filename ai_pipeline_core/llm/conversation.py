"""Immutable Conversation class for LLM interactions.

Provides a Document-aware, immutable Conversation class that wraps the
primitive _llm_core functions. All operations return NEW Conversation
instances with response properties derived from the last ModelResponse.

Usage:
    >>> conv = Conversation(model="gpt-5.1", context=[system_doc])
    >>> conv = await conv.send("Hello!")
    >>> print(conv.content)  # "Hi there!"
    >>> conv = await conv.send("Follow up")  # continues conversation
"""

import asyncio
import base64
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Generic, Literal, cast

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import TypeVar

from ai_pipeline_core._llm_core import CoreMessage, ModelOptions, ModelResponse, Role, TokenUsage
from ai_pipeline_core._llm_core import generate as core_generate
from ai_pipeline_core._llm_core import generate_structured as core_generate_structured
from ai_pipeline_core._llm_core.model_response import Citation
from ai_pipeline_core._llm_core.types import ContentPart, ImageContent, PDFContent, TextContent
from ai_pipeline_core.documents import Document
from ai_pipeline_core.logging import get_pipeline_logger

from ._substitutor import URLSubstitutor
from ._validation import validate_image, validate_pdf, validate_text

# Default system prompt injected when substitutor is active with patterns and no user system prompt
_DEFAULT_SUBSTITUTOR_PROMPT = (
    "Text uses ... (three dots) to indicate shortened content. "
    "For example, 0x7a250d56...c659F2488D is a shortened blockchain address, "
    "and https://example.com/very/long/path/to/page...resource.pdf is a shortened URL. "
    "When quoting or referencing such text, preserve the ... markers exactly as shown."
)

logger = get_pipeline_logger(__name__)

# Supported image formats that don't need conversion
_LLM_SUPPORTED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "GIF", "WEBP"})
ImageMimeType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
_FORMAT_TO_MIME: dict[str, ImageMimeType] = {"JPEG": "image/jpeg", "PNG": "image/png", "GIF": "image/gif", "WEBP": "image/webp"}

# Token count per image/PDF (per CLAUDE.md §3.6)
_TOKENS_PER_IMAGE = 1080

T = TypeVar("T", default=None)
U = TypeVar("U", bound=BaseModel)

# Content can be string, Document, or list of Documents
ConversationContent = str | Document | list[Document]

# Messages can be Documents (user/system content) or ModelResponses (assistant responses)
MessageType = Document | ModelResponse[Any]


# Regex matching XML tags whose names are wrapper elements used by document serialization.
# Only these tags are escaped in content bodies — everything else (JSON, YAML, HTML, code) passes through.
_WRAPPER_TAG_RE = re.compile(r"<(/?)(document|content|description|attachment|id|name)\b([^>]*)>", re.IGNORECASE)


def _escape_xml_metadata(text: str) -> str:
    """Escape < and > in metadata fields (names, IDs, descriptions) to prevent tag injection."""
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _escape_xml_content(text: str) -> str:
    """Escape only XML tags matching wrapper element names to prevent structural injection.

    Leaves all other content intact — JSON, YAML, URLs, code, HTML tags, quotes,
    ampersands are never modified. Only tags that could break the <document>/<content>/
    <attachment> wrapper structure are escaped.
    """
    return _WRAPPER_TAG_RE.sub(lambda m: f"&lt;{m.group(1)}{m.group(2)}{m.group(3)}&gt;", text)


def _document_to_xml_header(doc: Document) -> str:
    """Generate XML header for a document with proper escaping."""
    escaped_name = _escape_xml_metadata(doc.name)
    escaped_id = _escape_xml_metadata(doc.id)
    desc = f"<description>{_escape_xml_metadata(doc.description)}</description>\n" if doc.description else ""
    return f"<document>\n<id>{escaped_id}</id>\n<name>{escaped_name}</name>\n{desc}<content>\n"


def _get_image_preset(model: str):
    """Get ImagePreset for a model."""
    from ._images import ImagePreset

    model_lower = model.lower()
    if "gemini" in model_lower:
        return ImagePreset.GEMINI
    if "claude" in model_lower:
        return ImagePreset.CLAUDE
    if "gpt" in model_lower:
        return ImagePreset.GPT4V
    return ImagePreset.DEFAULT


def _image_needs_processing(data: bytes, model: str) -> bool:
    """Check if image exceeds model limits and needs processing."""
    from ._images import ImageProcessingConfig

    preset = _get_image_preset(model)
    config = ImageProcessingConfig.for_preset(preset)

    try:
        with Image.open(BytesIO(data)) as img:
            w, h = img.size
            fmt = img.format
            # Needs processing if exceeds limits OR unsupported format
            return w > config.max_dimension or h > config.max_dimension or w * h > config.max_pixels or fmt not in _LLM_SUPPORTED_IMAGE_FORMATS
    except (OSError, ValueError):
        return True  # Process to validate/fail


def _process_image_to_parts(data: bytes, model: str) -> list[ContentPart]:
    """Process image bytes and return ContentParts.

    Only re-encodes images that exceed model limits. Adds text labels when
    images are split into multiple parts.
    """
    from ._images import ImageProcessingError, process_image

    # Check if image needs processing
    if not _image_needs_processing(data, model):
        # Send as-is, preserving original format/quality
        try:
            with Image.open(BytesIO(data)) as img:
                mime_type: ImageMimeType = _FORMAT_TO_MIME.get(img.format or "", "image/jpeg")
            return cast(list[ContentPart], [ImageContent(data=base64.b64encode(data), mime_type=mime_type)])
        except (OSError, ValueError):
            pass  # Fall through to process_image

    # Process (split/compress) if needed
    try:
        preset = _get_image_preset(model)
        result = process_image(data, preset=preset)
        parts: list[ContentPart] = []

        # Add description if split into multiple parts
        if len(result.parts) > 1:
            parts.append(TextContent(text=f"[Image split into {len(result.parts)} sequential parts with overlap]\n"))

        for img_part in result.parts:
            # Add label for each part if multiple
            if img_part.total > 1:
                parts.append(TextContent(text=f"[{img_part.label}]\n"))
            parts.append(ImageContent(data=base64.b64encode(img_part.data), mime_type="image/webp"))

        return parts
    except ImageProcessingError as e:
        logger.warning(f"Image processing failed: {e}")
        return []


def _document_to_content_parts(doc: Document, model: str) -> list[ContentPart]:
    """Convert a Document to content parts for CoreMessage.

    Validates content before processing and logs warnings for invalid content.
    All text content is XML-escaped to prevent injection attacks.
    Attachments are included inside the document wrapper.
    """
    parts: list[ContentPart] = []
    header = _document_to_xml_header(doc)

    if doc.is_text:
        if err := validate_text(doc.content, doc.name):
            logger.warning(f"Skipping invalid document: {err}")
            return []
        text = _escape_xml_content(doc.content.decode("utf-8"))
        # Build document with attachments INSIDE the wrapper
        # Text attachments are inlined as strings, binary attachments become separate ContentParts
        text_fragments = [f"{header}{text}\n"]
        binary_att_parts: list[ContentPart] = []

        for att in doc.attachments:
            if att.is_text:
                att_content = _build_attachment_content(att)
                if att_content:
                    text_fragments.append(att_content)
            else:
                binary_att_parts.extend(_build_attachment_parts(att, model))

        if binary_att_parts:
            # Mixed content: emit text, then binary parts, then closing tag
            parts.append(TextContent(text="".join(text_fragments)))
            parts.extend(binary_att_parts)
            parts.append(TextContent(text="</content>\n</document>"))
        else:
            # Pure text: single TextContent
            text_fragments.append("</content>\n</document>")
            parts.append(TextContent(text="".join(text_fragments)))

    elif doc.is_image:
        if err := validate_image(doc.content, doc.name):
            logger.warning(f"Skipping invalid document: {err}")
            return []
        parts.append(TextContent(text=header))
        image_parts = _process_image_to_parts(doc.content, model)
        parts.extend(image_parts)

        # Add attachments inside document
        for att in doc.attachments:
            att_parts = _build_attachment_parts(att, model)
            parts.extend(att_parts)

        parts.append(TextContent(text="</content>\n</document>"))

    elif doc.is_pdf:
        if err := validate_pdf(doc.content, doc.name):
            logger.warning(f"Skipping invalid document: {err}")
            return []
        parts.append(TextContent(text=header))
        parts.append(PDFContent(data=base64.b64encode(doc.content)))

        # Add attachments inside document
        for att in doc.attachments:
            att_parts = _build_attachment_parts(att, model)
            parts.extend(att_parts)

        parts.append(TextContent(text="</content>\n</document>"))
    else:
        logger.warning(f"Skipping unsupported document type: {doc.name} - {doc.mime_type}")
        return []

    return parts


def _build_attachment_content(att: Any) -> str | None:
    """Build text content for a text attachment (returns string for embedding)."""
    if not att.is_text:
        return None
    if err := validate_text(att.content, att.name):
        logger.warning(f"Skipping invalid attachment: {err}")
        return None

    escaped_name = _escape_xml_metadata(att.name)
    desc_attr = f' description="{_escape_xml_metadata(att.description)}"' if att.description else ""
    att_text = _escape_xml_content(att.content.decode("utf-8"))
    return f'<attachment name="{escaped_name}"{desc_attr}>\n{att_text}\n</attachment>\n'


def _build_attachment_parts(att: Any, model: str) -> list[ContentPart]:
    """Build content parts for an attachment (for binary attachments)."""
    parts: list[ContentPart] = []
    escaped_name = _escape_xml_metadata(att.name)
    desc_attr = f' description="{_escape_xml_metadata(att.description)}"' if att.description else ""
    att_open = f'<attachment name="{escaped_name}"{desc_attr}>\n'

    if att.is_text:
        if err := validate_text(att.content, att.name):
            logger.warning(f"Skipping invalid attachment: {err}")
            return []
        att_text = _escape_xml_content(att.content.decode("utf-8"))
        parts.append(TextContent(text=f"{att_open}{att_text}\n</attachment>\n"))
    elif att.is_image:
        if err := validate_image(att.content, att.name):
            logger.warning(f"Skipping invalid attachment: {err}")
            return []
        parts.append(TextContent(text=att_open))
        image_parts = _process_image_to_parts(att.content, model)
        parts.extend(image_parts)
        parts.append(TextContent(text="</attachment>\n"))
    elif att.is_pdf:
        if err := validate_pdf(att.content, att.name):
            logger.warning(f"Skipping invalid attachment: {err}")
            return []
        parts.append(TextContent(text=att_open))
        parts.append(PDFContent(data=base64.b64encode(att.content)))
        parts.append(TextContent(text="</attachment>\n"))
    else:
        logger.warning(f"Skipping unsupported attachment type: {att.name} - {att.mime_type}")

    return parts


@dataclass(frozen=True, slots=True)
class _UserMessage:
    """Internal wrapper for user string messages (not a Document)."""

    text: str


# Extended message type that includes internal user messages
_InternalMessageType = Document | ModelResponse[Any] | _UserMessage


def _normalize_content(content: ConversationContent) -> tuple[Document | _UserMessage, ...]:
    """Normalize content to tuple of Documents or internal messages."""
    if isinstance(content, str):
        return (_UserMessage(content),)
    if isinstance(content, Document):
        return (content,)
    return tuple(content)


class Conversation(BaseModel, Generic[T]):
    """Immutable conversation state for LLM interactions.

    After calling send() or send_structured(), the returned Conversation has
    response properties accessible directly (content, reasoning_content, usage,
    cost, parsed, citations).

    Generic parameter T represents the type of `.parsed` from the last
    send_structured() call. For conversations created with send(), T is None.

    Attributes:
        model: The model identifier (e.g., "gpt-5.1", "gemini-3-flash").
        context: Cacheable prefix documents.
        messages: Conversation history (Documents and ModelResponses).
        model_options: Optional model configuration.
        enable_substitutor: Whether to enable URL/address shortening (default True).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model: str
    context: tuple[Document, ...] = ()
    messages: tuple[MessageType | _UserMessage, ...] = ()
    model_options: ModelOptions | None = None
    enable_substitutor: bool = True

    # Substitutor as a regular field, excluded from serialization
    # This allows passing it through constructor without object.__setattr__ hacks
    substitutor: URLSubstitutor | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="before")
    @classmethod
    def _ensure_substitutor(cls, data: Any) -> Any:
        """Auto-create substitutor if enabled and not provided."""
        if isinstance(data, dict):
            # Auto-disable for search models unless explicitly overridden
            explicitly_set = "enable_substitutor" in data
            model_name = data.get("model", "")
            if not explicitly_set and isinstance(model_name, str) and model_name.endswith("-search"):
                data["enable_substitutor"] = False

            enabled = data.get("enable_substitutor", True)
            if enabled and data.get("substitutor") is None:
                data["substitutor"] = URLSubstitutor()
            elif not enabled:
                data["substitutor"] = None
        return data

    @field_validator("model")
    @classmethod
    def validate_model_not_empty(cls, v: str) -> str:
        """Reject empty model name."""
        if not v:
            raise ValueError("model must be non-empty")
        return v

    @field_validator("context", mode="before")
    @classmethod
    def convert_context_to_tuple(cls, v: list[Document] | tuple[Document, ...] | None) -> tuple[Document, ...]:
        """Coerce context list or None to immutable tuple."""
        if v is None:
            return ()
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("messages", mode="before")
    @classmethod
    def convert_messages_to_tuple(cls, v: list[MessageType] | tuple[MessageType, ...] | None) -> tuple[MessageType, ...]:
        """Coerce messages list or None to immutable tuple."""
        if v is None:
            return ()
        if isinstance(v, list):
            return tuple(v)
        return v

    @property
    def _last_response(self) -> ModelResponse[Any] | None:
        """Get the last ModelResponse from messages."""
        for msg in reversed(self.messages):
            if isinstance(msg, ModelResponse):
                return msg
        return None

    @property
    def content(self) -> str:
        """Response text from last send() call."""
        if r := self._last_response:
            return r.content
        return ""

    @property
    def reasoning_content(self) -> str:
        """Reasoning content from last send() call (if model supports it)."""
        if r := self._last_response:
            return r.reasoning_content
        return ""

    @property
    def usage(self) -> TokenUsage:
        """Token usage from last send() call."""
        if r := self._last_response:
            return r.usage
        return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    @property
    def cost(self) -> float | None:
        """Cost from last send() call (if available)."""
        if r := self._last_response:
            return r.cost
        return None

    @property
    def parsed(self) -> T | None:
        """Parsed Pydantic model from last send_structured() call."""
        if r := self._last_response:
            # For ModelResponse[str], parsed is the content string
            # For ModelResponse[SomeModel], parsed is the model instance (or dict after deser)
            if isinstance(r.parsed, str):
                return None  # Unstructured response, no typed parsed
            return r.parsed  # type: ignore[return-value]
        return None

    @property
    def citations(self) -> tuple[Citation, ...]:
        """Citations from last send() call (for search-enabled models)."""
        if r := self._last_response:
            return tuple(r.citations)
        return ()

    def _to_core_messages(self, items: tuple[MessageType | _UserMessage, ...]) -> list[CoreMessage]:
        """Convert Documents, UserMessages, and ModelResponses to CoreMessages for the LLM layer."""
        core_messages: list[CoreMessage] = []

        for item in items:
            if isinstance(item, ModelResponse):
                # Use content as-is (provider fields are preserved in ModelResponse
                # but not re-sent to the model)
                core_messages.append(CoreMessage(role=Role.ASSISTANT, content=item.content))
            elif isinstance(item, _UserMessage):
                # Simple string message from user
                core_messages.append(CoreMessage(role=Role.USER, content=item.text))
            elif isinstance(item, Document):
                if item.name == "system_prompt":
                    core_messages.append(CoreMessage(role=Role.SYSTEM, content=item.text))
                else:
                    parts = _document_to_content_parts(item, self.model)
                    if parts:
                        if len(parts) == 1 and isinstance(parts[0], TextContent):
                            core_messages.append(CoreMessage(role=Role.USER, content=parts[0].text))
                        else:
                            core_messages.append(CoreMessage(role=Role.USER, content=tuple(parts)))

        return core_messages

    def _collect_text_for_substitutor(self, items: tuple[Document | _UserMessage, ...]) -> list[str]:
        """Collect text content from documents and user messages for substitutor preparation."""
        texts: list[str] = []
        for item in items:
            if isinstance(item, _UserMessage) or (isinstance(item, Document) and item.is_text):
                texts.append(item.text)
        return texts

    async def _execute_send(
        self,
        content: ConversationContent,
        response_format: type[BaseModel] | None,
        purpose: str | None,
        expected_cost: float | None,
    ) -> tuple[tuple[MessageType | _UserMessage, ...], ModelResponse[Any]]:
        """Common preparation, LLM call, and response restoration for send methods."""
        docs = _normalize_content(content)
        new_messages = self.messages + docs

        # Prepare substitutor with all text content
        if self.substitutor:
            all_items_for_sub = self.context + tuple(m for m in new_messages if isinstance(m, (Document, _UserMessage)))
            self.substitutor.prepare(self._collect_text_for_substitutor(all_items_for_sub))

        # Inject default substitutor prompt if substitutor is active with patterns and no user system prompt
        effective_options = self.model_options
        if self.substitutor and self.substitutor.pattern_count > 0:
            has_user_prompt = self.model_options is not None and self.model_options.system_prompt is not None
            if not has_user_prompt:
                if self.model_options is not None:
                    effective_options = self.model_options.model_copy(update={"system_prompt": _DEFAULT_SUBSTITUTOR_PROMPT})
                else:
                    effective_options = ModelOptions(system_prompt=_DEFAULT_SUBSTITUTOR_PROMPT)

        # Build CoreMessages in single thread call (CPU-bound image/PDF processing)
        context_core, messages_core = await asyncio.to_thread(lambda: (self._to_core_messages(self.context), self._to_core_messages(new_messages)))
        core_messages = context_core + messages_core
        context_count = len(context_core)

        # Apply substitution if enabled
        if self.substitutor:
            core_messages = self._apply_substitution(core_messages)

        if response_format is not None:
            response: ModelResponse[Any] = await core_generate_structured(
                core_messages,
                response_format,
                model=self.model,
                model_options=effective_options,
                purpose=purpose,
                expected_cost=expected_cost,
                context_count=context_count,
            )
            response = self._restore_response(response, response_format)
        else:
            response = await core_generate(
                core_messages,
                model=self.model,
                model_options=effective_options,
                purpose=purpose,
                expected_cost=expected_cost,
                context_count=context_count,
            )
            response = self._restore_response(response)

        return new_messages, response

    async def send(
        self,
        content: ConversationContent,
        *,
        purpose: str | None = None,
        expected_cost: float | None = None,
    ) -> "Conversation[None]":
        """Send message, returns NEW Conversation with response.

        Args:
            content: Message content - str, Document, or list of Documents.
            purpose: Optional semantic label for tracing.
            expected_cost: Optional expected cost for tracking.

        Returns:
            New Conversation[None] with response accessible via .content, .reasoning_content, etc.
        """
        new_messages, response = await self._execute_send(content, None, purpose, expected_cost)
        return Conversation[None](
            model=self.model,
            context=self.context,
            messages=new_messages + (response,),
            model_options=self.model_options,
            enable_substitutor=self.enable_substitutor,
            substitutor=self.substitutor,
        )

    async def send_structured(
        self,
        content: ConversationContent,
        response_format: type[U],
        *,
        purpose: str | None = None,
        expected_cost: float | None = None,
    ) -> "Conversation[U]":
        """Send message expecting structured response.

        Args:
            content: Message content - str, Document, or list of Documents.
            response_format: Pydantic model class for structured output.
            purpose: Optional semantic label for tracing.
            expected_cost: Optional expected cost for tracking.

        Returns:
            New Conversation[U] with .parsed returning U instance.
        """
        new_messages, response = await self._execute_send(content, response_format, purpose, expected_cost)
        return Conversation[U](
            model=self.model,
            context=self.context,
            messages=new_messages + (response,),
            model_options=self.model_options,
            enable_substitutor=self.enable_substitutor,
            substitutor=self.substitutor,
        )

    def _apply_substitution(self, core_messages: list[CoreMessage]) -> list[CoreMessage]:
        """Apply URL/address substitution to text content in messages."""
        if not self.substitutor:
            return core_messages

        result: list[CoreMessage] = []
        for msg in core_messages:
            if isinstance(msg.content, str):
                substituted = self.substitutor.substitute(msg.content)
                result.append(CoreMessage(role=msg.role, content=substituted))
            elif isinstance(msg.content, tuple):
                new_parts: list[ContentPart] = []
                for part in msg.content:
                    if isinstance(part, TextContent):
                        new_parts.append(TextContent(text=self.substitutor.substitute(part.text)))
                    else:
                        new_parts.append(part)
                result.append(CoreMessage(role=msg.role, content=tuple(new_parts)))
            else:
                result.append(msg)
        return result

    def _restore_response(self, response: ModelResponse[Any], response_format: type[BaseModel] | None = None) -> ModelResponse[Any]:
        """Restore shortened URLs/addresses in LLM response before storing in messages.

        For unstructured responses, updates content and parsed (both strings).
        For structured responses, re-parses the Pydantic model from restored JSON.
        """
        if not self.substitutor or self.substitutor.pattern_count == 0:
            return response
        restored = self.substitutor.restore(response.content)
        if restored == response.content:
            return response
        update: dict[str, Any] = {"content": restored}
        if response_format is not None and not isinstance(response.parsed, str):
            update["parsed"] = response_format.model_validate_json(restored)
        else:
            update["parsed"] = restored
        return response.model_copy(update=update)

    def restore_content(self, text: str) -> str:
        """Restore shortened URLs/addresses in text to originals.

        Use this if you need to restore content that was shortened by the substitutor.
        """
        if self.substitutor:
            return self.substitutor.restore(text)
        return text

    def with_document(self, doc: Document) -> "Conversation[T]":
        """Return NEW Conversation with document appended to messages."""
        return Conversation[T](
            model=self.model,
            context=self.context,
            messages=self.messages + (doc,),
            model_options=self.model_options,
            enable_substitutor=self.enable_substitutor,
            substitutor=self.substitutor,
        )

    def with_context(self, *docs: Document) -> "Conversation[T]":
        """Return NEW Conversation with documents added to context."""
        return Conversation[T](
            model=self.model,
            context=self.context + docs,
            messages=self.messages,
            model_options=self.model_options,
            enable_substitutor=self.enable_substitutor,
            substitutor=self.substitutor,
        )

    def with_model_options(self, options: ModelOptions) -> "Conversation[T]":
        """Return NEW Conversation with updated model options."""
        return Conversation[T](
            model=self.model,
            context=self.context,
            messages=self.messages,
            model_options=options,
            enable_substitutor=self.enable_substitutor,
            substitutor=self.substitutor,
        )

    def with_substitutor(self, enabled: bool = True) -> "Conversation[T]":
        """Return NEW Conversation with substitutor enabled/disabled."""
        return Conversation[T](
            model=self.model,
            context=self.context,
            messages=self.messages,
            model_options=self.model_options,
            enable_substitutor=enabled,
            substitutor=self.substitutor if enabled else None,
        )

    def to_json(self) -> str:
        """Serialize conversation to JSON string for debugging.

        Note: Deserialization is not supported. Conversation is designed for
        transient use within a single task, not for persistence/restore.
        """
        return self.model_dump_json()

    @property
    def approximate_tokens_count(self) -> int:
        """Approximate token count for all context and messages."""
        total = 0
        for item in self.context + self.messages:
            if isinstance(item, ModelResponse):
                total += len(item.content) // 4
                if reasoning := item.reasoning_content:
                    total += len(reasoning) // 4
            elif isinstance(item, Document):
                if item.is_text:
                    total += len(item.content) // 4
                elif item.is_image or item.is_pdf:
                    total += _TOKENS_PER_IMAGE
                for att in item.attachments:
                    if att.is_text:
                        total += len(att.content) // 4
                    elif att.is_image or att.is_pdf:
                        total += _TOKENS_PER_IMAGE
        return total
