"""Immutable Conversation class for LLM interactions.

Provides a Document-aware, immutable Conversation class that wraps the
primitive _llm_core functions. All operations return NEW Conversation
instances with response properties derived from the last ModelResponse.
"""

import asyncio
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Any, Generic, cast, overload

from lmnr import Laminar
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing_extensions import TypeVar

from ai_pipeline_core._llm_core import CoreMessage, ModelOptions, ModelResponse, Role, TokenUsage
from ai_pipeline_core._llm_core import generate as core_generate
from ai_pipeline_core._llm_core import generate_structured as core_generate_structured
from ai_pipeline_core._llm_core._validation import validate_text
from ai_pipeline_core._llm_core.model_response import Citation
from ai_pipeline_core._llm_core.types import TOKENS_PER_IMAGE, ContentPart, ImageContent, PDFContent, TextContent
from ai_pipeline_core.documents import Document
from ai_pipeline_core.llm._images import validated_binary_parts
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.prompt_compiler.render import RESULT_CLOSE, _extract_result, render_multi_line_messages, render_text
from ai_pipeline_core.prompt_compiler.spec import PromptSpec

from ._substitutor import URLSubstitutor

__all__ = [
    "Conversation",
    "ConversationContent",
]

# Instruction appended to system prompt when substitutor is active with patterns
_SUBSTITUTOR_INSTRUCTION = (
    "Text uses ... (three dots) to indicate shortened content. "
    "For example, 0x7a250d56...c659F2488D is a shortened blockchain address, "
    "and https://example.com/very/long/path/to/page...resource.pdf is a shortened URL. "
    "When quoting or referencing such text, preserve entire url or address with the ... markers exactly as shown. "
    "Never create shortened content yourself, you can only reuse existing one."
)

logger = get_pipeline_logger(__name__)

# Document name sentinel for system prompt documents — treated as role=SYSTEM in messages
SYSTEM_PROMPT_DOCUMENT_NAME = "system_prompt"

CHARS_PER_TOKEN = 4

T = TypeVar("T", default=None)
U = TypeVar("U", bound=BaseModel)

ConversationContent = str | Document | list[Document]
"""Content accepted by send() and send_structured(): plain text, a Document, or a list of Documents."""

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


def _document_to_content_parts(doc: Document, model: str) -> list[ContentPart]:
    """Convert a Document to content parts for CoreMessage."""
    parts: list[ContentPart] = []
    header = _document_to_xml_header(doc)

    if doc.is_text:
        if err := validate_text(doc.content, doc.name):
            logger.warning("Skipping invalid document: %s", err)
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

    elif doc.is_image or doc.is_pdf:
        binary_parts = validated_binary_parts(doc.content, doc.name, is_image=doc.is_image, model=model)
        if binary_parts is None:
            return []
        parts.append(TextContent(text=header))
        parts.extend(binary_parts)

        for att in doc.attachments:
            parts.extend(_build_attachment_parts(att, model))

        parts.append(TextContent(text="</content>\n</document>"))
    else:
        logger.warning("Skipping unsupported document type: %s - %s", doc.name, doc.mime_type)
        return []

    return parts


def _build_attachment_content(att: Any) -> str | None:
    """Build text content for a text attachment (returns string for embedding)."""
    if not att.is_text:
        return None
    if err := validate_text(att.content, att.name):
        logger.warning("Skipping invalid attachment: %s", err)
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
            logger.warning("Skipping invalid attachment: %s", err)
            return []
        att_text = _escape_xml_content(att.content.decode("utf-8"))
        parts.append(TextContent(text=f"{att_open}{att_text}\n</attachment>\n"))
    elif att.is_image or att.is_pdf:
        binary_parts = validated_binary_parts(att.content, att.name, is_image=att.is_image, model=model)
        if binary_parts is None:
            return []
        parts.append(TextContent(text=att_open))
        parts.extend(binary_parts)
        parts.append(TextContent(text="</attachment>\n"))
    else:
        logger.warning("Skipping unsupported attachment type: %s - %s", att.name, att.mime_type)

    return parts


@dataclass(frozen=True, slots=True)
class _UserMessage:
    """Internal wrapper for user string messages (not a Document)."""

    text: str


@dataclass(frozen=True, slots=True)
class _AssistantMessage:
    """Internal wrapper for injected assistant messages (not from an LLM call)."""

    text: str


# Union of all types that can appear in messages tuple
_AnyMessage = Document | ModelResponse[Any] | _UserMessage | _AssistantMessage


def _normalize_content(content: ConversationContent) -> tuple[Document | _UserMessage, ...]:
    """Normalize content to tuple of Documents or internal messages."""
    if isinstance(content, str):
        return (_UserMessage(content),)
    if isinstance(content, Document):
        return (content,)
    return tuple(content)


class Conversation(BaseModel, Generic[T]):
    """Immutable conversation state for LLM interactions.

    Every send()/send_structured() call returns a NEW Conversation instance.
    Never discard the return value — the original Conversation is unchanged.

    Images in Documents are automatically processed per model preset (splitting, downscaling).

    Content protection (URLs, addresses, high-entropy strings) is enabled by default,
    auto-disabled for `-search` suffix models. Both `.content` and `.parsed` are
    eagerly restored after each send.

    Attachment rendering in LLM context:
    - Text attachments: wrapped in <attachment name="..." description="..."> tags
    - Binary attachments (images, PDFs): inserted as separate content parts
    """

    model_config = ConfigDict(frozen=True)

    model: str
    context: tuple[Document, ...] = ()
    messages: tuple[_AnyMessage, ...] = ()
    model_options: ModelOptions | None = None
    enable_substitutor: bool = True
    extract_result_tags: bool = False

    @model_validator(mode="before")
    @classmethod
    def _disable_substitutor_for_search(cls, data: Any) -> Any:
        """Auto-disable substitutor for search models unless explicitly enabled."""
        if isinstance(data, dict):
            d = cast(dict[str, Any], data)
            if "enable_substitutor" not in d:
                model_name = d.get("model", "")
                if isinstance(model_name, str) and model_name.endswith("-search"):
                    d["enable_substitutor"] = False
        return data  # pyright: ignore[reportUnknownVariableType]

    @field_validator("model")
    @classmethod
    def validate_model_not_empty(cls, v: str) -> str:
        """Reject empty model name."""
        if not v:
            raise ValueError("model must be non-empty")
        return v

    @field_validator("context", "messages", mode="before")
    @classmethod
    def _coerce_to_tuple(cls, v: list[Any] | tuple[Any, ...] | None) -> tuple[Any, ...]:
        """Coerce list or None to immutable tuple."""
        if v is None:
            return ()
        if isinstance(v, list):
            return tuple(v)
        return v

    # --- Response properties (delegate to last ModelResponse) ---

    @property
    def _last_response(self) -> ModelResponse[Any] | None:
        """Get the last ModelResponse from messages."""
        for msg in reversed(self.messages):
            if isinstance(msg, ModelResponse):
                return msg
        return None

    @property
    def content(self) -> str:
        """Response text from last send() call.

        When extract_result_tags is True, strips <result>...</result> tags
        from the raw response (used by send_spec with output_structure).
        """
        if r := self._last_response:
            return _extract_result(r.content) if self.extract_result_tags else r.content
        return ""

    @property
    def reasoning_content(self) -> str:
        """Reasoning content from last send() call (if model supports it)."""
        return r.reasoning_content if (r := self._last_response) else ""

    @property
    def usage(self) -> TokenUsage:
        """Token usage from last send() call."""
        return r.usage if (r := self._last_response) else TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    @property
    def cost(self) -> float | None:
        """Cost from last send() call (if available)."""
        return r.cost if (r := self._last_response) else None

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
        return tuple(r.citations) if (r := self._last_response) else ()

    # --- Core message conversion ---

    def _to_core_messages(self, items: tuple[_AnyMessage, ...]) -> list[CoreMessage]:
        """Convert Documents, UserMessages, AssistantMessages, and ModelResponses to CoreMessages."""
        core_messages: list[CoreMessage] = []

        for item in items:
            if isinstance(item, ModelResponse):
                core_messages.append(CoreMessage(role=Role.ASSISTANT, content=item.content))
            elif isinstance(item, _AssistantMessage):
                core_messages.append(CoreMessage(role=Role.ASSISTANT, content=item.text))
            elif isinstance(item, _UserMessage):
                core_messages.append(CoreMessage(role=Role.USER, content=item.text))
            elif isinstance(item, Document):  # pyright: ignore[reportUnnecessaryIsInstance]
                if item.name == SYSTEM_PROMPT_DOCUMENT_NAME:
                    core_messages.append(CoreMessage(role=Role.SYSTEM, content=item.text))
                else:
                    parts = _document_to_content_parts(item, self.model)
                    if parts:
                        if len(parts) == 1 and isinstance(parts[0], TextContent):
                            core_messages.append(CoreMessage(role=Role.USER, content=parts[0].text))
                        else:
                            core_messages.append(CoreMessage(role=Role.USER, content=tuple(parts)))

        return core_messages

    @staticmethod
    def _core_messages_to_span_input(messages: list[CoreMessage]) -> list[dict[str, str | list[dict[str, str]]]]:
        """Convert CoreMessages to Laminar-compatible chat format, replacing binary content with placeholders."""
        result: list[dict[str, str | list[dict[str, str]]]] = []
        for msg in messages:
            role = msg.role.value
            if isinstance(msg.content, str):
                result.append({"role": role, "content": msg.content})
            elif isinstance(msg.content, tuple):
                parts: list[dict[str, str]] = []
                for part in msg.content:
                    if isinstance(part, TextContent):
                        parts.append({"type": "text", "text": part.text})
                    elif isinstance(part, ImageContent):
                        parts.append({"type": "text", "text": "[image]"})
                    elif isinstance(part, PDFContent):  # pyright: ignore[reportUnnecessaryIsInstance]
                        parts.append({"type": "text", "text": "[pdf]"})
                result.append({"role": role, "content": parts})
            else:
                result.append({"role": role, "content": str(msg.content)})
        return result

    # --- Substitution ---

    @staticmethod
    def _collect_text(items: tuple[_AnyMessage, ...]) -> list[str]:
        """Collect text content from documents and messages for substitutor preparation."""
        texts: list[str] = []
        for item in items:
            if isinstance(item, (_UserMessage, _AssistantMessage)) or (isinstance(item, Document) and item.is_text):
                texts.append(item.text)
        return texts

    @staticmethod
    def _apply_substitution(core_messages: list[CoreMessage], substitutor: URLSubstitutor) -> list[CoreMessage]:
        """Apply URL/address substitution to text content in messages."""
        result: list[CoreMessage] = []
        for msg in core_messages:
            if isinstance(msg.content, str):
                result.append(CoreMessage(role=msg.role, content=substitutor.substitute(msg.content)))
            elif isinstance(msg.content, tuple):
                new_parts: list[ContentPart] = []
                for part in msg.content:
                    if isinstance(part, TextContent):
                        new_parts.append(TextContent(text=substitutor.substitute(part.text)))
                    else:
                        new_parts.append(part)
                result.append(CoreMessage(role=msg.role, content=tuple(new_parts)))
            else:
                result.append(msg)
        return result

    @staticmethod
    def _restore_response(response: ModelResponse[Any], substitutor: URLSubstitutor, response_format: type[BaseModel] | None = None) -> ModelResponse[Any]:
        """Restore shortened URLs/addresses in LLM response."""
        if substitutor.pattern_count == 0:
            return response
        restored = substitutor.restore(response.content)
        if restored == response.content:
            return response
        update: dict[str, Any] = {"content": restored}
        if response_format is not None and not isinstance(response.parsed, str):
            update["parsed"] = response_format.model_validate_json(restored)
        else:
            update["parsed"] = restored
        return response.model_copy(update=update)  # nosemgrep: no-document-model-copy

    # --- Replay payload ---

    def _build_replay_payload(
        self,
        content: ConversationContent,
        response_format: type[BaseModel] | None,
        purpose: str | None,
        response: ModelResponse[Any],
    ) -> dict[str, Any]:
        """Build a replay payload dict capturing the full conversation state."""
        from ai_pipeline_core.replay._capture import build_conversation_replay_payload

        return build_conversation_replay_payload(
            content=content,
            response_format=response_format,
            purpose=purpose,
            response=response,
            context=self.context,
            model=self.model,
            model_options=self.model_options,
            messages=self.messages,
            enable_substitutor=self.enable_substitutor,
            extract_result_tags=self.extract_result_tags,
        )

    # --- Send methods ---

    async def _execute_send(
        self,
        content: ConversationContent,
        response_format: type[BaseModel] | None,
        purpose: str | None,
        expected_cost: float | None,
    ) -> tuple[tuple[_AnyMessage, ...], ModelResponse[Any]]:
        """Common preparation, LLM call, and response restoration for send methods."""
        docs = _normalize_content(content)
        new_messages = self.messages + docs

        # Prepare substitutor fresh each call — no mutable state stored on Conversation
        substitutor: URLSubstitutor | None = None
        if self.enable_substitutor:
            substitutor = URLSubstitutor()
            all_items = self.context + tuple(m for m in new_messages if isinstance(m, (Document, _UserMessage, _AssistantMessage)))
            substitutor.prepare(self._collect_text(all_items))

        # Adjust system prompt if substitution found patterns to shorten
        effective_options = self.model_options
        if substitutor and substitutor.pattern_count > 0:
            user_prompt = self.model_options.system_prompt if self.model_options else None
            combined = f"{user_prompt}\n\n{_SUBSTITUTOR_INSTRUCTION}" if user_prompt else _SUBSTITUTOR_INSTRUCTION
            effective_options = (self.model_options or ModelOptions()).model_copy(update={"system_prompt": combined})  # nosemgrep: no-document-model-copy

        # Build CoreMessages in thread (CPU-bound image/PDF processing)
        context_core, messages_core = await asyncio.to_thread(lambda: (self._to_core_messages(self.context), self._to_core_messages(new_messages)))
        core_messages = context_core + messages_core
        context_count = len(context_core)

        # Trace the full send operation — input captures pre-substitution content
        span_name = purpose or f"conversation.{'send_structured' if response_format else 'send'}"
        span_input = self._core_messages_to_span_input(core_messages)
        with Laminar.start_as_current_span(f"{span_name}:{self.model}", input=span_input) as span:
            if substitutor:
                core_messages = self._apply_substitution(core_messages, substitutor)

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
            else:
                response = await core_generate(
                    core_messages,
                    model=self.model,
                    model_options=effective_options,
                    purpose=purpose,
                    expected_cost=expected_cost,
                    context_count=context_count,
                )

            if substitutor:
                response = self._restore_response(response, substitutor, response_format)

            Laminar.set_span_output([response.content])

            span_attrs = response.get_laminar_metadata()
            if response.reasoning_content:
                span_attrs["reasoning_content"] = response.reasoning_content
            if response.citations:
                span_attrs["citations"] = json.dumps(
                    [{"title": c.title, "url": c.url} for c in response.citations],
                    indent=2,
                )
            if purpose:
                span_attrs["purpose"] = purpose
            span.set_attributes(span_attrs)  # pyright: ignore[reportArgumentType]

            # Build and attach replay payload for trace-based replay
            try:
                span.set_attribute(
                    "replay.payload",
                    json.dumps(
                        self._build_replay_payload(content, response_format, purpose, response),
                    ),
                )
            except Exception:
                logger.debug("Failed to build replay payload", exc_info=True)

            return new_messages, response

    async def send(
        self,
        content: ConversationContent,
        *,
        purpose: str | None = None,
        expected_cost: float | None = None,
    ) -> "Conversation[None]":
        """Send message, returns NEW Conversation with response.

        Document content is wrapped in <document> XML tags with id, name, description.
        """
        new_messages, response = await self._execute_send(content, None, purpose, expected_cost)
        return self.model_copy(update={"messages": new_messages + (response,)})  # type: ignore[return-value]

    async def send_structured(
        self,
        content: ConversationContent,
        response_format: type[U],
        *,
        purpose: str | None = None,
        expected_cost: float | None = None,
    ) -> "Conversation[U]":
        """Send message expecting structured response, returns NEW Conversation[U] with .parsed.

        Quality degrades beyond ~2-3K output tokens or nesting >2 levels.
        Never use dict types in response_format — use lists of typed models.
        Split complex structures across multiple calls.
        """
        new_messages, response = await self._execute_send(content, response_format, purpose, expected_cost)
        return self.model_copy(update={"messages": new_messages + (response,)})  # type: ignore[return-value]

    @overload
    async def send_spec(
        self,
        spec: PromptSpec[str],
        *,
        documents: Sequence[Document] | None = None,
        include_input_documents: bool = True,
        purpose: str | None = None,
        expected_cost: float | None = None,
    ) -> "Conversation[None]": ...

    @overload
    async def send_spec(
        self,
        spec: PromptSpec[U],
        *,
        documents: Sequence[Document] | None = None,
        include_input_documents: bool = True,
        purpose: str | None = None,
        expected_cost: float | None = None,
    ) -> "Conversation[U]": ...

    async def send_spec(
        self,
        spec: PromptSpec[Any],
        *,
        documents: Sequence[Document] | None = None,
        include_input_documents: bool = True,
        purpose: str | None = None,
        expected_cost: float | None = None,
    ) -> "Conversation[Any]":
        """Send a PromptSpec to the LLM.

        Adds documents to context (or messages for follow-up specs), renders the
        prompt, and dispatches to send() or send_structured() based on output_type.

        For specs with output_structure, sets stop sequences at </result> and
        auto-extracts content — conv.content returns clean text.

        For follow-up specs (follows is set), documents go to messages instead of
        context. If the follow-up declares input_documents and documents are passed,
        they are listed in the prompt text with their runtime id, name, description.
        """
        is_follow_up = spec.follows is not None

        # Warning for missing documents (only for non-follow-up specs)
        if not is_follow_up and spec.input_documents and not documents and include_input_documents:
            logger.warning(
                "PromptSpec '%s' declares input_documents (%s) but no documents were passed to send_spec().",
                spec.__class__.__name__,
                ", ".join(d.__name__ for d in spec.input_documents),
            )

        # Place documents in context (initial specs) or messages (follow-ups)
        if documents:
            if is_follow_up:
                conv = self.with_documents(documents)
            else:
                conv = self.with_context(*documents)
        else:
            conv = self

        # Set stop sequence when output_structure is set (result tag wrapping)
        if spec.output_structure is not None:
            opts = conv.model_options or ModelOptions()
            if RESULT_CLOSE not in (opts.stop or ()):
                stop = (*opts.stop, RESULT_CLOSE) if opts.stop else (RESULT_CLOSE,)
                conv = conv.with_model_options(opts.model_copy(update={"stop": stop}))  # nosemgrep: no-document-model-copy

        # Determine whether to include input documents in prompt text
        # Follow-ups: only include if the spec declares its own input_documents AND documents are passed
        if is_follow_up:
            effective_include_docs = bool(spec.input_documents) and documents is not None
        else:
            effective_include_docs = include_input_documents

        # Add multi-line field values as a single user message before the prompt
        ml_messages = render_multi_line_messages(spec)
        if ml_messages:
            combined = "\n".join(xml_block for _, xml_block in ml_messages)
            conv = conv.model_copy(update={"messages": conv.messages + (_UserMessage(combined),)})

        prompt_text = render_text(spec, documents=documents, include_input_documents=effective_include_docs)
        trace_purpose = purpose or spec.__class__.__name__

        # Dispatch to structured or text generation
        if spec.output_type is not str:
            return await conv.send_structured(
                prompt_text,
                response_format=cast(type[BaseModel], spec.output_type),
                purpose=trace_purpose,
                expected_cost=expected_cost,
            )

        result = await conv.send(prompt_text, purpose=trace_purpose, expected_cost=expected_cost)

        if spec.output_structure is not None:
            return result.model_copy(update={"extract_result_tags": True})

        return result

    # --- Builder methods (return NEW Conversation) ---

    def with_document(self, doc: Document) -> "Conversation[T]":
        """Return NEW Conversation with document appended to messages (dynamic suffix, not cached)."""
        return self.model_copy(update={"messages": self.messages + (doc,)})

    def with_documents(self, docs: Sequence[Document]) -> "Conversation[T]":
        """Return NEW Conversation with multiple documents appended to messages (not cached)."""
        return self.model_copy(update={"messages": self.messages + tuple(docs)})

    def with_assistant_message(self, content: str) -> "Conversation[T]":
        """Return NEW Conversation with an injected assistant turn in messages."""
        return self.model_copy(update={"messages": self.messages + (_AssistantMessage(content),)})

    def with_context(self, *docs: Document) -> "Conversation[T]":
        """Return NEW Conversation with documents added to the cacheable context prefix.

        Always set context before the first send() — adding context mid-conversation
        changes the prefix and invalidates existing cache.
        """
        return self.model_copy(update={"context": self.context + docs})

    def with_model_options(self, options: ModelOptions) -> "Conversation[T]":
        """Return NEW Conversation with updated model options."""
        return self.model_copy(update={"model_options": options})

    def with_model(self, model: str) -> "Conversation[T]":
        """Return NEW Conversation with a different model, preserving all state."""
        if not model:
            raise ValueError("model must be non-empty")
        return self.model_copy(update={"model": model})

    def with_substitutor(self, enabled: bool = True) -> "Conversation[T]":
        """Return NEW Conversation with content protection enabled/disabled.

        Shortens URLs, blockchain addresses, and high-entropy strings before sending.
        Both .content and .parsed are eagerly restored after each send.
        Auto-disabled for -search suffix models.
        """
        return self.model_copy(update={"enable_substitutor": enabled})

    # --- Utilities ---

    @property
    def approximate_tokens_count(self) -> int:
        """Approximate token count for all context and messages."""
        total = 0
        for item in chain(self.context, self.messages):
            if isinstance(item, ModelResponse):
                total += len(item.content) // CHARS_PER_TOKEN
                if reasoning := item.reasoning_content:
                    total += len(reasoning) // CHARS_PER_TOKEN
            elif isinstance(item, (_UserMessage, _AssistantMessage)):
                total += len(item.text) // CHARS_PER_TOKEN
            elif isinstance(item, Document):  # pyright: ignore[reportUnnecessaryIsInstance]
                if item.is_text:
                    total += len(item.content) // CHARS_PER_TOKEN
                elif item.is_image or item.is_pdf:
                    total += TOKENS_PER_IMAGE
                for att in item.attachments:
                    if att.is_text:
                        total += len(att.content) // CHARS_PER_TOKEN
                    elif att.is_image or att.is_pdf:
                        total += TOKENS_PER_IMAGE
        return total
