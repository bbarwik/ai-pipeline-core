"""Private message and content helpers for Conversation."""

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ai_pipeline_core._codec import UniversalCodec
from ai_pipeline_core._llm_core import CoreMessage
from ai_pipeline_core._llm_core._validation import validate_text
from ai_pipeline_core._llm_core.model_response import ModelResponse
from ai_pipeline_core._llm_core.types import ContentPart, ImageContent, PDFContent, TextContent
from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents._hashing import compute_content_sha256
from ai_pipeline_core.llm._images import validated_binary_parts
from ai_pipeline_core.logger import get_pipeline_logger

from .tools import Tool, to_snake_case

__all__ = [
    "AnyMessage",
    "AssistantMessage",
    "ConversationContent",
    "ToolResultMessage",
    "UserMessage",
    "_build_attachment_content",
    "_build_attachment_parts",
    "_core_messages_to_db_span_input",
    "_core_messages_to_span_input",
    "_document_to_content_parts",
    "_document_to_xml_header",
    "_escape_xml_content",
    "_escape_xml_metadata",
    "_finish_reason",
    "_normalize_content",
    "_prompt_parts",
    "_response_format_path",
    "_serialize_response_tool_calls",
    "_serialize_tool_config",
]

logger = get_pipeline_logger(__name__)

ConversationContent = str | Document | list[Document]

# Regex matching XML tags whose names are wrapper elements used by document serialization.
# Only these tags are escaped in content bodies; other content remains unchanged.
_WRAPPER_TAG_RE = re.compile(r"<(/?)(document|content|description|attachment|id|name)\b([^>]*)>", re.IGNORECASE)


def _escape_xml_metadata(text: str) -> str:
    """Escape < and > in metadata fields to prevent tag injection."""
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _escape_xml_content(text: str) -> str:
    """Escape only wrapper tags that could break document XML structure."""
    return _WRAPPER_TAG_RE.sub(lambda match: f"&lt;{match.group(1)}{match.group(2)}{match.group(3)}&gt;", text)


def _document_to_xml_header(doc: Document) -> str:
    """Generate XML header for a document with proper escaping."""
    escaped_name = _escape_xml_metadata(doc.name)
    escaped_id = _escape_xml_metadata(doc.id)
    description = f"<description>{_escape_xml_metadata(doc.description)}</description>\n" if doc.description else ""
    return f"<document>\n<id>{escaped_id}</id>\n<name>{escaped_name}</name>\n{description}<content>\n"


def _document_to_content_parts(doc: Document, model: str) -> list[ContentPart]:
    """Convert a Document to content parts for CoreMessage."""
    parts: list[ContentPart] = []
    header = _document_to_xml_header(doc)

    if doc.is_text:
        if err := validate_text(doc.content, doc.name):
            logger.warning("Skipping invalid document: %s", err)
            return []
        text = _escape_xml_content(doc.content.decode("utf-8"))
        text_fragments = [f"{header}{text}\n"]
        binary_attachment_parts: list[ContentPart] = []

        for attachment in doc.attachments:
            if attachment.is_text:
                attachment_content = _build_attachment_content(attachment)
                if attachment_content:
                    text_fragments.append(attachment_content)
                continue
            binary_attachment_parts.extend(_build_attachment_parts(attachment, model))

        if binary_attachment_parts:
            parts.append(TextContent(text="".join(text_fragments)))
            parts.extend(binary_attachment_parts)
            parts.append(TextContent(text="</content>\n</document>"))
        else:
            text_fragments.append("</content>\n</document>")
            parts.append(TextContent(text="".join(text_fragments)))
        return parts

    if doc.is_image or doc.is_pdf:
        binary_parts = validated_binary_parts(doc.content, doc.name, is_image=doc.is_image, model=model)
        if binary_parts is None:
            return []
        parts.append(TextContent(text=header))
        parts.extend(binary_parts)
        for attachment in doc.attachments:
            parts.extend(_build_attachment_parts(attachment, model))
        parts.append(TextContent(text="</content>\n</document>"))
        return parts

    logger.warning("Skipping unsupported document type: %s - %s", doc.name, doc.mime_type)
    return []


def _build_attachment_content(attachment: Any) -> str | None:
    """Build text content for a text attachment."""
    if not attachment.is_text:
        return None
    if err := validate_text(attachment.content, attachment.name):
        logger.warning("Skipping invalid attachment: %s", err)
        return None

    escaped_name = _escape_xml_metadata(attachment.name)
    description_attr = f' description="{_escape_xml_metadata(attachment.description)}"' if attachment.description else ""
    attachment_text = _escape_xml_content(attachment.content.decode("utf-8"))
    return f'<attachment name="{escaped_name}"{description_attr}>\n{attachment_text}\n</attachment>\n'


def _build_attachment_parts(attachment: Any, model: str) -> list[ContentPart]:
    """Build content parts for one attachment."""
    parts: list[ContentPart] = []
    escaped_name = _escape_xml_metadata(attachment.name)
    description_attr = f' description="{_escape_xml_metadata(attachment.description)}"' if attachment.description else ""
    attachment_open = f'<attachment name="{escaped_name}"{description_attr}>\n'

    if attachment.is_text:
        if err := validate_text(attachment.content, attachment.name):
            logger.warning("Skipping invalid attachment: %s", err)
            return []
        attachment_text = _escape_xml_content(attachment.content.decode("utf-8"))
        parts.append(TextContent(text=f"{attachment_open}{attachment_text}\n</attachment>\n"))
        return parts

    if attachment.is_image or attachment.is_pdf:
        binary_parts = validated_binary_parts(attachment.content, attachment.name, is_image=attachment.is_image, model=model)
        if binary_parts is None:
            return []
        parts.append(TextContent(text=attachment_open))
        parts.extend(binary_parts)
        parts.append(TextContent(text="</attachment>\n"))
        return parts

    logger.warning("Skipping unsupported attachment type: %s - %s", attachment.name, attachment.mime_type)
    return []


@dataclass(frozen=True, slots=True)
class UserMessage:
    """Internal wrapper for user string messages."""

    text: str


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    """Internal wrapper for injected assistant messages."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    """Internal wrapper for tool execution results."""

    tool_call_id: str
    function_name: str
    content: str


AnyMessage = Document | ModelResponse[Any] | UserMessage | AssistantMessage | ToolResultMessage


def _normalize_content(content: ConversationContent) -> tuple[Document | UserMessage, ...]:
    """Normalize send content to Documents or internal user messages."""
    if isinstance(content, str):
        return (UserMessage(content),)
    if isinstance(content, Document):
        return (content,)
    return tuple(content)


def _core_messages_to_span_input(messages: list[CoreMessage]) -> list[dict[str, Any]]:
    """Convert CoreMessages to Laminar-compatible input, replacing binary parts with placeholders."""
    result: list[dict[str, Any]] = []
    for message in messages:
        role = message.role.value
        if isinstance(message.content, str):
            result.append({"role": role, "content": message.content})
            continue
        if isinstance(message.content, tuple):
            parts: list[dict[str, str]] = []
            for part in message.content:
                if isinstance(part, TextContent):
                    parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ImageContent):
                    parts.append({"type": "text", "text": "[image]"})
                elif isinstance(part, PDFContent):
                    parts.append({"type": "text", "text": "[pdf]"})
            result.append({"role": role, "content": parts})
    return result


def _core_messages_to_db_span_input(messages: list[CoreMessage]) -> list[dict[str, Any]]:
    """Convert CoreMessages to database span input, replacing binary parts with blob refs."""
    result: list[dict[str, Any]] = []
    for message in messages:
        payload: dict[str, Any] = {"role": message.role.value}
        if isinstance(message.content, str):
            payload["content"] = message.content
        else:
            parts: list[dict[str, str]] = []
            content_parts = message.content if isinstance(message.content, tuple) else (message.content,)
            for part in content_parts:
                if isinstance(part, TextContent):
                    parts.append({"type": "text", "text": part.text})
                    continue
                if isinstance(part, ImageContent):
                    parts.append({
                        "type": "image",
                        "$doc_ref": compute_content_sha256(bytes(part.data)),
                        "media_type": part.mime_type,
                    })
                    continue
                if isinstance(part, PDFContent):
                    parts.append({
                        "type": "pdf",
                        "$doc_ref": compute_content_sha256(bytes(part.data)),
                        "media_type": "application/pdf",
                    })
            payload["content"] = parts
        if message.tool_calls is not None:
            payload["tool_calls"] = [tool_call.model_dump(mode="json") for tool_call in message.tool_calls]
        if message.tool_call_id is not None:
            payload["tool_call_id"] = message.tool_call_id
        if message.name is not None:
            payload["name"] = message.name
        result.append(payload)
    return result


def _response_format_path(response_format: type[BaseModel] | None) -> str:
    """Convert a response model class to an importable path."""
    if response_format is None:
        return ""
    return f"{response_format.__module__}:{response_format.__qualname__}"


def _prompt_parts(content: ConversationContent) -> tuple[str, tuple[str, ...]]:
    """Return prompt text and prompt document SHA256 values for one send call."""
    if isinstance(content, str):
        return content, ()
    if isinstance(content, Document):
        return "", (content.sha256,)
    return "", tuple(document.sha256 for document in content)


def _serialize_tool_config(tool: Tool) -> dict[str, Any]:
    """Serialize tool metadata for conversation detail_json."""
    tool_cls = type(tool)
    constructor_args = UniversalCodec().encode(tool.__dict__).value
    return {
        "name": to_snake_case(tool_cls.__name__),
        "class_path": f"{tool_cls.__module__}:{tool_cls.__qualname__}",
        "constructor_args": constructor_args,
    }


def _serialize_response_tool_calls(tool_calls: tuple[Any, ...]) -> list[dict[str, Any]]:
    """Serialize tool calls for llm_round detail_json."""
    serialized: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        try:
            arguments: Any = json.loads(tool_call.arguments)
        except json.JSONDecodeError:
            arguments = {"_raw": tool_call.arguments}
        serialized.append({
            "id": tool_call.id,
            "function_name": tool_call.function_name,
            "arguments": arguments,
        })
    return serialized


def _finish_reason(response: ModelResponse[Any]) -> str:
    """Infer a finish reason for span detail_json."""
    raw_finish_reason = response.metadata.get("finish_reason")
    if isinstance(raw_finish_reason, str) and raw_finish_reason:
        return raw_finish_reason
    if response.has_tool_calls:
        return "tool_calls"
    return "stop"
