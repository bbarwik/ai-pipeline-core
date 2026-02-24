"""Content writing for trace debugging.

Writes span input/output as YAML files with inline content trimming.
Large text is trimmed (first/last chars preserved), images are replaced with metadata placeholders,
and documents use the same trimming as the rest of the trace system.
"""

import hashlib
import json
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import yaml
from pydantic import BaseModel, SecretStr

from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._trimming import _CONTENT_TRIM_THRESHOLD, _is_binary_content, _trim_content_string

from ._config import TraceDebugConfig

logger = get_pipeline_logger(__name__)


class _BlockStringDumper(yaml.SafeDumper):
    """YAML dumper that uses block scalar style (|) for multiline strings."""


def _block_string_representer(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_BlockStringDumper.add_representer(str, _block_string_representer)


class ContentWriter:
    """Writes content as input.yaml / output.yaml with inline trimming."""

    def __init__(self, config: TraceDebugConfig):
        self._config = config
        self._compiled_patterns = [re.compile(p) for p in config.redact_patterns]

    def write(self, content: Any, span_dir: Path, name: str) -> dict[str, Any]:
        """Write content as {name}.yaml with inline trimming.

        Returns:
            Metadata dict with type, path, size_bytes
        """
        if content is None:
            return {"type": "none", "size_bytes": 0}

        structured = self._structure_content(content)

        serialized = yaml.dump(
            structured,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            Dumper=_BlockStringDumper,
        )
        serialized = self._redact(serialized)
        size = len(serialized.encode("utf-8"))

        if size > self._config.max_file_bytes:
            serialized = serialized[: self._config.max_file_bytes]
            max_bytes = self._config.max_file_bytes
            serialized += f"\n\n# [TRUNCATED: original {size} bytes exceeded {max_bytes} limit]\n"
            size = len(serialized.encode("utf-8"))

        file_path = span_dir / f"{name}.yaml"
        file_path.write_text(serialized, encoding="utf-8")

        return {
            "type": "file",
            "path": f"{name}.yaml",
            "size_bytes": size,
        }

    def _structure_content(self, content: Any) -> dict[str, Any]:
        """Convert raw content to structured YAML-ready format."""
        if self._is_llm_messages(content):
            return self._structure_llm_messages(content)
        if self._is_document_list(content):
            return self._structure_documents(content)
        return self._structure_generic(content)

    def _is_llm_messages(self, content: Any) -> bool:
        """Check if content looks like LLM messages."""
        if not isinstance(content, list):
            return False
        if not content:
            return False
        items = cast(list[Any], content)
        sample = items[: min(10, len(items))]
        if not all(isinstance(item, dict) for item in sample):
            return False
        return all("role" in item and "content" in item for item in sample)

    def _is_document_list(self, content: Any) -> bool:
        """Check if content looks like a list of serialized documents."""
        if not isinstance(content, list):
            return False
        if not content:
            return False
        items = cast(list[Any], content)
        sample = items[: min(10, len(items))]
        if not all(isinstance(item, dict) for item in sample):
            return False
        return all("class_name" in item and "content" in item for item in sample)

    def _structure_llm_messages(self, messages: list[Any]) -> dict[str, Any]:
        """Structure LLM messages with inline trimming."""
        message_entries: list[dict[str, Any]] = []

        total_text_bytes = 0
        total_image_bytes = 0
        total_tool_bytes = 0

        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content")

            msg_entry: dict[str, Any] = {
                "index": i,
                "role": role,
            }

            if isinstance(content, list):
                content_parts = cast(list[Any], content)
                msg_parts: list[dict[str, Any]] = []
                msg_entry["parts"] = msg_parts
                for j, part in enumerate(content_parts):
                    structured_part, part_bytes = self._structure_message_part(part, j)
                    msg_parts.append(structured_part)
                    part_type = structured_part.get("type", "")
                    if part_type == "text":
                        total_text_bytes += part_bytes
                    elif part_type == "image":
                        total_image_bytes += part_bytes
                    elif part_type in {"tool_use", "tool_result"}:
                        total_tool_bytes += part_bytes
            elif isinstance(content, str):
                text_entry = self._structure_text_element(content, 0)
                msg_entry["parts"] = [text_entry]
                total_text_bytes += text_entry.get("size_bytes", 0)
            elif content is None:
                msg_entry["parts"] = []
            else:
                msg_entry["parts"] = [{"type": "unknown", "sequence": 0, "raw": str(content)}]

            if "tool_calls" in msg:
                msg_entry["tool_calls"] = self._convert_types(msg["tool_calls"])
            if "function_call" in msg:
                msg_entry["function_call"] = self._convert_types(msg["function_call"])
            if "tool_call_id" in msg:
                msg_entry["tool_call_id"] = msg["tool_call_id"]
            if "name" in msg:
                msg_entry["name"] = msg["name"]

            message_entries.append(msg_entry)

        return {
            "format_version": 3,
            "type": "llm_messages",
            "message_count": len(messages),
            "messages": message_entries,
            "metadata": {
                "total_text_bytes": total_text_bytes,
                "total_image_bytes": total_image_bytes,
                "total_tool_bytes": total_tool_bytes,
            },
            "size_bytes": total_text_bytes + total_image_bytes + total_tool_bytes,
        }

    def _structure_message_part(self, part: dict[str, Any], sequence: int) -> tuple[dict[str, Any], int]:
        """Structure a single message part.

        Returns:
            Tuple of (structured_dict, size_bytes)
        """
        part_type = part.get("type", "")

        if part_type == "text":
            entry = self._structure_text_element(part.get("text", ""), sequence)
            return entry, entry.get("size_bytes", 0)
        if part_type in {"image_url", "image"}:
            entry = self._structure_image(part, part_type, sequence)
            return entry, entry.get("size_bytes", 0)
        if part_type == "tool_use":
            input_str = json.dumps(part.get("input", {}))
            size = len(input_str.encode("utf-8"))
            return {
                "type": "tool_use",
                "sequence": sequence,
                "id": part.get("id"),
                "name": part.get("name"),
                "input": self._convert_types(part.get("input")),
            }, size
        if part_type == "tool_result":
            result_content = part.get("content")
            entry: dict[str, Any] = {
                "type": "tool_result",
                "sequence": sequence,
                "tool_use_id": part.get("tool_use_id"),
                "is_error": part.get("is_error", False),
            }
            size = 0
            if isinstance(result_content, str):
                text_entry = self._structure_text_element(result_content, 0)
                entry["content"] = text_entry
                size = text_entry.get("size_bytes", 0)
            elif isinstance(result_content, list):
                result_parts = cast(list[Any], result_content)
                content_list: list[dict[str, Any]] = []
                entry["content"] = content_list
                for k, p in enumerate(result_parts):
                    part_entry, part_size = self._structure_message_part(p, k)
                    content_list.append(part_entry)
                    size += part_size
            else:
                entry["content"] = self._convert_types(result_content)
            return entry, size
        # Unknown type — preserve raw data
        raw = self._convert_types(part)
        raw_str = json.dumps(raw)
        size = len(raw_str.encode("utf-8"))
        return {
            "type": "unknown",
            "sequence": sequence,
            "original_type": part_type,
            "raw_data": raw,
        }, size

    def _structure_text_element(self, text: str, sequence: int) -> dict[str, Any]:
        """Structure a text element with inline trimming."""
        text = self._redact(text)
        text_bytes = len(text.encode("utf-8"))

        entry: dict[str, Any] = {
            "type": "text",
            "sequence": sequence,
            "size_bytes": text_bytes,
        }

        if text_bytes > _CONTENT_TRIM_THRESHOLD:
            entry["content"] = _trim_content_string(text)
            entry["truncated"] = True
            entry["original_size_bytes"] = text_bytes
        else:
            entry["content"] = text

        return entry

    def _structure_image(self, part: dict[str, Any], part_type: str, sequence: int) -> dict[str, Any]:
        """Structure image part as metadata-only placeholder."""
        if part_type == "image_url":
            url = part.get("image_url", {}).get("url", "")
            detail = part.get("image_url", {}).get("detail", "auto")

            if not url.startswith("data:image/"):
                return {"type": "image_url", "sequence": sequence, "url": url, "detail": detail, "size_bytes": 0}

            match = re.match(r"data:image/(\w+);base64,(.+)", url)
            if not match:
                return {"type": "image_parse_error", "sequence": sequence, "url_preview": url[:100], "size_bytes": 0}

            ext, b64_data = match.groups()
        else:
            detail = None
            source = part.get("source", {})
            mime_type = source.get("media_type", "image/png")
            ext = mime_type.split("/")[-1] if "/" in mime_type else "png"

            if source.get("type") != "base64":
                return {"type": "image", "sequence": sequence, "source_type": source.get("type"), "format": ext, "size_bytes": 0}

            b64_data = source.get("data", "")

        estimated_size = len(b64_data) * 3 // 4 if b64_data else 0
        content_hash = hashlib.sha256(b64_data.encode()).hexdigest()[:16] if b64_data else "empty"

        entry: dict[str, Any] = {
            "type": "image",
            "sequence": sequence,
            "format": ext,
            "size_bytes": estimated_size,
            "hash": content_hash,
            "preview": f"[{ext.upper()} image, {estimated_size} bytes]",
        }
        if detail is not None:
            entry["detail"] = detail
        return entry

    def _structure_documents(self, docs: list[Any]) -> dict[str, Any]:
        """Structure document list with inline trimming."""
        doc_entries: list[dict[str, Any]] = []

        for i, doc in enumerate(docs):
            doc_name = doc.get("name", f"doc_{i}")
            class_name = doc.get("class_name", "Document")
            content = doc.get("content", "")

            doc_entry: dict[str, Any] = {
                "index": i,
                "name": doc_name,
                "class_name": class_name,
            }

            if _is_binary_content(content):
                doc_entry["content"] = "[binary content removed]"
                doc_entry["size_bytes"] = 0
            elif isinstance(content, str):
                doc_entry["size_bytes"] = len(content.encode("utf-8"))
                doc_entry["content"] = self._redact(_trim_content_string(content))
            else:
                doc_entry["content"] = self._redact(str(content))
                doc_entry["size_bytes"] = len(str(content).encode("utf-8"))

            raw_attachments = doc.get("attachments")
            if isinstance(raw_attachments, list) and raw_attachments:
                att_entries: list[dict[str, Any]] = []
                for j, att in enumerate(cast(list[Any], raw_attachments)):
                    if not isinstance(att, dict):
                        continue
                    att_dict = cast(dict[str, Any], att)
                    att_content = att_dict.get("content", "")
                    att_entry: dict[str, Any] = {"index": j, "name": att_dict.get("name", f"attachment_{j}")}
                    if att_dict.get("description"):
                        att_entry["description"] = att_dict["description"]
                    if _is_binary_content(att_content):
                        att_entry["content"] = "[binary content removed]"
                        att_entry["size_bytes"] = 0
                    elif isinstance(att_content, str):
                        att_entry["size_bytes"] = len(att_content.encode("utf-8"))
                        att_entry["content"] = self._redact(_trim_content_string(att_content))
                    att_entries.append(att_entry)
                doc_entry["attachment_count"] = len(att_entries)
                doc_entry["attachments"] = att_entries

            doc_entries.append(doc_entry)

        return {
            "format_version": 3,
            "type": "document_list",
            "document_count": len(docs),
            "documents": doc_entries,
        }

    def _structure_generic(self, content: Any) -> dict[str, Any]:
        """Structure generic content."""
        converted = self._convert_types(content)
        serialized = json.dumps(converted)
        size = len(serialized.encode("utf-8"))

        return {
            "format_version": 3,
            "type": "generic",
            "size_bytes": size,
            "content": converted,
        }

    def _redact(self, text: str) -> str:
        """Apply redaction patterns to text."""
        for pattern in self._compiled_patterns:
            text = pattern.sub("[REDACTED]", text)
        return text

    def _convert_types(self, value: Any, seen: set[int] | None = None) -> Any:  # noqa: PLR0911
        """Convert non-serializable types recursively with cycle detection."""
        if seen is None:
            seen = set()

        obj_id = id(value)
        if obj_id in seen:
            return "[circular reference]"

        match value:
            case None | bool() | int() | float() | str():
                return value
            case SecretStr():
                return "[REDACTED:SecretStr]"
            case bytes():
                if len(value) < 100:
                    return f"[bytes: {len(value)} bytes, preview: {value[:50].hex()}...]"
                return f"[bytes: {len(value)} bytes]"
            case Path():
                return str(value)
            case UUID():
                return str(value)
            case datetime():
                return value.isoformat()
            case Enum():
                return value.value
            case set() | frozenset():
                return sorted(str(x) for x in cast(set[Any] | frozenset[Any], value))
            case BaseModel():
                try:
                    return value.model_dump(mode="json")
                except Exception:
                    return str(value)
            case dict():
                seen.add(obj_id)
                typed_dict = cast(dict[Any, Any], value)
                result = {str(k): self._convert_types(v, seen) for k, v in typed_dict.items()}
                seen.discard(obj_id)
                return result
            case list() | tuple():
                seen.add(obj_id)
                typed_seq = cast(list[Any] | tuple[Any, ...], value)
                result = [self._convert_types(x, seen) for x in typed_seq]
                seen.discard(obj_id)
                return result
            case _:
                try:
                    return str(value)
                except Exception:
                    return f"<{type(value).__name__}>"
