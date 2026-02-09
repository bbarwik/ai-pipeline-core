"""Document content trimming utilities for trace spans.

Trims large text content and removes binary data URIs to keep trace payloads
manageable while preserving enough context for debugging.
"""

import re
from typing import Any, cast

_CONTENT_TRIM_THRESHOLD = 250
_CONTENT_TRIM_KEEP = 100


def _is_binary_content(content: Any) -> bool:
    """Detect binary content by data URI prefix (RFC 2397 format with base64 encoding)."""
    return isinstance(content, str) and bool(re.match(r"^data:[a-zA-Z0-9.+/-]+;base64,", content))


def _trim_content_string(content: str) -> str:
    """Trim a text content string if over threshold, keeping first/last chars."""
    if len(content) <= _CONTENT_TRIM_THRESHOLD:
        return content
    trimmed_chars = len(content) - 2 * _CONTENT_TRIM_KEEP
    return content[:_CONTENT_TRIM_KEEP] + f" ... [trimmed {trimmed_chars} chars] ... " + content[-_CONTENT_TRIM_KEEP:]


def _trim_attachment_list(attachments: list[Any]) -> list[Any]:
    """Trim attachment content in a serialized attachment list.

    - Binary (data URI): replace content with placeholder
    - Text > threshold: keep first/last chars
    """
    trimmed: list[Any] = []
    for raw_att in attachments:
        if not isinstance(raw_att, dict):
            trimmed.append(raw_att)
            continue
        att: dict[str, Any] = cast(dict[str, Any], raw_att)
        content = att.get("content")
        if _is_binary_content(content):
            att = att.copy()
            att["content"] = "[binary content removed]"
        elif isinstance(content, str) and len(content) > _CONTENT_TRIM_THRESHOLD:
            att = att.copy()
            att["content"] = _trim_content_string(content)
        trimmed.append(att)
    return trimmed


def _trim_document_content(doc_dict: dict[str, Any]) -> dict[str, Any]:
    """Trim document content for traces. Binary removed, text trimmed."""
    if not isinstance(doc_dict, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        return doc_dict  # pyright: ignore[reportUnreachable]

    if "content" not in doc_dict or "class_name" not in doc_dict:
        return doc_dict

    doc_dict = doc_dict.copy()
    content = doc_dict.get("content")

    # Trim attachments
    if "attachments" in doc_dict and isinstance(doc_dict["attachments"], list):
        doc_dict["attachments"] = _trim_attachment_list(cast(list[Any], doc_dict["attachments"]))

    # Binary: remove content
    if _is_binary_content(content):
        doc_dict["content"] = "[binary content removed]"
        return doc_dict

    # Text: trim if over threshold
    if isinstance(content, str):
        doc_dict["content"] = _trim_content_string(content)

    return doc_dict


def _trim_documents_in_data(data: Any) -> Any:
    """Recursively trim document content in nested data structures."""
    if isinstance(data, dict):
        data_dict = cast(dict[str, Any], data)
        if "class_name" in data_dict and "content" in data_dict:
            return _trim_document_content(data_dict)
        return {k: _trim_documents_in_data(v) for k, v in data_dict.items()}
    if isinstance(data, list):
        return [_trim_documents_in_data(item) for item in cast(list[Any], data)]
    if isinstance(data, tuple):
        return tuple(_trim_documents_in_data(item) for item in cast(tuple[Any, ...], data))
    return data


__all__ = [
    "_is_binary_content",
    "_trim_attachment_list",
    "_trim_content_string",
    "_trim_document_content",
    "_trim_documents_in_data",
]
