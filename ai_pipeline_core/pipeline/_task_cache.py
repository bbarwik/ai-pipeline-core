"""Helpers for task cache keys and compact task descriptions."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from ai_pipeline_core._codec import UniversalCodec
from ai_pipeline_core.documents import Document

__all__ = [
    "build_task_cache_key",
    "build_task_description",
]

MAX_DESCRIPTION_PARTS = 4
MAX_DESCRIPTION_VALUE_CHARS = 80
MAX_DESCRIPTION_TOTAL_CHARS = 240
TASK_CACHE_FINGERPRINT_LENGTH = 16
_CONVERSATION_MODULE = "ai_pipeline_core.llm.conversation"
_CONVERSATION_CLASS = "Conversation"


def _trim_value(value: str) -> str:
    if len(value) <= MAX_DESCRIPTION_VALUE_CHARS:
        return value
    return f"{value[: MAX_DESCRIPTION_VALUE_CHARS - 3]}..."


def _describe_mapping_value(value: Mapping[Any, Any]) -> str | None:
    described_items: list[str] = []
    for key, item in value.items():
        if len(described_items) >= MAX_DESCRIPTION_PARTS:
            break
        nested = _describe_value(item)
        if nested:
            described_items.append(f"{key}={nested}")
    if not described_items:
        return None
    return "{" + ", ".join(described_items) + "}"


def _describe_sequence_value(value: Sequence[Any]) -> str | None:
    described_items = [item for item in (_describe_value(item) for item in value[:MAX_DESCRIPTION_PARTS]) if item]
    if not described_items:
        return None
    suffix = ", ..." if len(value) > MAX_DESCRIPTION_PARTS else ""
    return "[" + ", ".join(described_items) + suffix + "]"


def _describe_value(value: Any) -> str | None:
    result: str | None = None
    if isinstance(value, Document):
        result = value.name
    elif _is_conversation_like(value):
        result = f"Conversation(model={value.model})"
    elif isinstance(value, str):
        result = _trim_value(value) if value else None
    elif isinstance(value, (bool, int, float)):
        result = str(value)
    elif isinstance(value, BaseModel):
        result = _trim_value(value.__class__.__name__)
    elif isinstance(value, Mapping):
        result = _describe_mapping_value(value)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        result = _describe_sequence_value(value)
    return result


def _is_conversation_like(value: Any) -> bool:
    value_type = type(value)
    return value_type.__module__ == _CONVERSATION_MODULE and value_type.__name__ == _CONVERSATION_CLASS and isinstance(getattr(value, "model", None), str)


def build_task_description(arguments: Mapping[str, Any]) -> str:
    """Synthesize a compact invocation description from validated task inputs."""
    parts: list[str] = []
    for key, value in arguments.items():
        described = _describe_value(value)
        if not described:
            continue
        parts.append(f"{key}={described}")
        joined = ", ".join(parts)
        if len(joined) >= MAX_DESCRIPTION_TOTAL_CHARS or len(parts) >= MAX_DESCRIPTION_PARTS:
            break
    description = ", ".join(parts)
    if len(description) <= MAX_DESCRIPTION_TOTAL_CHARS:
        return description
    return f"{description[: MAX_DESCRIPTION_TOTAL_CHARS - 3]}..."


def build_task_cache_key(
    *,
    task_class_path: str,
    cache_version: int,
    arguments: Mapping[str, Any],
) -> str:
    """Build the deterministic cache key for a cacheable task invocation."""
    encoded_arguments = UniversalCodec().encode(dict(arguments)).value
    canonical_json = json.dumps(encoded_arguments, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:TASK_CACHE_FINGERPRINT_LENGTH]
    return f"task:{task_class_path}:v{cache_version}:{fingerprint}"
