"""Content writing and extraction for trace debugging V3.

Uses hash-based artifact storage with automatic deduplication.
"""

import base64
import hashlib
import json
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml
from pydantic import BaseModel, ConfigDict, SecretStr

from .config import TraceDebugConfig


class ContentRef(BaseModel):
    """Reference to content in artifact store."""

    hash: str  # "sha256:abcdef..."
    path: str  # "artifacts/sha256/ab/cd/abcdef...1234.txt"
    size_bytes: int
    mime_type: str | None = None
    encoding: str | None = None  # "utf-8" | "binary"

    model_config = ConfigDict(frozen=True)


class ArtifactStore:
    """Hash-based artifact storage with automatic deduplication.

    Stores large content elements in artifacts/sha256/<first2>/<next2>/<hash>.<ext>
    Identical content automatically deduplicates (same hash = same file).
    """

    def __init__(self, trace_path: Path):
        """Initialize artifact store for given trace path."""
        self._artifacts_path = trace_path / "artifacts" / "sha256"
        self._artifacts_path.mkdir(parents=True, exist_ok=True)
        self._known_hashes: dict[str, ContentRef] = {}
        self._trace_path = trace_path

    def store_text(self, text: str, mime_type: str = "text/plain") -> ContentRef:
        """Store text content, return reference."""
        data = text.encode("utf-8")
        content_hash = hashlib.sha256(data).hexdigest()

        if content_hash in self._known_hashes:
            return self._known_hashes[content_hash]

        # Create sharded path: ab/cd/abcdef...1234.txt
        file_path = (
            self._artifacts_path / content_hash[:2] / content_hash[2:4] / f"{content_hash}.txt"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not file_path.exists():
            file_path.write_bytes(data)

        ref = ContentRef(
            hash=f"sha256:{content_hash}",
            path=str(file_path.relative_to(self._trace_path)),
            size_bytes=len(data),
            mime_type=mime_type,
            encoding="utf-8",
        )

        self._known_hashes[content_hash] = ref
        return ref

    def store_binary(self, data: bytes, mime_type: str = "application/octet-stream") -> ContentRef:
        """Store binary content, return reference."""
        content_hash = hashlib.sha256(data).hexdigest()

        if content_hash in self._known_hashes:
            return self._known_hashes[content_hash]

        # Determine extension from mime type
        ext_map = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/gif": ".gif",
            "application/pdf": ".pdf",
        }
        ext = ext_map.get(mime_type, ".bin")

        file_path = (
            self._artifacts_path / content_hash[:2] / content_hash[2:4] / f"{content_hash}{ext}"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not file_path.exists():
            file_path.write_bytes(data)

        ref = ContentRef(
            hash=f"sha256:{content_hash}",
            path=str(file_path.relative_to(self._trace_path)),
            size_bytes=len(data),
            mime_type=mime_type,
            encoding="binary",
        )

        self._known_hashes[content_hash] = ref
        return ref

    def get_stats(self) -> dict[str, int | float]:
        """Get deduplication statistics."""
        total_files = len(list(self._artifacts_path.rglob("*.*")))
        total_size = sum(f.stat().st_size for f in self._artifacts_path.rglob("*.*") if f.is_file())
        total_refs = len(self._known_hashes)

        return {
            "unique_artifacts": total_files,
            "total_references": total_refs,
            "total_bytes": total_size,
            "dedup_ratio": total_refs / total_files if total_files > 0 else 1.0,
        }


class ContentWriter:
    """Writes content as input.yaml / output.yaml with artifact externalization."""

    def __init__(self, config: TraceDebugConfig, artifact_store: ArtifactStore | None = None):
        """Initialize content writer with config and optional artifact store."""
        self._config = config
        self._compiled_patterns = [re.compile(p) for p in config.redact_patterns]
        self._artifact_store = artifact_store

    def write(self, content: Any, span_dir: Path, name: str) -> dict[str, Any]:
        """Write content as {name}.yaml with artifact externalization.

        Args:
            content: Raw content (LLM messages, documents, dicts, etc.)
            span_dir: Span directory
            name: "input" or "output"

        Returns:
            Metadata dict with type, path, size_bytes, breakdown
        """
        if content is None:
            return {"type": "none", "size_bytes": 0}

        # Structure content (recursive processing with externalization)
        structured = self._structure_content(content)

        # Serialize to YAML
        serialized = yaml.dump(
            structured,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        serialized = self._redact(serialized)
        size = len(serialized.encode("utf-8"))

        # Check file size limit
        if size > self._config.max_file_bytes:
            # Reduce preview sizes to fit under limit
            structured = self._reduce_previews(structured)
            serialized = yaml.dump(
                structured, default_flow_style=False, allow_unicode=True, sort_keys=False
            )
            serialized = self._redact(serialized)
            size = len(serialized.encode("utf-8"))

            # If still over, truncate with warning
            if size > self._config.max_file_bytes:
                serialized = serialized[: self._config.max_file_bytes]
                max_bytes = self._config.max_file_bytes
                serialized += (
                    f"\n\n# [TRUNCATED: original {size} bytes exceeded {max_bytes} limit]\n"
                )
                size = len(serialized.encode("utf-8"))

        # Write file
        file_path = span_dir / f"{name}.yaml"
        file_path.write_text(serialized, encoding="utf-8")

        return {
            "type": "file",
            "path": f"{name}.yaml",
            "size_bytes": size,
            "breakdown": self._extract_breakdown(structured),
        }

    def _structure_content(self, content: Any) -> dict[str, Any]:
        """Convert raw content to structured YAML-ready format."""
        if self._is_llm_messages(content):
            return self._structure_llm_messages(content)
        elif self._is_document_list(content):
            return self._structure_documents(content)
        else:
            return self._structure_generic(content)

    def _is_llm_messages(self, content: Any) -> bool:
        """Check if content looks like LLM messages."""
        if not isinstance(content, list):
            return False
        if not content:
            return False
        first = content[0]
        if not isinstance(first, dict):
            return False
        return "role" in first and "content" in first

    def _is_document_list(self, content: Any) -> bool:
        """Check if content looks like a DocumentList."""
        if not isinstance(content, list):
            return False
        if not content:
            return False
        first = content[0]
        if not isinstance(first, dict):
            return False
        return "base_type" in first and "content" in first

    def _structure_llm_messages(self, messages: list[Any]) -> dict[str, Any]:
        """Structure LLM messages preserving ALL parts losslessly."""
        result = {
            "format_version": 3,
            "type": "llm_messages",
            "message_count": len(messages),
            "messages": [],
        }

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
                # Multimodal: preserve each part separately
                msg_entry["parts"] = []
                for j, part in enumerate(content):
                    structured_part, part_bytes = self._structure_message_part(part, j)
                    msg_entry["parts"].append(structured_part)
                    part_type = structured_part.get("type", "")
                    if part_type == "text":
                        total_text_bytes += part_bytes
                    elif part_type == "image":
                        total_image_bytes += part_bytes
                    elif part_type in ("tool_use", "tool_result"):
                        total_tool_bytes += part_bytes
            elif isinstance(content, str):
                # Simple text message
                text_entry = self._structure_text_element(content, 0)
                msg_entry["parts"] = [text_entry]
                total_text_bytes += text_entry.get("size_bytes", 0)
            elif content is None:
                msg_entry["parts"] = []
            else:
                msg_entry["parts"] = [{"type": "unknown", "sequence": 0, "raw": str(content)}]

            # Preserve tool_calls at message level (OpenAI format)
            if "tool_calls" in msg:
                msg_entry["tool_calls"] = self._convert_types(msg["tool_calls"])
            if "function_call" in msg:
                msg_entry["function_call"] = self._convert_types(msg["function_call"])
            if "tool_call_id" in msg:
                msg_entry["tool_call_id"] = msg["tool_call_id"]
            if "name" in msg:
                msg_entry["name"] = msg["name"]

            result["messages"].append(msg_entry)

        # Add metadata
        result["metadata"] = {
            "total_text_bytes": total_text_bytes,
            "total_image_bytes": total_image_bytes,
            "total_tool_bytes": total_tool_bytes,
        }
        result["size_bytes"] = total_text_bytes + total_image_bytes + total_tool_bytes

        return result

    def _structure_message_part(
        self, part: dict[str, Any], sequence: int
    ) -> tuple[dict[str, Any], int]:
        """Structure a single message part losslessly.

        Returns:
            Tuple of (structured_dict, size_bytes)
        """
        part_type = part.get("type", "")

        if part_type == "text":
            entry = self._structure_text_element(part.get("text", ""), sequence)
            return entry, entry.get("size_bytes", 0)
        elif part_type == "image_url":
            entry = self._structure_image_openai(part, sequence)
            return entry, entry.get("size_bytes", 0)
        elif part_type == "image":
            entry = self._structure_image_anthropic(part, sequence)
            return entry, entry.get("size_bytes", 0)
        elif part_type == "tool_use":
            input_str = json.dumps(part.get("input", {}))
            size = len(input_str.encode("utf-8"))
            return {
                "type": "tool_use",
                "sequence": sequence,
                "id": part.get("id"),
                "name": part.get("name"),
                "input": self._convert_types(part.get("input")),
            }, size
        elif part_type == "tool_result":
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
                entry["content"] = []
                for k, p in enumerate(result_content):
                    part_entry, part_size = self._structure_message_part(p, k)
                    entry["content"].append(part_entry)
                    size += part_size
            else:
                entry["content"] = self._convert_types(result_content)
            return entry, size
        else:
            # Unknown type — preserve raw data, never drop
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
        """Structure a text element, optionally externalizing large content."""
        text = self._redact(text)
        text_bytes = len(text.encode("utf-8"))

        entry: dict[str, Any] = {
            "type": "text",
            "sequence": sequence,
            "size_bytes": text_bytes,
        }

        if text_bytes > self._config.max_element_bytes:
            # Store full content in artifact store
            if self._artifact_store:
                ref = self._artifact_store.store_text(text, "text/plain")
                excerpt_len = self._config.element_excerpt_bytes
                entry["content_ref"] = {
                    "hash": ref.hash,
                    "path": ref.path,
                    "mime_type": ref.mime_type,
                    "encoding": ref.encoding,
                }
                entry["excerpt"] = (
                    text[:excerpt_len] + "\n[TRUNCATED - see artifact for full content]"
                )
            else:
                # No artifact store — truncate with marker
                entry["content"] = text[: self._config.max_element_bytes]
                entry["truncated"] = True
                entry["original_size_bytes"] = text_bytes
        else:
            entry["content"] = text

        return entry

    def _structure_image_openai(self, part: dict[str, Any], sequence: int) -> dict[str, Any]:
        """Structure OpenAI format image part."""
        url = part.get("image_url", {}).get("url", "")
        detail = part.get("image_url", {}).get("detail", "auto")

        if not url.startswith("data:image/"):
            return {
                "type": "image_url",
                "sequence": sequence,
                "url": url,
                "detail": detail,
                "size_bytes": 0,
            }

        match = re.match(r"data:image/(\w+);base64,(.+)", url)
        if not match:
            return {
                "type": "image_parse_error",
                "sequence": sequence,
                "url_preview": url[:100],
                "size_bytes": 0,
            }

        ext, b64_data = match.groups()
        estimated_size = len(b64_data) * 3 // 4
        content_hash = hashlib.sha256(b64_data.encode()).hexdigest()

        entry: dict[str, Any] = {
            "type": "image",
            "sequence": sequence,
            "format": ext,
            "size_bytes": estimated_size,
            "hash": content_hash[:16],
            "detail": detail,
        }

        # Extract if configured
        if self._config.extract_base64_images and self._artifact_store:
            try:
                image_bytes = base64.b64decode(b64_data)
                ref = self._artifact_store.store_binary(image_bytes, f"image/{ext}")
                entry["content_ref"] = {
                    "hash": ref.hash,
                    "path": ref.path,
                    "mime_type": ref.mime_type,
                    "encoding": ref.encoding,
                }
                entry["preview"] = f"[{ext.upper()} image, {estimated_size} bytes]"
                entry["extracted"] = True
            except Exception as e:
                entry["extract_error"] = str(e)
                entry["extracted"] = False
        else:
            entry["extracted"] = False

        return entry

    def _structure_image_anthropic(self, part: dict[str, Any], sequence: int) -> dict[str, Any]:
        """Structure Anthropic format image part."""
        source = part.get("source", {})
        media_type = source.get("media_type", "image/png")
        ext = media_type.split("/")[-1] if "/" in media_type else "png"

        if source.get("type") != "base64":
            return {
                "type": "image",
                "sequence": sequence,
                "source_type": source.get("type"),
                "format": ext,
                "size_bytes": 0,
            }

        b64_data = source.get("data", "")
        estimated_size = len(b64_data) * 3 // 4 if b64_data else 0
        content_hash = hashlib.sha256(b64_data.encode()).hexdigest() if b64_data else "empty"

        entry: dict[str, Any] = {
            "type": "image",
            "sequence": sequence,
            "format": ext,
            "size_bytes": estimated_size,
            "hash": content_hash[:16],
        }

        if self._config.extract_base64_images and self._artifact_store and b64_data:
            try:
                image_bytes = base64.b64decode(b64_data)
                ref = self._artifact_store.store_binary(image_bytes, media_type)
                entry["content_ref"] = {
                    "hash": ref.hash,
                    "path": ref.path,
                    "mime_type": ref.mime_type,
                    "encoding": ref.encoding,
                }
                entry["preview"] = f"[{ext.upper()} image, {estimated_size} bytes]"
                entry["extracted"] = True
            except Exception as e:
                entry["extract_error"] = str(e)
                entry["extracted"] = False
        else:
            entry["extracted"] = False

        return entry

    def _structure_documents(self, docs: list[Any]) -> dict[str, Any]:
        """Structure document list."""
        result = {
            "format_version": 3,
            "type": "document_list",
            "document_count": len(docs),
            "documents": [],
        }

        for i, doc in enumerate(docs):
            doc_name = doc.get("name", f"doc_{i}")
            base_type = doc.get("base_type", "unknown")
            content = doc.get("content", "")
            content_encoding = doc.get("content_encoding", "utf-8")

            doc_entry: dict[str, Any] = {
                "index": i,
                "name": doc_name,
                "base_type": base_type,
            }

            if content_encoding == "base64":
                # Binary content
                try:
                    binary_data = base64.b64decode(content)
                    size = len(binary_data)
                    doc_entry["size_bytes"] = size
                    doc_entry["encoding"] = "base64"

                    if size > self._config.max_element_bytes and self._artifact_store:
                        # Externalize binary
                        mime_type = doc.get("mime_type", "application/octet-stream")
                        ref = self._artifact_store.store_binary(binary_data, mime_type)
                        doc_entry["content_ref"] = {
                            "hash": ref.hash,
                            "path": ref.path,
                            "mime_type": ref.mime_type,
                            "encoding": ref.encoding,
                        }
                        doc_entry["preview"] = f"[Binary content, {size} bytes]"
                    else:
                        doc_entry["content"] = content  # Keep base64 inline
                except Exception:
                    doc_entry["content"] = "[binary content - decode failed]"
                    doc_entry["size_bytes"] = 0
            else:
                # Text content
                text = self._redact(str(content))
                text_bytes = len(text.encode("utf-8"))
                doc_entry["size_bytes"] = text_bytes

                if text_bytes > self._config.max_element_bytes and self._artifact_store:
                    ref = self._artifact_store.store_text(text)
                    excerpt_len = self._config.element_excerpt_bytes
                    doc_entry["content_ref"] = {
                        "hash": ref.hash,
                        "path": ref.path,
                        "mime_type": ref.mime_type,
                        "encoding": ref.encoding,
                    }
                    doc_entry["excerpt"] = (
                        text[:excerpt_len] + "\n[TRUNCATED - see artifact for full content]"
                    )
                else:
                    doc_entry["content"] = text

            result["documents"].append(doc_entry)

        return result

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

    def _extract_breakdown(self, structured: dict[str, Any]) -> dict[str, int]:
        """Extract size breakdown from already-structured content.

        Uses metadata computed during structuring (which has access to full
        image data) rather than recalculating from LMNR attributes (where
        base64 image data is stripped).
        """
        if structured.get("type") == "llm_messages":
            metadata = structured.get("metadata", {})
            return {
                "text_bytes": metadata.get("total_text_bytes", 0),
                "image_bytes": metadata.get("total_image_bytes", 0),
                "tool_bytes": metadata.get("total_tool_bytes", 0),
            }
        elif "size_bytes" in structured:
            return {"total_bytes": structured["size_bytes"]}
        else:
            serialized = json.dumps(self._convert_types(structured))
            return {"total_bytes": len(serialized.encode("utf-8"))}

    def _reduce_previews(self, structured: dict[str, Any]) -> dict[str, Any]:
        """Reduce preview/excerpt sizes to fit file under max_file_bytes."""
        if structured.get("type") == "llm_messages":
            # Reduce excerpt sizes in messages
            for msg in structured.get("messages", []):
                for part in msg.get("parts", []):
                    if "excerpt" in part:
                        # Reduce to 500 bytes
                        part["excerpt"] = part["excerpt"][:500] + "\n[TRUNCATED]"
        return structured

    def _redact(self, text: str) -> str:
        """Apply redaction patterns to text."""
        for pattern in self._compiled_patterns:
            text = pattern.sub("[REDACTED]", text)
        return text

    def _convert_types(self, value: Any, seen: set | None = None) -> Any:
        """Convert non-serializable types recursively with cycle detection."""
        # Cycle detection
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
                return sorted(str(x) for x in value)
            case BaseModel():
                try:
                    return value.model_dump(mode="json")
                except Exception:
                    return str(value)
            case dict():
                seen.add(obj_id)
                result = {str(k): self._convert_types(v, seen) for k, v in value.items()}
                seen.discard(obj_id)
                return result
            case list() | tuple():
                seen.add(obj_id)
                result = [self._convert_types(x, seen) for x in value]
                seen.discard(obj_id)
                return result
            case _:
                # Try str() as fallback
                try:
                    return str(value)
                except Exception:
                    return f"<{type(value).__name__}>"


def reconstruct_span_content(trace_root: Path, span_dir: Path, content_type: str) -> dict[str, Any]:
    """Reconstruct full content from input.yaml/output.yaml + artifacts.

    Args:
        trace_root: Trace root directory
        span_dir: Span directory containing input.yaml or output.yaml
        content_type: "input" or "output"

    Returns:
        Complete reconstructed content with all artifact refs resolved
    """
    content_path = span_dir / f"{content_type}.yaml"
    if not content_path.exists():
        return {}

    content = yaml.safe_load(content_path.read_text(encoding="utf-8"))
    return _rehydrate(content, trace_root)


def _rehydrate(obj: Any, trace_root: Path) -> Any:
    """Recursively replace content_ref entries with actual content."""
    if isinstance(obj, dict):
        if "content_ref" in obj:
            # This is an artifact reference - load the full content
            ref = obj["content_ref"]
            artifact_path = trace_root / ref["path"]

            if ref.get("encoding") == "utf-8":
                full_content = artifact_path.read_text(encoding="utf-8")
            else:
                full_content = artifact_path.read_bytes()

            # Replace ref with full content
            obj = obj.copy()
            obj["content"] = full_content
            del obj["content_ref"]
            if "excerpt" in obj:
                del obj["excerpt"]

        return {k: _rehydrate(v, trace_root) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [_rehydrate(v, trace_root) for v in obj]

    return obj
