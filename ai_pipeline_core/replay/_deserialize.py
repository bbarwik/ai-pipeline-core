"""Two-pass deserialization for replay payloads.

Pass 1: Recursively walk parsed YAML, replacing $doc_ref dicts with Documents.
Pass 2: For task kwargs, validate dict values against function type hints (BaseModel).
"""

import importlib
import typing
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ai_pipeline_core.documents._context_vars import _suppress_document_registration
from ai_pipeline_core.documents.document import Document

from ._resolve import _find_document_class, resolve_document_ref
from .types import DocumentRef

__all__ = ["resolve_doc_refs", "resolve_task_kwargs"]


def _import_by_path(path: str) -> Any:
    """Import an object from 'module:qualname' format.

    Used for both function paths (e.g. 'my_package.tasks:extract') and
    class paths (e.g. 'my_package.models:SummaryOutput').
    """
    module_path, qualname = path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    obj: Any = module
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


def _create_inline_document(data: dict[str, Any]) -> Document:
    """Create an ephemeral Document from inline content (no $doc_ref)."""
    class_name = data["class_name"]
    name = data.get("name", "inline.txt")
    content = data["content"]
    doc_cls = _find_document_class(class_name)

    content_bytes = content.encode("utf-8") if isinstance(content, str) else content
    with _suppress_document_registration():
        return doc_cls(
            name=name,
            content=content_bytes,
            description=data.get("description"),
        )


def resolve_doc_refs(data: Any, store_base: Path) -> Any:
    """Recursively walk parsed YAML data, replacing $doc_ref dicts with Documents.

    Also handles inline content escape hatch: dicts with class_name + content
    but no $doc_ref create ephemeral Documents without store access.
    """
    if isinstance(data, dict):
        if "$doc_ref" in data:
            ref = DocumentRef.model_validate(data)
            return resolve_document_ref(ref, store_base)
        if "class_name" in data and "content" in data and "$doc_ref" not in data:
            return _create_inline_document(data)
        return {k: resolve_doc_refs(v, store_base) for k, v in data.items()}
    if isinstance(data, list):
        return [resolve_doc_refs(item, store_base) for item in data]
    return data


def _is_basemodel_subclass(hint: Any) -> bool:
    """Check if a type hint is a concrete BaseModel subclass (not Document)."""
    return isinstance(hint, type) and issubclass(hint, BaseModel) and not issubclass(hint, Document)


def resolve_task_kwargs(function_path: str, raw_kwargs: dict[str, Any], store_base: Path) -> dict[str, Any]:
    """Resolve task kwargs: $doc_ref -> Documents, dicts -> BaseModel via type hints.

    First resolves all document references, then inspects the target function's
    type hints to validate dict values as BaseModel instances.
    """
    fn = _import_by_path(function_path)
    hints = typing.get_type_hints(fn)
    resolved = resolve_doc_refs(raw_kwargs, store_base)

    for key, value in resolved.items():
        if not isinstance(value, dict):
            continue
        hint = hints.get(key)
        if hint is not None and _is_basemodel_subclass(hint):
            resolved[key] = hint.model_validate(value)

    return resolved
