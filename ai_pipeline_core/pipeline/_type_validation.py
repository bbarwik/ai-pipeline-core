"""Type validation helpers for pipeline decorator annotation checking.

Extracted from decorators.py to keep file sizes manageable.
Used at decoration time to validate input/output type annotations
on @pipeline_task and @pipeline_flow targets.
"""

__all__ = [
    "callable_name",
    "find_non_document_leaves",
    "flatten_union",
    "is_already_traced",
    "parse_document_types_from_annotation",
    "resolve_type_hints",
    "validate_input_types",
]

import inspect
import types
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from uuid import UUID

from pydantic import BaseModel

from ai_pipeline_core.documents import Document
from ai_pipeline_core.pipeline.options import FlowOptions


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def callable_name(obj: Any, fallback: str) -> str:
    """Safely extract callable's name for error messages."""
    try:
        n = getattr(obj, "__name__", None)
        return n if isinstance(n, str) else fallback
    except Exception:
        return fallback


_MAX_WRAPPED_DEPTH = 10


def is_already_traced(func: Callable[..., Any]) -> bool:
    """Check if a function has already been wrapped by the trace decorator."""
    if hasattr(func, "__is_traced__") and func.__is_traced__:  # type: ignore[attr-defined]
        return True
    current = func
    depth = 0
    while hasattr(current, "__wrapped__") and depth < _MAX_WRAPPED_DEPTH:
        wrapped = current.__wrapped__  # type: ignore[attr-defined]
        if hasattr(wrapped, "__is_traced__") and wrapped.__is_traced__:
            return True
        current = wrapped
        depth += 1
    return False


# --------------------------------------------------------------------------- #
# Annotation parsing helpers
# --------------------------------------------------------------------------- #
def flatten_union(tp: Any) -> list[Any]:
    """Flatten Union / X | Y annotations into a list of constituent types."""
    origin = get_origin(tp)
    if origin is Union or isinstance(tp, types.UnionType):
        result: list[Any] = []
        for arg in get_args(tp):
            result.extend(flatten_union(arg))
        return result
    return [tp]


def find_non_document_leaves(tp: Any) -> list[Any]:
    """Walk a return type annotation and collect leaf types that are not Document subclasses or NoneType.

    Returns empty list when all leaf types are valid (Document subclasses or None).
    Used by @pipeline_task to validate return annotations at decoration time.
    """
    if tp is type(None) or (isinstance(tp, type) and issubclass(tp, Document)):
        return []

    origin = get_origin(tp)

    # Union / X | Y: all branches must be valid
    if origin is Union or isinstance(tp, types.UnionType):
        return [leaf for arg in get_args(tp) for leaf in find_non_document_leaves(arg)]

    # list[X]: recurse into element type
    if origin is list:
        args = get_args(tp)
        return find_non_document_leaves(args[0]) if args else [tp]

    # tuple[X, Y] or tuple[X, ...]
    if origin is tuple:
        args = get_args(tp)
        if not args:
            return [tp]
        elements = (args[0],) if (len(args) == 2 and args[1] is Ellipsis) else args
        return [leaf for arg in elements for leaf in find_non_document_leaves(arg)]

    # Everything else is invalid (int, str, Any, object, dict, etc.)
    return [tp]


def parse_document_types_from_annotation(annotation: Any) -> list[type[Document]]:
    """Extract Document subclasses from a list[...] type annotation.

    Handles list[DocA], list[DocA | DocB], list[Union[DocA, DocB]].
    Returns empty list if annotation is not a list of Document subclasses.
    """
    origin = get_origin(annotation)
    if origin is not list:
        return []

    args = get_args(annotation)
    if not args:
        return []

    inner = args[0]
    flat = flatten_union(inner)

    return [t for t in flat if isinstance(t, type) and issubclass(t, Document)]


def resolve_type_hints(fn: Callable[..., Any]) -> dict[str, Any]:
    """Resolve type hints, raising TypeError on failure with the original cause."""
    try:
        return get_type_hints(fn, include_extras=True)
    except Exception as e:
        name = callable_name(fn, "unknown")
        raise TypeError(f"Failed to resolve type hints for '{name}': {e}") from e


# --------------------------------------------------------------------------- #
# Input type validation
# --------------------------------------------------------------------------- #

# Types allowed in pipeline_task/pipeline_flow input annotations
_VALID_SCALAR_TYPES: tuple[type, ...] = (str, int, float, bool, type(None), UUID, Path)


_VALID_CONTAINER_ORIGINS: set[type] = {list, tuple}


def _is_valid_input_type(tp: Any) -> bool:  # noqa: PLR0911
    """Check if a type annotation is a valid JSON-serializable input type."""
    if tp is type(None):
        return True

    if isinstance(tp, type):
        if tp in _VALID_SCALAR_TYPES or issubclass(tp, (Document, FlowOptions, Enum)):
            return True
        if issubclass(tp, BaseModel):
            return tp.model_config.get("frozen", False) is True
        return False

    origin = get_origin(tp)

    if origin is Union or isinstance(tp, types.UnionType):
        return all(_is_valid_input_type(arg) for arg in get_args(tp))

    if origin in _VALID_CONTAINER_ORIGINS:
        args = get_args(tp)
        if not args:
            return True
        if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
            return _is_valid_input_type(args[0])
        return all(_is_valid_input_type(a) for a in args if a is not Ellipsis)

    if origin is dict:
        args = get_args(tp)
        if not args:
            return True
        if len(args) == 2 and args[0] is str:
            return _is_valid_input_type(args[1])
        return False

    return tp is Any


def validate_input_types(fn: Callable[..., Any], hints: dict[str, Any]) -> None:
    """Validate that all input parameter types are JSON-serializable.

    Raises TypeError if any parameter has an unsupported type annotation.
    """
    sig = inspect.signature(fn)
    name = callable_name(fn, "unknown")

    for param_name in sig.parameters:
        if param_name == "return":
            continue
        tp = hints.get(param_name, Any)
        if tp is Any or tp is inspect.Parameter.empty:
            continue
        if not _is_valid_input_type(tp):
            raise TypeError(
                f"Parameter '{param_name}' of '{name}' has unsupported type {tp!r}. "
                f"Pipeline inputs must be JSON-serializable: str, int, float, bool, None, "
                f"UUID, Path, Enum, list, tuple, dict[str, ...], Document, BaseModel, or FlowOptions."
            )
