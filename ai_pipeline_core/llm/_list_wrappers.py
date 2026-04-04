"""List output wrapper utilities for LLM structured output.

All LLM providers (OpenAI, Anthropic, OpenRouter) require object-root JSON schemas
in strict mode. When the user requests ``list[BaseModel]`` output, the framework
wraps it in a single-field object with a semantically derived key name, sends that
to the provider, and unwraps the response transparently.
"""

import json
import re
from functools import lru_cache
from typing import Any, get_args, get_origin

from pydantic import BaseModel, ConfigDict, create_model

from ai_pipeline_core._llm_core.model_response import ModelResponse

__all__ = [
    "get_list_field_name",
    "get_list_item_type",
    "get_or_create_wrapper",
    "is_list_output_type",
    "unwrap_list_response",
]

# Sibilant endings that need -es for pluralization
_SIBILANT_RE = re.compile(r"(s|sh|ch|x|z)$")


def is_list_output_type(output_type: Any) -> bool:
    """Check if the output type is ``list[BaseModel]``."""
    if get_origin(output_type) is not list:
        return False
    args = get_args(output_type)
    return bool(args) and isinstance(args[0], type) and issubclass(args[0], BaseModel)


def get_list_item_type(output_type: Any) -> type[BaseModel]:
    """Extract the item type T from ``list[T]``. Caller must verify with ``is_list_output_type`` first."""
    return get_args(output_type)[0]


def _derive_list_field_name(item_type: type[BaseModel]) -> str:
    """Derive a semantic plural field name from the item type.

    ``TaskGroupModel`` -> ``task_groups``
    ``Finding`` -> ``findings``
    ``Analysis`` -> ``analysiss`` (imperfect but deterministic)
    """
    name = item_type.__name__
    # Strip common suffixes
    for suffix in ("Model", "Schema", "Output", "Result"):
        if name.endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]
            break
    # PascalCase to snake_case
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    snake = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", snake).lower()
    # Pluralize conservatively
    if snake.endswith("y") and not snake.endswith(("ay", "ey", "oy", "uy")):
        return snake[:-1] + "ies"
    if _SIBILANT_RE.search(snake):
        return snake + "es"
    return snake + "s"


@lru_cache(maxsize=128)
def get_or_create_wrapper(item_type: type[BaseModel]) -> type[BaseModel]:
    """Get or create a wrapper model for the given item type.

    The wrapper has a single field with a semantically derived name containing
    a tuple of the item type. Cached by item_type identity.
    """
    field_name = _derive_list_field_name(item_type)
    wrapper_name = f"_ListWrapper_{item_type.__name__}"
    wrapper: type[BaseModel] = create_model(
        wrapper_name,
        __config__=ConfigDict(frozen=True, extra="forbid"),
        **{field_name: (tuple[item_type, ...], ...)},  # type: ignore[valid-type]
    )
    return wrapper


def get_list_field_name(wrapper_type: type[BaseModel]) -> str:
    """Get the field name containing the list in a wrapper model."""
    fields = list(wrapper_type.model_fields.keys())
    return fields[0]


def unwrap_list_response(response: ModelResponse[Any], item_type: type[BaseModel]) -> ModelResponse[Any]:
    """Unwrap a wrapper ModelResponse into a list ModelResponse.

    Extracts the list from the wrapper's field, re-serializes content as JSON array,
    and returns a new ModelResponse with the unwrapped list as parsed.
    """
    wrapper = get_or_create_wrapper(item_type)
    field_name = get_list_field_name(wrapper)
    parsed_wrapper = response.parsed
    unwrapped_items = list(getattr(parsed_wrapper, field_name))
    array_json = json.dumps([item.model_dump(mode="json") for item in unwrapped_items], indent=2)
    return response.model_copy(update={"content": array_json, "parsed": unwrapped_items})  # nosemgrep: no-document-model-copy
