"""Tool base class for LLM function calling.

Provides the public Tool class with import-time validation, schema generation,
and supporting types (ToolOutput, ToolCallRecord).

Tools are regular Python classes (not Pydantic models) with runtime state in __init__
and an async execute() method called by the tool loop.
"""

import inspect
import re
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

__all__ = ["Tool", "ToolCallRecord", "ToolOutput", "generate_tool_schema", "to_snake_case"]

_SNAKE_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_RESERVED_FIELD_NAMES = frozenset({"strict", "additionalProperties"})


def to_snake_case(name: str) -> str:
    """Convert PascalCase to snake_case, handling consecutive capitals.

    >>> to_snake_case("HTTPClient")
    'http_client'
    >>> to_snake_case("GetWeather")
    'get_weather'
    """
    return _SNAKE_RE.sub("_", name).lower()


class ToolOutput(BaseModel):
    """Base for tool outputs. ``content`` is sent to the LLM as the tool result.

    Subclass to add metadata fields accessible to the caller but not sent to the LLM.
    """

    model_config = ConfigDict(frozen=True)

    content: str


class Tool:
    """Base class for LLM tools with import-time validation.

    Subclasses must define:
    - A non-empty docstring (becomes the LLM tool description)
    - An ``Input`` inner class (BaseModel with Field(description=...) on every field)
    - An ``async def execute(self, input: Input) -> ToolOutput`` method

    Optionally define an ``Output`` inner class extending ToolOutput for typed metadata.

    Tools are regular classes — use ``__init__`` for runtime state (API clients,
    lookup tables, other Conversations). ``execute()`` is called once per tool
    invocation by the tool loop.
    """

    Input: type[BaseModel]
    Output: type[ToolOutput]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        name = cls.__name__

        if not cls.__doc__ or not cls.__doc__.strip():
            raise TypeError(f"Tool '{name}' must define a non-empty docstring. The docstring becomes the LLM tool description.")

        if "Input" not in cls.__dict__:
            raise TypeError(
                f"Tool '{name}' must define an 'Input' inner class (BaseModel). Example: class Input(BaseModel): query: str = Field(description='...')"
            )
        input_cls = cls.__dict__["Input"]
        if not isinstance(input_cls, type) or not issubclass(input_cls, BaseModel):
            raise TypeError(f"Tool '{name}'.Input must be a BaseModel subclass")

        # Validate all Input fields have descriptions and no reserved names
        for field_name, field_info in input_cls.model_fields.items():
            if field_info.description is None:
                raise TypeError(
                    f"Tool '{name}'.Input field '{field_name}' must use Field(description='...'). All Input fields require descriptions for the LLM."
                )
            if field_name in _RESERVED_FIELD_NAMES:
                raise TypeError(
                    f"Tool '{name}'.Input field '{field_name}' uses a reserved name that collides "
                    f"with JSON Schema or LiteLLM keywords. LiteLLM strips '{field_name}' from "
                    f"schemas for some providers (Gemini, xAI), causing required/properties mismatches. "
                    f"Rename the field (e.g., '{field_name}_value', '{field_name}_mode')."
                )

        # Validate Input schema is compatible with OpenAI strict mode
        _validate_strict_mode_compatibility(input_cls.model_json_schema(), name)

        # Validate Output if defined
        if "Output" in cls.__dict__:
            output_cls = cls.__dict__["Output"]
            if not isinstance(output_cls, type) or not issubclass(output_cls, ToolOutput):
                raise TypeError(f"Tool '{name}'.Output must extend ToolOutput")
        else:
            cls.Output = ToolOutput

        # Validate execute method
        if "execute" not in cls.__dict__:
            raise TypeError(f"Tool '{name}' must define an 'async def execute(self, input: Input) -> ToolOutput' method")
        execute_method = cls.__dict__["execute"]
        if not inspect.iscoroutinefunction(execute_method):
            raise TypeError(f"Tool '{name}'.execute must be async (async def execute)")

    async def execute(self, input: Any) -> ToolOutput:
        """Execute the tool with validated input and return the result."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """Record of a single tool call execution within the tool loop."""

    tool: type[Tool]
    input: BaseModel
    output: ToolOutput
    round: int


def _validate_strict_mode_compatibility(schema: dict[str, Any], tool_name: str) -> None:
    """Validate schema has no dict[str, V] types incompatible with OpenAI strict mode."""
    _check_strict_node(schema, tool_name, "Input")


def _check_strict_node(node: dict[str, Any], tool_name: str, path: str) -> None:
    """Recursively check schema node for dict types incompatible with strict mode."""
    empty: dict[str, Any] = {}
    if node.get("type") == "object" and isinstance(node.get("additionalProperties"), dict) and not node.get("properties"):
        raise TypeError(
            f"Tool '{tool_name}'.Input has a dict type at '{path}' which is incompatible "
            f"with OpenAI strict mode. dict[str, V] produces dynamic-key objects that "
            f"cannot be represented in strict-mode JSON schemas. "
            f"Replace with list[SomeModel] where SomeModel has explicit fields."
        )
    for prop_name, prop_schema in node.get("properties", empty).items():
        if isinstance(prop_schema, dict):
            _check_strict_node(prop_schema, tool_name, f"{path}.{prop_name}")
    for def_name, def_schema in node.get("$defs", empty).items():
        if isinstance(def_schema, dict):
            _check_strict_node(def_schema, tool_name, f"$defs.{def_name}")
    for key in ("allOf", "anyOf", "oneOf"):
        for i, item in enumerate(node.get(key, [])):
            if isinstance(item, dict):
                _check_strict_node(item, tool_name, f"{path}.{key}[{i}]")
    items = node.get("items")
    if isinstance(items, dict):
        _check_strict_node(items, tool_name, f"{path}.items")


def _make_strict_schema(schema: dict[str, Any]) -> None:
    """Recursively add additionalProperties: false and ensure all properties are required.

    Required by OpenAI strict mode. LiteLLM silently strips strict/additionalProperties
    for providers that don't support it (Gemini, xAI).
    """
    empty_dict: dict[str, Any] = {}
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        properties = cast(dict[str, Any], schema.get("properties", empty_dict))
        # Strict mode requires ALL properties in the required list
        schema["required"] = [str(key) for key in properties]
        for prop in properties.values():
            if isinstance(prop, dict):
                _make_strict_schema(cast(dict[str, Any], prop))
    # Handle $defs for nested models
    definitions = cast(dict[str, Any], schema.get("$defs", empty_dict))
    for definition in definitions.values():
        if isinstance(definition, dict):
            _make_strict_schema(cast(dict[str, Any], definition))
    # Handle allOf, anyOf, oneOf
    for key in ("allOf", "anyOf", "oneOf"):
        branch_items = cast(list[Any], schema.get(key, []))
        for item in branch_items:
            if isinstance(item, dict):
                _make_strict_schema(cast(dict[str, Any], item))
    # Handle items for arrays
    items = schema.get("items")
    if isinstance(items, dict):
        _make_strict_schema(cast(dict[str, Any], items))


def generate_tool_schema(tool: Tool) -> dict[str, Any]:
    """Generate OpenAI Chat Completions tool schema from a Tool instance.

    Example::

        schema = generate_tool_schema(my_tool_instance)
        # {"type": "function", "function": {"name": "...", "description": "...", ...}}
    """
    tool_cls = type(tool)
    schema = tool_cls.Input.model_json_schema()
    _make_strict_schema(schema)
    return {
        "type": "function",
        "function": {
            "name": to_snake_case(tool_cls.__name__),
            "description": dedent(tool_cls.__doc__ or "").strip(),
            "parameters": schema,
            "strict": True,
        },
    }
