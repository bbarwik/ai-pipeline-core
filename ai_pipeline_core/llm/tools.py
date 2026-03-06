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
from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = ["Tool", "ToolCallRecord", "ToolOutput", "generate_tool_schema", "to_snake_case"]

_SNAKE_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


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

        # Validate all Input fields have descriptions
        for field_name, field_info in input_cls.model_fields.items():
            if field_info.description is None:
                raise TypeError(
                    f"Tool '{name}'.Input field '{field_name}' must use Field(description='...'). All Input fields require descriptions for the LLM."
                )

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


def _make_strict_schema(schema: dict[str, Any]) -> None:
    """Recursively add additionalProperties: false and ensure all properties are required.

    Required by OpenAI strict mode. LiteLLM silently strips strict/additionalProperties
    for providers that don't support it (Gemini, xAI).
    """
    empty_dict: dict[str, Any] = {}
    empty_list: list[Any] = []
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        # Strict mode requires ALL properties in the required list
        if "properties" in schema:
            schema["required"] = list(schema["properties"].keys())
        for prop in schema.get("properties", empty_dict).values():
            if isinstance(prop, dict):
                _make_strict_schema(prop)
    # Handle $defs for nested models
    for definition in schema.get("$defs", empty_dict).values():
        if isinstance(definition, dict):
            _make_strict_schema(definition)
    # Handle allOf, anyOf, oneOf
    for key in ("allOf", "anyOf", "oneOf"):
        for item in schema.get(key, empty_list):
            if isinstance(item, dict):
                _make_strict_schema(item)
    # Handle items for arrays
    if "items" in schema and isinstance(schema["items"], dict):
        _make_strict_schema(schema["items"])


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
