"""Tool base class for LLM function calling.

Provides the public Tool class with import-time validation, schema generation,
and supporting types (ToolOutput, ToolCallRecord).

Tools are regular Python classes (not Pydantic models) with runtime state in __init__
and a framework-owned async execute() method wrapped around user-defined run().
"""

import asyncio
import inspect
import json
import logging
import re
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict

__all__ = ["Tool", "ToolCallRecord", "ToolOutput"]

logger = logging.getLogger(__name__)

_SNAKE_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_RESERVED_FIELD_NAMES = frozenset({"strict", "additionalProperties"})


def _to_snake_case(name: str) -> str:
    """Convert PascalCase to snake_case, handling consecutive capitals.

    >>> _to_snake_case("HTTPClient")
    'http_client'
    >>> _to_snake_case("GetWeather")
    'get_weather'
    """
    return _SNAKE_RE.sub("_", name).lower()


class ToolOutput(BaseModel):
    """Container for serialized tool output sent to the LLM and caller metadata."""

    model_config = ConfigDict(frozen=True)

    content: str
    data: Any = None


class Tool:
    """Base class for LLM tools with import-time validation and a sealed lifecycle.

    Subclasses must define:
    - A non-empty docstring (becomes the LLM tool description)
    - An ``Input`` inner class (BaseModel with Field(description=...) on every field)
    - An ``Output`` inner class (BaseModel)
    - An ``async def run(self, input: Input) -> Output`` method

    Tool authors should not override ``execute()``. The framework owns:
    retries, timeout, structured error handling, and serialization.
    """

    Input: type[BaseModel]
    Output: type[BaseModel]
    _abstract_tool: ClassVar[bool] = False
    name: ClassVar[str]
    _tool_spec: ClassVar[Any]
    retries: ClassVar[int] = 0
    retry_delay_seconds: ClassVar[float] = 2.0
    timeout_seconds: ClassVar[int] = 120
    max_response_bytes: ClassVar[int | None] = None
    handled_exceptions: ClassVar[tuple[type[Exception], ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.name = _to_snake_case(cls.__name__)

        if cls.__dict__.get("_abstract_tool", False) is True:
            return

        if "execute" in cls.__dict__:
            raise TypeError(
                f"Tool '{cls.name}' must not override execute(). "
                f"Implement 'async def run(self, input: Input) -> Output' instead. "
                f"The framework owns execute() and wraps run() with retry, timeout, and error handling."
            )

        _validate_tool_class(cls)

    async def run(self, input: Any) -> BaseModel:
        """Override this method with your tool logic. Return self.Output(...)."""
        raise NotImplementedError

    def _is_retryable(self, error: Exception) -> bool:  # noqa: PLR6301 — overridable hook, self used by subclasses
        """Classify whether a handled exception should be retried. Default: False."""
        return False

    def handle_error(self, error: Exception) -> ToolOutput:
        """Format a handled exception into a caller-facing ToolOutput."""
        return ToolOutput(content=f"Error: Tool '{self.name}' failed: {error}")

    async def execute(self, input: BaseModel) -> ToolOutput:
        """Execute tool lifecycle (retry, timeout, errors, serialization)."""
        result: BaseModel | None = None
        for attempt in range(self.retries + 1):
            try:
                result = await asyncio.wait_for(self.run(input), timeout=self.timeout_seconds)
            except TimeoutError as timeout_exc:
                if attempt < self.retries:
                    logger.warning("Tool '%s' timed out (attempt %d/%d), retrying", self.name, attempt + 1, self.retries + 1, exc_info=timeout_exc)
                    await asyncio.sleep(self.retry_delay_seconds)
                    continue
                logger.warning("Tool '%s' timed out after %d attempts", self.name, self.retries + 1, exc_info=timeout_exc)
                return ToolOutput(content=f"Error: Tool '{self.name}' timed out after {self.timeout_seconds}s ({self.retries + 1} attempts).")
            except self.handled_exceptions as error:
                if self._is_retryable(error) and attempt < self.retries:
                    logger.warning("Tool '%s' failed (attempt %d/%d), retrying", self.name, attempt + 1, self.retries + 1, exc_info=error)
                    await asyncio.sleep(self.retry_delay_seconds)
                    continue
                try:
                    return self.handle_error(error)
                except Exception:  # noqa: BLE001 — re-raise original if handle_error bugs
                    raise error from None
            else:
                break

        expected_output = type(self).Output
        if not isinstance(result, expected_output):
            raise TypeError(
                f"Tool '{self.name}'.run() must return {expected_output.__name__}, got {type(result).__name__}. Return self.Output(...) from run()."
            )

        data = result.model_dump(mode="json")
        content = json.dumps(data, indent=2)

        content_bytes = len(content.encode("utf-8"))
        if self.max_response_bytes is not None and content_bytes > self.max_response_bytes:
            return ToolOutput(
                content=json.dumps({
                    "error": "response_too_large",
                    "message": f"Tool '{self.name}' response exceeds {self.max_response_bytes} bytes. Use narrower filters or pagination.",
                    "actual_bytes": content_bytes,
                }),
                data=result,
            )

        return ToolOutput(content=content, data=result)


def _validate_tool_class(cls: type[Tool]) -> None:  # noqa: C901, PLR0912
    """Validate a concrete Tool subclass at definition time."""
    name = cls.name

    if not cls.__doc__ or not cls.__doc__.strip():
        raise TypeError(f"Tool '{name}' must define a non-empty docstring. The docstring becomes the LLM tool description.")

    if "Input" in cls.__dict__:
        input_cls = cls.__dict__["Input"]
        if not isinstance(input_cls, type) or not issubclass(input_cls, BaseModel):
            raise TypeError(f"Tool '{name}'.Input must be a BaseModel subclass")
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
        _validate_strict_mode_compatibility(input_cls.model_json_schema(), name)
    elif not getattr(cls, "_tool_spec", None):
        raise TypeError(f"Tool '{name}' must define an 'Input' inner class (BaseModel). Example: class Input(BaseModel): query: str = Field(description='...')")

    if "Output" in cls.__dict__:
        output_cls = cls.__dict__["Output"]
        if not isinstance(output_cls, type) or not issubclass(output_cls, BaseModel):
            raise TypeError(f"Tool '{name}'.Output must be a BaseModel subclass")
        if issubclass(output_cls, ToolOutput):
            raise TypeError(f"Tool '{name}'.Output must extend BaseModel, not ToolOutput. The framework creates ToolOutput internally.")
    elif not getattr(cls, "_tool_spec", None):
        raise TypeError(f"Tool '{name}' must define an 'Output' inner class (BaseModel) or inherit one from a validated parent tool.")

    if "run" in cls.__dict__:
        if not inspect.iscoroutinefunction(cls.__dict__["run"]):
            raise TypeError(f"Tool '{name}'.run must be async (async def run)")
    elif not getattr(cls, "_tool_spec", None):
        raise TypeError(f"Tool '{name}' must define an 'async def run(self, input: Input) -> Output' method or inherit one from a validated parent tool.")

    _validate_lifecycle_classvars(cls, name)
    cls._tool_spec = True


def _validate_lifecycle_classvars(cls: type[Tool], name: str) -> None:
    """Validate lifecycle ClassVars and error handling methods on a concrete Tool."""
    if cls.retries < 0:
        raise TypeError(f"Tool '{name}' has invalid retries={cls.retries}. Use a value >= 0.")
    if cls.retry_delay_seconds <= 0:
        raise TypeError(f"Tool '{name}' has invalid retry_delay_seconds={cls.retry_delay_seconds}. Use a value > 0.")
    if cls.timeout_seconds <= 0:
        raise TypeError(f"Tool '{name}' has invalid timeout_seconds={cls.timeout_seconds}. Use a value > 0.")
    if cls.max_response_bytes is not None and cls.max_response_bytes <= 0:
        raise TypeError(f"Tool '{name}' has invalid max_response_bytes={cls.max_response_bytes}. Use None or a value > 0.")
    if not isinstance(cls.handled_exceptions, tuple):
        raise TypeError(f"Tool '{name}' handled_exceptions must be a tuple of Exception subclasses.")
    if any(not isinstance(exc_type, type) or not issubclass(exc_type, Exception) for exc_type in cls.handled_exceptions):
        raise TypeError(f"Tool '{name}' handled_exceptions must contain only Exception subclasses.")
    if inspect.iscoroutinefunction(cls._is_retryable):
        raise TypeError(f"Tool '{name}' _is_retryable must be sync; async definitions break execute() control flow.")
    if inspect.iscoroutinefunction(cls.handle_error):
        raise TypeError(f"Tool '{name}' handle_error must be sync; async definitions break execute() control flow.")


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


def _generate_tool_schema(tool: Tool) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]  # used by conversation.py
    """Generate OpenAI Chat Completions tool schema from a Tool instance.

    Example::

        schema = _generate_tool_schema(my_tool_instance)
        # {"type": "function", "function": {"name": "...", "description": "...", ...}}
    """
    tool_cls = type(tool)
    schema = tool_cls.Input.model_json_schema()
    _make_strict_schema(schema)
    return {
        "type": "function",
        "function": {
            "name": tool_cls.name,
            "description": dedent(tool_cls.__doc__ or "").strip(),
            "parameters": schema,
            "strict": True,
        },
    }
