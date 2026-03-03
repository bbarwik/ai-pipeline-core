"""Tests for llm/tools.py: Tool base class, schema generation, and validation."""

import pytest
from pydantic import BaseModel, Field

from ai_pipeline_core.llm.tools import Tool, ToolCallRecord, ToolOutput, to_snake_case, generate_tool_schema


# ── Valid Tool for reuse across tests ────────────────────────────────────────


class GetWeather(Tool):
    """Get the current weather for a location."""

    class Input(BaseModel):
        city: str = Field(description="City name")
        unit: str = Field(default="celsius", description="Temperature unit")

    async def execute(self, input: BaseModel) -> ToolOutput:
        return ToolOutput(content=f"Weather in {input.city}: 20°")  # type: ignore[attr-defined]


class CustomOutputTool(Tool):
    """Tool with custom output."""

    class Input(BaseModel):
        query: str = Field(description="Search query")

    class Output(ToolOutput):
        source: str = "web"

    async def execute(self, input: BaseModel) -> ToolOutput:
        return self.Output(content="result", source="cache")


# ── Definition-time validation ───────────────────────────────────────────────


def test_tool_definition_valid() -> None:
    """Valid Tool subclass raises no errors."""
    assert GetWeather.__doc__
    assert issubclass(GetWeather.Input, BaseModel)


def test_tool_definition_missing_docstring() -> None:
    with pytest.raises(TypeError, match="must define a non-empty docstring"):

        class BadTool(Tool):
            class Input(BaseModel):
                x: int = Field(description="x value")

            async def execute(self, input: BaseModel) -> ToolOutput:
                return ToolOutput(content="")


def test_tool_definition_missing_input() -> None:
    with pytest.raises(TypeError, match="must define an 'Input' inner class"):

        class BadTool(Tool):
            """Missing Input."""

            async def execute(self, input: BaseModel) -> ToolOutput:
                return ToolOutput(content="")


def test_tool_definition_input_not_basemodel() -> None:
    with pytest.raises(TypeError, match="Input must be a BaseModel subclass"):
        # Dynamic class creation avoids static type checker flagging the intentionally invalid override
        type("BadTool", (Tool,), {"__doc__": "Bad input.", "Input": type("Input", (), {})})


def test_tool_definition_input_field_missing_description() -> None:
    with pytest.raises(TypeError, match="must use Field\\(description="):

        class BadTool(Tool):
            """Missing field description."""

            class Input(BaseModel):
                query: str  # no Field(description=...)

            async def execute(self, input: BaseModel) -> ToolOutput:
                return ToolOutput(content="")


def test_tool_definition_non_async_execute() -> None:
    with pytest.raises(TypeError, match="execute must be async"):

        class BadTool(Tool):
            """Sync execute."""

            class Input(BaseModel):
                x: int = Field(description="x")

            def execute(self, input: BaseModel) -> ToolOutput:  # type: ignore[override]
                return ToolOutput(content="")


def test_tool_definition_missing_execute() -> None:
    with pytest.raises(TypeError, match="must define an 'async def execute"):

        class BadTool(Tool):
            """No execute."""

            class Input(BaseModel):
                x: int = Field(description="x")


def test_tool_definition_invalid_output_class() -> None:
    class BadOutput(BaseModel):
        content: str

    with pytest.raises(TypeError, match="Output must extend ToolOutput"):
        # Dynamic class creation avoids static type checker flagging the intentionally invalid override
        type("BadTool", (Tool,), {"__doc__": "Bad output.", "Input": GetWeather.Input, "Output": BadOutput})


def test_tool_with_custom_output() -> None:
    """Tool with custom Output extending ToolOutput is valid."""
    assert issubclass(CustomOutputTool.Output, ToolOutput)


# ── Schema generation ────────────────────────────────────────────────────────


def test_generate_tool_schema_basic() -> None:
    schema = generate_tool_schema(GetWeather())
    assert schema["type"] == "function"
    func = schema["function"]
    assert func["name"] == "get_weather"
    assert "Get the current weather" in func["description"]
    assert func["strict"] is True
    params = func["parameters"]
    assert params["type"] == "object"
    assert "city" in params["properties"]
    assert "unit" in params["properties"]
    assert params["additionalProperties"] is False


def test_generate_tool_schema_strict_mode_nested() -> None:
    """Strict mode recursively adds additionalProperties: false."""

    class NestedTool(Tool):
        """Tool with nested schema."""

        class Input(BaseModel):
            location: GetWeather.Input = Field(description="Location data")

        async def execute(self, input: BaseModel) -> ToolOutput:
            return ToolOutput(content="")

    schema = generate_tool_schema(NestedTool())
    params = schema["function"]["parameters"]
    # Nested model must produce $defs with strict mode applied
    assert "$defs" in params, "Nested model should produce $defs"
    for definition in params["$defs"].values():
        assert definition.get("additionalProperties") is False


def test_generate_tool_schema_all_fields_required() -> None:
    """Strict mode ensures all fields are in required list."""
    schema = generate_tool_schema(GetWeather())
    params = schema["function"]["parameters"]
    assert set(params["required"]) == {"city", "unit"}


# ── to_snake_case ───────────────────────────────────────────────────────────


def test_snake_case_simple() -> None:
    assert to_snake_case("GetWeather") == "get_weather"


def test_snake_case_consecutive_capitals() -> None:
    assert to_snake_case("HTTPClient") == "http_client"


def test_snake_case_single_word() -> None:
    assert to_snake_case("Search") == "search"


def test_snake_case_already_lower() -> None:
    assert to_snake_case("search") == "search"


# ── ToolOutput ───────────────────────────────────────────────────────────────


def test_tool_output_frozen() -> None:
    from pydantic import ValidationError

    output = ToolOutput(content="hello")
    with pytest.raises(ValidationError):
        output.content = "world"  # type: ignore[misc]


# ── ToolCallRecord ───────────────────────────────────────────────────────────


def test_tool_call_record_frozen() -> None:
    record = ToolCallRecord(
        tool=GetWeather,
        input=GetWeather.Input(city="Paris"),
        output=ToolOutput(content="sunny"),
        round=1,
    )
    assert record.tool is GetWeather
    assert record.round == 1
