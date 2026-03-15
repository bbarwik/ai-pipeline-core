"""Bug-proving tests for Tool.Input schema incompatibilities.

Bugs A1 and A2 from NEW-BUGS-REPORT.md:
- A1: dict[str, V] in Tool.Input produces invalid OpenAI strict-mode schemas
- A2: Field named 'strict' collides with LiteLLM recursive key stripping
"""

import pytest
from pydantic import BaseModel, Field

from ai_pipeline_core.llm.tools import Tool


# ── A1: dict[str, V] in Tool.Input ──────────────────────────────────────────


def test_tool_definition_rejects_dict_field() -> None:
    """dict[str, V] produces dynamic-key objects incompatible with OpenAI strict mode.

    Tool.__init_subclass__ should reject dict fields at import time.
    """
    with pytest.raises(TypeError, match=r"dict type.*incompatible"):

        class DictTool(Tool):
            """Tool with dict field."""

            class Input(BaseModel):
                coins: dict[str, list[int]] = Field(description="coin -> timestamps mapping")

            class Output(BaseModel):
                result: str

            async def run(self, input: Input) -> Output:
                return self.Output(result="")


def test_tool_definition_rejects_nested_dict_in_list() -> None:
    """dict[str, V] nested inside list[...] is also incompatible with strict mode."""
    with pytest.raises(TypeError, match="dict type"):

        class NestedDictTool(Tool):
            """Tool with nested dict in list."""

            class Input(BaseModel):
                items: list[dict[str, str]] = Field(description="list of dicts")

            class Output(BaseModel):
                result: str

            async def run(self, input: Input) -> Output:
                return self.Output(result="")


def test_tool_definition_rejects_dict_in_referenced_model() -> None:
    """dict[str, V] inside a referenced model (appears in $defs) must also be rejected."""

    class ItemData(BaseModel):
        values: dict[str, int] = Field(description="key-value data")

    with pytest.raises(TypeError, match="dict type"):

        class RefDictTool(Tool):
            """Tool with dict in referenced model."""

            class Input(BaseModel):
                item: ItemData = Field(description="item with dict")

            class Output(BaseModel):
                result: str

            async def run(self, input: Input) -> Output:
                return self.Output(result="")


def test_tool_definition_rejects_optional_dict_field() -> None:
    """dict[str, V] | None wraps in anyOf — the dict branch must still be rejected."""
    with pytest.raises(TypeError, match="dict type"):

        class OptionalDictTool(Tool):
            """Tool with optional dict field."""

            class Input(BaseModel):
                data: dict[str, int] | None = Field(description="optional dict", default=None)

            class Output(BaseModel):
                result: str

            async def run(self, input: Input) -> Output:
                return self.Output(result="")


def test_tool_definition_allows_nested_basemodel() -> None:
    """Nested BaseModel with fixed properties remains valid."""

    class Entry(BaseModel):
        key: str = Field(description="key")
        value: int = Field(description="value")

    class GoodTool(Tool):
        """Tool with explicit nested model."""

        class Input(BaseModel):
            entries: list[Entry] = Field(description="entries")

        class Output(BaseModel):
            result: str

        async def run(self, input: Input) -> Output:
            return self.Output(result="")

    assert issubclass(GoodTool.Input, BaseModel)


def test_make_strict_schema_corrupts_dict_field() -> None:
    """Prove _make_strict_schema destroys dict[str, V] schema shape.

    This demonstrates the runtime consequence even without definition-time rejection:
    after _make_strict_schema, a dict field becomes an empty object with no properties
    but is still listed in required — causing OpenAI to reject it.
    """

    # Bypass __init_subclass__ validation by building schema directly from a plain model
    class DictInput(BaseModel):
        coins: dict[str, list[int]] = Field(description="coin timestamps")
        search_width: str | None = Field(description="search width", default=None)

    schema = DictInput.model_json_schema()
    # Before strict: coins has additionalProperties (dict schema) and no properties
    coins_before = schema["properties"]["coins"]
    assert "additionalProperties" in coins_before
    assert "properties" not in coins_before

    from ai_pipeline_core.llm.tools import _make_strict_schema

    _make_strict_schema(schema)

    # After strict: additionalProperties is clobbered to false, required is empty
    coins_after = schema["properties"]["coins"]
    assert coins_after.get("additionalProperties") is False
    assert coins_after.get("required") == []
    # Top-level required still has "coins" — schema mismatch
    assert "coins" in schema["required"]


# ── A2: Reserved field names ────────────────────────────────────────────────


def test_tool_definition_rejects_reserved_field_strict() -> None:
    """Field named 'strict' collides with LiteLLM recursive key stripping."""
    with pytest.raises(TypeError, match="reserved name"):

        class StrictTool(Tool):
            """Tool with reserved field name."""

            class Input(BaseModel):
                strict: bool = Field(description="strict mode", default=True)

            class Output(BaseModel):
                result: str

            async def run(self, input: Input) -> Output:
                return self.Output(result="")


def test_tool_definition_rejects_reserved_field_additional_properties() -> None:
    """Field named 'additionalProperties' collides with JSON Schema keyword."""
    with pytest.raises(TypeError, match="reserved name"):

        class APTool(Tool):
            """Tool with additionalProperties field."""

            class Input(BaseModel):
                additionalProperties: bool = Field(description="flag")

            class Output(BaseModel):
                result: str

            async def run(self, input: Input) -> Output:
                return self.Output(result="")


def test_tool_definition_allows_renamed_reserved_field() -> None:
    """Renamed variant of reserved name is valid."""

    class GoodTool(Tool):
        """Tool with renamed field."""

        class Input(BaseModel):
            strict_mode: bool = Field(description="Enable strict mode", default=True)

        class Output(BaseModel):
            result: str

        async def run(self, input: Input) -> Output:
            return self.Output(result="")

    assert "strict_mode" in GoodTool.Input.model_fields


def test_strict_field_collides_in_generated_schema() -> None:
    """Prove the schema collision: user field 'strict' appears inside properties
    at the same level as function-level 'strict': True.

    LiteLLM strips all 'strict' keys recursively — removing the user's property
    but leaving its name in 'required', causing provider rejection.
    """

    # Build schema manually from a plain model (bypassing __init_subclass__)
    class StrictInput(BaseModel):
        strict: bool = Field(description="strict mode", default=True)
        query: str = Field(description="search query")

    schema = StrictInput.model_json_schema()
    from ai_pipeline_core.llm.tools import _make_strict_schema

    _make_strict_schema(schema)

    # The generated tool envelope would have:
    tool_schema = {
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "test",
            "parameters": schema,
            "strict": True,  # function-level strict
        },
    }
    # Both function-level "strict" and property-level "strict" exist
    assert "strict" in tool_schema["function"]  # function-level
    assert "strict" in tool_schema["function"]["parameters"]["properties"]  # pyright: ignore[reportIndexIssue]
    # LiteLLM would strip BOTH, leaving 'strict' in required but missing from properties
    assert "strict" in tool_schema["function"]["parameters"]["required"]  # pyright: ignore[reportIndexIssue]
