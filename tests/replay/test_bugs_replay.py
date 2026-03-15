"""Bug-proving tests for replay override issues.

Bugs C1, C2, C3 from CORE-BUGS.md:
- C1: response_format override silently dropped on unstructured calls
- C2: New tools dropped when any recorded tool matches
- C3: Model override partially ignored for non-Conversation replay
"""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from ai_pipeline_core.llm.tools import Tool, ToolOutput
from ai_pipeline_core.replay._execute import _apply_overrides, _override_tools_in_recorded_order


# ── Helpers ─────────────────────────────────────────────────────────────────


@dataclass
class FakeOverrides:
    """Minimal stand-in for ExperimentOverrides."""

    model: str | None = None
    model_options: dict[str, Any] | None = None
    response_format: type[BaseModel] | None = None
    tools: dict[str, Tool] | None = None


class SearchTool(Tool):
    """Search tool."""

    class Input(BaseModel):
        query: str = Field(description="Query")

    async def execute(self, input: Input) -> ToolOutput:
        return ToolOutput(content="results")


class SummarizeTool(Tool):
    """Summarize tool."""

    class Input(BaseModel):
        text: str = Field(description="Text to summarize")

    async def execute(self, input: Input) -> ToolOutput:
        return ToolOutput(content="summary")


class NewTool(Tool):
    """Brand new tool not in recording."""

    class Input(BaseModel):
        data: str = Field(description="Data")

    async def execute(self, input: Input) -> ToolOutput:
        return ToolOutput(content="new result")


# ── C1: response_format override silently dropped ───────────────────────────


def test_response_format_override_applied_when_key_absent() -> None:
    """When overriding an unstructured call to structured, response_format must be applied
    even if 'response_format' key was not in the original arguments.

    Currently: line 119 requires 'response_format' in normalized_arguments — silently drops override.
    """

    class OutputModel(BaseModel):
        answer: str

    receiver = None
    arguments: dict[str, Any] = {"model": "gpt-5", "model_options": None}
    # No "response_format" key in arguments — this was an unstructured call

    overrides = FakeOverrides(response_format=OutputModel)
    _, new_args = _apply_overrides(receiver=receiver, arguments=arguments, overrides=overrides)

    assert isinstance(new_args, dict)
    assert new_args.get("response_format") is OutputModel, "response_format override should be applied even when key was absent in original arguments"


# ── C2: New tools dropped when any recorded tool matches ────────────────────


def test_override_tools_appends_new_tools_after_matching() -> None:
    """When override tools include both matching and new tools, the new ones
    must be appended — not silently dropped.

    Currently: line 64 'return ordered or list(override_tools.values())'
    fallback only fires when ordered is empty. If any tool matched, new-only
    tools are lost.
    """
    recorded_value = [
        {"name": "search_tool", "class_path": "...", "constructor_args": {}},
    ]
    override_tools = {
        "search_tool": SearchTool(),
        "new_tool": NewTool(),
    }

    result = _override_tools_in_recorded_order(recorded_value, override_tools)

    tool_types = [type(t) for t in result]
    assert SearchTool in tool_types, "Matched tool should be present"
    assert NewTool in tool_types, "New tool should be appended, not dropped"
    assert len(result) == 2


def test_override_tools_preserves_recorded_order_and_appends_new() -> None:
    """Matching tools appear in recorded order; new tools are appended at end."""
    recorded_value = [
        {"name": "summarize_tool", "class_path": "...", "constructor_args": {}},
        {"name": "search_tool", "class_path": "...", "constructor_args": {}},
    ]
    override_tools = {
        "search_tool": SearchTool(),
        "summarize_tool": SummarizeTool(),
        "new_tool": NewTool(),
    }

    result = _override_tools_in_recorded_order(recorded_value, override_tools)

    # SummarizeTool first (recorded order), then SearchTool, then NewTool (appended)
    assert isinstance(result[0], SummarizeTool)
    assert isinstance(result[1], SearchTool)
    assert len(result) == 3, f"Expected 3 tools (2 matched + 1 new), got {len(result)}"


# ── C3: Model override partially ignored for non-Conversation replay ────────


def test_model_override_applies_to_constructor_args_receiver() -> None:
    """When replaying a Task/Flow span, model override must apply to receiver
    constructor args (e.g., receiver['value']['model']), not just Conversation.

    Currently: _apply_overrides only handles isinstance(receiver['value'], Conversation).
    """
    receiver = {
        "mode": "constructor_args",
        "value": {"model": "old-model", "model_options": None},
    }
    arguments: dict[str, Any] = {"input_data": "test"}

    overrides = FakeOverrides(model="new-model")
    new_receiver, _ = _apply_overrides(receiver=receiver, arguments=arguments, overrides=overrides)

    assert isinstance(new_receiver, dict)
    assert new_receiver["value"]["model"] == "new-model", "Model override should apply to constructor_args receiver, not just Conversation"
