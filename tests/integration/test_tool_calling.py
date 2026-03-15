"""Integration tests for tool calling across providers.

Tests the complete tool loop with real LLM APIs: grok-4.1-fast, gemini-3-flash, gpt-5-mini.
Each test is parameterized by model AND streaming mode to verify both paths.
"""

import pytest
from pydantic import BaseModel, Field

from ai_pipeline_core import Conversation, ModelOptions, Tool, ToolOutput
from ai_pipeline_core.settings import settings

HAS_API_KEYS = bool(settings.openai_api_key and settings.openai_base_url)


class CityInfo(BaseModel):
    city: str = Field(description="Capital city name")
    country: str = Field(description="Country name")


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_API_KEYS, reason="OpenAI API keys not configured"),
]

# Models to test tool calling
TOOL_MODELS = ("grok-4.1-fast", "gemini-3-flash", "gpt-5-mini")
STREAM_MODES = (False, True)


@pytest.fixture(params=[pytest.param((model, stream), id=f"{model}-{'stream' if stream else 'nostream'}") for model in TOOL_MODELS for stream in STREAM_MODES])
def model_opts(request: pytest.FixtureRequest) -> tuple[str, ModelOptions]:
    """Parameterized (model_name, ModelOptions) fixture for all model x stream combos."""
    model, stream = request.param
    return model, ModelOptions(stream=stream)


# ── Test Tools ───────────────────────────────────────────────────────────────


class Calculator(Tool):
    """Perform basic arithmetic calculations."""

    class Input(BaseModel):
        expression: str = Field(description="Mathematical expression to evaluate, e.g. '2 + 3'")

    async def execute(self, input: Input) -> ToolOutput:
        try:
            result = eval(input.expression, {"__builtins__": {}}, {})
            return ToolOutput(content=str(result))
        except (SyntaxError, TypeError, ValueError, ZeroDivisionError, NameError, ArithmeticError) as e:
            return ToolOutput(content=f"Error: {e}")


class GetCapital(Tool):
    """Get the capital city of a country."""

    class Input(BaseModel):
        country: str = Field(description="Country name")

    async def execute(self, input: Input) -> ToolOutput:
        capitals = {
            "france": "Paris",
            "japan": "Tokyo",
            "brazil": "Brasília",
            "germany": "Berlin",
        }
        country = input.country.lower()
        if country in capitals:
            return ToolOutput(content=f"The capital of {input.country} is {capitals[country]}")
        return ToolOutput(content=f"Unknown country: {input.country}")


class FailingTool(Tool):
    """A tool that always fails to test error handling."""

    class Input(BaseModel):
        reason: str = Field(description="Reason for failure")

    async def execute(self, input: Input) -> ToolOutput:
        raise RuntimeError(f"Intentional failure: {input.reason}")


class WeatherLookup(Tool):
    """Look up weather for a city."""

    class Input(BaseModel):
        city: str = Field(description="City to look up weather for")

    async def execute(self, input: Input) -> ToolOutput:
        return ToolOutput(content=f"The weather in {input.city} is sunny, 22°C")


# ── Integration Tests ────────────────────────────────────────────────────────


async def test_single_tool_call(model_opts: tuple[str, ModelOptions]) -> None:
    """LLM calls a single tool and uses the result in its response."""
    model, opts = model_opts
    conv = await Conversation(model=model, model_options=opts).send(
        "What is the capital of France? Use the get_capital tool.",
        tools=[GetCapital()],
        purpose="test_single_tool",
    )

    assert "Paris" in conv.content
    assert len(conv.tool_call_records) >= 1
    assert conv.tool_call_records[0].tool is GetCapital


async def test_tool_with_calculation(model_opts: tuple[str, ModelOptions]) -> None:
    """LLM uses a calculator tool."""
    model, opts = model_opts
    conv = await Conversation(model=model, model_options=opts).send(
        "What is 137 * 29? Use the calculator tool to compute it.",
        tools=[Calculator()],
        purpose="test_calculator",
    )

    # Some models format with commas (3,973) or spaces
    assert "3973" in conv.content.replace(",", "").replace(" ", "")
    assert len(conv.tool_call_records) >= 1


async def test_multiple_tools_available(model_opts: tuple[str, ModelOptions]) -> None:
    """LLM correctly selects from multiple available tools."""
    model, opts = model_opts
    conv = await Conversation(model=model, model_options=opts).send(
        "What is the capital of Japan? Use the appropriate tool.",
        tools=[Calculator(), GetCapital()],
        purpose="test_multi_tool_selection",
    )

    assert "Tokyo" in conv.content
    assert any(r.tool is GetCapital for r in conv.tool_call_records)


async def test_tool_error_recovery(model_opts: tuple[str, ModelOptions]) -> None:
    """LLM receives tool error and gracefully handles it."""
    model, opts = model_opts
    conv = await Conversation(model=model, model_options=opts).send(
        "Use the failing_tool with reason 'test'. Then tell me what happened.",
        tools=[FailingTool()],
        max_tool_rounds=2,
        purpose="test_error_recovery",
    )

    # LLM should have received the error and responded about it
    assert conv.content  # non-empty response
    assert len(conv.messages) > 1  # at least user + error response + final


async def test_tool_choice_none(model_opts: tuple[str, ModelOptions]) -> None:
    """tool_choice='none' prevents tool usage."""
    model, opts = model_opts
    conv = await Conversation(model=model, model_options=opts).send(
        "What is the capital of Germany?",
        tools=[GetCapital()],
        tool_choice="none",
        purpose="test_tool_choice_none",
    )

    # Should have a response without any tool calls
    assert conv.content
    assert len(conv.tool_call_records) == 0


async def test_max_tool_rounds_forced_final(model_opts: tuple[str, ModelOptions]) -> None:
    """max_tool_rounds=1 forces a final response after one round."""
    model, opts = model_opts
    conv = await Conversation(model=model, model_options=opts).send(
        "What is the capital of France? Use the get_capital tool.",
        tools=[GetCapital()],
        max_tool_rounds=1,
        purpose="test_max_rounds",
    )

    # Should still get a response (forced final or natural completion)
    assert conv.content


async def test_structured_output_with_tools(model_opts: tuple[str, ModelOptions]) -> None:
    """Tools work with structured output (tool loop → final structured call)."""

    model, opts = model_opts
    conv = await Conversation(model=model, model_options=opts).send_structured(
        "Look up the capital of France using the tool, then return the result.",
        response_format=CityInfo,
        tools=[GetCapital()],
        purpose="test_structured_with_tools",
    )

    assert conv.parsed is not None
    result = conv.parsed
    assert isinstance(result, CityInfo)
    assert result.city.lower() == "paris"


async def test_multi_round_tool_conversation(model_opts: tuple[str, ModelOptions]) -> None:
    """Multi-round tool conversation within a single send call."""
    model, opts = model_opts
    # Use two tools that the LLM might call in sequence
    conv = await Conversation(model=model, model_options=opts).send(
        "First look up the weather in Paris using the weather_lookup tool, then calculate 22 * 3 using the calculator tool. Report both results.",
        tools=[WeatherLookup(), Calculator()],
        purpose="test_multi_round",
    )

    assert conv.content
    # Should have used at least one tool
    assert len(conv.tool_call_records) >= 1


async def test_tool_choice_required(model_opts: tuple[str, ModelOptions]) -> None:
    """tool_choice='required' forces at least one tool call."""
    model, opts = model_opts
    conv = await Conversation(model=model, model_options=opts).send(
        "Tell me about the weather.",
        tools=[WeatherLookup()],
        tool_choice="required",
        purpose="test_tool_choice_required",
    )

    assert conv.content
    assert len(conv.tool_call_records) >= 1


async def test_conversation_continues_after_tools() -> None:
    """Conversation state is preserved after tool usage for follow-up messages."""
    conv = await Conversation(
        model="gemini-3-flash",
        model_options=ModelOptions(stream=False),
    ).send(
        "What is the capital of France? Use the get_capital tool.",
        tools=[GetCapital()],
        purpose="test_continuation_1",
    )

    assert "Paris" in conv.content

    # Follow-up without tools should work and have conversation history
    conv = await conv.send(
        "What country was that capital in?",
        purpose="test_continuation_2",
    )

    assert "France" in conv.content or "france" in conv.content.lower()
