"""Integration tests for tool calling across providers.

Tests the complete tool loop with real LLM APIs: grok-4.1-fast, gemini-3-flash, gpt-5-mini.
Each test is parameterized by model AND streaming mode to verify both paths.
"""

import pytest
from pydantic import BaseModel, Field

from ai_pipeline_core import Conversation, ModelOptions, Tool
from ai_pipeline_core.llm._conversation_messages import UserMessage
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

    class Output(BaseModel):
        result: str

    async def run(self, input: Input) -> Output:
        try:
            result = eval(input.expression, {"__builtins__": {}}, {})
            return self.Output(result=str(result))
        except (SyntaxError, TypeError, ValueError, ZeroDivisionError, NameError, ArithmeticError) as e:
            return self.Output(result=f"Error: {e}")


class GetCapital(Tool):
    """Get the capital city of a country."""

    class Input(BaseModel):
        country: str = Field(description="Country name")

    class Output(BaseModel):
        answer: str

    async def run(self, input: Input) -> Output:
        capitals = {
            "france": "Paris",
            "japan": "Tokyo",
            "brazil": "Brasília",
            "germany": "Berlin",
        }
        country = input.country.lower()
        if country in capitals:
            return self.Output(answer=f"The capital of {input.country} is {capitals[country]}")
        return self.Output(answer=f"Unknown country: {input.country}")


class FailingTool(Tool):
    """A tool that always fails to test error handling."""

    class Input(BaseModel):
        reason: str = Field(description="Reason for failure")

    class Output(BaseModel):
        result: str

    async def run(self, input: Input) -> Output:
        raise RuntimeError(f"Intentional failure: {input.reason}")


class WeatherLookup(Tool):
    """Look up weather for a city."""

    class Input(BaseModel):
        city: str = Field(description="City to look up weather for")

    class Output(BaseModel):
        weather: str

    async def run(self, input: Input) -> Output:
        return self.Output(weather=f"The weather in {input.city} is sunny, 22°C")


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


# ── CounterAdd tool for deterministic max-rounds testing ─────────────────────


class CounterAdd(Tool):
    """Add 1 to the counter and return the new value."""

    class Input(BaseModel):
        pass

    class Output(BaseModel):
        value: int = Field(description="Counter value after increment")

    def __init__(self) -> None:
        self._counter = 0

    async def run(self, input: Input) -> Output:
        self._counter += 1
        return self.Output(value=self._counter)


# ── Regression: forced final must produce real response ───────────────────────


async def test_forced_final_produces_real_response_not_synthetic(model_opts: tuple[str, ModelOptions]) -> None:
    """When max_tool_rounds is exhausted, the forced final must produce a real LLM
    response, not the synthetic fallback string.

    Regression test: without the steering USER message, the forced final
    call with tools=[] caused the model to still generate tool calls, triggering
    ValueError in client.py → retry loop → LLMError → synthetic fallback.

    Uses tool_choice="required" + max_tool_rounds=1 to guarantee the forced final
    path is always triggered, regardless of model or streaming mode.
    """
    model, opts = model_opts
    tool = CounterAdd()

    conv = await Conversation(model=model, model_options=opts).send(
        "Use the counter_add tool to increment the counter. Keep incrementing until the counter value reaches 10.",
        tools=[tool],
        tool_choice="required",
        max_tool_rounds=1,
        purpose="test_forced_final_real_response",
    )

    # tool_choice=required guarantees at least one tool call in round 1
    assert tool._counter >= 1
    assert len(conv.tool_call_records) >= 1
    assert conv.content
    assert "[Final synthesis failed" not in conv.content
    # Steering message must NOT appear in conversation history
    # (it only exists in core_messages for the immediate forced final call)
    steering = [m for m in conv.messages if isinstance(m, UserMessage) and "no more tools" in m.text.lower()]
    assert len(steering) == 0, "Steering message must not persist in conversation history"


async def test_follow_up_tool_call_works_after_forced_final(model_opts: tuple[str, ModelOptions]) -> None:
    """Follow-up send() with tools works normally after a forced final.

    The steering message ("no more tools...") from the forced final is preserved in
    conversation history. This test verifies that a subsequent send() with tools is
    NOT suppressed by that message — the model still calls tools when they are available.
    """
    model, opts = model_opts

    # Trigger forced final by exhausting max_tool_rounds
    conv = await Conversation(model=model, model_options=opts).send(
        "Use the get_capital tool to look up the capital of France.",
        tools=[GetCapital()],
        tool_choice="required",
        max_tool_rounds=1,
        purpose="test_follow_up_1",
    )
    assert "[Final synthesis failed" not in conv.content

    # Follow-up send with tools — must still work despite steering message in history
    conv2 = await conv.send(
        "Now use the get_capital tool to look up the capital of Japan.",
        tools=[GetCapital()],
        purpose="test_follow_up_2",
    )

    assert "Tokyo" in conv2.content
    assert len(conv2.tool_call_records) >= 1


async def test_forced_final_steering_not_in_conversation_history(model_opts: tuple[str, ModelOptions]) -> None:
    """Steering message must NOT appear in conv.messages.

    Persisting it would suppress valid tool calls in follow-up send() calls
    (confirmed with grok-4.1-fast). The steering message exists only in
    core_messages for the immediate forced final LLM call.
    """
    model, opts = model_opts

    conv = await Conversation(model=model, model_options=opts).send(
        "What is the capital of France? Use the get_capital tool.",
        tools=[GetCapital()],
        tool_choice="required",
        max_tool_rounds=1,
        purpose="test_steering_not_in_history",
    )

    steering_msgs = [m for m in conv.messages if isinstance(m, UserMessage) and "no more tools" in m.text.lower()]
    assert len(steering_msgs) == 0, "Steering message must not persist in conversation history"
    assert conv.content
    assert "[Final synthesis failed" not in conv.content
