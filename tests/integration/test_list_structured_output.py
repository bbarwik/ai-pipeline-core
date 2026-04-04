"""Integration tests for list[BaseModel] structured output across all non-search models.

Tests real LLM calls with list output using the wrapper/unwrap mechanism.
Requires API keys configured in .env file.
"""

import pytest
from pydantic import BaseModel

from ai_pipeline_core.llm import Conversation
from ai_pipeline_core.prompt_compiler.components import Role
from ai_pipeline_core.prompt_compiler.spec import PromptSpec
from ai_pipeline_core.settings import settings
from tests.integration.model_categories import CORE_MODELS

pytestmark = pytest.mark.integration

HAS_API_KEYS = bool(settings.openai_api_key and settings.openai_base_url)


# --- Models ---


class CityInfo(BaseModel):
    name: str
    country: str


class MathStep(BaseModel):
    step_number: int
    description: str
    result: str


class TagModel(BaseModel):
    label: str


# --- PromptSpec for list output ---


class AnalystRole(Role):
    """Geography analyst."""

    text = "geography analyst"


class ListCitiesSpec(PromptSpec[list[CityInfo]]):
    """Extract cities from text."""

    input_documents = ()
    role = AnalystRole
    task = "Return exactly 3 European capital cities with their country names."


# --- Tests parametrized across all core models ---


@pytest.mark.skipif(not HAS_API_KEYS, reason="API keys not configured")
class TestListStructuredOutput:
    """Integration tests for list[BaseModel] structured output."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", CORE_MODELS)
    async def test_list_output_basic(self, model: str) -> None:
        """Each core model should return a list of CityInfo via send_structured."""
        conv = Conversation(model=model)
        conv = await conv.send_structured(
            "Return exactly 2 cities: Paris (France) and Berlin (Germany).",
            response_format=list[CityInfo],
        )
        assert isinstance(conv.parsed, list)
        assert len(conv.parsed) == 2
        assert all(isinstance(item, CityInfo) for item in conv.parsed)
        names = {item.name.lower() for item in conv.parsed}
        assert "paris" in names

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", CORE_MODELS)
    async def test_list_content_is_array_json(self, model: str) -> None:
        """conv.content should be JSON array, not wrapper object."""
        import json

        conv = Conversation(model=model)
        conv = await conv.send_structured(
            "Return exactly 1 tag with label 'test'.",
            response_format=list[TagModel],
        )
        parsed_content = json.loads(conv.content)
        assert isinstance(parsed_content, list), f"Expected array JSON, got: {conv.content[:100]}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", CORE_MODELS)
    async def test_list_output_single_item(self, model: str) -> None:
        """Single-item list should work correctly."""
        conv = Conversation(model=model)
        conv = await conv.send_structured(
            "Return exactly 1 city: Tokyo (Japan).",
            response_format=list[CityInfo],
        )
        assert isinstance(conv.parsed, list)
        assert len(conv.parsed) == 1
        assert conv.parsed[0].country.lower() == "japan"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", CORE_MODELS)
    async def test_list_output_nested_fields(self, model: str) -> None:
        """List items with multiple fields should all be populated."""
        conv = Conversation(model=model)
        conv = await conv.send_structured(
            "Return 2 math steps: step 1 is 'add 2+3' with result '5', step 2 is 'multiply by 2' with result '10'.",
            response_format=list[MathStep],
        )
        assert isinstance(conv.parsed, list)
        assert len(conv.parsed) == 2
        assert all(isinstance(s, MathStep) for s in conv.parsed)
        assert conv.parsed[0].step_number == 1
        assert conv.parsed[1].result == "10"


@pytest.mark.skipif(not HAS_API_KEYS, reason="API keys not configured")
class TestListSpecIntegration:
    """Integration tests for PromptSpec[list[T]] via send_spec."""

    @pytest.mark.asyncio
    async def test_send_spec_list_output(self) -> None:
        """send_spec with PromptSpec[list[T]] should unwrap correctly."""
        conv = Conversation(model="gemini-3-flash")
        spec = ListCitiesSpec()
        conv = await conv.send_spec(spec)
        assert isinstance(conv.parsed, list)
        assert len(conv.parsed) == 3
        assert all(isinstance(item, CityInfo) for item in conv.parsed)

    @pytest.mark.asyncio
    async def test_send_spec_list_output_different_model(self) -> None:
        """send_spec with list output on gpt model."""
        conv = Conversation(model="gpt-5.4")
        spec = ListCitiesSpec()
        conv = await conv.send_spec(spec)
        assert isinstance(conv.parsed, list)
        assert all(isinstance(item, CityInfo) for item in conv.parsed)
