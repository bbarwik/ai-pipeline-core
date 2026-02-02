"""Integration tests for search model citations."""

import uuid

import pytest

from ai_pipeline_core.llm import Citation, ModelOptions, generate

# Models that return structured url_citation annotations
CITATION_MODELS = [
    "sonar-pro-search",
    "gemini-3-flash-search",
    "gpt-5-mini-search",
]


@pytest.mark.integration
class TestSearchCitations:
    """Test that search models return citations via the citations property."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", CITATION_MODELS)
    async def test_citation_model_returns_citations(self, model: str):
        """Models with structured citations should return non-empty Citation list."""
        # Unique query to avoid proxy cache returning responses without annotations
        query = f"Who is the current pope? (ref:{uuid.uuid4().hex[:8]})"
        response = await generate(model, messages=query, purpose=f"{model}-citation-test")

        assert response.content, f"{model} returned empty content"

        # Verify search was actually performed — Pope Leo XIV was elected in 2025
        content_lower = response.content.lower()
        assert "leo" in content_lower or "xiv" in content_lower, f"{model} did not return current search results about the pope"

        assert len(response.citations) > 0, f"{model} returned no citations"
        for citation in response.citations:
            assert isinstance(citation, Citation)
            assert citation.url.startswith("http"), f"{model} citation URL invalid: {citation.url}"
            assert citation.title, f"{model} citation has empty title"

    @pytest.mark.asyncio
    async def test_grok_search_non_streaming(self):
        """Grok search works in non-streaming mode (streaming crashes OpenAI SDK delta accumulation)."""
        query = f"Who is the current pope? (ref:{uuid.uuid4().hex[:8]})"
        response = await generate(
            "grok-4.1-fast-search",
            messages=query,
            options=ModelOptions(stream=False),
            purpose="grok-search-test",
        )
        assert response.content
        content_lower = response.content.lower()
        assert "leo" in content_lower or "xiv" in content_lower, "Grok did not return current search results about the pope"
