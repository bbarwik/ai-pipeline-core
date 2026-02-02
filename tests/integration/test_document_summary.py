"""Integration tests for LLM-powered document and span summary generation."""

import pytest

from ai_pipeline_core.observability._summary import generate_document_summary, generate_span_summary
from ai_pipeline_core.settings import settings

HAS_API_KEYS = bool(settings.openai_api_key and settings.openai_base_url)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_API_KEYS, reason="OpenAI API keys not configured"),
]


@pytest.mark.asyncio
async def test_generate_document_summary():
    """Test document summary generation with a real LLM call."""
    summary = await generate_document_summary(
        name="quarterly_report_2025.md",
        excerpt=(
            "# Q3 2025 Revenue Report\n\n"
            "Total revenue reached $4.2M, a 23% increase over Q2. "
            "The SaaS segment grew 31% driven by enterprise contracts. "
            "Operating costs remained stable at $2.8M."
        ),
    )

    assert summary, "Summary should not be empty"
    assert len(summary.split()) <= 100, f"Summary too long ({len(summary.split())} words): {summary}"


@pytest.mark.asyncio
async def test_generate_span_summary():
    """Test span/task summary generation with a real LLM call."""
    summary = await generate_span_summary(
        label="verify_sources",
        output_hint="Verified 12 sources against the research report. 10 confirmed, 2 flagged as inconsistent.",
    )

    assert summary, "Summary should not be empty"
    assert len(summary.split()) <= 100, f"Summary too long ({len(summary.split())} words): {summary}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "excerpt"),
    [
        ("analysis.md", "The market analysis shows a 15% growth in the AI sector. Key players include..."),
        ("scraper.py", "import asyncio\nimport aiohttp\n\nasync def fetch_page(url: str) -> str:\n    async with aiohttp.ClientSession() as session:"),
        ("raw_data.txt", "John Smith, CEO of Acme Corp, stated that the company plans to expand into European markets by Q4 2025."),
    ],
    ids=["markdown", "code", "plain_text"],
)
async def test_document_summary_various_content_types(name: str, excerpt: str):
    """Test that summary generation handles different content types."""
    summary = await generate_document_summary(name=name, excerpt=excerpt)

    assert summary, f"Summary should not be empty for {name}"
    assert len(summary.split()) <= 100, f"Summary too long for {name}: {summary}"
