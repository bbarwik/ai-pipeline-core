"""Unit tests for LLM-powered document summary generation (mocked)."""

from unittest.mock import AsyncMock, patch

import pytest

from ai_pipeline_core.document_store._summary_llm import (
    DocumentSummary,
    generate_document_summary,
)


class TestGenerateDocumentSummary:
    @pytest.mark.asyncio
    async def test_success_returns_json(self):
        mock_summary = DocumentSummary(short_title="Revenue Report", summary="Quarterly financial results")
        mock_response = type("Resp", (), {"parsed": mock_summary})()

        with patch(
            "ai_pipeline_core.document_store._summary_llm.generate_structured",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_gen:
            result = await generate_document_summary("report.md", "Revenue data here")

        assert '"short_title"' in result
        assert "Revenue Report" in result
        mock_gen.assert_called_once()
        call_kwargs = mock_gen.call_args
        assert call_kwargs[1]["response_format"] is DocumentSummary
        assert call_kwargs[1]["purpose"] == "document_summary: report.md"

    @pytest.mark.asyncio
    async def test_formats_user_template_with_excerpt(self):
        mock_summary = DocumentSummary(short_title="Test", summary="Test summary")
        mock_response = type("Resp", (), {"parsed": mock_summary})()

        with patch(
            "ai_pipeline_core.document_store._summary_llm.generate_structured",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_gen:
            await generate_document_summary("doc.txt", "My excerpt content")

        messages = mock_gen.call_args[0][0]
        assert len(messages) == 2
        assert messages[0].role.value == "system"
        assert "My excerpt content" in messages[1].content

    @pytest.mark.asyncio
    async def test_exception_returns_empty_string(self):
        with patch(
            "ai_pipeline_core.document_store._summary_llm.generate_structured",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API error"),
        ):
            result = await generate_document_summary("fail.md", "content")

        assert result == ""

    @pytest.mark.asyncio
    async def test_uses_default_model(self):
        mock_summary = DocumentSummary(short_title="T", summary="S")
        mock_response = type("Resp", (), {"parsed": mock_summary})()

        with patch(
            "ai_pipeline_core.document_store._summary_llm.generate_structured",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_gen:
            await generate_document_summary("doc.txt", "content")

        assert mock_gen.call_args[1]["model"] == "gemini-3-flash"

    @pytest.mark.asyncio
    async def test_custom_model(self):
        mock_summary = DocumentSummary(short_title="T", summary="S")
        mock_response = type("Resp", (), {"parsed": mock_summary})()

        with patch(
            "ai_pipeline_core.document_store._summary_llm.generate_structured",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_gen:
            await generate_document_summary("doc.txt", "content", model="gpt-4o-mini")

        assert mock_gen.call_args[1]["model"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_model_options_configured_correctly(self):
        mock_summary = DocumentSummary(short_title="T", summary="S")
        mock_response = type("Resp", (), {"parsed": mock_summary})()

        with patch(
            "ai_pipeline_core.document_store._summary_llm.generate_structured",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_gen:
            await generate_document_summary("doc.txt", "content")

        options = mock_gen.call_args[1]["model_options"]
        assert options.cache_ttl is None
        assert options.retries == 3
        assert options.timeout == 30
