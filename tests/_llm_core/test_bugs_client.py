"""Bug-proving tests for LLM client response handling."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_pipeline_core._llm_core.client import _build_model_response
from ai_pipeline_core._llm_core.types import CoreMessage, Role
from ai_pipeline_core.exceptions import EmptyResponseError, LLMError


def _make_response(
    content: str = "Hello",
    finish_reason: str = "stop",
    prompt: int = 10,
    completion: int = 5,
    reasoning_content: str | None = None,
    annotations: list[Any] | None = None,
    provider_specific_fields: dict[str, Any] | None = None,
    thinking_blocks: list[dict[str, Any]] | None = None,
) -> SimpleNamespace:
    msg = SimpleNamespace(
        content=content,
        role="assistant",
        reasoning_content=reasoning_content,
        annotations=annotations or [],
        provider_specific_fields=provider_specific_fields,
        thinking_blocks=thinking_blocks,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=None,
        completion_tokens_details=None,
    )
    return SimpleNamespace(id="resp-1", choices=[choice], usage=usage)


def test_null_choices_raises_empty_response_error() -> None:
    """response.choices=None should raise EmptyResponseError, not TypeError."""
    resp = _make_response()
    resp.choices = None
    with pytest.raises(EmptyResponseError, match=r"no choices.*None"):
        _build_model_response(resp, {}, None, "test-model", None)


def test_empty_choices_raises_empty_response_error() -> None:
    """response.choices=[] should raise EmptyResponseError."""
    resp = _make_response()
    resp.choices = []
    with pytest.raises(EmptyResponseError, match=r"no choices.*empty"):
        _build_model_response(resp, {}, None, "test-model", None)


@patch("ai_pipeline_core._llm_core.client.settings")
@patch("ai_pipeline_core._llm_core.client.AsyncOpenAI")
async def test_empty_response_retried_with_cache_disabled(mock_aoai: Any, mock_settings: Any) -> None:
    """Empty response is retried with LiteLLM cache disabled via header.

    _build_model_response raises EmptyResponseError which is caught by the retry loop.
    On retry, cache is disabled via extra_body.cache = {"no-cache": True}.
    """
    from ai_pipeline_core._llm_core.client import _generate_impl
    from ai_pipeline_core._llm_core.types import ModelOptions

    mock_settings.openai_api_key = "key"
    mock_settings.openai_base_url = "http://localhost:4000"

    empty_resp = _make_response(content="")
    raw_response = MagicMock()
    raw_response.parse.return_value = empty_resp
    raw_response.headers = {}

    mock_client = AsyncMock()
    mock_client.chat.completions.with_raw_response.create = AsyncMock(return_value=raw_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_aoai.return_value = mock_client

    msg = CoreMessage(role=Role.USER, content="hi")
    opts = ModelOptions(retries=3, retry_delay_seconds=0, cache_ttl=None)
    with pytest.raises(LLMError):
        await _generate_impl([msg], model="test-model", model_options=opts)

    # Empty response IS retried (1 initial + 3 retries = 4 total)
    assert mock_client.chat.completions.with_raw_response.create.call_count == 4, (
        f"Empty response should be retried. Expected 4 calls (1 + 3 retries), got {mock_client.chat.completions.with_raw_response.create.call_count}"
    )


def test_empty_response_raises_empty_response_error() -> None:
    """Empty response raises EmptyResponseError with model info in message."""
    resp = _make_response(content="")
    with pytest.raises(EmptyResponseError, match="Empty response content"):
        _build_model_response(resp, {}, None, "test-model", None)


def test_empty_content_with_provider_fields_logs_warning(caplog: Any) -> None:
    """When content is empty but provider_specific_fields has data,
    the framework logs the available field keys for debugging.
    """
    import logging

    resp = _make_response(
        content="",
        provider_specific_fields={"output_text": "Actual response from provider"},
    )
    with caplog.at_level(logging.INFO), pytest.raises(EmptyResponseError):
        _build_model_response(resp, {}, None, "test-model", None)

    assert any("provider_specific_fields" in record.message for record in caplog.records), (
        "Should log available provider_specific_fields keys when content is empty"
    )
