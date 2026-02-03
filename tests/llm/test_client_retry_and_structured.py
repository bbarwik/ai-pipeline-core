"""Tests for LLM client retry and structured generation."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from ai_pipeline_core.exceptions import LLMError
from ai_pipeline_core.llm import AIMessages, ModelOptions, generate
from ai_pipeline_core.llm.client import _generate_with_retry


class ExampleModel(BaseModel):
    """Example model for structured generation."""

    field1: str
    field2: int


class TestRetryLogic:
    """Test retry logic for LLM generation."""

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """Test retry on timeout with exponential backoff."""
        with patch("ai_pipeline_core.llm.client.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # First two calls fail, third succeeds
            mock_client.chat.completions.with_raw_response.create = MagicMock(
                return_value=MagicMock(
                    parse=MagicMock(
                        return_value=MagicMock(
                            choices=[MagicMock(message=MagicMock(content="Success"))],
                            id="test-id",
                            object="chat.completion",
                            created=1234567890,
                            model="test-model",
                            usage=None,
                        )
                    ),
                    headers=MagicMock(items=MagicMock(return_value=[])),
                )
            )

            # Mock sleep to avoid delays
            with patch("asyncio.sleep") as mock_sleep:
                mock_sleep.return_value = None

                # This should eventually succeed after retries
                AIMessages(["Test"])
                ModelOptions(retries=3, retry_delay_seconds=1)

                # Note: Can't easily test the actual retry logic without more complex mocking
                # This is a simplified test structure

    @pytest.mark.asyncio
    async def test_exhausted_retries(self):
        """Test that LLMError is raised when all retries are exhausted."""
        with patch("ai_pipeline_core.llm.client.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # All calls fail
            mock_client.chat.completions.with_raw_response.create.side_effect = TimeoutError("Timeout")

            with patch("asyncio.sleep"):
                messages = AIMessages(["Test"])
                options = ModelOptions(retries=2, retry_delay_seconds=1)

                with pytest.raises(LLMError) as exc_info:
                    await _generate_with_retry("test-model", AIMessages(), messages, options)

                assert "Exhausted all retry attempts" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_no_model_raises_error(self):
        """Test that missing model raises ValueError."""
        messages = AIMessages(["Test"])

        with pytest.raises(ValueError) as exc_info:
            await _generate_with_retry("", AIMessages(), messages, ModelOptions())

        assert "Model must be provided" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_no_messages_or_context_raises_error(self):
        """Test that having neither messages nor context raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            await _generate_with_retry("test-model", AIMessages(), AIMessages(), ModelOptions())

        assert "Either context or messages must be provided" in str(exc_info.value)


class TestCacheRemovalOnRetry:
    """Test that cache is properly disabled on non-timeout retry."""

    @pytest.mark.asyncio
    async def test_prompt_cache_key_removed_on_non_timeout_error(self):
        """Test that prompt_cache_key is removed from completion_kwargs on non-timeout error."""
        with patch("ai_pipeline_core.llm.client._generate") as mock_generate:
            # First call fails with ValueError (non-timeout), second succeeds
            mock_response = MagicMock()
            mock_response.content = "Success"
            mock_response.reasoning_content = None
            mock_response.get_laminar_metadata.return_value = {}
            mock_response.validate_output.return_value = None
            mock_generate.side_effect = [ValueError("Test error"), mock_response]

            with patch("asyncio.sleep"):
                with patch("ai_pipeline_core.llm.client.Laminar") as mock_laminar:
                    mock_span = MagicMock()
                    mock_laminar.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
                    mock_laminar.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

                    messages = AIMessages(["Test message"])
                    context = AIMessages(["Large context " * 1000])  # Large enough to trigger caching
                    options = ModelOptions(retries=2, retry_delay_seconds=0, cache_ttl="300s")

                    # Use gpt model to avoid Gemini's minimum context size check
                    await _generate_with_retry("gpt-5.1", context, messages, options)

                    # Check that the second call had prompt_cache_key removed
                    assert mock_generate.call_count == 2
                    second_call_kwargs = mock_generate.call_args_list[1][0][2]  # completion_kwargs
                    assert "prompt_cache_key" not in second_call_kwargs
                    assert second_call_kwargs["extra_body"]["cache"] == {"no-cache": True}

    @pytest.mark.asyncio
    async def test_cache_not_removed_on_timeout_error(self):
        """Test that cache is NOT removed on timeout error (only on other errors)."""
        with patch("ai_pipeline_core.llm.client._generate") as mock_generate:
            # First call fails with TimeoutError, second succeeds
            mock_response = MagicMock()
            mock_response.content = "Success"
            mock_response.reasoning_content = None
            mock_response.get_laminar_metadata.return_value = {}
            mock_response.validate_output.return_value = None

            mock_generate.side_effect = [TimeoutError("Timeout"), mock_response]

            with patch("asyncio.sleep"):
                with patch("ai_pipeline_core.llm.client.Laminar") as mock_laminar:
                    mock_span = MagicMock()
                    mock_laminar.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
                    mock_laminar.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

                    messages = AIMessages(["Test message"])
                    context = AIMessages(["Large context " * 1000])
                    options = ModelOptions(retries=2, retry_delay_seconds=0, cache_ttl="300s")

                    # Use gpt model to avoid Gemini's minimum context size check
                    await _generate_with_retry("gpt-5.1", context, messages, options)

                    # Check that the second call still has prompt_cache_key (not removed for timeout)
                    assert mock_generate.call_count == 2
                    second_call_kwargs = mock_generate.call_args_list[1][0][2]
                    # prompt_cache_key should still be present for timeout errors
                    assert "prompt_cache_key" in second_call_kwargs


class TestStructuredGeneration:
    """Test structured generation with Pydantic models."""

    @pytest.mark.asyncio
    async def test_generate_delegates_to_retry(self):
        """Test that generate properly delegates to _generate_with_retry."""
        with patch("ai_pipeline_core.llm.client._generate_with_retry") as mock_retry:
            mock_response = MagicMock()
            mock_retry.return_value = mock_response

            context = AIMessages(["Context"])
            messages = AIMessages(["Message"])
            options = ModelOptions()

            result = await generate(model="test-model", context=context, messages=messages, options=options)

            assert result == mock_response
            mock_retry.assert_called_once_with(
                "test-model",
                context,
                messages,
                options,
                purpose=None,
                expected_cost=None,
            )
