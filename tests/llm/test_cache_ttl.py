"""Tests for cache_ttl functionality in ModelOptions and LLM client."""

from typing import Any, cast

from ai_pipeline_core import AIMessages, ModelOptions
from ai_pipeline_core.llm.client import _process_messages  # pyright: ignore[reportPrivateUsage]


class TestCacheTTL:
    """Test cache_ttl functionality."""

    def test_model_options_default_cache_ttl(self):
        """Test ModelOptions has default cache_ttl of 120s."""
        options = ModelOptions()
        assert options.cache_ttl == "120s"

    def test_model_options_custom_cache_ttl(self):
        """Test ModelOptions accepts custom cache_ttl values."""
        options = ModelOptions(cache_ttl="300s")
        assert options.cache_ttl == "300s"

        options = ModelOptions(cache_ttl="5m")
        assert options.cache_ttl == "5m"

    def test_model_options_none_cache_ttl(self):
        """Test ModelOptions accepts None to disable caching."""
        options = ModelOptions(cache_ttl=None)
        assert options.cache_ttl is None

    def test_process_messages_default_cache_ttl(self):
        """Test _process_messages uses default cache_ttl of 120s."""
        context = AIMessages(["context message"])
        messages = AIMessages(["query message"])

        result = _process_messages(context, messages)

        # Find the context message
        context_msg = next((msg for msg in result if msg.get("content") == "context message"), None)
        assert context_msg is not None
        context_msg_dict = cast(dict[str, Any], context_msg)
        assert "cache_control" in context_msg_dict
        assert context_msg_dict["cache_control"]["type"] == "ephemeral"
        assert context_msg_dict["cache_control"]["ttl"] == "120s"

    def test_process_messages_custom_cache_ttl(self):
        """Test _process_messages uses custom cache_ttl."""
        context = AIMessages(["context message"])
        messages = AIMessages(["query message"])

        result = _process_messages(context, messages, cache_ttl="300s")

        # Find the context message
        context_msg = next((msg for msg in result if msg.get("content") == "context message"), None)
        assert context_msg is not None
        context_msg_dict = cast(dict[str, Any], context_msg)
        assert "cache_control" in context_msg_dict
        assert context_msg_dict["cache_control"]["type"] == "ephemeral"
        assert context_msg_dict["cache_control"]["ttl"] == "300s"

    def test_process_messages_none_cache_ttl(self):
        """Test _process_messages with cache_ttl=None disables caching."""
        context = AIMessages(["context message"])
        messages = AIMessages(["query message"])

        result = _process_messages(context, messages, cache_ttl=None)

        # Find the context message
        context_msg = next((msg for msg in result if msg.get("content") == "context message"), None)
        assert context_msg is not None
        assert "cache_control" not in context_msg

    def test_process_messages_with_system_prompt(self):
        """Test _process_messages with system prompt and cache_ttl."""
        context = AIMessages(["context message"])
        messages = AIMessages(["query message"])

        result = _process_messages(
            context, messages, system_prompt="You are helpful", cache_ttl="180s"
        )

        # System prompt should be first
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful"
        assert "cache_control" not in result[0]  # System prompt not cached

        # Context message should have cache_control
        context_msg = next((msg for msg in result if msg.get("content") == "context message"), None)
        assert context_msg is not None
        context_msg_dict = cast(dict[str, Any], context_msg)
        assert context_msg_dict["cache_control"]["ttl"] == "180s"

    def test_process_messages_no_context(self):
        """Test _process_messages with no context doesn't add cache_control."""
        messages = AIMessages(["query message"])

        result = _process_messages(AIMessages(), messages, cache_ttl="300s")

        # No messages should have cache_control
        for msg in result:
            assert "cache_control" not in msg

    def test_process_messages_multiple_context_messages(self):
        """Test only last context message gets cache_control."""
        context = AIMessages(["context1", "context2", "context3"])
        messages = AIMessages(["query"])

        result = _process_messages(context, messages, cache_ttl="60s")

        # Count messages with cache_control
        cached_messages = [msg for msg in result if "cache_control" in msg]
        assert len(cached_messages) == 1

        # Last context message should have cache_control
        assert cached_messages[0]["content"] == "context3"
        cached_msg_dict = cast(dict[str, Any], cached_messages[0])
        assert cached_msg_dict["cache_control"]["ttl"] == "60s"

    def test_cache_ttl_various_formats(self):
        """Test various cache_ttl time formats."""
        # Test hours format
        options = ModelOptions(cache_ttl="1h")
        assert options.cache_ttl == "1h"

        # Test minutes format
        options = ModelOptions(cache_ttl="30m")
        assert options.cache_ttl == "30m"

        # Test mixed format
        options = ModelOptions(cache_ttl="90s")
        assert options.cache_ttl == "90s"

        # Test with _process_messages
        context = AIMessages(["context"])
        messages = AIMessages(["query"])

        for ttl in ["1h", "30m", "90s", "2h", "45m"]:
            result = _process_messages(context, messages, cache_ttl=ttl)
            context_msg = next((msg for msg in result if msg.get("content") == "context"), None)
            context_msg_dict = cast(dict[str, Any], context_msg)
            assert context_msg_dict["cache_control"]["ttl"] == ttl

    def test_cache_ttl_empty_string(self):
        """Test empty string cache_ttl disables caching (falsy value)."""
        options = ModelOptions(cache_ttl="")
        assert options.cache_ttl == ""

        # Test with _process_messages - empty string is falsy, so no cache_control
        context = AIMessages(["context"])
        messages = AIMessages(["query"])

        result = _process_messages(context, messages, cache_ttl="")

        # Empty string is falsy, so no cache_control is added
        context_msg = next((msg for msg in result if msg.get("content") == "context"), None)
        assert context_msg is not None
        assert "cache_control" not in context_msg

    def test_cache_ttl_from_model_options(self):
        """Test that cache_ttl from ModelOptions is passed to _process_messages."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from ai_pipeline_core.llm.client import (
            _generate_with_retry,  # pyright: ignore[reportPrivateUsage]
        )

        # Mock the _generate function to avoid actual API calls
        with patch("ai_pipeline_core.llm.client._generate") as mock_generate:
            mock_response = AsyncMock()
            mock_response.content = "test response"
            mock_response.get_laminar_metadata = lambda: {}
            mock_generate.return_value = mock_response

            # Test with custom cache_ttl in options
            options = ModelOptions(cache_ttl="600s", retries=1)
            context = AIMessages(["context"])
            messages = AIMessages(["query"])

            # Run the async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    _generate_with_retry("test-model", context, messages, options)
                )
            finally:
                loop.close()

            # Check that _generate was called with correct cache_ttl
            assert mock_generate.called
            call_args = mock_generate.call_args[0]
            processed_messages = call_args[1]

            # Find context message with cache_control
            context_msg = next(
                (msg for msg in processed_messages if msg.get("content") == "context"), None
            )
            assert context_msg is not None
            context_msg_dict = cast(dict[str, Any], context_msg)
            assert context_msg_dict["cache_control"]["ttl"] == "600s"

    def test_cache_ttl_integration_with_options(self):
        """Test that ModelOptions correctly integrates cache_ttl."""
        # Test default value
        default_options = ModelOptions()
        assert default_options.cache_ttl == "120s"

        # Test that to_openai_completion_kwargs doesn't include cache_ttl
        # (it's handled separately in _process_messages)
        kwargs = default_options.to_openai_completion_kwargs()
        assert "cache_ttl" not in kwargs
        assert "cache_control" not in kwargs

        # Test custom value
        custom_options = ModelOptions(cache_ttl="10m")
        assert custom_options.cache_ttl == "10m"
        kwargs = custom_options.to_openai_completion_kwargs()
        assert "cache_ttl" not in kwargs

    def test_cache_ttl_only_on_last_context_message(self):
        """Verify cache_control is only applied to the last context message."""
        context = AIMessages(["ctx1", "ctx2", "ctx3", "ctx4"])
        messages = AIMessages(["msg1", "msg2"])

        result = _process_messages(context, messages, cache_ttl="200s")

        # Check each message
        cache_control_count = 0
        for i, msg in enumerate(result):
            if "cache_control" in msg:
                cache_control_count += 1
                # Should be the last context message (ctx4)
                assert msg["content"] == "ctx4"
                msg_dict = cast(dict[str, Any], msg)
                assert msg_dict["cache_control"]["type"] == "ephemeral"
                assert msg_dict["cache_control"]["ttl"] == "200s"

        # Only one message should have cache_control
        assert cache_control_count == 1

        # Regular messages should not have cache_control
        regular_messages = [msg for msg in result if msg.get("content") in ["msg1", "msg2"]]
        for msg in regular_messages:
            assert "cache_control" not in msg
