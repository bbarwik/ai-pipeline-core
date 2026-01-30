"""Tests for LLM client cache control removal."""

# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportTypedDictNotRequiredAccess=false, reportArgumentType=false, reportOptionalSubscript=false

from ai_pipeline_core.llm.client import _remove_cache_control


class TestRemoveCacheControl:
    """Test cache control removal from messages."""

    def test_empty_messages(self):
        """Test with empty message list."""
        messages = []  # type: ignore
        result = _remove_cache_control(messages)  # type: ignore
        assert result == []

    def test_message_without_cache_control(self):
        """Test that messages without cache control are unchanged."""
        messages = [  # type: ignore
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
        ]
        result = _remove_cache_control(messages)  # type: ignore
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert "cache_control" not in result[0]
        assert "cache_control" not in result[1]

    def test_message_level_cache_control_removed(self):
        """Test that message-level cache_control is removed."""
        messages = [  # type: ignore
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}],
                "cache_control": {"type": "ephemeral", "ttl": "300s"},
            }
        ]
        result = _remove_cache_control(messages)  # type: ignore
        assert len(result) == 1
        assert "cache_control" not in result[0]
        assert result[0]["content"][0]["text"] == "Hello"

    def test_content_level_cache_control_removed(self):
        """Test that content-level cache_control is removed."""
        messages = [  # type: ignore
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello",
                        "cache_control": {"type": "ephemeral", "ttl": "300s"},
                    }
                ],
            }
        ]
        result = _remove_cache_control(messages)  # type: ignore
        assert len(result) == 1
        assert "cache_control" not in result[0]
        assert "cache_control" not in result[0]["content"][0]
        assert result[0]["content"][0]["text"] == "Hello"

    def test_both_levels_cache_control_removed(self):
        """Test that cache_control at both message and content level is removed."""
        messages = [  # type: ignore
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Context message",
                        "cache_control": {"type": "ephemeral", "ttl": "300s"},
                    }
                ],
                "cache_control": {"type": "ephemeral", "ttl": "300s"},
            }
        ]
        result = _remove_cache_control(messages)  # type: ignore
        assert len(result) == 1
        assert "cache_control" not in result[0]
        assert "cache_control" not in result[0]["content"][0]
        assert result[0]["content"][0]["text"] == "Context message"

    def test_multiple_content_parts_with_cache_control(self):
        """Test removing cache_control from multiple content parts."""
        messages = [  # type: ignore
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Part 1",
                        "cache_control": {"type": "ephemeral", "ttl": "300s"},
                    },
                    {"type": "text", "text": "Part 2"},
                    {
                        "type": "text",
                        "text": "Part 3",
                        "cache_control": {"type": "ephemeral", "ttl": "300s"},
                    },
                ],
                "cache_control": {"type": "ephemeral", "ttl": "300s"},
            }
        ]
        result = _remove_cache_control(messages)  # type: ignore
        assert len(result) == 1
        assert "cache_control" not in result[0]
        assert len(result[0]["content"]) == 3
        assert "cache_control" not in result[0]["content"][0]
        assert "cache_control" not in result[0]["content"][1]
        assert "cache_control" not in result[0]["content"][2]

    def test_multiple_messages_mixed_cache_control(self):
        """Test removing cache_control from multiple messages with mixed patterns."""
        messages = [  # type: ignore
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Message 1",
                        "cache_control": {"type": "ephemeral", "ttl": "300s"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Response 1",
                        "cache_control": {"type": "ephemeral", "ttl": "300s"},
                    }
                ],
                "cache_control": {"type": "ephemeral", "ttl": "300s"},
            },
            {"role": "user", "content": [{"type": "text", "text": "Message 2"}]},
        ]
        result = _remove_cache_control(messages)  # type: ignore
        assert len(result) == 3
        for msg in result:
            assert "cache_control" not in msg
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    assert "cache_control" not in item

    def test_string_content_unchanged(self):
        """Test that string content (non-list) is handled correctly."""
        messages = [  # type: ignore
            {
                "role": "system",
                "content": "System prompt",
                "cache_control": {"type": "ephemeral", "ttl": "300s"},
            }
        ]
        result = _remove_cache_control(messages)  # type: ignore
        assert len(result) == 1
        assert result[0]["content"] == "System prompt"
        assert "cache_control" not in result[0]

    def test_preserves_other_message_fields(self):
        """Test that other message fields are preserved."""
        messages = [  # type: ignore
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello",
                        "cache_control": {"type": "ephemeral", "ttl": "300s"},
                    }
                ],
                "cache_control": {"type": "ephemeral", "ttl": "300s"},
                "name": "test_user",
            }
        ]
        result = _remove_cache_control(messages)  # type: ignore
        assert len(result) == 1
        assert "cache_control" not in result[0]
        assert result[0].get("name") == "test_user"
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["text"] == "Hello"

    def test_modifies_in_place(self):
        """Test that the function modifies messages in place."""
        messages = [  # type: ignore
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello",
                        "cache_control": {"type": "ephemeral", "ttl": "300s"},
                    }
                ],
                "cache_control": {"type": "ephemeral", "ttl": "300s"},
            }
        ]
        result = _remove_cache_control(messages)  # type: ignore
        # Result should be the same list object (in-place modification)
        assert result is messages
        assert "cache_control" not in messages[0]
        assert "cache_control" not in messages[0]["content"][0]
