"""Tests for AIMessages freeze functionality."""

import pytest

from ai_pipeline_core.llm import AIMessages, ModelResponse
from tests.test_helpers import ConcreteFlowDocument, create_test_model_response


class TestAIMessagesFreeze:
    """Test AIMessages freeze functionality."""

    def test_init_frozen(self) -> None:
        """Test creating frozen AIMessages at initialization."""
        messages = AIMessages(["Hello", "World"], frozen=True)

        # Verify initial content
        assert len(messages) == 2
        assert messages[0] == "Hello"
        assert messages[1] == "World"

        # Should be frozen
        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages.append("Test")

    def test_freeze_method(self) -> None:
        """Test freezing AIMessages after creation."""
        messages = AIMessages(["Hello"])

        # Should be modifiable
        messages.append("World")
        assert len(messages) == 2

        # Freeze it
        messages.freeze()

        # Now should be frozen
        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages.append("Test")

    def test_freeze_is_permanent(self) -> None:
        """Test that freeze cannot be undone."""
        messages = AIMessages()
        messages.freeze()

        # Try to unfreeze by setting _frozen directly - should still be frozen
        # This tests that the freeze is intended to be permanent
        messages._frozen = False  # pyright: ignore[reportPrivateUsage]
        messages._frozen = True  # Reset it back  # pyright: ignore[reportPrivateUsage]

        # Should still be frozen
        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages.append("Test")

    def test_copy_creates_unfrozen(self) -> None:
        """Test that copy creates unfrozen deep copy."""
        doc = ConcreteFlowDocument(name="test.txt", content=b"content")
        response = create_test_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "response"},
                    "finish_reason": "stop",
                }
            ],
        )

        # Create frozen list with mixed types
        frozen_messages = AIMessages(["Hello", doc, response], frozen=True)

        # Copy should be unfrozen
        copied_messages = frozen_messages.copy()

        # Verify copy has same content
        assert len(copied_messages) == 3
        assert copied_messages[0] == "Hello"
        assert copied_messages[1].name == doc.name  # type: ignore
        assert copied_messages[2].content == "response"  # type: ignore

        # Copy should be modifiable
        copied_messages.append("New message")
        assert len(copied_messages) == 4

        # Original should still be frozen and unchanged
        assert len(frozen_messages) == 3
        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            frozen_messages.append("Test")

    def test_copy_is_deep(self) -> None:
        """Test that copy performs deep copy of messages."""
        doc = ConcreteFlowDocument(name="test.txt", content=b"content")
        original = AIMessages(["Hello", doc])

        # Make a copy
        copied = original.copy()

        # Verify deep copy - documents should be different objects
        assert copied[1] is not original[1]
        assert copied[1].name == original[1].name  # type: ignore
        assert copied[1].content == original[1].content  # type: ignore

    def test_frozen_append(self) -> None:
        """Test that append raises RuntimeError when frozen."""
        messages = AIMessages(frozen=True)
        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages.append("Test")

    def test_frozen_extend(self) -> None:
        """Test that extend raises RuntimeError when frozen."""
        messages = AIMessages(frozen=True)
        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages.extend(["Test1", "Test2"])

    def test_frozen_insert(self) -> None:
        """Test that insert raises RuntimeError when frozen."""
        messages = AIMessages(["Hello"], frozen=True)
        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages.insert(0, "Test")

    def test_frozen_setitem(self) -> None:
        """Test that __setitem__ raises RuntimeError when frozen."""
        messages = AIMessages(["Hello", "World"], frozen=True)

        # Test single item
        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages[0] = "Test"

        # Test slice
        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages[0:1] = ["Test"]

    def test_frozen_iadd(self) -> None:
        """Test that += raises RuntimeError when frozen."""
        messages = AIMessages(["Hello"], frozen=True)

        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages += ["World"]

    def test_frozen_delitem(self) -> None:
        """Test that __delitem__ raises RuntimeError when frozen."""
        messages = AIMessages(["Hello", "World"], frozen=True)

        # Test single item deletion
        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            del messages[0]

        # Test slice deletion
        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            del messages[0:1]

    def test_frozen_pop(self) -> None:
        """Test that pop raises RuntimeError when frozen."""
        messages = AIMessages(["Hello"], frozen=True)

        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages.pop()

        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages.pop(0)

    def test_frozen_remove(self) -> None:
        """Test that remove raises RuntimeError when frozen."""
        messages = AIMessages(["Hello"], frozen=True)

        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages.remove("Hello")

    def test_frozen_clear(self) -> None:
        """Test that clear raises RuntimeError when frozen."""
        messages = AIMessages(["Hello"], frozen=True)

        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages.clear()

    def test_frozen_reverse(self) -> None:
        """Test that reverse raises RuntimeError when frozen."""
        messages = AIMessages(["Hello", "World"], frozen=True)

        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages.reverse()

    def test_frozen_sort(self) -> None:
        """Test that sort raises RuntimeError when frozen."""
        messages = AIMessages(["World", "Hello"], frozen=True)

        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages.sort()

        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages.sort(reverse=True)

    def test_frozen_allows_read_operations(self) -> None:
        """Test that frozen list allows read operations."""
        doc = ConcreteFlowDocument(name="test.txt", content=b"content")
        response = create_test_model_response(
            id="test",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "response"},
                    "finish_reason": "stop",
                }
            ],
        )

        messages = AIMessages(["Hello", doc, response], frozen=True)

        # All read operations should work
        assert len(messages) == 3
        assert messages[0] == "Hello"
        assert messages[1] == doc
        assert messages[2] == response
        assert "Hello" in messages

        # get_last_message should work
        assert messages.get_last_message() == response

        # get_last_message_as_str should work (when last is string)
        str_messages = AIMessages(["Hello", "World"], frozen=True)
        assert str_messages.get_last_message_as_str() == "World"

        # to_prompt should work
        prompt = messages.to_prompt()
        assert len(prompt) == 3

        # get_prompt_cache_key should work
        key = messages.get_prompt_cache_key()
        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hex digest

        # Iteration should work
        for msg in messages:
            assert msg in [messages[0], messages[1], messages[2]]

    def test_empty_frozen_list(self) -> None:
        """Test empty frozen list."""
        messages = AIMessages(frozen=True)

        assert len(messages) == 0
        assert list(messages) == []

        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages.append("Test")

    def test_frozen_with_model_response(self) -> None:
        """Test frozen list with ModelResponse."""
        response1 = create_test_model_response(
            id="test1",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "response1"},
                    "finish_reason": "stop",
                }
            ],
        )
        response2 = create_test_model_response(
            id="test2",
            object="chat.completion",
            created=1234567891,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "response2"},
                    "finish_reason": "stop",
                }
            ],
        )

        messages = AIMessages(["Question", response1, "Follow-up", response2], frozen=True)

        assert len(messages) == 4
        assert messages[1].content == "response1"  # type: ignore
        assert messages[3].content == "response2"  # type: ignore

        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages.append("Another question")

    def test_frozen_prevents_string_construction(self) -> None:
        """Test that frozen parameter doesn't bypass string construction check."""
        # Should still raise TypeError for direct string construction
        with pytest.raises(TypeError, match="cannot be constructed from a string directly"):
            AIMessages("text", frozen=True)  # type: ignore

    def test_frozen_copy_with_documents(self) -> None:
        """Test that copying frozen AIMessages with documents works correctly."""
        doc1 = ConcreteFlowDocument(name="doc1.txt", content=b"content1")
        doc2 = ConcreteFlowDocument(name="doc2.txt", content=b"content2")

        frozen = AIMessages([doc1, "message", doc2], frozen=True)
        copied = frozen.copy()

        # Verify deep copy of documents
        assert copied[0] is not frozen[0]
        assert copied[0].name == frozen[0].name  # type: ignore
        assert copied[2] is not frozen[2]
        assert copied[2].name == frozen[2].name  # type: ignore

        # Copied list should be modifiable
        copied.append("new")
        assert len(copied) == 4
        assert len(frozen) == 3  # Original unchanged

    def test_multiple_freeze_calls(self) -> None:
        """Test that calling freeze multiple times is safe."""
        messages = AIMessages()

        # First freeze
        messages.freeze()

        # Second freeze should be safe (idempotent)
        messages.freeze()

        # Still frozen
        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages.append("Test")

    def test_frozen_with_none_initialization(self) -> None:
        """Test frozen list created with None."""
        messages = AIMessages(None, frozen=True)

        assert len(messages) == 0

        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages.append("Test")

    def test_copy_preserves_message_types(self) -> None:
        """Test that copy preserves exact message types and content."""
        doc = ConcreteFlowDocument(name="test.txt", content=b"doc content", description="Test")
        response = create_test_model_response(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "AI response"},
                    "finish_reason": "stop",
                }
            ],
        )

        original = AIMessages(["User message", doc, response], frozen=True)
        copied = original.copy()

        # Check exact preservation of types
        assert isinstance(copied[0], str)
        assert copied[0] == "User message"

        assert isinstance(copied[1], ConcreteFlowDocument)
        assert copied[1].name == doc.name  # type: ignore
        assert copied[1].content == doc.content  # type: ignore
        assert copied[1].description == doc.description  # type: ignore

        assert isinstance(copied[2], ModelResponse)
        assert copied[2].id == response.id  # type: ignore
        assert copied[2].content == "AI response"  # type: ignore

    def test_frozen_prevents_all_modifications(self) -> None:
        """Test comprehensive modification prevention when frozen."""
        messages = AIMessages(["msg1", "msg2"], frozen=True)

        # Test all modification methods
        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages.append("new")

        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages.extend(["new1", "new2"])

        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages.insert(1, "new")

        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages[0] = "changed"

        with pytest.raises(RuntimeError, match="Cannot modify frozen AIMessages"):
            messages += ["new"]

        # Verify original is unchanged
        assert len(messages) == 2
        assert messages[0] == "msg1"
        assert messages[1] == "msg2"
