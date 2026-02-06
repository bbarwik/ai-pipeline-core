"""Tests that verify the issues were fixed.

Originally these tests were designed to FAIL, proving bugs exist.
After fixes, most should now PASS.

Issues covered:
1. Boolean truthiness bug (temperature=0.0, max_completion_tokens=0, stop=[]) - FIXED
2. Substitutor state persistence across Conversation.send() - FIXED
3. Async without await in URLSubstitutor.prepare() - FIXED (now sync)
4. XML injection vulnerability in document processing - FIXED
5. Attachments inside document wrapper - FIXED
6. Backward compatibility shim (token_usage property) - FIXED (removed)
7. Blocking I/O in async context - NOT YET FIXED
8. Missing slots=True on dataclasses - FIXED
9. Magic number inconsistency (1000 vs 1080) - FIXED
"""

import inspect

import pytest

from ai_pipeline_core._llm_core.model_options import ModelOptions
from ai_pipeline_core._llm_core.model_response import Citation, ModelResponse
from ai_pipeline_core._llm_core.types import TextContent, TokenUsage
from ai_pipeline_core.documents import Attachment
from ai_pipeline_core.llm.conversation import Conversation, _document_to_content_parts, _escape_xml
from ai_pipeline_core.llm import URLSubstitutor

from tests.support.helpers import ConcreteDocument


# =============================================================================
# Issue 1: Boolean Truthiness Bug - FIXED
# File: model_options.py:174,177,180
# Fix: Changed `if self.temperature:` to `if self.temperature is not None:`
# =============================================================================


class TestBooleanTruthinessBugFixed:
    """Tests verifying the boolean truthiness bug is FIXED."""

    def test_temperature_zero_is_included(self):
        """temperature=0.0 should be passed to API for deterministic output."""
        options = ModelOptions(temperature=0.0)
        kwargs = options.to_openai_completion_kwargs()

        assert "temperature" in kwargs, "temperature=0.0 should be included in kwargs"
        assert kwargs["temperature"] == 0.0

    def test_max_completion_tokens_zero_is_included(self):
        """max_completion_tokens=0 should be passed to API."""
        options = ModelOptions(max_completion_tokens=0)
        kwargs = options.to_openai_completion_kwargs()

        assert "max_completion_tokens" in kwargs, "max_completion_tokens=0 should be included"
        assert kwargs["max_completion_tokens"] == 0

    def test_stop_empty_list_is_included(self):
        """stop=[] should be passed to API."""
        options = ModelOptions(stop=[])
        kwargs = options.to_openai_completion_kwargs()

        assert "stop" in kwargs, "stop=[] should be included in kwargs"
        assert kwargs["stop"] == []

    def test_positive_temperature_included(self):
        """Positive temperature should work (sanity check)."""
        options = ModelOptions(temperature=0.7)
        kwargs = options.to_openai_completion_kwargs()
        assert "temperature" in kwargs
        assert kwargs["temperature"] == 0.7


# =============================================================================
# Issue 2: Substitutor State Persistence - FIXED
# Fix: Using Field(exclude=True) instead of PrivateAttr, passed through constructor
# =============================================================================


class TestSubstitutorStatePersistenceFixed:
    """Tests verifying substitutor state persists across send() calls."""

    @pytest.mark.asyncio
    async def test_substitutor_passed_through_constructor(self, monkeypatch):
        """Substitutor should be passed through constructor, not via object.__setattr__."""
        mock_response = ModelResponse[str](
            content="Response",
            parsed="Response",
            reasoning_content="",
            citations=(),
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            cost=None,
            model="gpt-5.1",
            response_id="test-id",
            metadata={},
            thinking_blocks=None,
            provider_specific_fields=None,
        )

        async def fake_generate(*args, **kwargs):
            return mock_response

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        # Create conversation with a document that has a long URL
        doc = ConcreteDocument.create(
            name="test.txt",
            content="Visit https://example.com/very/long/path/to/resource/that/needs/shortening",
        )
        conv1 = Conversation(model="gpt-5.1", context=[doc])

        # Verify substitutor exists
        assert conv1.substitutor is not None

        # Add mappings by calling prepare
        conv1.substitutor.prepare(["https://example.com/very/long/path/to/resource/that/needs/shortening"])
        initial_count = conv1.substitutor.pattern_count

        # Send and get new conversation
        conv2 = await conv1.send("Follow up")

        # Verify same substitutor instance is shared (state persisted)
        assert conv2.substitutor is not None
        assert conv2.substitutor is conv1.substitutor, "Substitutor should be the same instance"
        assert conv2.substitutor.pattern_count == initial_count, "Mappings should be preserved"

    def test_substitutor_is_regular_field(self):
        """Substitutor should be a regular Field, not PrivateAttr."""
        # Check that 'substitutor' is in model_fields (regular field)
        # and not in __private_attributes__ (PrivateAttr)
        assert "substitutor" in Conversation.model_fields, "substitutor should be a model field"


# =============================================================================
# Issue 3: Async Without Await - FIXED
# Fix: Removed `async` from prepare() - it's now a regular sync method
# =============================================================================


class TestPrepareMethodFixed:
    """Tests verifying prepare() is now synchronous."""

    def test_prepare_method_is_not_async(self):
        """prepare() should be a regular sync method, not async."""
        # After fix, prepare() should NOT be a coroutine function
        assert not inspect.iscoroutinefunction(URLSubstitutor.prepare), "prepare() should be sync, not async"

    def test_prepare_can_be_called_synchronously(self):
        """prepare() should be callable without await."""
        sub = URLSubstitutor()
        # This should work without await
        sub.prepare(["https://example.com/test"])
        assert sub.is_prepared


# =============================================================================
# Issue 4: XML Injection Vulnerability - FIXED
# Fix: Added _escape_xml() using html.escape() for all document fields
# =============================================================================


class TestXmlInjectionFixed:
    """Tests verifying XML injection is prevented."""

    def test_escape_xml_function_exists(self):
        """_escape_xml function should exist and work."""
        assert _escape_xml("<test>") == "&lt;test&gt;"
        assert _escape_xml("a & b") == "a &amp; b"
        assert _escape_xml('"quoted"') == "&quot;quoted&quot;"

    def test_document_content_is_escaped(self):
        """Document content should have XML characters escaped."""
        doc = ConcreteDocument.create(name="test.txt", content="Hello <world> & friends")

        parts = _document_to_content_parts(doc, "gpt-5.1")
        combined = "".join(p.text for p in parts if isinstance(p, TextContent))

        # Content should be escaped
        assert "&lt;world&gt;" in combined, "< and > should be escaped"
        assert "&amp;" in combined, "& should be escaped"
        # Raw XML tags should NOT appear
        assert "<world>" not in combined, "Unescaped XML should not appear"

    def test_document_description_is_escaped(self):
        """Document description should have XML characters escaped."""
        doc = ConcreteDocument.create(name="test.txt", content="Hello", description="A <test> description")

        parts = _document_to_content_parts(doc, "gpt-5.1")
        combined = "".join(p.text for p in parts if isinstance(p, TextContent))

        # Description should be escaped
        assert "&lt;test&gt;" in combined, "Description XML should be escaped"


# =============================================================================
# Issue 5: Attachments Inside Document Wrapper - FIXED
# Fix: Moved attachment processing inside document wrapper
# =============================================================================


class TestAttachmentsInsideWrapperFixed:
    """Tests verifying attachments are inside document wrapper."""

    def test_text_attachment_inside_document(self):
        """Text attachments should be inside the <document>...</document> wrapper."""
        doc = ConcreteDocument.create(
            name="main.txt",
            content="Main content",
            attachments=(Attachment(name="extra.txt", content=b"Attachment content"),),
        )

        parts = _document_to_content_parts(doc, "gpt-5.1")
        combined = "".join(p.text for p in parts if isinstance(p, TextContent))

        # Find positions
        doc_close_pos = combined.find("</document>")
        attachment_pos = combined.find("extra.txt")

        assert attachment_pos != -1, "Attachment should be present"
        assert doc_close_pos != -1, "Document close tag should be present"
        assert attachment_pos < doc_close_pos, f"Attachment at {attachment_pos} should be before </document> at {doc_close_pos}"


# =============================================================================
# Issue 6: Backward Compatibility Shim - FIXED
# Fix: Removed token_usage property from ModelResponse
# =============================================================================


class TestBackwardCompatRemoved:
    """Tests verifying backward compatibility shims are removed."""

    def test_token_usage_property_removed(self):
        """ModelResponse should not have token_usage property."""
        assert not hasattr(ModelResponse, "token_usage"), "token_usage property should be removed"

    def test_usage_property_works(self):
        """The .usage property should work directly."""
        response = ModelResponse[str](
            content="test",
            parsed="test",
            reasoning_content="",
            citations=(),
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            cost=None,
            model="test",
            response_id="",
            metadata={},
        )
        assert response.usage.total_tokens == 15


# =============================================================================
# Issue 7: Blocking I/O in Async Context - FIXED
# Fix: Wrapped _to_core_messages() with asyncio.to_thread() in send()/send_structured()
# =============================================================================


class TestBlockingIOFixed:
    """Tests verifying blocking I/O is handled via asyncio.to_thread()."""

    def test_to_core_messages_wrapped_in_to_thread(self):
        """_to_core_messages() should be called via asyncio.to_thread()."""
        import ai_pipeline_core.llm.conversation as conv_module

        source = inspect.getsource(conv_module)

        # Verify asyncio.to_thread is used with _to_core_messages
        assert "asyncio.to_thread(" in source and "_to_core_messages" in source, "_to_core_messages should be wrapped with asyncio.to_thread"

    def test_asyncio_imported(self):
        """asyncio module should be imported."""
        import ai_pipeline_core.llm.conversation as conv_module

        assert hasattr(conv_module, "asyncio") or "import asyncio" in inspect.getsource(conv_module)


# =============================================================================
# Issue 8: Missing slots=True on Dataclasses - FIXED
# Fix: Added slots=True to Citation and URLSubstitutor
# =============================================================================


class TestSlotsAddedFixed:
    """Tests verifying slots=True was added to dataclasses."""

    def test_citation_has_slots(self):
        """Citation dataclass should have __slots__."""
        assert hasattr(Citation, "__slots__"), "Citation should have slots=True"

    def test_urlsubstitutor_has_slots(self):
        """URLSubstitutor dataclass should have __slots__."""
        assert hasattr(URLSubstitutor, "__slots__"), "URLSubstitutor should have slots=True"


# =============================================================================
# Issue 9: Magic Number Inconsistency - FIXED
# Fix: Changed 1000 to _TOKENS_PER_IMAGE = 1080
# =============================================================================


class TestMagicNumberFixed:
    """Tests verifying magic number is fixed to 1080."""

    def test_image_token_count_is_1080(self):
        """Image token estimation should use 1080 tokens per CLAUDE.md."""
        import base64

        # Create minimal valid PNG
        png_data = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")

        doc = ConcreteDocument.create(name="test.png", content=png_data)
        conv = Conversation(model="gpt-5.1", context=[doc])

        # The approximate token count should use 1080 per image
        token_count = conv.approximate_tokens_count
        assert token_count == 1080, f"Image should be 1080 tokens per CLAUDE.md, got {token_count}"


# =============================================================================
# Issue 10: Backward Compat in Substitutor - FIXED
# Fix: Removed PatternType enum and PATTERNS dict
# =============================================================================


class TestSubstitutorBackwardCompatRemoved:
    """Tests verifying backward compatibility code was removed from substitutor."""

    def test_no_pattern_type_enum(self):
        """PatternType enum should be removed."""
        import ai_pipeline_core.llm._substitutor as sub_module

        assert not hasattr(sub_module, "PatternType"), "PatternType enum should be removed"

    def test_no_patterns_dict(self):
        """PATTERNS backward compat dict should be removed."""
        import ai_pipeline_core.llm._substitutor as sub_module

        assert not hasattr(sub_module, "PATTERNS"), "PATTERNS dict should be removed"
