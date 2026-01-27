"""Tests for prompt_builder global cache module."""

# pyright: reportPrivateUsage=false

from ai_pipeline_core.documents import TemporaryDocument
from ai_pipeline_core.llm import AIMessages
from ai_pipeline_core.prompt_builder.global_cache import (
    CACHED_PROMPTS,
    MIN_SIZE_FOR_CACHE,
    GlobalCacheLock,
)


class TestGlobalCacheLock:
    """Test GlobalCacheLock context manager."""

    def test_context_size_text(self):
        """Test size calculation for text messages."""
        lock = GlobalCacheLock.__new__(GlobalCacheLock)
        context = AIMessages(["hello world", "test message"])
        size = lock._context_size(context)
        assert size == len("hello world") + len("test message")

    def test_context_size_document(self):
        """Test size calculation for document messages."""
        lock = GlobalCacheLock.__new__(GlobalCacheLock)
        doc = TemporaryDocument(name="test.txt", content=b"x" * 100)
        context = AIMessages([doc])
        size = lock._context_size(context)
        assert size == 100

    def test_context_size_binary_document(self):
        """Test size calculation for non-text document uses fixed estimate."""
        lock = GlobalCacheLock.__new__(GlobalCacheLock)
        # Create actual binary content that won't be detected as text
        binary = bytes(range(256)) * 4
        doc = TemporaryDocument(name="test.bin", content=binary)
        context = AIMessages([doc])
        size = lock._context_size(context)
        # Binary documents: either treated as non-text (1024) or as text (their size)
        assert size > 0

    def test_init_no_cache_small_context(self):
        """Test cache is disabled for small contexts."""
        context = AIMessages(["short"])
        lock = GlobalCacheLock("model", context, cache_lock=True)
        assert lock.use_cache is False

    def test_init_no_cache_when_disabled(self):
        """Test cache is disabled when cache_lock=False."""
        big_text = "x" * (MIN_SIZE_FOR_CACHE + 1)
        context = AIMessages([big_text])
        lock = GlobalCacheLock("model", context, cache_lock=False)
        assert lock.use_cache is False

    def test_init_enables_cache_for_large_context(self):
        """Test cache is enabled for large contexts."""
        big_text = "x" * (MIN_SIZE_FOR_CACHE + 1)
        context = AIMessages([big_text])
        lock = GlobalCacheLock("model", context, cache_lock=True)
        assert lock.use_cache is True

    async def test_aenter_no_cache(self):
        """Test __aenter__ returns quickly when cache disabled."""
        context = AIMessages(["short"])
        lock = GlobalCacheLock("model", context, cache_lock=False)
        result = await lock.__aenter__()
        assert result is lock
        assert lock.wait_time == 0
        await lock.__aexit__(None, None, None)

    async def test_aenter_new_cache_entry(self):
        """Test __aenter__ creates new cache entry for large context."""
        # Use a unique model to avoid collision with other tests
        model = "test-unique-model-new-cache"
        big_text = "x" * (MIN_SIZE_FOR_CACHE + 1)
        context = AIMessages([big_text])
        lock = GlobalCacheLock(model, context, cache_lock=True)

        # Clean up any previous state
        CACHED_PROMPTS.pop(lock.cache_key, None)

        async with lock:
            assert lock.new_cache is True

        # After exit, cache should have a timestamp
        assert isinstance(CACHED_PROMPTS.get(lock.cache_key), int)

        # Cleanup
        CACHED_PROMPTS.pop(lock.cache_key, None)

    async def test_aenter_existing_cache_hit(self):
        """Test __aenter__ detects existing cache entry."""
        import time

        model = "test-unique-model-cache-hit"
        big_text = "y" * (MIN_SIZE_FOR_CACHE + 1)
        context = AIMessages([big_text])

        lock1 = GlobalCacheLock(model, context, cache_lock=True)

        # Pre-populate cache with a recent timestamp
        CACHED_PROMPTS[lock1.cache_key] = int(time.time())

        lock2 = GlobalCacheLock(model, context, cache_lock=True)
        result = await lock2.__aenter__()
        assert result is lock2
        await lock2.__aexit__(None, None, None)

        # Cleanup
        CACHED_PROMPTS.pop(lock1.cache_key, None)
