"""Prompt cache coordination for concurrent LLM calls."""

import asyncio
import time
from asyncio import Lock

from ai_pipeline_core.documents import Document
from ai_pipeline_core.llm import AIMessages, ModelName
from ai_pipeline_core.llm.model_response import ModelResponse

CACHED_PROMPTS: dict[str, Lock | int] = {}

_cache_lock = Lock()
CACHE_TTL = 600
MIN_SIZE_FOR_CACHE = 32 * 1024


class GlobalCacheLock:
    """Serialize first prompt per cache key so subsequent calls get cache hits.

    Waits for the first caller to complete before allowing others to execute,
    ensuring the prompt cache is populated.
    """

    wait_time: float = 0
    use_cache: bool = False

    def _context_size(self, context: AIMessages) -> int:
        length = 0
        for msg in context:
            if isinstance(msg, Document):
                if msg.is_text:
                    length += msg.size
                else:
                    length += 1024
            elif isinstance(msg, str):
                length += len(msg)
            elif isinstance(msg, ModelResponse):  # type: ignore[arg-type]
                length += len(msg.content)
        return length

    def __init__(self, model: ModelName, context: AIMessages, cache_lock: bool):  # noqa: D107
        self.use_cache = cache_lock and self._context_size(context) > MIN_SIZE_FOR_CACHE
        self.cache_key = f"{model}-{context.get_prompt_cache_key()}"
        self.new_cache = False

    async def __aenter__(self) -> "GlobalCacheLock":
        wait_start = time.time()
        if not self.use_cache:
            return self

        async with _cache_lock:
            cache = CACHED_PROMPTS.get(self.cache_key)
            if isinstance(cache, int):
                if time.time() > cache + CACHE_TTL:
                    cache = None
                else:
                    CACHED_PROMPTS[self.cache_key] = int(time.time())
                    self.wait_time = time.time() - wait_start
                    return self
            if not cache:
                self.new_cache = True
                CACHED_PROMPTS[self.cache_key] = Lock()
                await CACHED_PROMPTS[self.cache_key].acquire()  # type: ignore[union-attr]

        if not self.new_cache and isinstance(cache, Lock):
            async with cache:
                pass  # waiting for lock to be released

        self.wait_time = time.time() - wait_start
        return self

    async def __aexit__(self, exc_type: type | None, exc: BaseException | None, tb: object) -> None:
        if self.new_cache:
            await asyncio.sleep(1)  # give time for cache to be prepared
            async with _cache_lock:
                CACHED_PROMPTS[self.cache_key].release()  # type: ignore[union-attr]
                CACHED_PROMPTS[self.cache_key] = int(time.time())
