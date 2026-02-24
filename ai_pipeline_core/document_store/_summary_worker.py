"""Background worker for asynchronous document summary generation.

Also defines the summary-related type aliases and constants (merged from _summary.py).
"""

import asyncio
import contextlib
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from threading import Event, Thread

from lmnr.opentelemetry_lib.tracing import context as laminar_context
from opentelemetry import context as otel_context
from opentelemetry.context import Context

from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.documents.types import DocumentSha256
from ai_pipeline_core.logging import get_pipeline_logger

type SummaryGenerator = Callable[[str, str], Coroutine[None, None, str]]
"""Async callable: (document_name, content_excerpt) -> summary string.
Returns empty string on failure. Must handle recursion prevention internally."""

type SummaryUpdateFn = Callable[[DocumentSha256, str], Coroutine[None, None, None]]
"""Async callable: (document_sha256, summary) -> None. Persists summary to store."""

SUMMARY_EXCERPT_CHARS: int = 30_000
"""Total character budget for document excerpts (split across start/middle/end sections)."""

EXCERPT_SECTION_CHARS: int = 10_000
"""Characters per excerpt section (start, middle, end)."""

logger = get_pipeline_logger(__name__)

_SENTINEL = object()


def _build_excerpt(document: Document) -> str:
    """Build a rich excerpt for summary generation using XML structure.

    Mirrors the XML wrapping used in _llm_core for document context:
    <document>, <name>, <class>, <description>, <content> tags.
    For long text documents, content samples start, middle, and end sections.
    """
    parts: list[str] = ["<document>"]
    parts.append(f"<name>{document.name}</name>")
    parts.append(f"<class>{type(document).__name__}</class>")
    if document.description:
        parts.append(f"<description>{document.description}</description>")

    if not document.is_text:
        parts.append(f"<content>[Binary: {document.mime_type}, {len(document.content)} bytes]</content>")
        parts.append("</document>")
        return "\n".join(parts)

    text = document.text
    total_len = len(text)

    parts.append("<content>")
    if total_len <= SUMMARY_EXCERPT_CHARS:
        parts.append(text)
    else:
        sec = EXCERPT_SECTION_CHARS
        mid_start = (total_len - sec) // 2

        gap_after_start = mid_start - sec
        gap_after_middle = (total_len - sec) - (mid_start + sec)

        parts.append(text[:sec])
        parts.append(f"\n[... {gap_after_start:,} chars truncated ...]\n")
        parts.append(text[mid_start : mid_start + sec])
        parts.append(f"\n[... {gap_after_middle:,} chars truncated ...]\n")
        parts.append(text[-sec:])

    parts.append("</content>")
    parts.append("</document>")
    return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class _SummaryItem:
    sha256: DocumentSha256
    name: str
    excerpt: str
    parent_otel_context: Context | None = field(default=None, hash=False, compare=False)
    parent_laminar_context: Context | None = field(default=None, hash=False, compare=False)


class SummaryWorker:
    """Background daemon thread that generates summaries and writes them back to the store.

    Processes jobs in parallel on its own asyncio event loop. Thread-safe scheduling
    via ``loop.call_soon_threadsafe()``. Best-effort — failures are logged and skipped.
    """

    def __init__(
        self,
        *,
        generator: SummaryGenerator,
        update_fn: SummaryUpdateFn,
    ) -> None:
        self._generator = generator
        self._update_fn = update_fn
        self._inflight: set[DocumentSha256] = set()  # sha256 only — one summary per document globally
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[_SummaryItem | object] | None = None
        self._thread: Thread | None = None
        self._ready = Event()

    def start(self) -> None:
        """Start the background daemon thread for summary generation."""
        if self._thread is not None:
            return
        self._thread = Thread(target=self._thread_main, name="summary-worker", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            logger.warning("Summary worker thread did not start within 5 seconds")

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._queue = asyncio.Queue()
        self._ready.set()
        try:
            self._loop.run_until_complete(self._run())
        finally:
            self._loop.close()
            self._loop = None

    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, Event):
                item.set()
                continue
            assert isinstance(item, _SummaryItem)

            # Collect all immediately available items into a batch
            batch: list[_SummaryItem] = [item]
            sentinel_seen = False
            flush_events: list[Event] = []

            while not self._queue.empty():
                try:
                    next_item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if next_item is _SENTINEL:
                    sentinel_seen = True
                    break
                if isinstance(next_item, Event):
                    flush_events.append(next_item)
                    break
                assert isinstance(next_item, _SummaryItem)
                batch.append(next_item)

            await asyncio.gather(*[self._process_one(i) for i in batch])

            for event in flush_events:
                event.set()
            if sentinel_seen:
                break

    async def _process_one(self, item: _SummaryItem) -> None:
        try:
            otel_token = otel_context.attach(item.parent_otel_context) if item.parent_otel_context is not None else None
            laminar_token = laminar_context.attach_context(item.parent_laminar_context) if item.parent_laminar_context is not None else None
            try:
                summary = await self._generator(item.name, item.excerpt)
            finally:
                if laminar_token is not None:
                    laminar_context.detach_context(laminar_token)
                if otel_token is not None:
                    otel_context.detach(otel_token)
            if summary:
                await self._update_fn(item.sha256, summary)
        except Exception as e:
            logger.warning("Summary generation failed for '%s': %s", item.name, e)
        finally:
            self._inflight.discard(item.sha256)

    def schedule(self, document: Document) -> None:
        """Schedule summary generation for a document. Thread-safe, non-blocking."""
        if self._loop is None or self._queue is None:
            return
        key = document.sha256
        if key in self._inflight:
            return
        self._inflight.add(key)
        excerpt = _build_excerpt(document)
        item = _SummaryItem(
            sha256=document.sha256,
            name=document.name,
            excerpt=excerpt,
            parent_otel_context=otel_context.get_current(),
            parent_laminar_context=laminar_context.get_current_context(),
        )
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except RuntimeError:
            self._inflight.discard(key)

    def flush(self, timeout: float = 60.0) -> None:
        """Block until all queued items are processed."""
        if self._loop is None or self._queue is None:
            return
        barrier = Event()
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, barrier)
        except RuntimeError:
            return
        if not barrier.wait(timeout=timeout):
            logger.warning("Summary worker flush timed out after %.0fs — some summaries may still be processing", timeout)

    def shutdown(self, timeout: float = 60.0) -> None:
        """Send stop sentinel and join the worker thread. Pending items are drained before stop."""
        if self._loop is not None and self._queue is not None:
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._queue.put_nowait, _SENTINEL)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
