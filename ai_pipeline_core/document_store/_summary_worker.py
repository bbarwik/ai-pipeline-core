"""Background worker for asynchronous document summary generation."""

import asyncio
import contextlib
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from threading import Event, Thread

from lmnr.opentelemetry_lib.tracing import context as laminar_context
from opentelemetry import context as otel_context
from opentelemetry.context import Context

from ai_pipeline_core.document_store._summary import SUMMARY_EXCERPT_CHARS, SummaryGenerator
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)

_SENTINEL = object()


@dataclass(frozen=True, slots=True)
class _SummaryItem:
    sha256: str
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
        update_fn: Callable[[str, str], Coroutine[None, None, None]],
    ) -> None:
        self._generator = generator
        self._update_fn = update_fn
        self._inflight: set[str] = set()  # sha256 only — one summary per document globally
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
            logger.warning(f"Summary generation failed for '{item.name}': {e}")
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
        if document.is_text:
            excerpt = document.text[:SUMMARY_EXCERPT_CHARS]
        else:
            excerpt = f"[Binary document: {document.mime_type}, {len(document.content)} bytes]"
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
