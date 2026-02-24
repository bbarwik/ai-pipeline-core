"""Background writer thread for ClickHouse inserts."""

import asyncio
import contextlib
from dataclasses import dataclass, field
from threading import Event, Thread

from pydantic import BaseModel

from ai_pipeline_core.logging import get_pipeline_logger

from ._client import ClickHouseClient

logger = get_pipeline_logger(__name__)


@dataclass(frozen=True, slots=True)
class InsertBatch:
    """Batch of rows to insert into a single table."""

    table: str
    rows: list[BaseModel] = field(default_factory=list)


_SENTINEL = object()


class ClickHouseWriter:
    """Background writer that batches inserts to ClickHouse.

    Uses a dedicated thread with its own asyncio event loop. External callers
    push work via ``write()`` which uses ``loop.call_soon_threadsafe()``.
    The writer drains the queue in batches for efficiency.
    """

    def __init__(
        self,
        client: ClickHouseClient,
        *,
        batch_size: int = 100,
        flush_interval_seconds: float = 2.0,
    ) -> None:
        """Store config. Does NOT start the writer thread."""
        self._client = client
        self._batch_size = batch_size
        self._flush_interval = flush_interval_seconds

        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[InsertBatch | object] | None = None
        self._thread: Thread | None = None
        self._shutdown = False
        self._disabled = False
        self._ready = Event()

    def start(self) -> None:
        """Start the background writer thread."""
        if self._thread is not None:
            return
        self._thread = Thread(target=self._thread_main, name="ch-writer", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            logger.warning("ClickHouse writer thread did not start within 10 seconds")

    def _thread_main(self) -> None:
        """Entry point for background thread — creates event loop and runs."""
        self._loop = asyncio.new_event_loop()
        self._queue = asyncio.Queue()
        self._ready.set()
        try:
            self._loop.run_until_complete(self._run())
        finally:
            self._loop.close()
            self._loop = None

    async def _run(self) -> None:
        """Main async loop: connect, create tables, then drain queue."""
        assert self._queue is not None, "_run() must be called after _queue is initialized"

        if not await self._connect_with_retry():
            return

        pending: dict[str, list[BaseModel]] = {}

        while True:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=self._flush_interval)
            except TimeoutError:
                self._flush_batches(pending)
                continue

            if item is _SENTINEL:
                self._flush_batches(pending)
                break

            if isinstance(item, InsertBatch):
                pending.setdefault(item.table, []).extend(item.rows)
                if sum(len(v) for v in pending.values()) >= self._batch_size:
                    self._flush_batches(pending)
            elif isinstance(item, Event):
                self._flush_batches(pending)
                item.set()

    async def _connect_with_retry(self, max_retries: int = 5, base_delay: float = 1.0) -> bool:
        """Attempt to connect to ClickHouse with exponential backoff.

        Returns True on success, False if all retries exhausted (writer is disabled).
        """
        for attempt in range(max_retries):
            try:
                self._client.connect()
                self._client.ensure_tables()
                return True
            except Exception as e:
                delay = base_delay * (2**attempt)
                if attempt < max_retries - 1:
                    logger.warning("ClickHouse connection attempt %d/%d failed: %s. Retrying in %.0fs", attempt + 1, max_retries, e, delay)
                    await asyncio.sleep(delay)
                else:
                    logger.warning("ClickHouse connection failed after %d attempts, tracking disabled: %s", max_retries, e)
                    self._disabled = True
                    return False
        return False  # unreachable

    def _flush_batches(self, pending: dict[str, list[BaseModel]]) -> None:
        """Flush all pending inserts to ClickHouse."""
        flushed: list[str] = []
        for table, rows in list(pending.items()):
            if rows:
                try:
                    self._client._insert_rows(table, rows)
                    flushed.append(table)
                except Exception as e:
                    logger.warning("Failed to insert %d rows into %s: %s", len(rows), table, e)
        for table in flushed:
            del pending[table]

    @property
    def _inactive(self) -> bool:
        """True when the writer cannot accept work."""
        return self._disabled or self._shutdown or self._loop is None or self._queue is None

    def _enqueue(self, item: object) -> None:
        """Enqueue an item on the writer's event loop. Thread-safe, non-blocking."""
        if self._inactive:
            return
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)  # type: ignore[union-attr]

    def write(self, table: str, rows: list[BaseModel]) -> None:
        """Enqueue rows for insertion. Thread-safe, non-blocking."""
        self._enqueue(InsertBatch(table=table, rows=rows))

    def flush(self, timeout: float = 30.0) -> None:
        """Block until all queued items are processed."""
        if self._inactive:
            return
        barrier = Event()
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, barrier)  # type: ignore[union-attr]
        except RuntimeError:
            return
        barrier.wait(timeout=timeout)

    def shutdown(self, timeout: float = 60.0) -> None:
        """Signal shutdown and wait for the writer thread to finish."""
        if self._shutdown:
            return
        self._shutdown = True
        if self._loop is not None and self._queue is not None:
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._queue.put_nowait, _SENTINEL)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
