"""Background writer thread for ClickHouse inserts and summary jobs."""

import asyncio
import contextlib
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from threading import Event, Thread

from lmnr.opentelemetry_lib.tracing import context as laminar_context
from opentelemetry import context as otel_context
from opentelemetry.context import Context
from pydantic import BaseModel

from ai_pipeline_core.logging import get_pipeline_logger

from ._client import ClickHouseClient
from ._models import TrackedSpanRow

type SpanSummaryFn = Callable[[str, str, str], Coroutine[None, None, str]]

logger = get_pipeline_logger(__name__)


@dataclass(frozen=True)
class InsertBatch:
    """Batch of rows to insert into a single table."""

    table: str
    rows: list[BaseModel] = field(default_factory=list)


@dataclass(frozen=True)
class SummaryJob:
    """Job requesting LLM-generated summary for a span."""

    span_id: str
    label: str
    output_hint: str
    summary_model: str = "gemini-3-flash"
    parent_otel_context: Context | None = field(default=None, hash=False, compare=False)
    parent_laminar_context: Context | None = field(default=None, hash=False, compare=False)


_SENTINEL = object()


class ClickHouseWriter:
    """Background writer that batches inserts and processes summary jobs.

    Uses a dedicated thread with its own asyncio event loop. External callers
    push work via ``write()`` and ``write_job()`` which use
    ``loop.call_soon_threadsafe()``. The writer drains the queue in batches
    for efficiency.
    """

    def __init__(
        self,
        client: ClickHouseClient,
        *,
        summary_row_builder: Callable[[str, str], TrackedSpanRow | None] | None = None,
        span_summary_fn: SpanSummaryFn | None = None,
        batch_size: int = 100,
        flush_interval_seconds: float = 2.0,
    ) -> None:
        """Store config. Does NOT start the writer thread."""
        self._client = client
        self._summary_row_builder = summary_row_builder
        self._span_summary_fn = span_summary_fn
        self._batch_size = batch_size
        self._flush_interval = flush_interval_seconds

        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[InsertBatch | SummaryJob | object] | None = None
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
        pending_jobs: list[SummaryJob] = []

        while True:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=self._flush_interval)
            except TimeoutError:
                self._flush_batches(pending)
                await self._process_summary_jobs(pending_jobs)
                continue

            if item is _SENTINEL:
                self._flush_batches(pending)
                await self._process_summary_jobs(pending_jobs)
                break

            if isinstance(item, InsertBatch):
                pending.setdefault(item.table, []).extend(item.rows)
                if sum(len(v) for v in pending.values()) >= self._batch_size:
                    self._flush_batches(pending)
            elif isinstance(item, SummaryJob):
                pending_jobs.append(item)
            elif isinstance(item, Event):
                self._flush_batches(pending)
                await self._process_summary_jobs(pending_jobs)
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
                    logger.warning(f"ClickHouse connection attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay:.0f}s")
                    await asyncio.sleep(delay)
                else:
                    logger.warning(f"ClickHouse connection failed after {max_retries} attempts, tracking disabled: {e}")
                    self._disabled = True
                    return False
        return False  # unreachable

    def _flush_batches(self, pending: dict[str, list[BaseModel]]) -> None:
        """Flush all pending inserts to ClickHouse."""
        flushed: list[str] = []
        for table, rows in list(pending.items()):
            if rows:
                try:
                    self._client._insert_rows(table, rows)  # pyright: ignore[reportPrivateUsage]
                    flushed.append(table)
                except Exception as e:
                    logger.warning(f"Failed to insert {len(rows)} rows into {table}: {e}")
        for table in flushed:
            del pending[table]

    async def _process_summary_jobs(self, jobs: list[SummaryJob]) -> None:
        """Process all pending summary jobs in parallel."""
        if not jobs:
            return
        await asyncio.gather(*[self._process_summary_job(job) for job in jobs])
        jobs.clear()

    async def _process_summary_job(self, job: SummaryJob) -> None:
        """Process a span summary generation job."""
        try:
            if self._span_summary_fn:
                otel_token = otel_context.attach(job.parent_otel_context) if job.parent_otel_context is not None else None
                laminar_token = laminar_context.attach_context(job.parent_laminar_context) if job.parent_laminar_context is not None else None
                try:
                    summary = await self._span_summary_fn(job.label, job.output_hint, job.summary_model)
                finally:
                    if laminar_token is not None:
                        laminar_context.detach_context(laminar_token)
                    if otel_token is not None:
                        otel_context.detach(otel_token)
                if summary and self._summary_row_builder:
                    row = self._summary_row_builder(job.span_id, summary)
                    if row:
                        self._client.update_span(row)
        except Exception as e:
            logger.warning(f"Summary job failed: {e}")

    def write(self, table: str, rows: list[BaseModel]) -> None:
        """Enqueue rows for insertion. Thread-safe, non-blocking."""
        if self._disabled or self._shutdown or self._loop is None or self._queue is None:
            return
        batch = InsertBatch(table=table, rows=rows)
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._queue.put_nowait, batch)

    def write_job(self, job: SummaryJob) -> None:
        """Enqueue a summary job. Thread-safe, non-blocking."""
        if self._disabled or self._shutdown or self._loop is None or self._queue is None:
            return
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._queue.put_nowait, job)

    def flush(self, timeout: float = 30.0) -> None:
        """Block until all queued items (including summary jobs) are processed."""
        if self._disabled or self._shutdown or self._loop is None or self._queue is None:
            return
        barrier = Event()
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, barrier)
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
