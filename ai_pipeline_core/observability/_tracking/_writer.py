"""Background writer thread for ClickHouse inserts.

Manages connection lifecycle, DDL, and batched columnar inserts for
the two tracking tables: pipeline_runs and pipeline_spans.
"""

import asyncio
import contextlib
from dataclasses import dataclass, field
from threading import Event, Thread

import clickhouse_connect
from pydantic import BaseModel

from ai_pipeline_core.logging import get_pipeline_logger

from ._models import TABLE_PIPELINE_RUNS, TABLE_PIPELINE_SPANS

logger = get_pipeline_logger(__name__)

_CREATE_TABLES_SQL = [
    f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PIPELINE_RUNS}
    (
        execution_id       UUID,
        run_id             LowCardinality(String),
        flow_name          LowCardinality(String),
        run_scope          String          DEFAULT '',
        status             LowCardinality(String),
        start_time         DateTime64(3, 'UTC'),
        end_time           Nullable(DateTime64(3, 'UTC')),
        total_cost         Float64         DEFAULT 0,
        total_tokens       UInt64          DEFAULT 0,
        metadata_json      String          DEFAULT '{{}}' CODEC(ZSTD(3)),
        version            UInt64          DEFAULT 1
    )
    ENGINE = ReplacingMergeTree(version)
    PARTITION BY toYYYYMM(start_time)
    ORDER BY (execution_id)
    SETTINGS index_granularity = 8192
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PIPELINE_SPANS}
    (
        -- Identity
        execution_id           UUID,
        span_id                String,
        trace_id               String,
        parent_span_id         Nullable(String),

        -- Run context (denormalized)
        run_id                 LowCardinality(String),
        flow_name              LowCardinality(String),
        run_scope              String,

        -- Classification
        name                   String,
        span_type              LowCardinality(String),
        status                 LowCardinality(String),

        -- Timing
        start_time             DateTime64(3, 'UTC'),
        end_time               DateTime64(3, 'UTC'),
        duration_ms            UInt64          DEFAULT 0,
        span_order             UInt32          DEFAULT 0,

        -- LLM metrics
        cost                   Float64         DEFAULT 0,
        tokens_input           UInt64          DEFAULT 0,
        tokens_output          UInt64          DEFAULT 0,
        tokens_cached          UInt64          DEFAULT 0,
        llm_model              LowCardinality(Nullable(String)),

        -- Error
        error_message          String          DEFAULT '',

        -- Content (ZSTD-compressed)
        input_json             String          DEFAULT '' CODEC(ZSTD(3)),
        output_json            String          DEFAULT '' CODEC(ZSTD(3)),
        replay_payload         String          DEFAULT '' CODEC(ZSTD(3)),
        attributes_json        String          DEFAULT '{{}}' CODEC(ZSTD(3)),
        events_json            String          DEFAULT '[]' CODEC(ZSTD(3)),

        -- Document lineage
        input_doc_sha256s      Array(String),
        output_doc_sha256s     Array(String),

        -- Indexes
        INDEX idx_name          name              TYPE bloom_filter GRANULARITY 1,
        INDEX idx_span_type     span_type         TYPE set(10)      GRANULARITY 4,
        INDEX idx_trace         trace_id          TYPE bloom_filter  GRANULARITY 1,
        INDEX idx_status        status            TYPE set(10)       GRANULARITY 4,
        INDEX idx_input_docs    input_doc_sha256s TYPE bloom_filter  GRANULARITY 1,
        INDEX idx_output_docs   output_doc_sha256s TYPE bloom_filter GRANULARITY 1
    )
    ENGINE = ReplacingMergeTree()
    PARTITION BY toYYYYMM(start_time)
    ORDER BY (execution_id, span_id)
    SETTINGS index_granularity = 8192
    """,
]


@dataclass(frozen=True, slots=True)
class InsertBatch:
    """Batch of rows to insert into a single table."""

    table: str
    rows: list[BaseModel] = field(default_factory=list)


_SENTINEL = object()

_BATCH_SIZE = 100


class ClickHouseWriter:
    """Background writer that batches inserts to ClickHouse.

    Manages connection lifecycle, DDL, and batched columnar inserts.
    Uses a dedicated thread with its own asyncio event loop.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int = 8443,
        database: str = "default",
        username: str = "default",
        password: str = "",
        secure: bool = True,
        connect_timeout: int = 10,
        send_receive_timeout: int = 30,
        flush_interval_seconds: float = 2.0,
    ) -> None:
        self._conn_params = {
            "host": host,
            "port": port,
            "database": database,
            "username": username,
            "password": password,
            "secure": secure,
            "connect_timeout": connect_timeout,
            "send_receive_timeout": send_receive_timeout,
        }
        self._flush_interval = flush_interval_seconds

        self._client: object | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[InsertBatch | object] | None = None
        self._thread: Thread | None = None
        self._shutdown = False
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
        """Entry point for background thread."""
        self._loop = asyncio.new_event_loop()
        self._queue = asyncio.Queue()
        self._ready.set()
        try:
            self._loop.run_until_complete(self._run())
        finally:
            self._loop.close()
            self._loop = None

    async def _run(self) -> None:
        """Main async loop: connect (with indefinite retry), create tables, then drain queue."""
        assert self._queue is not None
        await self._connect_until_ready()

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
                if sum(len(v) for v in pending.values()) >= _BATCH_SIZE:
                    self._flush_batches(pending)
            elif isinstance(item, Event):
                self._flush_batches(pending)
                item.set()

    async def _connect_until_ready(self) -> None:
        """Connect to ClickHouse, retrying indefinitely until success or shutdown.

        Initial attempts use exponential backoff (5 attempts). After that,
        sleeps 60s between retry rounds. Queue accumulates during outage.
        """
        INITIAL_RETRIES = 5
        RECONNECT_SLEEP = 60

        while not self._shutdown:
            if await self._connect_with_retry(max_retries=INITIAL_RETRIES):
                return
            logger.warning(
                "ClickHouse unavailable after %d attempts, retrying in %ds. Queued spans will be flushed on reconnect.",
                INITIAL_RETRIES,
                RECONNECT_SLEEP,
            )
            for _ in range(RECONNECT_SLEEP):
                if self._shutdown:
                    return
                await asyncio.sleep(1)

    async def _connect_with_retry(self, max_retries: int = 5, base_delay: float = 1.0) -> bool:
        """Attempt to connect to ClickHouse with exponential backoff."""
        for attempt in range(max_retries):
            if self._shutdown:
                return False
            try:
                self._client = clickhouse_connect.get_client(**self._conn_params)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
                self._ensure_tables()
                logger.info("Connected to ClickHouse at %s:%s", self._conn_params["host"], self._conn_params["port"])
                return True
            except Exception as e:
                delay = base_delay * (2**attempt)
                if attempt < max_retries - 1:
                    logger.warning("ClickHouse connection attempt %d/%d failed: %s. Retrying in %.0fs", attempt + 1, max_retries, e, delay)
                    await asyncio.sleep(delay)
                else:
                    logger.warning("ClickHouse connection failed after %d attempts: %s", max_retries, e)
        return False

    def _ensure_tables(self) -> None:
        """Create tracking tables if they don't exist."""
        client = self._client
        if client is None:
            return
        for sql in _CREATE_TABLES_SQL:
            client.command(sql)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        logger.info("ClickHouse tracking tables verified/created")

    def _flush_batches(self, pending: dict[str, list[BaseModel]]) -> None:
        """Flush all pending inserts to ClickHouse.

        Failed batches are discarded to prevent unbounded memory growth.
        Tracking failures are non-fatal — the pipeline continues regardless.
        """
        for table, rows in list(pending.items()):
            if rows:
                try:
                    self._insert_rows(table, rows)
                except Exception as e:
                    logger.warning(
                        "Failed to insert %d rows into %s (discarding batch): %s. Check ClickHouse connectivity and schema compatibility.",
                        len(rows),
                        table,
                        e,
                    )
        pending.clear()

    def _insert_rows(self, table: str, rows: list[BaseModel]) -> None:
        """Insert rows into a table using columnar format."""
        client = self._client
        if not rows or client is None:
            return
        row_type = type(rows[0])
        if not all(type(r) is row_type for r in rows):
            raise ValueError(f"Mixed row types in batch for table {table}")
        column_names = list(row_type.model_fields.keys())
        data = [[getattr(row, col) for row in rows] for col in column_names]
        client.insert(table, data, column_names=column_names, column_oriented=True)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]

    @property
    def _inactive(self) -> bool:
        return self._shutdown or self._loop is None or self._queue is None

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
