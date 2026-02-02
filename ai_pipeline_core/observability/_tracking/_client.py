"""ClickHouse client with lazy connection and table management."""

import clickhouse_connect
from pydantic import BaseModel

from ai_pipeline_core.logging import get_pipeline_logger

from ._models import (
    TABLE_DOCUMENT_EVENTS,
    TABLE_PIPELINE_RUNS,
    TABLE_SPAN_EVENTS,
    TABLE_TRACKED_SPANS,
    TrackedSpanRow,
)

logger = get_pipeline_logger(__name__)

# SQL statements for table creation
_CREATE_TABLES_SQL = [
    f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PIPELINE_RUNS}
    (
        run_id           UUID,
        project_name     LowCardinality(String),
        flow_name        LowCardinality(String),
        run_scope        String         DEFAULT '',
        status           LowCardinality(String),
        start_time       DateTime64(3, 'UTC'),
        end_time         Nullable(DateTime64(3, 'UTC')),
        total_cost       Float64        DEFAULT 0,
        total_tokens     UInt64         DEFAULT 0,
        metadata         String         DEFAULT '{{}}' CODEC(ZSTD(3)),
        version          UInt64         DEFAULT 1
    )
    ENGINE = ReplacingMergeTree(version)
    PARTITION BY toYYYYMM(start_time)
    ORDER BY (run_id)
    SETTINGS index_granularity = 8192
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {TABLE_TRACKED_SPANS}
    (
        span_id                  String,
        trace_id                 String,
        run_id                   UUID,
        parent_span_id           Nullable(String),
        name                     String,
        span_type                LowCardinality(String),
        status                   LowCardinality(String),
        start_time               DateTime64(3, 'UTC'),
        end_time                 Nullable(DateTime64(3, 'UTC')),
        duration_ms              UInt64         DEFAULT 0,
        cost                     Float64        DEFAULT 0,
        tokens_input             UInt64         DEFAULT 0,
        tokens_output            UInt64         DEFAULT 0,
        llm_model                LowCardinality(Nullable(String)),
        user_summary             Nullable(String) CODEC(ZSTD(3)),
        user_visible             Bool           DEFAULT false,
        user_label               Nullable(String),
        input_document_sha256s   Array(String),
        output_document_sha256s  Array(String),
        version                  UInt64         DEFAULT 1,
        INDEX idx_trace trace_id TYPE bloom_filter GRANULARITY 1
    )
    ENGINE = ReplacingMergeTree(version)
    PARTITION BY toYYYYMM(start_time)
    ORDER BY (run_id, span_id)
    SETTINGS index_granularity = 8192
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {TABLE_DOCUMENT_EVENTS}
    (
        event_id           UUID,
        run_id             UUID,
        document_sha256    String,
        span_id            String,
        event_type         LowCardinality(String),
        timestamp          DateTime64(3, 'UTC'),
        metadata           String         DEFAULT '{{}}' CODEC(ZSTD(3))
    )
    ENGINE = MergeTree
    PARTITION BY toYYYYMM(timestamp)
    ORDER BY (run_id, document_sha256, timestamp)
    SETTINGS index_granularity = 8192
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {TABLE_SPAN_EVENTS}
    (
        event_id       UUID,
        run_id         UUID,
        span_id        String,
        name           String,
        timestamp      DateTime64(3, 'UTC'),
        attributes     String         DEFAULT '{{}}' CODEC(ZSTD(3)),
        level          LowCardinality(Nullable(String))
    )
    ENGINE = MergeTree
    PARTITION BY toYYYYMM(timestamp)
    ORDER BY (run_id, span_id, timestamp)
    SETTINGS index_granularity = 8192
    """,
]


class ClickHouseClient:
    """Synchronous ClickHouse client with lazy connection.

    All methods are synchronous and must be called from the writer background
    thread — never from the async event loop. Connection is deferred to
    ``connect()`` which is called from the writer thread's ``_run()`` startup.
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
    ) -> None:
        """Store connection params. Does NOT connect yet."""
        self._params = {
            "host": host,
            "port": port,
            "database": database,
            "username": username,
            "password": password,
            "secure": secure,
        }
        self._client: object | None = None
        self._tables_initialized = False

    def connect(self) -> None:
        """Connect to ClickHouse. Call from writer thread, not async context."""
        self._client = clickhouse_connect.get_client(**self._params)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
        logger.info(f"Connected to ClickHouse at {self._params['host']}:{self._params['port']}")

    def ensure_tables(self) -> None:
        """Create tables if they don't exist. Call after connect()."""
        if self._client is None:
            raise RuntimeError("Not connected — call connect() first")
        if self._tables_initialized:
            return
        for sql in _CREATE_TABLES_SQL:
            self._client.command(sql)  # type: ignore[union-attr]

        self._tables_initialized = True
        logger.info("ClickHouse tables verified/created")

    def _insert_rows(self, table: str, rows: list[BaseModel]) -> None:
        """Insert rows into a table using columnar format."""
        if not rows or self._client is None:
            return
        column_names = list(type(rows[0]).model_fields.keys())
        data = [[getattr(row, col) for row in rows] for col in column_names]
        self._client.insert(table, data, column_names=column_names, column_oriented=True)  # type: ignore[union-attr]

    def insert_runs(self, rows: list[BaseModel]) -> None:
        """Insert pipeline run rows."""
        self._insert_rows(TABLE_PIPELINE_RUNS, rows)

    def insert_spans(self, rows: list[BaseModel]) -> None:
        """Insert tracked span rows."""
        self._insert_rows(TABLE_TRACKED_SPANS, rows)

    def insert_document_events(self, rows: list[BaseModel]) -> None:
        """Insert document event rows."""
        self._insert_rows(TABLE_DOCUMENT_EVENTS, rows)

    def insert_span_events(self, rows: list[BaseModel]) -> None:
        """Insert span event rows."""
        self._insert_rows(TABLE_SPAN_EVENTS, rows)

    def update_span(self, row: TrackedSpanRow) -> None:
        """Insert a single replacement span row (ReplacingMergeTree update)."""
        self.insert_spans([row])
