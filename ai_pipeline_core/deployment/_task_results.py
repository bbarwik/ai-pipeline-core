"""TaskResultStore implementations for durable completion result backup.

ClickHouseTaskResultStore persists results to a ClickHouse table.
MemoryTaskResultStore stores results in-memory for testing.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import clickhouse_connect

from ai_pipeline_core.logging import get_pipeline_logger

from ._types import TaskResultRecord

logger = get_pipeline_logger(__name__)

TABLE_TASK_RESULTS = "task_results"

_DDL_TASK_RESULTS = f"""
CREATE TABLE IF NOT EXISTS {TABLE_TASK_RESULTS}
(
    run_id          String,
    result          String CODEC(ZSTD(3)),
    chain_context   String CODEC(ZSTD(3)),
    stored_at       DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree(stored_at)
ORDER BY (run_id)
SETTINGS index_granularity = 8192
"""


class ClickHouseTaskResultStore:
    """Persists completion results to ClickHouse for durable backup.

    Uses a single-thread executor for sync clickhouse_connect operations.
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
    ) -> None:
        self._params = {
            "host": host,
            "port": port,
            "database": database,
            "username": username,
            "password": password,
            "secure": secure,
            "connect_timeout": connect_timeout,
            "send_receive_timeout": send_receive_timeout,
        }
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ch-taskresults")
        self._client: Any = None
        self._tables_initialized = False

    async def _run(self, fn: Any, *args: Any) -> Any:
        """Run a sync function on the dedicated executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    def _connect(self) -> None:
        """Establish connection to ClickHouse."""
        self._client = clickhouse_connect.get_client(  # pyright: ignore[reportUnknownMemberType]
            **self._params,  # pyright: ignore[reportArgumentType]
        )
        logger.info("Task result store connected to ClickHouse at %s:%d", self._params["host"], self._params["port"])

    def _ensure_tables(self) -> None:
        """Create tables if not already initialized."""
        if self._tables_initialized:
            return
        if self._client is None:
            self._connect()
        self._client.command(_DDL_TASK_RESULTS)
        self._tables_initialized = True
        logger.info("Task result store table verified/created")

    def _sync_write(self, run_id: str, result: str, chain_context: str) -> None:
        """Insert a result row (sync, executor thread)."""
        self._ensure_tables()
        now = datetime.now(UTC)
        self._client.insert(
            TABLE_TASK_RESULTS,
            [[run_id, result, chain_context, now]],
            column_names=["run_id", "result", "chain_context", "stored_at"],
        )

    def _sync_read(self, run_id: str) -> TaskResultRecord | None:
        """Read a result row by run_id (sync, executor thread)."""
        self._ensure_tables()
        query = (
            f"SELECT run_id, result, chain_context, stored_at FROM {TABLE_TASK_RESULTS} FINAL WHERE run_id = {{run_id:String}} ORDER BY stored_at DESC LIMIT 1"
        )
        result = self._client.query(query, parameters={"run_id": run_id})
        rows = result.result_rows
        if not rows:
            return None
        row = rows[0]
        return TaskResultRecord(
            run_id=row[0],
            result=row[1],
            chain_context=row[2],
            stored_at=row[3],
        )

    async def write_result(self, run_id: str, result: str, chain_context: str) -> None:
        """Write a completion result to ClickHouse."""
        await self._run(self._sync_write, run_id, result, chain_context)

    async def read_result(self, run_id: str) -> TaskResultRecord | None:
        """Read a completion result from ClickHouse."""
        return await self._run(self._sync_read, run_id)

    def shutdown(self) -> None:
        """Shut down the executor."""
        self._executor.shutdown(wait=False)


class MemoryTaskResultStore:
    """In-memory task result store for testing."""

    def __init__(self) -> None:
        self._results: dict[str, TaskResultRecord] = {}

    async def write_result(self, run_id: str, result: str, chain_context: str) -> None:
        """Store a result in memory."""
        self._results[run_id] = TaskResultRecord(
            run_id=run_id,
            result=result,
            chain_context=chain_context,
            stored_at=datetime.now(UTC),
        )

    async def read_result(self, run_id: str) -> TaskResultRecord | None:
        """Retrieve a result from memory."""
        return self._results.get(run_id)

    def shutdown(self) -> None:
        """No resources to release in memory store."""


__all__ = [
    "ClickHouseTaskResultStore",
    "MemoryTaskResultStore",
]
