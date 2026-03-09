"""Shared ClickHouse connection management for the unified database.

Provides a connection factory and circuit breaker for resilient ClickHouse access.
All database backends share this connection infrastructure.
"""

import time
from typing import Any

import clickhouse_connect
from clickhouse_connect.driver.exceptions import DatabaseError as ClickHouseDatabaseError

from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.settings import Settings

logger = get_pipeline_logger(__name__)

__all__ = [
    "ClickHouseCircuitBreaker",
    "get_clickhouse_client",
]

_FAILURE_THRESHOLD = 3
_RECONNECT_INTERVAL_SEC = 60


def get_clickhouse_client(s: Settings | None = None) -> Any:
    """Create a new ClickHouse client from settings.

    Returns a clickhouse_connect client instance.
    """
    if s is None:
        s = Settings()
    return clickhouse_connect.get_client(  # pyright: ignore[reportUnknownMemberType]
        host=s.clickhouse_host,
        port=s.clickhouse_port,
        database=s.clickhouse_database,
        username=s.clickhouse_user,
        password=s.clickhouse_password,
        secure=s.clickhouse_secure,
        connect_timeout=s.clickhouse_connect_timeout,
        send_receive_timeout=s.clickhouse_send_receive_timeout,
    )


class ClickHouseCircuitBreaker:
    """Simple circuit breaker for ClickHouse connections.

    Tracks consecutive failures. Opens circuit after FAILURE_THRESHOLD failures.
    Auto-resets after RECONNECT_INTERVAL_SEC seconds.

    NOT thread-safe — designed for single-thread executor usage.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._client: Any = None
        self._tables_initialized = False
        self._consecutive_failures = 0
        self._circuit_open = False
        self._last_reconnect_attempt = 0.0

    @property
    def client(self) -> Any:
        """Return the current ClickHouse client, connecting if needed."""
        if self._client is None:
            self._connect()
        return self._client

    @property
    def is_open(self) -> bool:
        """True when circuit is open (ClickHouse unavailable)."""
        return self._circuit_open

    def ensure_connected(self) -> bool:
        """Ensure connection and tables exist. Returns True on success."""
        try:
            if self._client is None:
                self._connect()
            return True
        except (ClickHouseDatabaseError, ConnectionError, OSError) as e:
            logger.warning("ClickHouse connection failed: %s", e)
            self._client = None
            self._tables_initialized = False
            return False

    def ensure_tables(self, ddl_statements: list[str]) -> None:
        """Create tables if not already initialized."""
        if self._tables_initialized:
            return
        if self._client is None:
            self._connect()
        for ddl in ddl_statements:
            self._client.command(ddl)
        self._tables_initialized = True
        logger.info("Database tables verified/created")

    def try_reconnect(self) -> bool:
        """Attempt reconnection after circuit open. Returns True on success.

        Rate-limited to one attempt per RECONNECT_INTERVAL_SEC.
        """
        now = time.monotonic()
        if now - self._last_reconnect_attempt < _RECONNECT_INTERVAL_SEC:
            return False
        self._last_reconnect_attempt = now
        if self.ensure_connected():
            self._circuit_open = False
            self._consecutive_failures = 0
            logger.info("ClickHouse reconnected after circuit break")
            return True
        return False

    def record_success(self) -> None:
        """Record a successful operation. Closes circuit if open."""
        self._consecutive_failures = 0
        if self._circuit_open:
            self._circuit_open = False
            logger.info("Circuit breaker closed")

    def record_failure(self) -> None:
        """Record a failed operation. Opens circuit after threshold."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= _FAILURE_THRESHOLD and not self._circuit_open:
            self._circuit_open = True
            self._client = None
            self._tables_initialized = False
            logger.warning("Circuit breaker opened after %d consecutive failures", self._consecutive_failures)

    def close(self) -> None:
        """Close the client connection."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.debug("Error closing ClickHouse client (ignored during cleanup)")
            self._client = None
            self._tables_initialized = False

    def _connect(self) -> None:
        """Create a new ClickHouse client."""
        self._client = get_clickhouse_client(self._settings)
        logger.info("Connected to ClickHouse at %s:%s", self._settings.clickhouse_host, self._settings.clickhouse_port)
