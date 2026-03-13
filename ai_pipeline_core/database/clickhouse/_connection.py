"""Shared ClickHouse async client helpers."""

import asyncio
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

from clickhouse_connect import get_async_client
from clickhouse_connect.driver.asyncclient import AsyncClient
from clickhouse_connect.driver.exceptions import DatabaseError as ClickHouseDatabaseError

from ai_pipeline_core.database.clickhouse._ddl import DDL_STATEMENTS, SCHEMA_META_TABLE, SCHEMA_VERSION
from ai_pipeline_core.settings import Settings

__all__ = [
    "SchemaVersionError",
    "get_async_clickhouse_client",
    "reset_schema_check",
]


@dataclass(slots=True)
class _SchemaState:
    """Process-level schema verification flag."""

    verified: bool = False


_schema_state = _SchemaState()
_schema_lock = asyncio.Lock()


async def _table_exists(client: AsyncClient, table: str, database: str) -> bool:
    result = await client.query(
        "SELECT 1 FROM system.tables WHERE database = {db:String} AND name = {tbl:String}",
        parameters={"db": database, "tbl": table},
    )
    return len(result.result_rows) > 0


async def _ensure_schema(client: AsyncClient, database: str) -> None:
    """Check schema version or auto-initialize an empty database.

    - No schema_meta table → fresh database → create all tables and stamp version.
    - schema_meta exists, version matches → proceed.
    - schema_meta exists, DB version < framework → raise error (DB needs upgrade).
    - schema_meta exists, DB version > framework → raise error (framework needs upgrade).
    """
    if _schema_state.verified:
        return

    async with _schema_lock:
        if _schema_state.verified:
            return

        meta_exists = await _table_exists(client, SCHEMA_META_TABLE, database)

        if not meta_exists:
            fw_version = _get_framework_version()
            for ddl in DDL_STATEMENTS:
                await client.command(ddl)
            await client.command(
                f"INSERT INTO {SCHEMA_META_TABLE} (version, applied_at, framework_version) VALUES ({{version:UInt32}}, now64(3), {{fw:String}})",
                parameters={"version": SCHEMA_VERSION, "fw": fw_version},
            )
            _schema_state.verified = True
            return

        result = await client.query(f"SELECT max(version) FROM {SCHEMA_META_TABLE}")
        db_version = int(result.result_rows[0][0]) if result.result_rows and result.result_rows[0][0] else 0

        if db_version < SCHEMA_VERSION:
            raise SchemaVersionError(
                f"Database schema version ({db_version}) is older than the framework expects ({SCHEMA_VERSION}). "
                "Update the database schema to match the current framework version."
            )
        if db_version > SCHEMA_VERSION:
            raise SchemaVersionError(
                f"Database schema version ({db_version}) is newer than the framework supports ({SCHEMA_VERSION}). "
                f"Upgrade ai-pipeline-core to a version that supports schema version {db_version}."
            )

        _schema_state.verified = True


def _get_framework_version() -> str:
    try:
        return package_version("ai-pipeline-core")
    except PackageNotFoundError:
        return "unknown"


class SchemaVersionError(Exception):
    """Raised when the database schema version does not match the framework's expected version."""


def reset_schema_check() -> None:
    """Reset the process-level schema verification flag. For use in tests only."""
    _schema_state.verified = False


async def get_async_clickhouse_client(settings: Settings | None = None) -> AsyncClient:
    """Create a new ClickHouse async client and verify schema version."""
    active_settings = settings or Settings()
    client = await get_async_client(
        host=active_settings.clickhouse_host,
        port=active_settings.clickhouse_port,
        database=active_settings.clickhouse_database,
        username=active_settings.clickhouse_user,
        password=active_settings.clickhouse_password,
        secure=active_settings.clickhouse_secure,
        connect_timeout=active_settings.clickhouse_connect_timeout,
        send_receive_timeout=active_settings.clickhouse_send_receive_timeout,
    )
    try:
        await _ensure_schema(client, active_settings.clickhouse_database)
    except ClickHouseDatabaseError:
        await client.close()
        raise
    return client
