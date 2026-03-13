"""Tests for ClickHouseDatabase DDL, schema versioning, and basic availability checks."""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from ai_pipeline_core.database.clickhouse._connection import SchemaVersionError, _ensure_schema, reset_schema_check
from ai_pipeline_core.database.clickhouse._ddl import (
    BLOBS_DDL,
    DDL_STATEMENTS,
    DOCUMENTS_DDL,
    LOGS_DDL,
    SCHEMA_META_DDL,
    SCHEMA_META_TABLE,
    SCHEMA_VERSION,
    SPANS_DDL,
)

HTTP_PORT = 8123


@pytest.fixture(scope="module")
def clickhouse_container(require_docker):
    """Start a ClickHouse container once per module, exposing the HTTP port."""
    from testcontainers.clickhouse import ClickHouseContainer

    container = ClickHouseContainer(port=HTTP_PORT)
    container.with_exposed_ports(HTTP_PORT)
    with container:
        yield container


@pytest.fixture(scope="module")
def clickhouse_settings(clickhouse_container):
    """Build Settings pointing at the testcontainer ClickHouse instance."""
    from ai_pipeline_core.settings import Settings

    return Settings(
        clickhouse_host=clickhouse_container.get_container_host_ip(),
        clickhouse_port=int(clickhouse_container.get_exposed_port(HTTP_PORT)),
        clickhouse_database=clickhouse_container.dbname,
        clickhouse_user=clickhouse_container.username,
        clickhouse_password=clickhouse_container.password,
        clickhouse_secure=False,
    )


def _extract_table_body(ddl: str) -> str:
    start = ddl.find("(")
    if start == -1:
        raise AssertionError(f"DDL is missing a column list: {ddl}")
    depth = 0
    for index in range(start, len(ddl)):
        char = ddl[index]
        if char == "(":
            depth += 1
            continue
        if char != ")":
            continue
        depth -= 1
        if depth == 0:
            return ddl[start + 1 : index]
    raise AssertionError(f"DDL has unbalanced parentheses: {ddl}")


def _extract_column_lines(ddl: str) -> list[str]:
    lines: list[str] = []
    for raw_line in _extract_table_body(ddl).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("INDEX "):
            continue
        lines.append(line.rstrip(","))
    return lines


def _extract_index_lines(ddl: str) -> list[str]:
    return [line.strip().rstrip(",") for line in _extract_table_body(ddl).splitlines() if line.strip().startswith("INDEX ")]


def test_ddl_statement_list_includes_all_tables() -> None:
    assert DDL_STATEMENTS == [SCHEMA_META_DDL, SPANS_DDL, DOCUMENTS_DDL, BLOBS_DDL, LOGS_DDL]


def test_spans_ddl_matches_expected_shape() -> None:
    assert len(_extract_column_lines(SPANS_DDL)) == 30
    assert "ENGINE = ReplacingMergeTree(version)" in SPANS_DDL
    assert "ORDER BY (root_deployment_id, deployment_id, span_id)" in SPANS_DDL
    assert len(_extract_index_lines(SPANS_DDL)) == 11
    assert "detail_json" not in SPANS_DDL


def test_documents_ddl_matches_expected_shape() -> None:
    assert "description String DEFAULT ''" in DOCUMENTS_DDL
    assert "mime_type LowCardinality(String) DEFAULT ''" in DOCUMENTS_DDL
    assert "attachments Nested(" in DOCUMENTS_DDL
    assert "ENGINE = ReplacingMergeTree(created_at)" in DOCUMENTS_DDL
    assert "detail_json" not in DOCUMENTS_DDL
    assert "version" not in DOCUMENTS_DDL
    assert "CODEC(ZSTD(3))" not in DOCUMENTS_DDL.split("summary String DEFAULT ''", 1)[1].split("\n", 1)[0]
    assert len(_extract_index_lines(DOCUMENTS_DDL)) == 4


def test_blobs_and_logs_ddl_match_expected_shape() -> None:
    assert len(_extract_column_lines(BLOBS_DDL)) == 3
    assert "ORDER BY (content_sha256)" in BLOBS_DDL
    assert len(_extract_column_lines(LOGS_DDL)) == 11
    assert "ORDER BY (deployment_id, span_id, timestamp, sequence_no)" in LOGS_DDL


def test_schema_meta_ddl_shape() -> None:
    assert f"CREATE TABLE IF NOT EXISTS {SCHEMA_META_TABLE}" in SCHEMA_META_DDL
    assert "version UInt32" in SCHEMA_META_DDL
    assert "applied_at DateTime64(3, 'UTC')" in SCHEMA_META_DDL
    assert "framework_version String" in SCHEMA_META_DDL
    assert "ENGINE = MergeTree()" in SCHEMA_META_DDL
    assert "ORDER BY version" in SCHEMA_META_DDL


def test_schema_version_is_positive_integer() -> None:
    assert isinstance(SCHEMA_VERSION, int)
    assert SCHEMA_VERSION >= 1


def test_schema_meta_ddl_is_first_in_ddl_statements() -> None:
    assert DDL_STATEMENTS[0] is SCHEMA_META_DDL


# ---------------------------------------------------------------------------
# Unit tests for _ensure_schema (mocked ClickHouse client)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_schema_flag():
    """Reset the process-level schema check flag before and after each test."""
    reset_schema_check()
    yield
    reset_schema_check()


def _mock_client(*, table_exists: bool, db_version: int | None = None) -> AsyncMock:
    """Build a mock AsyncClient for _ensure_schema tests."""
    client = AsyncMock()

    # _table_exists queries system.tables
    query_result = MagicMock()
    if table_exists:
        if db_version is not None:
            query_result.result_rows = [(db_version,)]
        else:
            query_result.result_rows = [(0,)]
        # First call: system.tables check (returns rows = table exists)
        # Second call: SELECT max(version)
        system_result = MagicMock()
        system_result.result_rows = [(1,)]
        client.query = AsyncMock(side_effect=[system_result, query_result])
    else:
        # system.tables check returns no rows = table doesn't exist
        system_result = MagicMock()
        system_result.result_rows = []
        client.query = AsyncMock(return_value=system_result)
    return client


@pytest.mark.asyncio
async def test_ensure_schema_creates_tables_on_fresh_db() -> None:
    client = _mock_client(table_exists=False)

    await _ensure_schema(client, "default")

    # DDL_STATEMENTS has 5 entries + 1 INSERT for the version stamp
    assert client.command.call_count == len(DDL_STATEMENTS) + 1
    last_call_sql = client.command.call_args_list[-1].args[0]
    assert SCHEMA_META_TABLE in last_call_sql
    assert "INSERT" in last_call_sql


@pytest.mark.asyncio
async def test_ensure_schema_passes_on_matching_version() -> None:
    client = _mock_client(table_exists=True, db_version=SCHEMA_VERSION)

    await _ensure_schema(client, "default")

    # No DDL should have been run — only 2 queries (system.tables + max(version))
    client.command.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_schema_raises_on_outdated_db() -> None:
    client = _mock_client(table_exists=True, db_version=SCHEMA_VERSION - 1)

    with pytest.raises(SchemaVersionError, match="older than the framework expects"):
        await _ensure_schema(client, "default")


@pytest.mark.asyncio
async def test_ensure_schema_raises_on_newer_db() -> None:
    client = _mock_client(table_exists=True, db_version=SCHEMA_VERSION + 1)

    with pytest.raises(SchemaVersionError, match="newer than the framework supports"):
        await _ensure_schema(client, "default")


@pytest.mark.asyncio
async def test_ensure_schema_runs_only_once_per_process() -> None:
    client = _mock_client(table_exists=True, db_version=SCHEMA_VERSION)

    await _ensure_schema(client, "default")
    initial_query_count = client.query.call_count

    # Second call should be a no-op (process-level flag is set)
    await _ensure_schema(client, "default")
    assert client.query.call_count == initial_query_count


@pytest.mark.asyncio
async def test_ensure_schema_raises_on_zero_version_in_existing_table() -> None:
    client = _mock_client(table_exists=True, db_version=0)

    with pytest.raises(SchemaVersionError, match="older than the framework expects"):
        await _ensure_schema(client, "default")


@pytest.mark.asyncio
async def test_reset_schema_check_allows_recheck() -> None:
    client = _mock_client(table_exists=True, db_version=SCHEMA_VERSION)

    await _ensure_schema(client, "default")
    reset_schema_check()

    # After reset, a new client should trigger a fresh check
    client2 = _mock_client(table_exists=True, db_version=SCHEMA_VERSION)
    await _ensure_schema(client2, "default")
    assert client2.query.call_count == 2  # system.tables + max(version)


# ---------------------------------------------------------------------------
# Integration tests (require running ClickHouse)
# ---------------------------------------------------------------------------


@pytest.mark.clickhouse
def test_clickhouse_database_can_connect(clickhouse_settings) -> None:
    from ai_pipeline_core.database.clickhouse._backend import ClickHouseDatabase

    database = ClickHouseDatabase(settings=clickhouse_settings)
    assert database is not None


@pytest.mark.clickhouse
def test_clickhouse_schema_auto_init_creates_tables_and_stamps_version(clickhouse_settings) -> None:
    """On a fresh database, get_async_clickhouse_client auto-creates all tables and stamps the schema version."""
    from clickhouse_connect import get_async_client

    from ai_pipeline_core.database.clickhouse._connection import get_async_clickhouse_client

    test_database = f"schema_test_{uuid4().hex[:12]}"

    async def _run() -> None:
        admin_client = await get_async_client(
            host=clickhouse_settings.clickhouse_host,
            port=clickhouse_settings.clickhouse_port,
            database=clickhouse_settings.clickhouse_database,
            username=clickhouse_settings.clickhouse_user,
            password=clickhouse_settings.clickhouse_password,
            secure=clickhouse_settings.clickhouse_secure,
            connect_timeout=clickhouse_settings.clickhouse_connect_timeout,
            send_receive_timeout=clickhouse_settings.clickhouse_send_receive_timeout,
        )
        try:
            await admin_client.command(f"CREATE DATABASE IF NOT EXISTS {test_database}")
        finally:
            await admin_client.close()

        try:
            reset_schema_check()
            from ai_pipeline_core.settings import Settings

            test_settings = Settings(
                clickhouse_host=clickhouse_settings.clickhouse_host,
                clickhouse_port=clickhouse_settings.clickhouse_port,
                clickhouse_database=test_database,
                clickhouse_user=clickhouse_settings.clickhouse_user,
                clickhouse_password=clickhouse_settings.clickhouse_password,
                clickhouse_secure=clickhouse_settings.clickhouse_secure,
            )
            client = await get_async_clickhouse_client(test_settings)
            try:
                result = await client.query(f"SELECT max(version) FROM {SCHEMA_META_TABLE}")
                assert result.result_rows[0][0] == SCHEMA_VERSION

                for table in ("spans", "documents", "blobs", "logs"):
                    exists_result = await client.query(
                        "SELECT 1 FROM system.tables WHERE database = {db:String} AND name = {tbl:String}",
                        parameters={"db": test_database, "tbl": table},
                    )
                    assert len(exists_result.result_rows) > 0, f"Table {table} was not created"
            finally:
                await client.close()
        finally:
            cleanup_client = await get_async_client(
                host=clickhouse_settings.clickhouse_host,
                port=clickhouse_settings.clickhouse_port,
                database=clickhouse_settings.clickhouse_database,
                username=clickhouse_settings.clickhouse_user,
                password=clickhouse_settings.clickhouse_password,
                secure=clickhouse_settings.clickhouse_secure,
                connect_timeout=clickhouse_settings.clickhouse_connect_timeout,
                send_receive_timeout=clickhouse_settings.clickhouse_send_receive_timeout,
            )
            try:
                await cleanup_client.command(f"DROP DATABASE IF EXISTS {test_database} SYNC")
            finally:
                await cleanup_client.close()
            reset_schema_check()

    asyncio.run(_run())


@pytest.mark.clickhouse
def test_clickhouse_schema_version_mismatch_raises(clickhouse_settings) -> None:
    """When schema_meta exists with a higher version than the framework, SchemaVersionError is raised."""
    from clickhouse_connect import get_async_client

    from ai_pipeline_core.database.clickhouse._connection import get_async_clickhouse_client

    test_database = f"schema_mismatch_{uuid4().hex[:12]}"

    async def _run() -> None:
        admin_client = await get_async_client(
            host=clickhouse_settings.clickhouse_host,
            port=clickhouse_settings.clickhouse_port,
            database=clickhouse_settings.clickhouse_database,
            username=clickhouse_settings.clickhouse_user,
            password=clickhouse_settings.clickhouse_password,
            secure=clickhouse_settings.clickhouse_secure,
            connect_timeout=clickhouse_settings.clickhouse_connect_timeout,
            send_receive_timeout=clickhouse_settings.clickhouse_send_receive_timeout,
        )
        try:
            await admin_client.command(f"CREATE DATABASE IF NOT EXISTS {test_database}")
            await admin_client.command(
                f"CREATE TABLE {test_database}.{SCHEMA_META_TABLE} "
                "(version UInt32, applied_at DateTime64(3, 'UTC'), framework_version String) "
                "ENGINE = MergeTree() ORDER BY version"
            )
            future_version = SCHEMA_VERSION + 99
            await admin_client.command(f"INSERT INTO {test_database}.{SCHEMA_META_TABLE} VALUES ({future_version}, now64(3), 'future-version')")
        finally:
            await admin_client.close()

        try:
            reset_schema_check()
            from ai_pipeline_core.settings import Settings

            test_settings = Settings(
                clickhouse_host=clickhouse_settings.clickhouse_host,
                clickhouse_port=clickhouse_settings.clickhouse_port,
                clickhouse_database=test_database,
                clickhouse_user=clickhouse_settings.clickhouse_user,
                clickhouse_password=clickhouse_settings.clickhouse_password,
                clickhouse_secure=clickhouse_settings.clickhouse_secure,
            )
            with pytest.raises(SchemaVersionError, match="newer than the framework supports"):
                await get_async_clickhouse_client(test_settings)
        finally:
            cleanup_client = await get_async_client(
                host=clickhouse_settings.clickhouse_host,
                port=clickhouse_settings.clickhouse_port,
                database=clickhouse_settings.clickhouse_database,
                username=clickhouse_settings.clickhouse_user,
                password=clickhouse_settings.clickhouse_password,
                secure=clickhouse_settings.clickhouse_secure,
                connect_timeout=clickhouse_settings.clickhouse_connect_timeout,
                send_receive_timeout=clickhouse_settings.clickhouse_send_receive_timeout,
            )
            try:
                await cleanup_client.command(f"DROP DATABASE IF EXISTS {test_database} SYNC")
            finally:
                await cleanup_client.close()
            reset_schema_check()

    asyncio.run(_run())
