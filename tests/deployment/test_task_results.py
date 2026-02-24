"""Tests for ClickHouseTaskResultStore and MemoryTaskResultStore."""

# pyright: reportPrivateUsage=false

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from ai_pipeline_core.deployment._task_results import (
    ClickHouseTaskResultStore,
    MemoryTaskResultStore,
    TABLE_TASK_RESULTS,
)
from ai_pipeline_core.deployment._types import TaskResultStore


class TestMemoryTaskResultStore:
    """Test in-memory task result store."""

    def test_satisfies_protocol(self):
        """MemoryTaskResultStore must satisfy the TaskResultStore protocol."""
        assert isinstance(MemoryTaskResultStore(), TaskResultStore)

    async def test_write_and_read_roundtrip(self):
        """Written result can be read back."""
        store = MemoryTaskResultStore()
        result_json = json.dumps({"success": True})
        chain_json = json.dumps({"version": 1})

        await store.write_result("run-1", result_json, chain_json)
        record = await store.read_result("run-1")

        assert record is not None
        assert record.run_id == "run-1"
        assert record.result == result_json
        assert record.chain_context == chain_json
        assert isinstance(record.stored_at, datetime)

    async def test_read_nonexistent_returns_none(self):
        """Reading a nonexistent run_id returns None."""
        store = MemoryTaskResultStore()
        assert await store.read_result("nonexistent") is None

    async def test_overwrite_replaces_record(self):
        """Writing the same run_id again replaces the previous record."""
        store = MemoryTaskResultStore()
        await store.write_result("run-1", '{"v": 1}', '{"ctx": 1}')
        await store.write_result("run-1", '{"v": 2}', '{"ctx": 2}')

        record = await store.read_result("run-1")
        assert record is not None
        assert record.result == '{"v": 2}'

    async def test_multiple_run_ids(self):
        """Different run_ids are stored independently."""
        store = MemoryTaskResultStore()
        await store.write_result("run-1", '{"a": 1}', '{"c": 1}')
        await store.write_result("run-2", '{"a": 2}', '{"c": 2}')

        r1 = await store.read_result("run-1")
        r2 = await store.read_result("run-2")
        assert r1 is not None and r1.result == '{"a": 1}'
        assert r2 is not None and r2.result == '{"a": 2}'

    async def test_record_is_frozen(self):
        """TaskResultRecord is immutable."""
        store = MemoryTaskResultStore()
        await store.write_result("run-1", '{"a": 1}', '{"c": 1}')
        record = await store.read_result("run-1")
        assert record is not None
        with pytest.raises(AttributeError):
            record.run_id = "modified"  # type: ignore[misc]


class TestClickHouseTaskResultStore:
    """Test ClickHouseTaskResultStore with mocked ClickHouse client."""

    def test_satisfies_protocol(self):
        """ClickHouseTaskResultStore must satisfy the TaskResultStore protocol."""
        store = ClickHouseTaskResultStore(host="localhost")
        assert isinstance(store, TaskResultStore)
        store.shutdown()

    async def test_write_result(self):
        """write_result inserts a row into ClickHouse."""
        store = ClickHouseTaskResultStore(host="localhost")
        mock_client = MagicMock()
        store._client = mock_client
        store._tables_initialized = True

        await store.write_result("run-1", '{"success": true}', '{"version": 1}')

        mock_client.insert.assert_called_once()
        call_args = mock_client.insert.call_args
        assert call_args[0][0] == TABLE_TASK_RESULTS
        row = call_args[0][1][0]
        assert row[0] == "run-1"
        assert row[1] == '{"success": true}'
        assert row[2] == '{"version": 1}'
        store.shutdown()

    async def test_read_result_found(self):
        """read_result returns TaskResultRecord when row exists."""
        store = ClickHouseTaskResultStore(host="localhost")
        mock_client = MagicMock()
        store._client = mock_client
        store._tables_initialized = True

        now = datetime.now(UTC)
        mock_result = MagicMock()
        mock_result.result_rows = [("run-1", '{"success": true}', '{"version": 1}', now)]
        mock_client.query.return_value = mock_result

        record = await store.read_result("run-1")
        assert record is not None
        assert record.run_id == "run-1"
        assert record.result == '{"success": true}'
        assert record.chain_context == '{"version": 1}'
        assert record.stored_at == now
        store.shutdown()

    async def test_read_result_not_found(self):
        """read_result returns None when no rows match."""
        store = ClickHouseTaskResultStore(host="localhost")
        mock_client = MagicMock()
        store._client = mock_client
        store._tables_initialized = True

        mock_result = MagicMock()
        mock_result.result_rows = []
        mock_client.query.return_value = mock_result

        record = await store.read_result("nonexistent")
        assert record is None
        store.shutdown()

    async def test_ensure_tables_creates_ddl(self):
        """First operation triggers table creation DDL."""
        store = ClickHouseTaskResultStore(host="localhost")
        mock_client = MagicMock()
        store._client = mock_client
        store._tables_initialized = False

        mock_result = MagicMock()
        mock_result.result_rows = []
        mock_client.query.return_value = mock_result

        await store.read_result("run-1")

        # DDL command should have been called
        mock_client.command.assert_called_once()
        ddl_sql = mock_client.command.call_args[0][0]
        assert TABLE_TASK_RESULTS in ddl_sql
        assert "ReplacingMergeTree" in ddl_sql
        store.shutdown()

    def test_lazy_connection(self):
        """Client is not connected at construction time."""
        store = ClickHouseTaskResultStore(host="localhost")
        assert store._client is None
        store.shutdown()

    def test_shutdown_executor(self):
        """shutdown() shuts down the thread pool executor."""
        store = ClickHouseTaskResultStore(host="localhost")
        store.shutdown()
        assert store._executor._shutdown
