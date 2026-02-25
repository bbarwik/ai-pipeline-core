"""Unit tests for ClickHouseWriter."""

from unittest.mock import MagicMock, patch

import pytest

clickhouse_connect = pytest.importorskip("clickhouse_connect")

from ai_pipeline_core.observability._tracking._writer import ClickHouseWriter, InsertBatch


class TestInsertBatch:
    def test_dataclass_fields(self):
        batch = InsertBatch(table="test_table", rows=[MagicMock()])
        assert batch.table == "test_table"
        assert len(batch.rows) == 1

    def test_default_empty_rows(self):
        batch = InsertBatch(table="t")
        assert batch.rows == []


class TestWriterLifecycle:
    def test_start_sets_ready(self):
        mock_client = MagicMock()
        writer = ClickHouseWriter(mock_client)
        with patch.object(writer, "_thread_main"):
            writer.start()
        assert writer._thread is not None

    def test_start_idempotent(self):
        mock_client = MagicMock()
        writer = ClickHouseWriter(mock_client)
        writer._thread = MagicMock()
        writer.start()  # should not create a second thread

    def test_inactive_when_disabled(self):
        mock_client = MagicMock()
        writer = ClickHouseWriter(mock_client)
        writer._disabled = True
        assert writer._inactive is True

    def test_inactive_when_shutdown(self):
        mock_client = MagicMock()
        writer = ClickHouseWriter(mock_client)
        writer._shutdown = True
        assert writer._inactive is True

    def test_inactive_when_no_loop(self):
        mock_client = MagicMock()
        writer = ClickHouseWriter(mock_client)
        assert writer._inactive is True  # _loop is None initially


class TestWriterWrite:
    def test_write_enqueues_when_active(self):
        mock_client = MagicMock()
        writer = ClickHouseWriter(mock_client)
        writer._loop = MagicMock()
        writer._queue = MagicMock()
        writer._shutdown = False
        writer._disabled = False
        writer.write("table", [MagicMock()])
        writer._loop.call_soon_threadsafe.assert_called_once()

    def test_write_noop_when_inactive(self):
        mock_client = MagicMock()
        writer = ClickHouseWriter(mock_client)
        writer._disabled = True
        writer.write("table", [MagicMock()])
        # no error, just a no-op


class TestWriterFlush:
    def test_flush_noop_when_inactive(self):
        mock_client = MagicMock()
        writer = ClickHouseWriter(mock_client)
        writer._disabled = True
        writer.flush()  # should not raise

    def test_flush_runtime_error_handled(self):
        mock_client = MagicMock()
        writer = ClickHouseWriter(mock_client)
        writer._shutdown = False
        writer._disabled = False
        mock_loop = MagicMock()
        mock_loop.call_soon_threadsafe.side_effect = RuntimeError("closed")
        writer._loop = mock_loop
        writer._queue = MagicMock()
        writer.flush()  # should not raise


class TestWriterShutdown:
    def test_shutdown_idempotent(self):
        mock_client = MagicMock()
        writer = ClickHouseWriter(mock_client)
        writer._shutdown = True
        writer.shutdown()  # should return early

    def test_shutdown_sends_sentinel(self):
        mock_client = MagicMock()
        writer = ClickHouseWriter(mock_client)
        writer._loop = MagicMock()
        writer._queue = MagicMock()
        writer._thread = MagicMock()
        writer.shutdown()
        assert writer._shutdown is True
        writer._loop.call_soon_threadsafe.assert_called_once()
        writer._thread.join.assert_called_once()


class TestFlushBatches:
    def test_flush_inserts_rows(self):
        mock_client = MagicMock()
        writer = ClickHouseWriter(mock_client)
        row = MagicMock()
        pending = {"table1": [row]}
        writer._flush_batches(pending)
        mock_client._insert_rows.assert_called_once_with("table1", [row])
        assert "table1" not in pending

    def test_flush_handles_insert_error(self):
        mock_client = MagicMock()
        mock_client._insert_rows.side_effect = RuntimeError("insert failed")
        writer = ClickHouseWriter(mock_client)
        row = MagicMock()
        pending = {"table1": [row]}
        writer._flush_batches(pending)
        # Rows should remain in pending on failure
        assert "table1" in pending


class TestConnectWithRetry:
    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        mock_client = MagicMock()
        writer = ClickHouseWriter(mock_client)
        result = await writer._connect_with_retry(max_retries=1, base_delay=0)
        assert result is True
        mock_client.connect.assert_called_once()
        mock_client.ensure_tables.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        mock_client = MagicMock()
        mock_client.connect.side_effect = RuntimeError("fail")
        writer = ClickHouseWriter(mock_client)
        result = await writer._connect_with_retry(max_retries=2, base_delay=0)
        assert result is False
        assert writer._disabled is True
