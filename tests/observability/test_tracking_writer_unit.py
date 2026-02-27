"""Unit tests for ClickHouseWriter."""

from unittest.mock import MagicMock

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
    def test_start_idempotent(self):
        writer = ClickHouseWriter(host="localhost")
        writer._thread = MagicMock()
        writer.start()  # should not create a second thread

    def test_inactive_when_disabled(self):
        writer = ClickHouseWriter(host="localhost")
        writer._disabled = True
        assert writer._inactive is True

    def test_inactive_when_shutdown(self):
        writer = ClickHouseWriter(host="localhost")
        writer._shutdown = True
        assert writer._inactive is True

    def test_inactive_when_no_loop(self):
        writer = ClickHouseWriter(host="localhost")
        assert writer._inactive is True  # _loop is None initially


class TestWriterWrite:
    def test_write_enqueues_when_active(self):
        writer = ClickHouseWriter(host="localhost")
        writer._loop = MagicMock()
        writer._queue = MagicMock()
        writer._shutdown = False
        writer._disabled = False
        writer.write("table", [MagicMock()])
        writer._loop.call_soon_threadsafe.assert_called_once()

    def test_write_noop_when_inactive(self):
        writer = ClickHouseWriter(host="localhost")
        writer._disabled = True
        writer.write("table", [MagicMock()])


class TestWriterFlush:
    def test_flush_noop_when_inactive(self):
        writer = ClickHouseWriter(host="localhost")
        writer._disabled = True
        writer.flush()

    def test_flush_runtime_error_handled(self):
        writer = ClickHouseWriter(host="localhost")
        writer._shutdown = False
        writer._disabled = False
        mock_loop = MagicMock()
        mock_loop.call_soon_threadsafe.side_effect = RuntimeError("closed")
        writer._loop = mock_loop
        writer._queue = MagicMock()
        writer.flush()


class TestWriterShutdown:
    def test_shutdown_idempotent(self):
        writer = ClickHouseWriter(host="localhost")
        writer._shutdown = True
        writer.shutdown()

    def test_shutdown_sends_sentinel(self):
        writer = ClickHouseWriter(host="localhost")
        writer._loop = MagicMock()
        writer._queue = MagicMock()
        writer._thread = MagicMock()
        writer.shutdown()
        assert writer._shutdown is True
        writer._loop.call_soon_threadsafe.assert_called_once()
        writer._thread.join.assert_called_once()


class TestFlushBatches:
    def test_flush_inserts_rows(self):
        writer = ClickHouseWriter(host="localhost")
        writer._client = MagicMock()
        row = MagicMock()
        row_type = type(row)
        row_type.model_fields = {"col": MagicMock()}
        pending = {"table1": [row]}
        writer._flush_batches(pending)
        writer._client.insert.assert_called_once()
        assert "table1" not in pending

    def test_flush_discards_on_insert_error(self):
        writer = ClickHouseWriter(host="localhost")
        writer._client = MagicMock()
        writer._client.insert.side_effect = RuntimeError("insert failed")
        row = MagicMock()
        row_type = type(row)
        row_type.model_fields = {"col": MagicMock()}
        pending = {"table1": [row]}
        writer._flush_batches(pending)
        assert "table1" not in pending


class TestConnectWithRetry:
    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        import unittest.mock

        writer = ClickHouseWriter(host="localhost")
        with unittest.mock.patch("ai_pipeline_core.observability._tracking._writer.clickhouse_connect") as mock_cc:
            mock_cc.get_client.side_effect = RuntimeError("fail")
            result = await writer._connect_with_retry(max_retries=2, base_delay=0)
        assert result is False
