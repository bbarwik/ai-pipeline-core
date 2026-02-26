"""Unit tests for ClickHouseClient."""

from unittest.mock import MagicMock, patch

import pytest

clickhouse_connect = pytest.importorskip("clickhouse_connect")

from ai_pipeline_core.observability._tracking._client import ClickHouseClient


class TestClickHouseClientInit:
    def test_stores_params(self):
        client = ClickHouseClient(host="myhost", port=9000, database="mydb", username="user", password="pass", secure=False)
        assert client._params["host"] == "myhost"
        assert client._params["port"] == 9000
        assert client._client is None
        assert client._tables_initialized is False


class TestConnect:
    @patch("ai_pipeline_core.observability._tracking._client.clickhouse_connect")
    def test_connect_creates_client(self, mock_cc):
        mock_cc.get_client.return_value = MagicMock()
        client = ClickHouseClient(host="h", port=1)
        client.connect()
        mock_cc.get_client.assert_called_once()
        assert client._client is not None


class TestEnsureTables:
    def test_raises_without_connect(self):
        client = ClickHouseClient(host="h")
        with pytest.raises(RuntimeError, match="Not connected"):
            client.ensure_tables()

    def test_creates_tables_once(self):
        client = ClickHouseClient(host="h")
        mock_inner = MagicMock()
        client._client = mock_inner
        client.ensure_tables()
        assert client._tables_initialized is True
        call_count = mock_inner.command.call_count
        # Second call should be a no-op
        client.ensure_tables()
        assert mock_inner.command.call_count == call_count


class TestInsertRows:
    def test_inserts_columnar_data(self):
        from pydantic import BaseModel

        class TestRow(BaseModel):
            col_a: str
            col_b: int

        client = ClickHouseClient(host="h")
        mock_inner = MagicMock()
        client._client = mock_inner

        rows = [TestRow(col_a="x", col_b=1), TestRow(col_a="y", col_b=2)]
        client._insert_rows("test_table", rows)
        mock_inner.insert.assert_called_once()
        call_kwargs = mock_inner.insert.call_args
        assert call_kwargs[0][0] == "test_table"
        assert call_kwargs[1]["column_oriented"] is True

    def test_empty_rows_noop(self):
        client = ClickHouseClient(host="h")
        mock_inner = MagicMock()
        client._client = mock_inner
        client._insert_rows("test_table", [])
        mock_inner.insert.assert_not_called()

    def test_no_client_noop(self):
        client = ClickHouseClient(host="h")
        from pydantic import BaseModel

        class Row(BaseModel):
            val: int

        client._insert_rows("t", [Row(val=1)])  # should not raise

    def test_mixed_types_raises(self):
        from pydantic import BaseModel

        class TypeA(BaseModel):
            val: int

        class TypeB(BaseModel):
            val: int

        client = ClickHouseClient(host="h")
        client._client = MagicMock()
        with pytest.raises(ValueError, match="Mixed row types"):
            client._insert_rows("t", [TypeA(val=1), TypeB(val=2)])


class TestCreateTablesSql:
    def test_create_tables_sql_includes_trace_span_content(self):
        from ai_pipeline_core.observability._tracking._client import _CREATE_TABLES_SQL

        combined = " ".join(_CREATE_TABLES_SQL)
        assert "trace_span_content" in combined

    def test_trace_span_content_ddl_has_correct_columns(self):
        from ai_pipeline_core.observability._tracking._client import _CREATE_TABLES_SQL

        combined = " ".join(_CREATE_TABLES_SQL)
        expected_cols = (
            "span_id",
            "trace_id",
            "execution_id",
            "span_order",
            "input_json",
            "output_json",
            "replay_payload",
            "attributes_json",
            "events_json",
            "stored_at",
        )
        for col in expected_cols:
            assert col in combined, f"Column {col} not found in DDL"


class TestInsertTraceContent:
    def test_insert_trace_content(self):
        client = ClickHouseClient(host="h")
        client._client = MagicMock()
        with patch.object(client, "_insert_rows") as mock:
            client.insert_trace_content([MagicMock()])
            mock.assert_called_once()
            assert mock.call_args[0][0] == "trace_span_content"


class TestConvenienceMethods:
    def test_insert_runs(self):
        client = ClickHouseClient(host="h")
        client._client = MagicMock()
        with patch.object(client, "_insert_rows") as mock:
            client.insert_runs([MagicMock()])
            mock.assert_called_once()

    def test_insert_spans(self):
        client = ClickHouseClient(host="h")
        client._client = MagicMock()
        with patch.object(client, "_insert_rows") as mock:
            client.insert_spans([MagicMock()])
            mock.assert_called_once()

    def test_insert_document_events(self):
        client = ClickHouseClient(host="h")
        client._client = MagicMock()
        with patch.object(client, "_insert_rows") as mock:
            client.insert_document_events([MagicMock()])
            mock.assert_called_once()

    def test_insert_span_events(self):
        client = ClickHouseClient(host="h")
        client._client = MagicMock()
        with patch.object(client, "_insert_rows") as mock:
            client.insert_span_events([MagicMock()])
            mock.assert_called_once()
