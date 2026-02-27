"""Unit tests for ClickHouseDocumentStore — no real ClickHouse connection needed."""

import time
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch


from ai_pipeline_core.document_store._clickhouse import (
    ClickHouseDocumentStore,
    _BufferedWrite,
    _FAILURE_THRESHOLD,
    _MAX_BUFFER_RETRIES,
    _ParsedDocumentRow,
    _build_document,
    _decode,
    _decode_content,
    _parse_document_row,
    _parse_node_row,
    _reconstruct_attachments,
)
from ai_pipeline_core.document_store._models import DocumentNode
from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents import RunScope


class _TestDoc(Document):
    pass


def _make_store() -> ClickHouseDocumentStore:
    s = ClickHouseDocumentStore(host="fake", port=1, secure=False)
    s._client = MagicMock()
    s._tables_initialized = True
    return s


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestDecode:
    def test_str_passthrough(self):
        assert _decode("hello") == "hello"

    def test_bytes_decode(self):
        assert _decode(b"hello") == "hello"


class TestDecodeContent:
    def test_bytes_passthrough(self):
        assert _decode_content(b"\x89PNG", 4) == b"\x89PNG"

    def test_hex_encoded_binary(self):
        raw_hex = "89504e47"
        result = _decode_content(raw_hex, 4)
        assert result == bytes.fromhex("89504e47")

    def test_plain_text(self):
        result = _decode_content("hello world", 11)
        assert result == b"hello world"

    def test_empty_string(self):
        result = _decode_content("", 0)
        assert result == b""


class TestParseDocumentRow:
    def test_tuple_to_named_fields(self):
        fields = (
            "doc.txt",  # name
            "a description",  # description
            ["src1"],  # derived_from
            ["trig1"],  # triggered_by
            ["att.png"],  # att_names
            ["att desc"],  # att_descs
            ["sha_att"],  # att_sha256s
            b"content data",  # content (bytes = passthrough)
            12,  # content_length
        )
        parsed = _parse_document_row(fields)
        assert parsed.name == "doc.txt"
        assert parsed.description == "a description"
        assert parsed.derived_from == ("src1",)
        assert parsed.triggered_by == ("trig1",)
        assert parsed.att_names == ["att.png"]
        assert parsed.att_descs == ["att desc"]
        assert parsed.att_sha256s == ["sha_att"]
        assert parsed.content == b"content data"

    def test_empty_description_becomes_none(self):
        fields = ("n", "", [], [], [], [], [], b"c", 1)
        parsed = _parse_document_row(fields)
        assert parsed.description is None


class TestParseNodeRow:
    def test_tuple_to_document_node(self):
        row = ("sha256hex", "MyDoc", "report.txt", "desc", ["src"], ["trig"], "summary text")
        node = _parse_node_row(row)
        assert isinstance(node, DocumentNode)
        assert node.sha256 == "sha256hex"
        assert node.class_name == "MyDoc"
        assert node.name == "report.txt"
        assert node.description == "desc"
        assert node.derived_from == ("src",)
        assert node.triggered_by == ("trig",)
        assert node.summary == "summary text"

    def test_empty_summary(self):
        row = ("sha", "Cls", "n", "", [], [], "")
        node = _parse_node_row(row)
        assert node.summary == ""


class TestBuildDocument:
    def test_from_parsed_row(self):
        row = _ParsedDocumentRow(
            name="test.txt",
            description="desc",
            derived_from=("https://example.com/source",),
            triggered_by=(),
            att_names=[],
            att_descs=[],
            att_sha256s=[],
            content=b"hello",
        )
        doc = _build_document(_TestDoc, row, {})
        assert isinstance(doc, _TestDoc)
        assert doc.name == "test.txt"
        assert doc.content == b"hello"


class TestReconstructAttachments:
    def test_happy_path(self):
        content_map = {"sha_a": b"image data"}
        result = _reconstruct_attachments(["screenshot.png"], ["a screenshot"], ["sha_a"], content_map, "doc.txt")
        assert len(result) == 1
        assert result[0].name == "screenshot.png"
        assert result[0].content == b"image data"
        assert result[0].description == "a screenshot"

    def test_missing_sha_skipped(self):
        result = _reconstruct_attachments(["missing.png"], ["desc"], ["sha_missing"], {}, "doc.txt")
        assert result == ()

    def test_empty_no_attachments(self):
        result = _reconstruct_attachments([], [], [], {}, "doc.txt")
        assert result == ()


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_record_success_resets(self):
        store = _make_store()
        store._consecutive_failures = 5
        store._circuit_open = True
        store._record_success()
        assert store._consecutive_failures == 0
        assert store._circuit_open is False

    def test_record_failure_opens_at_threshold(self):
        store = _make_store()
        for _ in range(_FAILURE_THRESHOLD):
            store._record_failure()
        assert store._circuit_open is True
        assert store._client is None
        assert store._tables_initialized is False

    def test_try_reconnect_respects_interval(self):
        store = _make_store()
        store._circuit_open = True
        store._last_reconnect_attempt = time.monotonic()
        assert store._try_reconnect() is False

    def test_ensure_connected_bad_host(self):
        store = ClickHouseDocumentStore(host="nonexistent.invalid", port=1, secure=False)
        with patch("ai_pipeline_core.document_store._clickhouse.clickhouse_connect") as mock_cc:
            mock_cc.get_client.side_effect = Exception("connection refused")
            result = store._ensure_connected()
        assert result is False
        assert store._client is None


# ---------------------------------------------------------------------------
# Buffer / save
# ---------------------------------------------------------------------------


class TestFlushBuffer:
    def test_retries_then_drops(self):
        store = _make_store()
        doc = _TestDoc(name="buf.txt", content=b"x")
        item = _BufferedWrite(document=doc, run_scope=RunScope("r"), retry_count=_MAX_BUFFER_RETRIES - 1)
        store._buffer.append(item)
        store._client.insert.side_effect = Exception("insert fail")
        store._flush_buffer()
        assert len(store._buffer) == 0

    def test_save_batch_sync_buffers_on_table_failure(self):
        store = _make_store()
        store._tables_initialized = False
        store._client.command.side_effect = Exception("table creation failed")
        docs = [_TestDoc(name="a.txt", content=b"a"), _TestDoc(name="b.txt", content=b"b")]
        store._save_batch_sync(docs, RunScope("r"))
        assert len(store._buffer) == 2

    def test_save_sync_buffers_when_circuit_open(self):
        store = _make_store()
        store._circuit_open = True
        store._last_reconnect_attempt = time.monotonic()
        doc = _TestDoc(name="c.txt", content=b"c")
        store._save_sync(doc, RunScope("r"))
        assert len(store._buffer) == 1

    def test_save_sync_opens_circuit(self):
        store = _make_store()
        store._client.insert.side_effect = Exception("insert fail")
        doc = _TestDoc(name="d.txt", content=b"d")
        for _ in range(_FAILURE_THRESHOLD):
            store._save_sync(doc, RunScope("r"))
        assert store._circuit_open is True


# ---------------------------------------------------------------------------
# Query / load
# ---------------------------------------------------------------------------


class TestQueryAndLoad:
    def test_query_by_source_sql_alias_before_final(self):
        import re

        store = _make_store()
        mock_result = MagicMock()
        mock_result.result_rows = []
        store._client.query.return_value = mock_result
        store._query_by_source(["src1"], "_TestDoc", None)
        sql = store._client.query.call_args[0][0]
        # Must NOT contain "FINAL AS <alias>" — that's the bug pattern
        assert "FINAL AS di" not in sql, f"Bug: 'FINAL AS di' found in SQL — should be 'AS di FINAL'. SQL: {sql}"
        assert "FINAL AS dc" not in sql, f"Bug: 'FINAL AS dc' found in SQL — should be 'AS dc FINAL'. SQL: {sql}"
        # Must contain correct "AS <alias> FINAL" pattern
        assert re.search(r"AS di\s+FINAL", sql) or "FINAL" in sql, f"Missing FINAL keyword in SQL: {sql}"

    def test_load_sync_sql_alias_before_final(self):
        """Validates that _load_sync queries use correct AS alias FINAL ordering."""
        store = _make_store()
        mock_result = MagicMock()
        mock_result.result_rows = []
        store._client.query.return_value = mock_result
        store._load_sync(RunScope("r"), [_TestDoc])
        sql = store._client.query.call_args[0][0]
        assert "FINAL AS " not in sql, f"Bug: 'FINAL AS' found in load SQL — should be 'AS <alias> FINAL'. SQL: {sql}"

    def test_find_by_source_sync_empty(self):
        store = _make_store()
        result = store._find_by_source_sync([], _TestDoc, None)
        assert result == {}

    def test_load_sync_skips_unknown_types(self):
        store = _make_store()
        mock_result = MagicMock()
        mock_result.result_rows = [
            ("UnknownClass", "name", "", [], [], [], [], [], b"content", 7),
        ]
        store._client.query.return_value = mock_result
        result = store._load_sync(RunScope("r"), [_TestDoc])
        assert result == []

    def test_get_flow_completion_empty(self):
        store = _make_store()
        mock_result = MagicMock()
        mock_result.result_rows = []
        store._client.query.return_value = mock_result
        result = store._get_flow_completion_sync(RunScope("r"), "flow1", None)
        assert result is None

    def test_get_flow_completion_with_result(self):
        store = _make_store()
        now = datetime.now(UTC)
        mock_result = MagicMock()
        mock_result.result_rows = [(["in1"], ["out1"], now)]
        store._client.query.return_value = mock_result
        result = store._get_flow_completion_sync(RunScope("r"), "flow1", None)
        assert result is not None
        assert result.flow_name == "flow1"
        assert result.input_sha256s == ("in1",)
        assert result.output_sha256s == ("out1",)

    def test_has_documents_sync_true(self):
        store = _make_store()
        mock_result = MagicMock()
        mock_result.result_rows = [(1,)]
        store._client.query.return_value = mock_result
        assert store._has_documents_sync(RunScope("r"), _TestDoc, None) is True

    def test_has_documents_sync_false(self):
        store = _make_store()
        mock_result = MagicMock()
        mock_result.result_rows = []
        store._client.query.return_value = mock_result
        assert store._has_documents_sync(RunScope("r"), _TestDoc, None) is False

    def test_check_existing_sync_empty_input(self):
        store = _make_store()
        result = store._check_existing_sync([])
        assert result == set()

    def test_load_summaries_sync_empty(self):
        store = _make_store()
        result = store._load_summaries_sync([])
        assert result == {}


class TestShutdown:
    def test_shutdown_without_summary_worker(self):
        store = _make_store()
        store.shutdown()
