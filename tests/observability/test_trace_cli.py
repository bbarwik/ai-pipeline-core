"""Tests for the ai-trace CLI tool."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

clickhouse_connect = pytest.importorskip("clickhouse_connect")

from ai_pipeline_core.observability.cli import (
    _parse_execution_id,
    _resolve_connection,
    _resolve_identifier,
    main,
)

SAMPLE_UUID = "550e8400-e29b-41d4-a716-446655440000"
SAMPLE_UUID_OBJ = UUID(SAMPLE_UUID)

# Column names matching _DOWNLOAD_SPANS_QUERY / _DOWNLOAD_SPANS_BY_RUN_PREFIX_QUERY
_SPAN_COLUMNS = [
    "span_id",
    "trace_id",
    "parent_span_id",
    "name",
    "span_type",
    "status",
    "start_time",
    "end_time",
    "duration_ms",
    "span_order",
    "cost",
    "tokens_input",
    "tokens_output",
    "tokens_cached",
    "llm_model",
    "error_message",
    "input_json",
    "output_json",
    "replay_payload",
    "attributes_json",
    "events_json",
    "execution_id",
    "run_id",
    "flow_name",
    "run_scope",
    "input_doc_sha256s",
    "output_doc_sha256s",
]

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
_LATER = datetime(2024, 6, 1, 12, 0, 1, tzinfo=UTC)


def _make_span_row(
    *,
    span_id: str = "span1",
    trace_id: str = "trace1",
    name: str = "test_span",
    run_id: str = "run-001",
    execution_id: UUID | None = None,
) -> tuple[Any, ...]:
    """Build a tuple matching _SPAN_COLUMNS for mock query results."""
    return (
        span_id,
        trace_id,
        None,
        name,
        "trace",
        "completed",
        _NOW,
        _LATER,
        1000,
        1,
        0.0,
        0,
        0,
        0,
        None,
        "",
        "",
        "",
        "",
        "{}",
        "[]",
        execution_id or SAMPLE_UUID_OBJ,
        run_id,
        "my_flow",
        "scope/run-001",
        [],
        [],
    )


_SETTINGS_PATCH = "ai_pipeline_core.settings.Settings"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock ClickHouse client."""
    client = MagicMock()
    client.query.return_value = MagicMock(result_rows=[])
    return client


@pytest.fixture
def _patch_create_client(mock_client: MagicMock) -> Any:
    """Patch _create_client to return mock_client."""
    with patch("ai_pipeline_core.observability.cli._create_client", return_value=mock_client):
        yield


# ---------------------------------------------------------------------------
# _parse_execution_id
# ---------------------------------------------------------------------------


class TestParseExecutionId:
    def test_valid_uuid(self):
        result = _parse_execution_id(SAMPLE_UUID)
        assert result == SAMPLE_UUID_OBJ

    def test_invalid_uuid_exits(self):
        with pytest.raises(SystemExit):
            _parse_execution_id("not-a-uuid")


# ---------------------------------------------------------------------------
# _resolve_connection
# ---------------------------------------------------------------------------


class TestResolveConnection:
    def test_host_from_args(self):
        args = MagicMock()
        args.host = "my-host.example.com"
        args.port = 9440
        args.database = "mydb"
        args.user = "admin"
        args.password = "secret"
        args.no_secure = False

        with patch(_SETTINGS_PATCH) as MockSettings:
            MockSettings.return_value = MagicMock(
                clickhouse_host="",
                clickhouse_port=8443,
                clickhouse_database="default",
                clickhouse_user="default",
                clickhouse_password="",
                clickhouse_secure=True,
            )
            result = _resolve_connection(args)

        assert result["host"] == "my-host.example.com"
        assert result["port"] == 9440
        assert result["database"] == "mydb"
        assert result["username"] == "admin"
        assert result["password"] == "secret"
        assert result["secure"] is True

    def test_host_from_settings(self):
        args = MagicMock()
        args.host = None
        args.port = None
        args.database = None
        args.user = None
        args.password = None
        args.no_secure = False

        with patch(_SETTINGS_PATCH) as MockSettings:
            MockSettings.return_value = MagicMock(
                clickhouse_host="settings-host.example.com",
                clickhouse_port=8443,
                clickhouse_database="default",
                clickhouse_user="default",
                clickhouse_password="pw",
                clickhouse_secure=True,
            )
            result = _resolve_connection(args)

        assert result["host"] == "settings-host.example.com"
        assert result["port"] == 8443

    def test_missing_host_exits(self):
        args = MagicMock()
        args.host = None
        args.port = None
        args.database = None
        args.user = None
        args.password = None
        args.no_secure = False

        with (
            patch(_SETTINGS_PATCH) as MockSettings,
            pytest.raises(SystemExit),
        ):
            MockSettings.return_value = MagicMock(clickhouse_host="")
            _resolve_connection(args)


# ---------------------------------------------------------------------------
# _cmd_download
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_create_client")
class TestCmdDownload:
    def test_no_spans_exits(self, mock_client: MagicMock, tmp_path: Path) -> None:
        mock_client.query.side_effect = [
            # _resolve_identifier queries _RUN_METADATA_QUERY for UUID input
            MagicMock(result_rows=[("run-001", "my_flow", "", "completed", _NOW, _LATER, 0, 0, "{}")]),
            # _DOWNLOAD_SPANS_QUERY returns no spans
            MagicMock(result_rows=[], column_names=_SPAN_COLUMNS),
        ]

        with pytest.raises(SystemExit):
            main(["download", SAMPLE_UUID, "-o", str(tmp_path / "out")])

    def test_download_error_returns_1(self, mock_client: MagicMock, tmp_path: Path) -> None:
        mock_client.query.side_effect = [
            # _resolve_identifier succeeds
            MagicMock(result_rows=[("run-001", "my_flow", "", "completed", _NOW, _LATER, 0, 0, "{}")]),
            # Span query fails
            RuntimeError("connection lost"),
        ]

        result = main(["download", SAMPLE_UUID, "-o", str(tmp_path / "out")])
        assert result == 1


# ---------------------------------------------------------------------------
# _cmd_list
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_create_client")
class TestCmdList:
    def test_successful_listing(self, mock_client: MagicMock) -> None:
        exec_id = uuid4()
        now = datetime.now(UTC)
        mock_client.query.return_value = MagicMock(
            result_rows=[
                (exec_id, "run-001", "my_flow", "completed", now, now, 0.05, 5000),
            ]
        )

        result = main(["list"])
        assert result == 0

    def test_empty_results(self, mock_client: MagicMock) -> None:
        mock_client.query.return_value = MagicMock(result_rows=[])

        result = main(["list"])
        assert result == 0

    def test_status_filter(self, mock_client: MagicMock) -> None:
        mock_client.query.return_value = MagicMock(result_rows=[])

        main(["list", "--status", "completed"])

        call_args = mock_client.query.call_args
        assert "status" in call_args[1]["parameters"]
        assert call_args[1]["parameters"]["status"] == "completed"

    def test_flow_filter(self, mock_client: MagicMock) -> None:
        mock_client.query.return_value = MagicMock(result_rows=[])

        main(["list", "--flow", "my_flow"])

        call_args = mock_client.query.call_args
        assert "flow_name" in call_args[1]["parameters"]
        assert call_args[1]["parameters"]["flow_name"] == "my_flow"


# ---------------------------------------------------------------------------
# _cmd_show
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_create_client")
class TestCmdShow:
    def test_successful_show(self, mock_client: MagicMock) -> None:
        now = datetime.now(UTC)
        mock_client.query.side_effect = [
            MagicMock(
                result_rows=[
                    ("run-001", "my_flow", "scope/run-001", "completed", now, now, 0.123, 10000, "{}"),
                ]
            ),
            MagicMock(
                result_rows=[
                    ("llm", 3, 5000, 0.1, 8000, 2000),
                    ("default", 5, 1000, 0.0, 0, 0),
                ]
            ),
        ]

        result = main(["show", SAMPLE_UUID])
        assert result == 0

    def test_no_run_found(self, mock_client: MagicMock) -> None:
        mock_client.query.return_value = MagicMock(result_rows=[])

        result = main(["show", SAMPLE_UUID])
        assert result == 1


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    def test_no_args_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = main([])
        assert result == 1

    def test_unknown_command_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = main([])
        assert result == 1


# ---------------------------------------------------------------------------
# _resolve_identifier
# ---------------------------------------------------------------------------


class TestResolveIdentifier:
    def test_uuid_returns_execution_id_and_run_id(self, mock_client: MagicMock) -> None:
        """UUID input queries pipeline_runs for run_id."""
        mock_client.query.return_value = MagicMock(result_rows=[("run-001", "my_flow", "", "completed", _NOW, _LATER, 0, 0, "{}")])
        execution_id, run_id = _resolve_identifier(SAMPLE_UUID, mock_client)
        assert execution_id == SAMPLE_UUID_OBJ
        assert run_id == "run-001"

    def test_uuid_with_no_run_metadata_returns_empty_run_id(self, mock_client: MagicMock) -> None:
        """UUID input with no pipeline_runs entry returns empty run_id."""
        mock_client.query.return_value = MagicMock(result_rows=[])
        execution_id, run_id = _resolve_identifier(SAMPLE_UUID, mock_client)
        assert execution_id == SAMPLE_UUID_OBJ
        assert run_id == ""

    def test_run_id_resolves_via_pipeline_runs(self, mock_client: MagicMock) -> None:
        """Non-UUID input queries pipeline_runs by run_id."""
        mock_client.query.return_value = MagicMock(result_rows=[(SAMPLE_UUID_OBJ, "my-run-001")])
        execution_id, run_id = _resolve_identifier("my-run-001", mock_client)
        assert execution_id == SAMPLE_UUID_OBJ
        assert run_id == "my-run-001"

    def test_run_id_not_found_exits(self, mock_client: MagicMock) -> None:
        """Non-UUID input with no match exits with error."""
        mock_client.query.return_value = MagicMock(result_rows=[])
        with pytest.raises(SystemExit):
            _resolve_identifier("nonexistent-run", mock_client)

    def test_short_uuid_treated_as_run_id(self, mock_client: MagicMock) -> None:
        """Partial UUID (e.g., '550e8400') is not a valid UUID, treated as run_id."""
        mock_client.query.return_value = MagicMock(result_rows=[(SAMPLE_UUID_OBJ, "550e8400")])
        execution_id, resolved_run_id = _resolve_identifier("550e8400", mock_client)
        assert execution_id == SAMPLE_UUID_OBJ
        assert resolved_run_id == "550e8400"


# ---------------------------------------------------------------------------
# _cmd_download with run_id
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_create_client")
class TestCmdDownloadByRunId:
    def test_download_by_run_id(self, mock_client: MagicMock, tmp_path: Path) -> None:
        """Non-UUID identifier resolves run_id then downloads spans."""
        mock_client.query.side_effect = [
            # _resolve_identifier: _RESOLVE_RUN_ID_QUERY
            MagicMock(result_rows=[(SAMPLE_UUID_OBJ, "my-run-001")]),
            # _DOWNLOAD_SPANS_QUERY
            MagicMock(result_rows=[_make_span_row(run_id="my-run-001")], column_names=_SPAN_COLUMNS),
        ]
        result = main(["download", "my-run-001", "-o", str(tmp_path / "out")])
        assert result == 0
        assert (tmp_path / "out").is_dir()

    def test_download_by_run_id_not_found(self, mock_client: MagicMock, tmp_path: Path) -> None:
        """Unknown run_id returns 1."""
        mock_client.query.return_value = MagicMock(result_rows=[])
        with pytest.raises(SystemExit):
            main(["download", "nonexistent", "-o", str(tmp_path / "out")])


# ---------------------------------------------------------------------------
# _cmd_download with --children
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_create_client")
class TestCmdDownloadChildren:
    def test_children_queries_by_run_prefix(self, mock_client: MagicMock, tmp_path: Path) -> None:
        """--children uses LIKE prefix query and creates child directories."""
        parent_span = _make_span_row(span_id="s1", trace_id="t1", name="parent_task", run_id="job-1")
        child_span = _make_span_row(span_id="s2", trace_id="t2", name="child_task", run_id="job-1-subtask")
        mock_client.query.side_effect = [
            # _resolve_identifier: _RESOLVE_RUN_ID_QUERY
            MagicMock(result_rows=[(SAMPLE_UUID_OBJ, "job-1")]),
            # _DOWNLOAD_SPANS_BY_RUN_PREFIX_QUERY
            MagicMock(result_rows=[parent_span, child_span], column_names=_SPAN_COLUMNS),
        ]
        out = tmp_path / "out"
        result = main(["download", "job-1", "--children", "-o", str(out)])
        assert result == 0
        # Parent spans go to root
        assert any(d.name[:1].isdigit() for d in out.iterdir() if d.is_dir())
        # Child spans go to child_{run_id}/ subdirectory
        child_dir = out / "child_job-1-subtask"
        assert child_dir.is_dir()
        assert any(d.name[:1].isdigit() for d in child_dir.iterdir() if d.is_dir())

        # Verify the LIKE query was used
        spans_query_call = mock_client.query.call_args_list[1]
        assert "run_id_prefix" in spans_query_call[1]["parameters"]
        assert spans_query_call[1]["parameters"]["run_id_prefix"] == "job-1%"

    def test_children_no_children_found(self, mock_client: MagicMock, tmp_path: Path) -> None:
        """--children with only parent spans produces no child directories."""
        parent_span = _make_span_row(span_id="s1", run_id="job-1")
        mock_client.query.side_effect = [
            MagicMock(result_rows=[(SAMPLE_UUID_OBJ, "job-1")]),
            MagicMock(result_rows=[parent_span], column_names=_SPAN_COLUMNS),
        ]
        out = tmp_path / "out"
        result = main(["download", "job-1", "--children", "-o", str(out)])
        assert result == 0
        child_dirs = [d for d in out.iterdir() if d.is_dir() and d.name.startswith("child_")]
        assert child_dirs == []

    def test_children_with_uuid_resolves_run_id(self, mock_client: MagicMock, tmp_path: Path) -> None:
        """UUID + --children resolves run_id from pipeline_runs first."""
        parent_span = _make_span_row(span_id="s1", trace_id="t1", run_id="job-1")
        child_span = _make_span_row(span_id="s2", trace_id="t2", run_id="job-1-sub1")
        mock_client.query.side_effect = [
            # _resolve_identifier: _RUN_METADATA_QUERY for UUID
            MagicMock(result_rows=[("job-1", "my_flow", "", "completed", _NOW, _LATER, 0, 0, "{}")]),
            # _DOWNLOAD_SPANS_BY_RUN_PREFIX_QUERY
            MagicMock(result_rows=[parent_span, child_span], column_names=_SPAN_COLUMNS),
        ]
        out = tmp_path / "out"
        result = main(["download", SAMPLE_UUID, "--children", "-o", str(out)])
        assert result == 0
        assert (out / "child_job-1-sub1").is_dir()

    def test_children_without_run_id_falls_back_to_execution_id(self, mock_client: MagicMock, tmp_path: Path) -> None:
        """UUID + --children but no run_id in pipeline_runs falls back to execution_id query."""
        parent_span = _make_span_row(span_id="s1", run_id="")
        mock_client.query.side_effect = [
            # _resolve_identifier: no run metadata
            MagicMock(result_rows=[]),
            # Falls back to _DOWNLOAD_SPANS_QUERY (execution_id based)
            MagicMock(result_rows=[parent_span], column_names=_SPAN_COLUMNS),
        ]
        out = tmp_path / "out"
        result = main(["download", SAMPLE_UUID, "--children", "-o", str(out)])
        assert result == 0
        # No children since run_id was empty, so --children had no effect
        spans_query_call = mock_client.query.call_args_list[1]
        assert "execution_id" in spans_query_call[1]["parameters"]
