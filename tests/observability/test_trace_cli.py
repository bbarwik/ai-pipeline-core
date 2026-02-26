"""Tests for the ai-trace CLI tool."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

clickhouse_connect = pytest.importorskip("clickhouse_connect")

from ai_pipeline_core.observability.cli import (
    _parse_execution_id,
    _resolve_connection,
    main,
)

SAMPLE_UUID = "550e8400-e29b-41d4-a716-446655440000"
SAMPLE_UUID_OBJ = UUID(SAMPLE_UUID)

_SETTINGS_PATCH = "ai_pipeline_core.settings.Settings"
_DOWNLOADER_PATCH = "ai_pipeline_core.observability._debug._reconstruction.TraceDownloader"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client():
    """Create a mock ClickHouse client."""
    client = MagicMock()
    client.query.return_value = MagicMock(result_rows=[])
    return client


@pytest.fixture
def _patch_create_client(mock_client):
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
    def test_successful_download(self, mock_client, tmp_path):
        output_dir = tmp_path / "trace_out"
        output_dir.mkdir()
        (output_dir / "001_root_flow").mkdir()
        (output_dir / "summary.md").write_text("# test_flow\nSummary here")

        with patch(_DOWNLOADER_PATCH) as MockDL:
            instance = MockDL.return_value
            instance.download_trace.return_value = output_dir

            result = main(["download", SAMPLE_UUID, "-o", str(output_dir)])

        assert result == 0
        instance.download_trace.assert_called_once()

    def test_default_output_path(self, mock_client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "550e8400_trace"
        output_dir.mkdir()
        (output_dir / "summary.md").write_text("# flow\n")

        with patch(_DOWNLOADER_PATCH) as MockDL:
            instance = MockDL.return_value
            instance.download_trace.return_value = output_dir

            result = main(["download", SAMPLE_UUID])

        assert result == 0
        call_args = instance.download_trace.call_args
        assert call_args[0][1].name == "550e8400_trace"

    def test_custom_output_path(self, mock_client, tmp_path):
        custom_dir = tmp_path / "custom_output"
        custom_dir.mkdir()
        (custom_dir / "summary.md").write_text("# flow\n")

        with patch(_DOWNLOADER_PATCH) as MockDL:
            instance = MockDL.return_value
            instance.download_trace.return_value = custom_dir

            result = main(["download", SAMPLE_UUID, "-o", str(custom_dir)])

        assert result == 0
        call_args = instance.download_trace.call_args
        assert call_args[0][1] == custom_dir

    def test_documents_flag(self, mock_client, tmp_path):
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        (output_dir / "summary.md").write_text("# flow\n")

        with patch(_DOWNLOADER_PATCH) as MockDL:
            instance = MockDL.return_value
            instance.download_trace.return_value = output_dir

            main(["download", SAMPLE_UUID, "-o", str(output_dir), "--documents"])

        call_kwargs = instance.download_trace.call_args[1]
        assert call_kwargs["include_documents"] is True

    def test_children_flag(self, mock_client, tmp_path):
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        (output_dir / "summary.md").write_text("# flow\n")

        with patch(_DOWNLOADER_PATCH) as MockDL:
            instance = MockDL.return_value
            instance.download_trace.return_value = output_dir

            main(["download", SAMPLE_UUID, "-o", str(output_dir), "--children"])

        call_kwargs = instance.download_trace.call_args[1]
        assert call_kwargs["follow_children"] is True

    def test_download_error_returns_1(self, mock_client, tmp_path):
        with patch(_DOWNLOADER_PATCH) as MockDL:
            instance = MockDL.return_value
            instance.download_trace.side_effect = RuntimeError("connection lost")

            result = main(["download", SAMPLE_UUID, "-o", str(tmp_path / "out")])

        assert result == 1


# ---------------------------------------------------------------------------
# _cmd_list
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_create_client")
class TestCmdList:
    def test_successful_listing(self, mock_client):
        exec_id = uuid4()
        now = datetime.now(UTC)
        mock_client.query.return_value = MagicMock(
            result_rows=[
                (exec_id, "run-001", "my_flow", "completed", now, now, 0.05, 5000),
            ]
        )

        result = main(["list"])
        assert result == 0

    def test_empty_results(self, mock_client):
        mock_client.query.return_value = MagicMock(result_rows=[])

        result = main(["list"])
        assert result == 0

    def test_status_filter(self, mock_client):
        mock_client.query.return_value = MagicMock(result_rows=[])

        main(["list", "--status", "completed"])

        call_args = mock_client.query.call_args
        assert "status" in call_args[1]["parameters"]
        assert call_args[1]["parameters"]["status"] == "completed"

    def test_flow_filter(self, mock_client):
        mock_client.query.return_value = MagicMock(result_rows=[])

        main(["list", "--flow", "my_flow"])

        call_args = mock_client.query.call_args
        assert "flow_name" in call_args[1]["parameters"]
        assert call_args[1]["parameters"]["flow_name"] == "my_flow"

    def test_custom_limit(self, mock_client):
        mock_client.query.return_value = MagicMock(result_rows=[])

        main(["list", "--limit", "5"])

        call_args = mock_client.query.call_args
        assert call_args[1]["parameters"]["limit"] == 5


# ---------------------------------------------------------------------------
# _cmd_show
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_create_client")
class TestCmdShow:
    def test_successful_show(self, mock_client):
        now = datetime.now(UTC)
        mock_client.query.side_effect = [
            # run metadata
            MagicMock(
                result_rows=[
                    ("run-001", "my_flow", "scope/run-001", "completed", now, now, 0.123, 10000, "{}"),
                ]
            ),
            # span summary
            MagicMock(
                result_rows=[
                    ("llm", 3, 5000, 0.1, 8000, 2000),
                    ("default", 5, 1000, 0.0, 0, 0),
                ]
            ),
        ]

        result = main(["show", SAMPLE_UUID])
        assert result == 0

    def test_no_run_found(self, mock_client):
        mock_client.query.return_value = MagicMock(result_rows=[])

        result = main(["show", SAMPLE_UUID])
        assert result == 1


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    def test_no_args_prints_help(self, capsys):
        result = main([])
        assert result == 1
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "ai-trace" in captured.out.lower()

    def test_unknown_command_prints_help(self, capsys):
        result = main([])
        assert result == 1
