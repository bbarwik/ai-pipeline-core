"""Tests for the ai-trace CLI tool."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from ai_pipeline_core.database import ExecutionLog, ExecutionNode, NodeKind, NodeStatus
from ai_pipeline_core.database._filesystem import FilesystemDatabase
from ai_pipeline_core.observability.cli import (
    _parse_execution_id,
    _resolve_connection,
    _resolve_identifier,
    main,
)

SAMPLE_UUID = "550e8400-e29b-41d4-a716-446655440000"
SAMPLE_UUID_OBJ = UUID(SAMPLE_UUID)


def _make_node(**kwargs: object) -> ExecutionNode:
    deployment_id = kwargs.pop("_deployment_id", None) or uuid4()
    root_deployment_id = kwargs.pop("_root_deployment_id", None) or deployment_id
    defaults: dict[str, object] = {
        "node_id": uuid4(),
        "node_kind": NodeKind.DEPLOYMENT,
        "deployment_id": deployment_id,
        "root_deployment_id": root_deployment_id,
        "run_id": "run-001",
        "run_scope": "run-001/scope",
        "deployment_name": "trace-cli-test",
        "name": "trace-cli-test",
        "sequence_no": 0,
        "status": NodeStatus.COMPLETED,
        "started_at": datetime(2024, 1, 1, tzinfo=UTC),
        "ended_at": datetime(2024, 1, 1, 0, 0, 5, tzinfo=UTC),
    }
    defaults.update(kwargs)
    return ExecutionNode(**defaults)  # type: ignore[arg-type]


def _make_log(**kwargs: object) -> ExecutionLog:
    deployment_id = kwargs.pop("_deployment_id", None) or uuid4()
    defaults: dict[str, object] = {
        "node_id": uuid4(),
        "deployment_id": deployment_id,
        "root_deployment_id": deployment_id,
        "flow_id": None,
        "task_id": None,
        "timestamp": datetime(2024, 1, 1, tzinfo=UTC),
        "sequence_no": 0,
        "level": "INFO",
        "category": "framework",
        "logger_name": "ai_pipeline_core.tests",
        "message": "test log",
    }
    defaults.update(kwargs)
    return ExecutionLog(**defaults)  # type: ignore[arg-type]


def _seed_trace_database(base_path: Path) -> tuple[FilesystemDatabase, ExecutionNode, ExecutionNode]:
    database = FilesystemDatabase(base_path)
    deployment = _make_node(node_id=SAMPLE_UUID_OBJ, run_id="run-001", deployment_name="trace-cli-test")
    task = _make_node(
        node_kind=NodeKind.TASK,
        _deployment_id=deployment.deployment_id,
        _root_deployment_id=deployment.root_deployment_id,
        parent_node_id=deployment.node_id,
        sequence_no=1,
        name="task-1",
        started_at=datetime(2024, 1, 1, 0, 0, 1, tzinfo=UTC),
        ended_at=datetime(2024, 1, 1, 0, 0, 3, tzinfo=UTC),
    )

    async def _seed() -> None:
        await database.insert_node(deployment)
        await database.insert_node(task)
        await database.save_logs_batch([
            _make_log(node_id=deployment.node_id, _deployment_id=deployment.deployment_id, message="deployment started"),
            _make_log(node_id=task.node_id, _deployment_id=deployment.deployment_id, sequence_no=1, message="task finished"),
        ])

    asyncio.run(_seed())
    return database, deployment, task


class TestParseExecutionId:
    def test_valid_uuid(self) -> None:
        assert _parse_execution_id(SAMPLE_UUID) == SAMPLE_UUID_OBJ

    def test_invalid_uuid_exits(self) -> None:
        with pytest.raises(SystemExit):
            _parse_execution_id("not-a-uuid")


class TestResolveConnection:
    def test_db_path_returns_filesystem_database(self, tmp_path: Path) -> None:
        args = type("Args", (), {"db_path": str(tmp_path)})()
        result = _resolve_connection(args)
        assert isinstance(result, FilesystemDatabase)


class TestResolveIdentifier:
    def test_uuid_returns_deployment_id_and_run_id(self, tmp_path: Path) -> None:
        database, deployment, task = _seed_trace_database(tmp_path)
        result = _resolve_identifier(str(task.node_id), database)
        assert result == (deployment.deployment_id, deployment.run_id)

    def test_run_id_resolves_deployment(self, tmp_path: Path) -> None:
        database, deployment, _task = _seed_trace_database(tmp_path)
        result = _resolve_identifier(deployment.run_id, database)
        assert result == (deployment.deployment_id, deployment.run_id)

    def test_missing_identifier_exits(self, tmp_path: Path) -> None:
        database = FilesystemDatabase(tmp_path)
        with pytest.raises(SystemExit):
            _resolve_identifier("missing-run", database)


class TestMain:
    def test_list_command(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _seed_trace_database(tmp_path)
        result = main(["list", "--db-path", str(tmp_path)])
        assert result == 0
        assert "trace-cli-test" in capsys.readouterr().out

    def test_show_command(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _seed_trace_database(tmp_path)
        result = main(["show", "run-001", "--db-path", str(tmp_path)])
        assert result == 0
        output = capsys.readouterr().out
        assert "# trace-cli-test / run-001" in output
        assert "deployment started" in output
        assert "task finished" in output

    def test_show_command_renders_log_fields_and_exception_text(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        database = FilesystemDatabase(tmp_path)
        deployment = _make_node(node_id=SAMPLE_UUID_OBJ, run_id="run-001", deployment_name="trace-cli-test")

        async def _seed() -> None:
            await database.insert_node(deployment)
            await database.save_logs_batch([
                _make_log(
                    node_id=deployment.node_id,
                    _deployment_id=deployment.deployment_id,
                    message="deployment failed",
                    fields='{"attempt": 3, "phase": "publish"}',
                    exception_text="Traceback line 1\nTraceback line 2",
                )
            ])

        asyncio.run(_seed())

        result = main(["show", "run-001", "--db-path", str(tmp_path)])
        assert result == 0
        output = capsys.readouterr().out
        assert '"attempt": 3' in output
        assert '"phase": "publish"' in output
        assert "Traceback line 1" in output
        assert "Traceback line 2" in output

    def test_download_command(self, tmp_path: Path) -> None:
        _seed_trace_database(tmp_path / "source")
        output_dir = tmp_path / "download"
        result = main(["download", "run-001", "--db-path", str(tmp_path / "source"), "--output-dir", str(output_dir)])
        assert result == 0
        assert (output_dir / "summary.md").exists()
        assert (output_dir / "costs.md").exists()
        assert (output_dir / "logs.jsonl").exists()
        assert (output_dir / "runs").is_dir()
