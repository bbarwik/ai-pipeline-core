"""Tests for deployment execution-log flush helpers."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from ai_pipeline_core.database import ExecutionLog
from ai_pipeline_core.deployment._helpers import _flush_execution_logs_once
from ai_pipeline_core.logging import ExecutionLogBuffer


def _make_log() -> ExecutionLog:
    deployment_id = uuid4()
    return ExecutionLog(
        node_id=uuid4(),
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        flow_id=None,
        task_id=None,
        timestamp=datetime.now(UTC),
        sequence_no=0,
        level="INFO",
        category="framework",
        logger_name="ai_pipeline_core.tests",
        message="test log",
    )


class _FailingDatabase:
    def __init__(self) -> None:
        self.calls = 0

    async def save_logs_batch(self, logs: list[ExecutionLog]) -> None:
        self.calls += 1
        raise RuntimeError(f"failing save {len(logs)}")


class _RecordingDatabase:
    def __init__(self) -> None:
        self.saved_batches: list[list[ExecutionLog]] = []

    async def save_logs_batch(self, logs: list[ExecutionLog]) -> None:
        self.saved_batches.append(list(logs))


class TestFlushExecutionLogsOnce:
    @pytest.mark.asyncio
    async def test_empty_inputs_return_empty_pending_logs(self) -> None:
        assert await _flush_execution_logs_once(None, None, []) == []

    @pytest.mark.asyncio
    async def test_flush_failure_preserves_pending_logs(self) -> None:
        buffer = ExecutionLogBuffer()
        log = _make_log()
        buffer.append(log)
        database = _FailingDatabase()

        pending = await _flush_execution_logs_once(database, buffer, [])

        assert database.calls == 1
        assert pending == [log]

    @pytest.mark.asyncio
    async def test_flush_success_drains_buffer(self) -> None:
        buffer = ExecutionLogBuffer()
        buffer.append(_make_log())
        database = _RecordingDatabase()

        pending = await _flush_execution_logs_once(database, buffer, [])

        assert pending == []
        assert len(database.saved_batches) == 1
