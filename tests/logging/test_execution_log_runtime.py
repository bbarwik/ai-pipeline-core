"""Focused tests for execution log buffering and handler classification."""

from datetime import UTC, datetime
import logging
from types import MappingProxyType
from uuid import uuid4

from ai_pipeline_core.database import ExecutionLog
from ai_pipeline_core.deployment._types import _NoopPublisher
from ai_pipeline_core.documents import RunScope
from ai_pipeline_core.logging import ExecutionLogBuffer, ExecutionLogHandler
from ai_pipeline_core.pipeline._execution_context import ExecutionContext, reset_execution_context, set_execution_context
from ai_pipeline_core.pipeline.limits import _SharedStatus


def _make_log(**kwargs: object) -> ExecutionLog:
    deployment_id = kwargs.pop("_deployment_id", None) or uuid4()
    defaults: dict[str, object] = {
        "node_id": uuid4(),
        "deployment_id": deployment_id,
        "root_deployment_id": deployment_id,
        "flow_id": None,
        "task_id": None,
        "timestamp": datetime.now(UTC),
        "sequence_no": 0,
        "level": "INFO",
        "category": "framework",
        "logger_name": "ai_pipeline_core.tests",
        "message": "test log",
    }
    defaults.update(kwargs)
    return ExecutionLog(**defaults)  # type: ignore[arg-type]


def test_execution_log_buffer_sequences_summaries_and_flush_request() -> None:
    requested_flushes: list[str] = []
    deployment_id = uuid4()
    node_id = uuid4()
    buffer = ExecutionLogBuffer(flush_size=2, request_flush=lambda: requested_flushes.append("flush"))

    buffer.append(_make_log(node_id=node_id, _deployment_id=deployment_id, level="WARNING", message="warn"))
    buffer.append(_make_log(node_id=node_id, _deployment_id=deployment_id, level="ERROR", message="error"))

    drained = buffer.drain()
    assert [log.sequence_no for log in drained] == [0, 1]
    assert requested_flushes == ["flush"]
    assert buffer.get_summary(node_id) == {
        "total": 2,
        "warnings": 1,
        "errors": 1,
        "last_error": "error",
    }


def test_execution_log_handler_classifies_and_filters_levels() -> None:
    root_logger = logging.getLogger()
    original_root_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)
    handler = next((item for item in root_logger.handlers if isinstance(item, ExecutionLogHandler)), None)
    added_handler = False
    if handler is None:
        handler = ExecutionLogHandler()
        root_logger.addHandler(handler)
        added_handler = True

    framework_logger = logging.getLogger("ai_pipeline_core.test_runtime")
    dependency_logger = logging.getLogger("httpx")
    application_logger = logging.getLogger("my_app.runtime")
    original_framework_level = framework_logger.level
    original_dependency_level = dependency_logger.level
    original_application_level = application_logger.level
    framework_logger.setLevel(logging.DEBUG)
    dependency_logger.setLevel(logging.DEBUG)
    application_logger.setLevel(logging.DEBUG)

    deployment_id = uuid4()
    buffer = ExecutionLogBuffer()
    ctx = ExecutionContext(
        run_id="test-run",
        run_scope=RunScope("test-run/scope"),
        execution_id=None,
        publisher=_NoopPublisher(),
        limits=MappingProxyType({}),
        limits_status=_SharedStatus(),
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        current_node_id=uuid4(),
        log_buffer=buffer,
    )
    token = set_execution_context(ctx)
    try:
        framework_logger.debug("framework debug")
        dependency_logger.info("dependency info should be filtered")
        dependency_logger.warning("dependency warning")
        application_logger.debug("application debug should be filtered")
        application_logger.info("application info")
        logs = buffer.drain()
        assert {(log.category, log.message) for log in logs} == {
            ("framework", "framework debug"),
            ("dependency", "dependency warning"),
            ("application", "application info"),
        }
    finally:
        reset_execution_context(token)
        framework_logger.setLevel(original_framework_level)
        dependency_logger.setLevel(original_dependency_level)
        application_logger.setLevel(original_application_level)
        root_logger.setLevel(original_root_level)
        if added_handler:
            root_logger.removeHandler(handler)


def test_execution_log_handler_ignores_records_without_execution_context() -> None:
    handler = ExecutionLogHandler()
    record = logging.LogRecord(
        name="my_app.runtime",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="outside context",
        args=(),
        exc_info=None,
    )

    handler.emit(record)


def test_execution_log_buffer_drops_oldest_when_capacity_is_exceeded(monkeypatch) -> None:
    monkeypatch.setattr("ai_pipeline_core.logging._buffer.MAX_PENDING_EXECUTION_LOGS", 2)
    node_id = uuid4()
    deployment_id = uuid4()
    buffer = ExecutionLogBuffer(max_pending_logs=2)

    buffer.append(_make_log(node_id=node_id, _deployment_id=deployment_id, message="first"))
    buffer.append(_make_log(node_id=node_id, _deployment_id=deployment_id, message="second"))
    buffer.append(_make_log(node_id=node_id, _deployment_id=deployment_id, message="third"))

    drained = buffer.drain()
    assert [log.message for log in drained] == ["second", "third"]
    assert buffer.consume_dropped_count() == 1
