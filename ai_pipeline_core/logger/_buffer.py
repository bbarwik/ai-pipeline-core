"""In-memory buffering for execution logs captured from the root logger."""

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from threading import Lock
from uuid import UUID

from ai_pipeline_core.logger._types import LogRecord

__all__ = [
    "ExecutionLogBuffer",
]

DEFAULT_LOG_BUFFER_FLUSH_SIZE = 500
MAX_PENDING_EXECUTION_LOGS = 50_000
WARNING_LEVEL_NAME = logging.getLevelName(logging.WARNING)
EMPTY_LOG_SUMMARY = {
    "total": 0,
    "warnings": 0,
    "errors": 0,
    "last_error": "",
}


class ExecutionLogBuffer:
    """Thread-safe execution log buffer with per-span ordering and summaries."""

    def __init__(
        self,
        *,
        flush_size: int = DEFAULT_LOG_BUFFER_FLUSH_SIZE,
        max_pending_logs: int = MAX_PENDING_EXECUTION_LOGS,
        request_flush: Callable[[], None] | None = None,
    ) -> None:
        self._flush_size = flush_size
        self._max_pending_logs = max_pending_logs
        self._request_flush = request_flush
        self._lock = Lock()
        self._pending: deque[LogRecord] = deque()
        self._sequence_by_span: dict[UUID, int] = {}
        self._summary_by_span: dict[UUID, dict[str, int | str]] = {}
        self._dropped_count = 0

    def append(self, log: LogRecord) -> None:
        """Assign sequence_no, update summaries, and queue the log for flush."""
        should_request_flush = False
        with self._lock:
            sequence_no = self._sequence_by_span.get(log.span_id, 0)
            self._sequence_by_span[log.span_id] = sequence_no + 1
            stored_log = replace(log, sequence_no=sequence_no)
            self._pending.append(stored_log)
            if len(self._pending) > self._max_pending_logs:
                self._pending.popleft()
                self._dropped_count += 1
            self._update_summary(stored_log)
            should_request_flush = len(self._pending) >= self._flush_size

        if should_request_flush and self._request_flush is not None:
            self._request_flush()

    def drain(self) -> list[LogRecord]:
        """Return all pending logs and clear the buffer."""
        with self._lock:
            drained = list(self._pending)
            self._pending.clear()
        return drained

    def get_summary(self, span_id: UUID) -> dict[str, int | str]:
        """Return lightweight log counters for a span."""
        with self._lock:
            summary = self._summary_by_span.get(span_id)
            if summary is None:
                return dict(EMPTY_LOG_SUMMARY)
            return dict(summary)

    def consume_summary(self, span_id: UUID) -> dict[str, int | str]:
        """Return and forget a span summary once its terminal payload has been persisted."""
        with self._lock:
            summary = self._summary_by_span.pop(span_id, None)
            self._sequence_by_span.pop(span_id, None)
            if summary is None:
                return dict(EMPTY_LOG_SUMMARY)
            return dict(summary)

    def consume_dropped_count(self) -> int:
        """Return and reset the count of logs dropped due to local buffer overflow."""
        with self._lock:
            dropped_count = self._dropped_count
            self._dropped_count = 0
        return dropped_count

    def _update_summary(self, log: LogRecord) -> None:
        summary = self._summary_by_span.setdefault(log.span_id, dict(EMPTY_LOG_SUMMARY))
        summary["total"] = int(summary["total"]) + 1
        level_no = logging.getLevelNamesMapping().get(log.level, logging.INFO)

        if log.level == WARNING_LEVEL_NAME:
            summary["warnings"] = int(summary["warnings"]) + 1

        if level_no >= logging.ERROR:
            summary["errors"] = int(summary["errors"]) + 1
            summary["last_error"] = log.message
