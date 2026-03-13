"""Shared sort-key functions for span and log records."""

from datetime import datetime

from ai_pipeline_core.database._types import LogRecord, SpanRecord

__all__ = [
    "child_span_sort_key",
    "deployment_sort_key",
    "log_sort_key",
    "span_sort_key",
    "tree_child_sort_key",
]


def span_sort_key(span: SpanRecord) -> tuple[datetime, int, int, int, str]:
    return (
        span.started_at,
        span.deployment_id.int,
        span.sequence_no,
        span.version,
        str(span.span_id),
    )


def child_span_sort_key(span: SpanRecord) -> tuple[int, datetime, str]:
    return span.sequence_no, span.started_at, str(span.span_id)


def deployment_sort_key(span: SpanRecord) -> tuple[datetime, str]:
    return span.started_at, str(span.span_id)


def tree_child_sort_key(span: SpanRecord) -> tuple[datetime, int, str]:
    return span.started_at, span.sequence_no, str(span.span_id)


def log_sort_key(log: LogRecord) -> tuple[datetime, int, str]:
    return log.timestamp, log.sequence_no, str(log.span_id)
