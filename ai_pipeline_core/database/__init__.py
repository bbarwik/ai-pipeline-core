"""Unified database module for the span-based schema."""

from ai_pipeline_core.database._factory import Database
from ai_pipeline_core.database._protocol import DatabaseReader
from ai_pipeline_core.database._types import (
    CostTotals,
    DocumentRecord,
    HydratedDocument,
    LogRecord,
    SpanKind,
    SpanRecord,
    SpanStatus,
)

__all__ = [
    "CostTotals",
    "Database",
    "DatabaseReader",
    "DocumentRecord",
    "HydratedDocument",
    "LogRecord",
    "SpanKind",
    "SpanRecord",
    "SpanStatus",
]
