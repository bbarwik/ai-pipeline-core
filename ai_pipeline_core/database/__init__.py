"""Unified database module for the span-based schema."""

from ai_pipeline_core.database._factory import Database
from ai_pipeline_core.database._memory import MemoryDatabase
from ai_pipeline_core.database._protocol import DatabaseReader
from ai_pipeline_core.database._types import (
    BlobRecord,
    CostTotals,
    DocumentRecord,
    HydratedDocument,
    LogRecord,
    SpanKind,
    SpanRecord,
    SpanStatus,
)

__all__ = [
    "BlobRecord",
    "CostTotals",
    "Database",
    "DatabaseReader",
    "DocumentRecord",
    "HydratedDocument",
    "LogRecord",
    "MemoryDatabase",
    "SpanKind",
    "SpanRecord",
    "SpanStatus",
]
