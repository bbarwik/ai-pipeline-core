"""Data types for the logging subsystem."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

__all__ = [
    "LogRecord",
]


@dataclass(frozen=True, slots=True)
class LogRecord:
    """Row from the logs table."""

    deployment_id: UUID
    span_id: UUID
    timestamp: datetime
    sequence_no: int
    level: str
    category: str
    logger_name: str
    message: str
    event_type: str = ""
    fields_json: str = "{}"
    exception_text: str = ""
