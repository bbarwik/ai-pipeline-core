"""Pipeline Core - Shared infrastructure for AI pipelines."""

from .documents import Document, DocumentList, FlowDocument, TaskDocument
from .flow import FlowConfig
from .logging import (
    LoggerMixin,
    LoggingConfig,
    StructuredLoggerMixin,
    get_pipeline_logger,
    setup_logging,
)
from .logging import (
    get_pipeline_logger as get_logger,
)
from .prompt_manager import PromptManager
from .settings import settings
from .tracing import trace

__version__ = "0.1.2"

__all__ = [
    "Document",
    "DocumentList",
    "FlowConfig",
    "FlowDocument",
    "get_logger",
    "get_pipeline_logger",
    "LoggerMixin",
    "LoggingConfig",
    "PromptManager",
    "settings",
    "setup_logging",
    "StructuredLoggerMixin",
    "TaskDocument",
    "trace",
]
