"""Pipeline Core - Shared infrastructure for AI pipelines."""

from . import llm
from .documents import (
    Document,
    DocumentList,
    FlowDocument,
    TaskDocument,
    TemporaryDocument,
    canonical_name_key,
    sanitize_url,
)
from .flow import FlowConfig, FlowOptions
from .llm import (
    AIMessages,
    AIMessageType,
    ModelName,
    ModelOptions,
    ModelResponse,
    StructuredModelResponse,
)
from .logging import (
    LoggerMixin,
    LoggingConfig,
    StructuredLoggerMixin,
    get_pipeline_logger,
    setup_logging,
)
from .logging import get_pipeline_logger as get_logger
from .pipeline import pipeline_flow, pipeline_task
from .prefect import disable_run_logger, prefect_test_harness
from .prompt_manager import PromptManager
from .settings import settings
from .tracing import TraceInfo, TraceLevel, trace

__version__ = "0.1.10"

__all__ = [
    # Config/Settings
    "settings",
    # Logging
    "get_logger",
    "get_pipeline_logger",
    "LoggerMixin",
    "LoggingConfig",
    "setup_logging",
    "StructuredLoggerMixin",
    # Documents
    "Document",
    "DocumentList",
    "FlowDocument",
    "TaskDocument",
    "TemporaryDocument",
    "canonical_name_key",
    "sanitize_url",
    # Flow/Task
    "FlowConfig",
    "FlowOptions",
    # Pipeline decorators (with tracing)
    "pipeline_task",
    "pipeline_flow",
    # Prefect decorators (clean, no tracing)
    "prefect_test_harness",
    "disable_run_logger",
    # LLM
    "llm",
    "ModelName",
    "ModelOptions",
    "ModelResponse",
    "StructuredModelResponse",
    "AIMessages",
    "AIMessageType",
    # Tracing
    "trace",
    "TraceLevel",
    "TraceInfo",
    # Utils
    "PromptManager",
]
