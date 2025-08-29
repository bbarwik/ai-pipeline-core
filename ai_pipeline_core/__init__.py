"""AI Pipeline Core - Production-ready framework for building AI pipelines with LLMs.

@public

AI Pipeline Core is a high-performance async framework for building type-safe AI pipelines.
It combines document processing, LLM integration, and workflow orchestration into a unified
system designed for production use.

The framework enforces best practices through strong typing (Pydantic), automatic retries,
cost tracking, and distributed tracing. All I/O operations are async for maximum throughput.

Core Capabilities:
    - **Document Processing**: Type-safe handling of text, JSON, YAML, PDFs, and images
    - **LLM Integration**: Unified interface to any model via LiteLLM with caching
    - **Structured Output**: Type-safe generation with Pydantic model validation
    - **Workflow Orchestration**: Prefect-based flows and tasks with retries
    - **Observability**: Distributed tracing via Laminar (LMNR) for debugging
    - **Local Development**: Simple runner for testing without infrastructure

Quick Start:
    >>> from ai_pipeline_core import pipeline_flow, FlowDocument, DocumentList
    >>> from ai_pipeline_core.flow import FlowOptions
    >>> from ai_pipeline_core.llm import generate
    >>>
    >>> class OutputDoc(FlowDocument):
    ...     '''Analysis result document.'''
    >>>
    >>> @pipeline_flow
    >>> async def analyze_flow(
    ...     project_name: str,
    ...     documents: DocumentList,
    ...     flow_options: FlowOptions
    ... ) -> DocumentList:
    ...     response = await generate(
    ...         model="gpt-5",
    ...         messages=documents[0].text
    ...     )
    ...     result = OutputDoc.create(
    ...         name="analysis.txt",
    ...         content=response.content
    ...     )
    ...     return DocumentList([result])

Environment Variables (when using LiteLLM proxy):
    - OPENAI_BASE_URL: LiteLLM proxy endpoint (e.g., http://localhost:4000)
    - OPENAI_API_KEY: API key for LiteLLM proxy

Optional Environment Variables:
    - PREFECT_API_URL: Prefect server for orchestration
    - LMNR_PROJECT_API_KEY: Laminar (LMNR) API key for tracing
"""

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
