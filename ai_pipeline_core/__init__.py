"""AI Pipeline Core - High-performance async library for AI pipelines.

@public

AI Pipeline Core provides a comprehensive framework for building production-ready
AI pipelines with strong typing, observability, and efficient LLM integration.
Built on Prefect for orchestration and LiteLLM for model access.

Key Features:
    - **Async-first**: All I/O operations are async for maximum performance
    - **Type safety**: Pydantic models and strong typing throughout
    - **Document abstraction**: Unified handling of text, images, PDFs
    - **LLM integration**: Smart context caching and structured outputs
    - **Observability**: Built-in tracing with LMNR (Laminar)
    - **Orchestration**: Prefect integration for flows and tasks
    - **Simple runner**: Local execution without full orchestration

Core Components:
    Documents: Type-safe document handling with automatic encoding
    LLM: Unified interface for language models via LiteLLM proxy
    Flow/Task: Pipeline orchestration with Prefect integration
    Tracing: Distributed tracing and observability with LMNR
    Logging: Unified logging with Prefect integration
    Settings: Centralized configuration management

Quick Start:
    >>> from ai_pipeline_core import (
    ...     pipeline_flow,
    ...     FlowDocument,
    ...     DocumentList,
    ...     FlowOptions,
    ...     ModelOptions,
    ...     llm
    ... )
    >>>
    >>> class InputDoc(FlowDocument):
    ...     '''Input document for analysis.'''
    >>>
    >>> @pipeline_flow
    >>> async def analyze_flow(
    ...     project_name: str,
    ...     documents: DocumentList,
    ...     flow_options: FlowOptions
    ... ) -> DocumentList:
    ...     # Your pipeline logic here
    ...     response = await llm.generate(
    ...         model="gpt-5",
    ...         messages=documents[0].text
    ...     )
    ...     return DocumentList([...])

Environment Setup:
    Required environment variables:
    - OPENAI_BASE_URL: LiteLLM proxy endpoint
    - OPENAI_API_KEY: API key for LiteLLM

    Optional:
    - PREFECT_API_URL: Prefect server for orchestration
    - LMNR_PROJECT_API_KEY: Laminar project key for tracing

Documentation:
    Full documentation: https://github.com/jxnl/ai-pipeline-core
    Examples: See examples/ directory in repository

Version History:
    - 0.1.10: Enhanced CLI error handling, improved documentation
    - 0.1.9: Document class enhancements with type safety
    - 0.1.8: Major refactoring and validation improvements
    - 0.1.7: Pipeline decorators and simple runner module

Current Version: 0.1.10
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
