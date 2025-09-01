"""AI Pipeline Core - Production-ready framework for building AI pipelines with LLMs.

@public

AI Pipeline Core is a high-performance async framework for building type-safe AI pipelines.
It combines document processing, LLM integration, and workflow orchestration into a unified
system designed for production use.

The framework enforces best practices through strong typing (Pydantic), automatic retries,
cost tracking, and distributed tracing. All I/O operations are async for maximum throughput.

**CRITICAL IMPORT RULE**:
    Always import from the top-level package:
        **CORRECT**:
        from ai_pipeline_core import llm, pipeline_flow, FlowDocument, DocumentList

        **WRONG** - Never import from submodules:
        from ai_pipeline_core.llm import generate  # NO!
        from ai_pipeline_core.documents import FlowDocument  # NO!

FRAMEWORK RULES (90% Use Cases):
    1. Decorators: Use @trace, @pipeline_task, @pipeline_flow WITHOUT parameters
    2. Logging: Use get_pipeline_logger(__name__) - NEVER print() or logging module
    3. LLM calls: Use AIMessages or str. Wrap Documents in AIMessages; do not call .text yourself
    4. Options: Omit ModelOptions unless specifically needed (defaults are optimal)
    5. Documents: Create with just name and content - skip description
    6. FlowConfig: OUTPUT_DOCUMENT_TYPE must differ from all INPUT_DOCUMENT_TYPES
    7. Initialization: PromptManager and logger at module scope, not in functions
    8. DocumentList: Use default constructor - no validation flags needed
    9. setup_logging(): Only in application main(), never at import time

Messages parameter type: AIMessages or str. Do not pass Document or DocumentList directly.

Core Capabilities:
    - **Document Processing**: Type-safe handling of text, JSON, YAML, PDFs, and images
    - **LLM Integration**: Unified interface to any model via LiteLLM with caching
    - **Structured Output**: Type-safe generation with Pydantic model validation
    - **Workflow Orchestration**: Prefect-based flows and tasks with retries
    - **Observability**: Distributed tracing via Laminar (LMNR) for debugging
    - **Local Development**: Simple runner for testing without infrastructure

Quick Start:
    >>> from ai_pipeline_core import (
    ...     pipeline_flow, FlowDocument, DocumentList, FlowOptions, llm, AIMessages
    ... )
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
    ...     # Messages accept AIMessages or str. Wrap documents: AIMessages([doc])
    ...     response = await llm.generate(
    ...         model="gpt-5",
    ...         messages=AIMessages([documents[0]])
    ...     )
    ...     result = OutputDoc.create(
    ...         name="analysis.txt",
    ...         content=response.content
    ...     )
    ...     return DocumentList([result])

Environment Variables (when using LiteLLM proxy):
    - OPENAI_BASE_URL: LiteLLM proxy endpoint (e.g., http://localhost:4000)
    - OPENAI_API_KEY: API key for LiteLLM proxy

    Note: LiteLLM proxy uses OpenAI-compatible API format, hence the OPENAI_*
    variable names are correct regardless of which LLM provider you're using.

Optional Environment Variables:
    - PREFECT_API_URL: Prefect server for orchestration
    - PREFECT_API_KEY: Prefect API authentication key
    - LMNR_PROJECT_API_KEY: Laminar (LMNR) API key for tracing
    - LMNR_DEBUG: Set to "true" to enable debug-level traces
    - LMNR_SESSION_ID: Default session ID for traces
    - LMNR_USER_ID: Default user ID for traces
"""

from . import llm
from .documents import (
    Document,
    DocumentList,
    FlowDocument,
    TaskDocument,
    TemporaryDocument,
    canonical_name_key,
    is_document_sha256,
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
from .settings import Settings
from .tracing import TraceInfo, TraceLevel, set_trace_cost, trace

__version__ = "0.1.14"

__all__ = [
    # Config/Settings
    "Settings",
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
    "is_document_sha256",
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
    "set_trace_cost",
    # Utils
    "PromptManager",
]
