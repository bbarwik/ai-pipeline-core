"""AI Pipeline Core - Production-ready framework for building AI pipelines with LLMs.

@public

AI Pipeline Core is a high-performance async framework for building type-safe AI pipelines.
It combines document processing, LLM integration, and workflow orchestration into a unified
system designed for production use.

The framework enforces best practices through strong typing (Pydantic), automatic retries,
and cost tracking. All I/O operations are async for maximum throughput.

**CRITICAL IMPORT RULE**:
    Always import from the top-level package:
        **CORRECT**:
        from ai_pipeline_core import llm, pipeline_flow, FlowDocument, DocumentList

        **WRONG** - Never import from submodules:
        from ai_pipeline_core.llm import generate  # NO!
        from ai_pipeline_core.documents import FlowDocument  # NO!

FRAMEWORK RULES (Use by default, unless instructed otherwise):
    1. Decorators: Use @pipeline_task WITHOUT parameters, @pipeline_flow WITH config
    2. Logging: Use get_pipeline_logger(__name__) - NEVER print() or logging module
    3. LLM calls: Use AIMessages or str. Wrap Documents in AIMessages; do not call .text yourself
    4. Options: DO NOT use options parameter - omit it entirely (defaults are optimal)
    5. Documents: Create with just name and content - skip description unless needed
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
    - **Observability**: Built-in monitoring and debugging capabilities
    - **Local Development**: Simple runner for testing without infrastructure

Quick Start:
    >>> from ai_pipeline_core import (
    ...     pipeline_flow, FlowDocument, DocumentList, FlowOptions, FlowConfig, llm, AIMessages
    ... )
    >>>
    >>> class OutputDoc(FlowDocument):
    ...     '''Analysis result document.'''
    >>>
    >>> class MyFlowConfig(FlowConfig):
    ...     INPUT_DOCUMENT_TYPES = []
    ...     OUTPUT_DOCUMENT_TYPE = OutputDoc
    >>>
    >>> @pipeline_flow(config=MyFlowConfig)
    >>> async def analyze_flow(
    ...     project_name: str,
    ...     documents: DocumentList,
    ...     flow_options: FlowOptions
    ... ) -> DocumentList:
    ...     # Messages accept AIMessages or str. Wrap documents: AIMessages([doc])
    ...     response = await llm.generate(
    ...         "gpt-5.1",
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
"""

import os
import sys

# Disable Prefect's built-in OpenTelemetry spans to prevent duplicates.
# All tracing is handled by our @trace decorator and Laminar SDK.
# Must be set before Prefect is imported by submodules below.
os.environ.setdefault("PREFECT_CLOUD_ENABLE_ORCHESTRATION_TELEMETRY", "false")

# If Prefect was already imported (user imported it before us), refresh its cached settings.
if "prefect" in sys.modules:
    try:
        from prefect.settings import get_current_settings  # noqa: PLC0415

        if get_current_settings().cloud.enable_orchestration_telemetry:
            from prefect.context import refresh_global_settings_context  # noqa: PLC0415

            refresh_global_settings_context()
    except (ImportError, AttributeError):
        pass

from . import llm, progress
from .deployment import DeploymentContext, DeploymentResult, PipelineDeployment
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
from .images import (
    ImagePart,
    ImagePreset,
    ImageProcessingConfig,
    ImageProcessingError,
    ProcessedImage,
    process_image,
    process_image_to_documents,
)
from .llm import (
    AIMessages,
    AIMessageType,
    ModelName,
    ModelOptions,
    ModelResponse,
    StructuredModelResponse,
    generate,
    generate_structured,
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
from .prompt_builder import EnvironmentVariable, PromptBuilder
from .prompt_manager import PromptManager
from .settings import Settings
from .tracing import TraceInfo, TraceLevel, set_trace_cost, trace
from .utils.remote_deployment import remote_deployment

__version__ = "0.3.4"

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
    # Deployment
    "PipelineDeployment",
    "DeploymentContext",
    "DeploymentResult",
    "remote_deployment",
    "progress",
    # LLM
    "llm",  # for backward compatibility
    "generate",
    "generate_structured",
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
    # Prompt Builder
    "PromptBuilder",
    "EnvironmentVariable",
    # Images
    "process_image",
    "process_image_to_documents",
    "ImagePreset",
    "ImageProcessingConfig",
    "ProcessedImage",
    "ImagePart",
    "ImageProcessingError",
    # Utils
    "PromptManager",
]
