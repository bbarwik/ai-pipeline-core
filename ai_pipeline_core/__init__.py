"""AI Pipeline Core - Production-ready framework for building AI pipelines with LLMs."""

import os
import sys

from prefect.context import refresh_global_settings_context
from prefect.settings import get_current_settings

# Disable Prefect's built-in OpenTelemetry spans to prevent duplicates.
# All tracing is handled by our @trace decorator and Laminar SDK.
# Must be set before Prefect is imported by submodules below.
os.environ.setdefault("PREFECT_CLOUD_ENABLE_ORCHESTRATION_TELEMETRY", "false")

# If Prefect was already imported (user imported it before us), refresh its cached settings.
if "prefect" in sys.modules and get_current_settings().cloud.enable_orchestration_telemetry:
    refresh_global_settings_context()

from . import llm
from .deployment import DeploymentContext, DeploymentResult, PipelineDeployment, progress
from .deployment.remote import remote_deployment
from .document_store import DocumentStore, SummaryGenerator, create_document_store, get_document_store, set_document_store
from .documents import (
    Attachment,
    Document,
    RunContext,
    TaskDocumentContext,
    canonical_name_key,
    get_run_context,
    is_document_sha256,
    reset_run_context,
    sanitize_url,
    set_run_context,
)
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
from .observability.tracing import TraceInfo, TraceLevel, set_trace_cost, trace
from .pipeline import FlowOptions, pipeline_flow, pipeline_task
from .prompt_manager import PromptManager
from .settings import Settings
from .testing import disable_run_logger, prefect_test_harness

__version__ = "0.4.8"

__all__ = [
    "AIMessageType",
    "AIMessages",
    "Attachment",
    "DeploymentContext",
    "DeploymentResult",
    "Document",
    "DocumentStore",
    "FlowOptions",
    "ImagePart",
    "ImagePreset",
    "ImageProcessingConfig",
    "ImageProcessingError",
    "LoggerMixin",
    "LoggingConfig",
    "ModelName",
    "ModelOptions",
    "ModelResponse",
    "PipelineDeployment",
    "ProcessedImage",
    "PromptManager",
    "RunContext",
    "Settings",
    "StructuredLoggerMixin",
    "StructuredModelResponse",
    "SummaryGenerator",
    "TaskDocumentContext",
    "TraceInfo",
    "TraceLevel",
    "canonical_name_key",
    "create_document_store",
    "disable_run_logger",
    "generate",
    "generate_structured",
    "get_document_store",
    "get_logger",
    "get_pipeline_logger",
    "get_run_context",
    "is_document_sha256",
    "llm",
    "pipeline_flow",
    "pipeline_task",
    "prefect_test_harness",
    "process_image",
    "process_image_to_documents",
    "progress",
    "remote_deployment",
    "reset_run_context",
    "sanitize_url",
    "set_document_store",
    "set_run_context",
    "set_trace_cost",
    "setup_logging",
    "trace",
]
