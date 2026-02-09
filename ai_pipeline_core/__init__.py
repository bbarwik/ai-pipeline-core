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
from .deployment.remote import RemoteDeployment, run_remote_deployment
from .document_store import DocumentStore, SummaryGenerator, create_document_store, get_document_store, set_document_store
from .documents import (
    Attachment,
    Document,
    DocumentSha256,
    RunContext,
    RunScope,
    TaskDocumentContext,
    get_run_context,
    is_document_sha256,
    reset_run_context,
    sanitize_url,
    set_run_context,
)
from .llm import (
    Citation,
    Conversation,
    ConversationContent,
    ImagePart,
    ImagePreset,
    ImageProcessingConfig,
    ImageProcessingError,
    ModelName,
    ModelOptions,
    ModelResponse,
    ProcessedImage,
    TokenUsage,
    URLSubstitutor,
    generate,
    generate_structured,
    process_image,
)
from .logging import (
    LoggingConfig,
    get_pipeline_logger,
    setup_logging,
)
from .logging import get_pipeline_logger as get_logger
from .observability.tracing import TraceInfo, TraceLevel, set_trace_cost, trace
from .pipeline import FlowOptions, pipeline_flow, pipeline_task
from .prompt_manager import PromptManager
from .settings import Settings

__version__ = "0.8.2"

__all__ = [
    "Attachment",
    "Citation",
    "Conversation",
    "ConversationContent",
    "DeploymentContext",
    "DeploymentResult",
    "Document",
    "DocumentSha256",
    "DocumentStore",
    "FlowOptions",
    "ImagePart",
    "ImagePreset",
    "ImageProcessingConfig",
    "ImageProcessingError",
    "LoggingConfig",
    "ModelName",
    "ModelOptions",
    "ModelResponse",
    "PipelineDeployment",
    "ProcessedImage",
    "PromptManager",
    "RemoteDeployment",
    "RunContext",
    "RunScope",
    "Settings",
    "SummaryGenerator",
    "TaskDocumentContext",
    "TokenUsage",
    "TraceInfo",
    "TraceLevel",
    "URLSubstitutor",
    "create_document_store",
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
    "process_image",
    "progress",
    "reset_run_context",
    "run_remote_deployment",
    "sanitize_url",
    "set_document_store",
    "set_run_context",
    "set_trace_cost",
    "setup_logging",
    "trace",
]
