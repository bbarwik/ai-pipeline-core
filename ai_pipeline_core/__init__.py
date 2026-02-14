"""AI Pipeline Core - Production-ready framework for building AI pipelines with LLMs."""

import os
import sys

# Disable Prefect telemetry and analytics before importing Prefect.
# All tracing is handled by our @trace decorator and Laminar SDK.
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("PREFECT_CLOUD_ENABLE_ORCHESTRATION_TELEMETRY", "false")

from prefect.context import refresh_global_settings_context
from prefect.settings import get_current_settings

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
from .pipeline import FlowOptions, LimitKind, PipelineLimit, pipeline_concurrency, pipeline_flow, pipeline_task
from .prompt_compiler import Guide, OutputRule, OutputT, Phase, PromptSpec, Role, Rule, extract_result, render_preview, render_text, send_spec
from .prompt_manager import PromptManager
from .settings import Settings
from .testing import disable_run_logger, prefect_test_harness

__version__ = "0.9.1"

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
    "Guide",
    "ImagePart",
    "ImagePreset",
    "ImageProcessingConfig",
    "ImageProcessingError",
    "LimitKind",
    "LoggingConfig",
    "ModelName",
    "ModelOptions",
    "ModelResponse",
    "OutputRule",
    "OutputT",
    "Phase",
    "PipelineDeployment",
    "PipelineLimit",
    "ProcessedImage",
    "PromptManager",
    "PromptSpec",
    "RemoteDeployment",
    "Role",
    "Rule",
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
    "disable_run_logger",
    "extract_result",
    "generate",
    "generate_structured",
    "get_document_store",
    "get_logger",
    "get_pipeline_logger",
    "get_run_context",
    "is_document_sha256",
    "llm",
    "pipeline_concurrency",
    "pipeline_flow",
    "pipeline_task",
    "prefect_test_harness",
    "process_image",
    "progress",
    "render_preview",
    "render_text",
    "reset_run_context",
    "run_remote_deployment",
    "sanitize_url",
    "send_spec",
    "set_document_store",
    "set_run_context",
    "set_trace_cost",
    "setup_logging",
    "trace",
]
