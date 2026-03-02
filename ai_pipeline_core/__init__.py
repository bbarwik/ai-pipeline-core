"""AI Pipeline Core - Production-ready framework for building AI pipelines with LLMs."""

import importlib.metadata
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
from .deployment import DeploymentResult, PipelineDeployment, progress
from .deployment.remote import RemoteDeployment, run_remote_deployment
from .document_store import DocumentReader, get_document_store
from .documents import (
    Attachment,
    Document,
    DocumentSha256,
    RunContext,
    RunScope,
    ensure_extension,
    find_document,
    get_run_context,
    is_document_sha256,
    replace_extension,
    reset_run_context,
    sanitize_url,
    set_run_context,
)
from .exceptions import (
    DocumentNameError,
    DocumentSizeError,
    DocumentValidationError,
    LLMError,
    OutputDegenerationError,
    PipelineCoreError,
)
from .llm import (
    Citation,
    Conversation,
    ConversationContent,
    ModelName,
    ModelOptions,
    TokenUsage,
)
from .logging import (
    LoggingConfig,
    get_pipeline_logger,
    setup_logging,
)
from .observability.tracing import TraceInfo, TraceLevel, set_trace_cost, trace
from .pipeline import FlowOptions, LimitKind, PipelineLimit, pipeline_concurrency, pipeline_flow, pipeline_task, safe_gather, safe_gather_indexed
from .prompt_compiler import Guide, MultiLineField, OutputRule, OutputT, PromptSpec, Role, Rule, render_preview, render_text
from .settings import Settings
from .testing import disable_run_logger, prefect_test_harness

__version__ = importlib.metadata.version("ai-pipeline-core")

__all__ = [
    "Attachment",
    "Citation",
    "Conversation",
    "ConversationContent",
    "DeploymentResult",
    "Document",
    "DocumentNameError",
    "DocumentReader",
    "DocumentSha256",
    "DocumentSizeError",
    "DocumentValidationError",
    "FlowOptions",
    "Guide",
    "LLMError",
    "LimitKind",
    "LoggingConfig",
    "ModelName",
    "ModelOptions",
    "MultiLineField",
    "OutputDegenerationError",
    "OutputRule",
    "OutputT",
    "PipelineCoreError",
    "PipelineDeployment",
    "PipelineLimit",
    "PromptSpec",
    "RemoteDeployment",
    "Role",
    "Rule",
    "RunContext",
    "RunScope",
    "Settings",
    "TokenUsage",
    "TraceInfo",
    "TraceLevel",
    "disable_run_logger",
    "ensure_extension",
    "find_document",
    "get_document_store",
    "get_pipeline_logger",
    "get_run_context",
    "is_document_sha256",
    "llm",
    "pipeline_concurrency",
    "pipeline_flow",
    "pipeline_task",
    "prefect_test_harness",
    "progress",
    "render_preview",
    "render_text",
    "replace_extension",
    "reset_run_context",
    "run_remote_deployment",
    "safe_gather",
    "safe_gather_indexed",
    "sanitize_url",
    "set_run_context",
    "set_trace_cost",
    "setup_logging",
    "trace",
]
