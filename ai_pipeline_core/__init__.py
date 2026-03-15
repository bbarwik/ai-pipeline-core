"""AI Pipeline Core - Production-ready framework for building AI pipelines with LLMs."""

import importlib.metadata
import os
import sys

# Disable Prefect telemetry and analytics before importing Prefect.
# Execution tracking is handled by the framework database.
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("PREFECT_CLOUD_ENABLE_ORCHESTRATION_TELEMETRY", "false")

from prefect.context import refresh_global_settings_context
from prefect.settings import get_current_settings

# If Prefect was already imported (user imported it before us), refresh its cached settings.
if "prefect" in sys.modules and get_current_settings().cloud.enable_orchestration_telemetry:
    refresh_global_settings_context()

from . import llm
from ._codec import CodecError, UniversalCodec
from .database import DatabaseReader
from .deployment import DeploymentResult, PipelineDeployment
from .deployment.remote import RemoteDeployment
from .documents import (
    Attachment,
    Document,
    DocumentSha256,
    ensure_extension,
    find_all,
    find_document,
    find_latest,
    is_document_sha256,
    replace_extension,
    sanitize_url,
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
    ModelName,
    ModelOptions,
    TokenUsage,
    Tool,
    ToolOutput,
)
from .logger import (
    LoggingConfig,
    get_pipeline_logger,
    setup_logging,
)
from .pipeline import (
    FlowOptions,
    LimitKind,
    PipelineFlow,
    PipelineLimit,
    PipelineTask,
    TaskBatch,
    TaskHandle,
    add_cost,
    as_task_completed,
    collect_tasks,
    get_run_id,
    pipeline_concurrency,
    pipeline_test_context,
    run_tasks_until,
    safe_gather,
    safe_gather_indexed,
    traced_operation,
)
from .prompt_compiler import Guide, ListField, MultiLineField, OutputRule, OutputT, PromptSpec, Role, Rule, StructuredField, render_preview, render_text
from .replay import ExperimentOverrides, ExperimentResult, execute_span, experiment_batch, experiment_span
from .settings import Settings
from .testing import disable_run_logger, prefect_test_harness

__version__ = importlib.metadata.version("ai-pipeline-core")

__all__ = [
    "Attachment",
    "Citation",
    "CodecError",
    "Conversation",
    "DatabaseReader",
    "DeploymentResult",
    "Document",
    "DocumentNameError",
    "DocumentSha256",
    "DocumentSizeError",
    "DocumentValidationError",
    "ExperimentOverrides",
    "ExperimentResult",
    "FlowOptions",
    "Guide",
    "LLMError",
    "LimitKind",
    "ListField",
    "LoggingConfig",
    "ModelName",
    "ModelOptions",
    "MultiLineField",
    "OutputDegenerationError",
    "OutputRule",
    "OutputT",
    "PipelineCoreError",
    "PipelineDeployment",
    "PipelineFlow",
    "PipelineLimit",
    "PipelineTask",
    "PromptSpec",
    "RemoteDeployment",
    "Role",
    "Rule",
    "Settings",
    "StructuredField",
    "TaskBatch",
    "TaskHandle",
    "TokenUsage",
    "Tool",
    "ToolOutput",
    "UniversalCodec",
    "add_cost",
    "as_task_completed",
    "collect_tasks",
    "disable_run_logger",
    "ensure_extension",
    "execute_span",
    "experiment_batch",
    "experiment_span",
    "find_all",
    "find_document",
    "find_latest",
    "get_pipeline_logger",
    "get_run_id",
    "is_document_sha256",
    "llm",
    "pipeline_concurrency",
    "pipeline_test_context",
    "prefect_test_harness",
    "render_preview",
    "render_text",
    "replace_extension",
    "run_tasks_until",
    "safe_gather",
    "safe_gather_indexed",
    "sanitize_url",
    "setup_logging",
    "traced_operation",
]
