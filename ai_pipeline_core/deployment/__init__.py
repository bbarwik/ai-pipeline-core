"""Pipeline deployment utilities for unified, type-safe deployments."""

from ._resolve import (
    AttachmentInput,
    DocumentInput,
    OutputAttachment,
    OutputDocument,
    build_output_document,
    resolve_document_inputs,
)
from .base import DeploymentContext, DeploymentResult, PipelineDeployment
from .contract import (
    CompletedRun,
    DeploymentResultData,
    FailedRun,
    PendingRun,
    ProgressRun,
    RunResponse,
)
from .progress import update as progress_update
from .remote import ProgressCallback, RemoteDeployment

__all__ = [
    "AttachmentInput",
    "CompletedRun",
    "DeploymentContext",
    "DeploymentResult",
    "DeploymentResultData",
    "DocumentInput",
    "FailedRun",
    "OutputAttachment",
    "OutputDocument",
    "PendingRun",
    "PipelineDeployment",
    "ProgressCallback",
    "ProgressRun",
    "RemoteDeployment",
    "RunResponse",
    "build_output_document",
    "progress_update",
    "resolve_document_inputs",
]
