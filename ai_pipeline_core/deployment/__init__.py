"""Pipeline deployment utilities for unified, type-safe deployments.

@public

This module provides the PipelineDeployment base class and related types
for creating pipeline deployments that work seamlessly across local testing,
CLI execution, and production Prefect deployments.

Example:
    >>> from ai_pipeline_core import PipelineDeployment, DeploymentContext, DeploymentResult
    >>>
    >>> class MyResult(DeploymentResult):
    ...     report: str
    >>>
    >>> class MyPipeline(PipelineDeployment[MyOptions, MyResult]):
    ...     flows = [step_01, step_02]
    ...
    ...     @staticmethod
    ...     def build_result(project_name, documents, options):
    ...         return MyResult(success=True, report="Done")
    >>>
    >>> pipeline = MyPipeline()
    >>> result = pipeline.run_local("test", documents, options)
"""

from .base import DeploymentContext, DeploymentResult, PipelineDeployment
from .contract import (
    CompletedRun,
    DeploymentResultData,
    FailedRun,
    PendingRun,
    ProgressRun,
    RunResponse,
)

__all__ = [
    "CompletedRun",
    "DeploymentContext",
    "DeploymentResult",
    "DeploymentResultData",
    "FailedRun",
    "PendingRun",
    "PipelineDeployment",
    "ProgressRun",
    "RunResponse",
]
