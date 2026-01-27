"""Unified pipeline run response contract.

@public

Single source of truth for the response shape used by both
webhook push (ai-pipeline-core) and polling pull (unified-middleware).
"""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Discriminator


class _RunBase(BaseModel):
    """Common fields on every run response variant."""

    type: str
    flow_run_id: UUID
    project_name: str
    state: str  # PENDING, RUNNING, COMPLETED, FAILED, CRASHED, CANCELLED
    timestamp: datetime
    storage_uri: str = ""

    model_config = ConfigDict(frozen=True)


class PendingRun(_RunBase):
    """Pipeline queued or running but no progress reported yet."""

    type: Literal["pending"] = "pending"  # pyright: ignore[reportIncompatibleVariableOverride]


class ProgressRun(_RunBase):
    """Pipeline running with step-level progress data."""

    type: Literal["progress"] = "progress"  # pyright: ignore[reportIncompatibleVariableOverride]
    step: int
    total_steps: int
    flow_name: str
    status: str  # "started", "completed", "cached"
    progress: float  # overall 0.0–1.0
    step_progress: float  # within step 0.0–1.0
    message: str


class DeploymentResultData(BaseModel):
    """Typed result payload — always has success + optional error."""

    success: bool
    error: str | None = None

    model_config = ConfigDict(frozen=True, extra="allow")


class CompletedRun(_RunBase):
    """Pipeline finished (Prefect COMPLETED). Check result.success for business outcome."""

    type: Literal["completed"] = "completed"  # pyright: ignore[reportIncompatibleVariableOverride]
    result: DeploymentResultData


class FailedRun(_RunBase):
    """Pipeline crashed — execution error, not business logic."""

    type: Literal["failed"] = "failed"  # pyright: ignore[reportIncompatibleVariableOverride]
    error: str
    result: DeploymentResultData | None = None


RunResponse = Annotated[
    PendingRun | ProgressRun | CompletedRun | FailedRun,
    Discriminator("type"),
]

__all__ = [
    "CompletedRun",
    "DeploymentResultData",
    "FailedRun",
    "PendingRun",
    "ProgressRun",
    "RunResponse",
]
