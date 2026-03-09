"""Flow options base class for pipeline execution."""

from pydantic import BaseModel, ConfigDict


class FlowOptions(BaseModel):
    """Base configuration for pipeline flows.

    Use FlowOptions for deployment/environment configuration that may
    differ between environments (dev/staging/production).

    Never inherit from FlowOptions for task-level options, writer configs,
    or programmatically-constructed parameter objects — use BaseModel instead.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


__all__ = ["FlowOptions"]
