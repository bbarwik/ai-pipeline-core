"""Flow options base class for pipeline execution."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class FlowOptions(BaseSettings):
    """Base configuration for pipeline flows.

    Subclass to add flow-specific parameters. Uses pydantic-settings
    for environment variable overrides. Immutable after creation.
    """

    model_config = SettingsConfigDict(frozen=True, extra="allow")


__all__ = ["FlowOptions"]
