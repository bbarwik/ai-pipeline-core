"""Flow options base class for pipeline execution."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class FlowOptions(BaseSettings):
    """Base configuration for pipeline flows. Uses pydantic-settings.

    Every field defined on a FlowOptions subclass is automatically
    overridable via environment variables (e.g., a field named 'mode'
    reads from the MODE env var at instantiation time).

    Use FlowOptions for deployment/environment configuration that may
    differ between environments (dev/staging/production).

    Never inherit from FlowOptions for task-level options, writer configs,
    or programmatically-constructed parameter objects — use BaseModel instead.
    FlowOptions fields are always subject to env var override, which causes
    silent, hard-to-debug behavior when field names collide with common
    env vars (MODE, HOST, PORT, etc.).
    """

    model_config = SettingsConfigDict(frozen=True, extra="forbid")


__all__ = ["FlowOptions"]
