"""Flow options configuration for pipeline execution.

@public

Provides base configuration settings for AI pipeline flows,
including model selection and runtime parameters.
"""

from typing import TypeVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_pipeline_core.llm import ModelName

T = TypeVar("T", bound="FlowOptions")


class FlowOptions(BaseSettings):
    """Base configuration settings for AI pipeline flows.

    @public

    FlowOptions provides runtime configuration for pipeline flows,
    including model selection and other parameters. It uses pydantic-settings
    to support environment variable overrides and is immutable (frozen) by default.

    This class is designed to be subclassed for flow-specific configuration:

    Example:
        >>> class MyFlowOptions(FlowOptions):
        ...     temperature: float = Field(0.7, ge=0, le=2)
        ...     batch_size: int = Field(10, gt=0)
        ...     custom_param: str = "default"

        >>> # Use in CLI with run_cli:
        >>> run_cli(
        ...     flows=[my_flow],
        ...     options_cls=MyFlowOptions  # Will parse CLI args
        ... )

        >>> # Or create programmatically:
        >>> options = MyFlowOptions(
        ...     core_model="gemini-2.5-pro",
        ...     temperature=0.9
        ... )

    Attributes:
        core_model: Primary LLM for complex tasks (default: gpt-5)
        small_model: Fast model for simple tasks (default: gpt-5-mini)

    Configuration:
        - Frozen (immutable) after creation
        - Extra fields ignored (not strict)
        - Can be populated from environment variables
        - Used by simple_runner.cli for command-line parsing

    Note:
        The base class provides model selection. Subclasses should
        add flow-specific parameters with appropriate validation.
    """

    core_model: ModelName = Field(
        default="gpt-5",
        description="Primary model for complex analysis and generation tasks.",
    )
    small_model: ModelName = Field(
        default="gpt-5-mini",
        description="Fast, cost-effective model for simple tasks and orchestration.",
    )

    model_config = SettingsConfigDict(frozen=True, extra="ignore")


__all__ = ["FlowOptions"]
