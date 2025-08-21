from typing import TypeVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_pipeline_core.llm import ModelName

T = TypeVar("T", bound="FlowOptions")


class FlowOptions(BaseSettings):
    """Base configuration for AI Pipeline flows."""

    core_model: ModelName | str = Field(
        default="gpt-5",
        description="Primary model for complex analysis and generation tasks.",
    )
    small_model: ModelName | str = Field(
        default="gpt-5-mini",
        description="Fast, cost-effective model for simple tasks and orchestration.",
    )

    model_config = SettingsConfigDict(frozen=True, extra="ignore")


__all__ = ["FlowOptions"]
