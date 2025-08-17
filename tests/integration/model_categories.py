"""Model categorization utilities for tests."""

from typing import cast, get_args

from ai_pipeline_core.llm.model_types import ModelName

# Get all models from the Literal type
ModelNameTuple = tuple[ModelName, ...]
ALL_MODELS: ModelNameTuple = cast(ModelNameTuple, get_args(ModelName))

CORE_MODELS: ModelNameTuple = tuple(model for model in ALL_MODELS if not model.endswith("-search"))
SEARCH_MODELS: ModelNameTuple = tuple(model for model in ALL_MODELS if model.endswith("-search"))


def is_core_model(model: ModelName) -> bool:
    """Check if a model is a core model."""
    return model not in SEARCH_MODELS


def is_search_model(model: ModelName) -> bool:
    """Check if a model is a search model."""
    return model in SEARCH_MODELS
