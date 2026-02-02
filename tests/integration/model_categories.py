"""Model categorization utilities for tests."""

from typing import cast, get_args

from ai_pipeline_core.llm.model_types import ModelName

# Get all models from the Literal type
# ModelName is now Literal[...] | str, so we need to extract the Literal part
ModelNameTuple = tuple[str, ...]
model_name_args = get_args(ModelName.__value__)  # Returns (Literal[...], str)

if model_name_args and hasattr(model_name_args[0], "__args__"):
    # First element is the Literal, get its args
    all_models = cast(ModelNameTuple, get_args(model_name_args[0]))
else:
    # Fallback if structure changes
    all_models = ()

ALL_MODELS: ModelNameTuple = all_models

CORE_MODELS: ModelNameTuple = tuple(model for model in ALL_MODELS if not model.endswith("-search"))
SEARCH_MODELS: ModelNameTuple = tuple(model for model in ALL_MODELS if model.endswith("-search"))


def is_core_model(model: str) -> bool:
    """Check if a model is a core model."""
    return model not in SEARCH_MODELS


def is_search_model(model: str) -> bool:
    """Check if a model is a search model."""
    return model in SEARCH_MODELS
