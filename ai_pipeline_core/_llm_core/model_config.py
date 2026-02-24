"""Per-model-family configuration for LLM behavior.

Centralizes model-specific knowledge that was previously scattered as hardcoded
string checks across conversation.py, api.py, and client.py.
"""

from dataclasses import dataclass

# Image preset names matching the ImagePreset enum in llm/_images.py
IMAGE_PRESET_GEMINI = "gemini"
IMAGE_PRESET_CLAUDE = "claude"
IMAGE_PRESET_GPT4V = "gpt4v"
IMAGE_PRESET_DEFAULT = "default"

# Minimum context tokens for Gemini caching to be effective
GEMINI_CACHE_MIN_TOKENS = 10_000


@dataclass(frozen=True, slots=True)
class ModelFamilyConfig:
    """Configuration for a model family (e.g., gemini, claude, gpt)."""

    image_preset: str = IMAGE_PRESET_DEFAULT
    supports_stop_sequences: bool = False
    openrouter_provider: str | None = None
    cache_min_tokens: int = 0


_MODEL_FAMILIES: dict[str, ModelFamilyConfig] = {
    "gemini": ModelFamilyConfig(
        image_preset=IMAGE_PRESET_GEMINI,
        supports_stop_sequences=True,
        openrouter_provider="google",
        cache_min_tokens=GEMINI_CACHE_MIN_TOKENS,
    ),
    "claude": ModelFamilyConfig(
        image_preset=IMAGE_PRESET_CLAUDE,
        openrouter_provider="anthropic",
    ),
    "gpt": ModelFamilyConfig(
        image_preset=IMAGE_PRESET_GPT4V,
        openrouter_provider="openai",
    ),
    "grok": ModelFamilyConfig(
        openrouter_provider="x-ai",
    ),
    "qwen3": ModelFamilyConfig(
        openrouter_provider="qwen",
    ),
    "deepseek-": ModelFamilyConfig(
        openrouter_provider="deepseek",
    ),
    "glm-": ModelFamilyConfig(
        openrouter_provider="z-ai",
    ),
    "kimi-": ModelFamilyConfig(
        openrouter_provider="moonshotai",
    ),
}


def get_model_family_config(model: str) -> ModelFamilyConfig:
    """Look up configuration for a model by matching its name against known families.

    Matches the first family prefix found in the lowercased model name.
    Returns default config if no family matches.
    """
    model_lower = model.lower()
    for prefix, config in _MODEL_FAMILIES.items():
        if prefix in model_lower:
            return config
    return ModelFamilyConfig()


def get_image_preset(model: str) -> str:
    """Get the image preset name for a model."""
    return get_model_family_config(model).image_preset


def supports_stop_sequences(model: str) -> bool:
    """Check if a model supports stop sequences."""
    return get_model_family_config(model).supports_stop_sequences


def get_openrouter_provider(model: str) -> str | None:
    """Get the OpenRouter provider prefix for a model, or None if unknown."""
    return get_model_family_config(model).openrouter_provider


def get_cache_min_tokens(model: str) -> int:
    """Get minimum context tokens for caching to be effective. 0 means no minimum."""
    return get_model_family_config(model).cache_min_tokens


__all__ = [
    "ModelFamilyConfig",
    "get_cache_min_tokens",
    "get_image_preset",
    "get_model_family_config",
    "get_openrouter_provider",
    "supports_stop_sequences",
]
