"""Model type definitions for LLM interactions.

This module defines type aliases for model names used throughout
the AI Pipeline Core system. The ModelName type provides type safety
and IDE support for supported model identifiers.

Model categories:
- Core models: High-capability general-purpose models
- Small models: Efficient, cost-effective models
- Search models: Models with web search capabilities
"""

from typing import Literal, TypeAlias

ModelName: TypeAlias = (
    Literal[
        # Core models
        "gemini-2.5-pro",
        "gpt-5",
        "grok-4",
        # Small models
        "gemini-2.5-flash",
        "gpt-5-mini",
        "grok-3-mini",
        # Search models
        "gemini-2.5-flash-search",
        "sonar-pro-search",
        "gpt-4o-search",
        "grok-3-mini-search",
    ]
    | str
)
"""Type-safe model name identifiers with support for custom models.

@public

Provides IDE autocompletion for common model names while allowing any
string for custom models. The type is a union of predefined literals
and str, giving you the best of both worlds: suggestions for known
models and flexibility for custom ones.

Note: These are example common model names as of Q3 2025. Actual availability
depends on your LiteLLM proxy configuration and provider access.

Model categories:
    Core models (gemini-2.5-pro, gpt-5, grok-4):
        High-capability models for complex tasks requiring deep reasoning,
        nuanced understanding, or creative generation.

    Small models (gemini-2.5-flash, gpt-5-mini, grok-3-mini):
        Efficient models optimized for speed and cost, suitable for
        simpler tasks or high-volume processing.

    Search models (*-search suffix):
        Models with integrated web search capabilities for retrieving
        and synthesizing current information.

Using custom models:
    ModelName now includes str, so you can use any model name directly:
    - Predefined models get IDE autocomplete and validation
    - Custom models work seamlessly as strings
    - No need for Union types or additional type aliases

Example:
    >>> from ai_pipeline_core import llm, ModelName
    >>>
    >>> # Predefined model with IDE autocomplete
    >>> model: ModelName = "gpt-5"  # IDE suggests common models
    >>> response = await llm.generate(model, messages="Hello")
    >>>
    >>> # Custom model works directly
    >>> model: ModelName = "custom-model-v2"  # Any string is valid
    >>> response = await llm.generate(model, messages="Hello")
    >>>
    >>> # Both types work seamlessly
    >>> models: list[ModelName] = ["gpt-5", "custom-llm", "gemini-2.5-pro"]

Note:
    The ModelName type includes both predefined literals and str,
    allowing full flexibility while maintaining IDE support for
    common models.

See Also:
    - llm.generate: Main generation function
    - ModelOptions: Model configuration options
"""
