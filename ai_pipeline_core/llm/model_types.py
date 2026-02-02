"""Model type definitions for LLM interactions.

This module defines type aliases for model names used throughout
the AI Pipeline Core system. The ModelName type provides type safety
and IDE support for supported model identifiers.

Model categories:
- Core models: High-capability general-purpose models
- Small models: Efficient, cost-effective models
- Search models: Models with web search capabilities
"""

from typing import Literal

type ModelName = (
    Literal[
        # Core models
        "gemini-3-pro",
        "gpt-5.1",
        # Small models
        "gemini-3-flash",
        "gpt-5-mini",
        "grok-4.1-fast",
        # Search models
        "gemini-3-flash-search",
        "gpt-5-mini-search",
        "grok-4.1-fast-search",
        "sonar-pro-search",
    ]
    | str
)
"""Type-safe model name identifiers with support for custom models.

Provides IDE autocompletion for common model names while allowing any
string for custom models. The type is a union of predefined literals
and str, giving you the best of both worlds: suggestions for known
models and flexibility for custom ones.

These are example common model names as of Q1 2026. Actual availability
depends on your LiteLLM proxy configuration and provider access.

Model categories:
    Core models (gemini-3-pro, gpt-5.1):
        High-capability models for complex tasks requiring deep reasoning,
        nuanced understanding, or creative generation.

    Small models (gemini-3-flash, gpt-5-mini, grok-4.1-fast):
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

The ModelName type includes both predefined literals and str,
allowing full flexibility while maintaining IDE support for
common models.
"""
