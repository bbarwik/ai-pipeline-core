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

ModelName: TypeAlias = Literal[
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
"""Type-safe model name identifiers.

@public

Provides compile-time validation and IDE autocompletion for supported
language model names. Used throughout the library to prevent typos
and ensure only valid models are referenced.

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

Extending with custom models:
    The generate functions accept any string, not just ModelName literals.
    To add custom models for type safety:
    1. Create a new type alias: CustomModel = Literal["my-model"]
    2. Use Union: model: ModelName | CustomModel = "my-model"
    3. Or simply use strings: model = "any-model-via-litellm"

Example:
    >>> from ai_pipeline_core.llm import generate, ModelName
    >>>
    >>> # Type-safe model selection
    >>> model: ModelName = "gpt-5"  # IDE autocomplete works
    >>> response = await generate(model, messages="Hello")
    >>>
    >>> # Also accepts string for custom models
    >>> response = await generate("custom-model-v2", messages="Hello")
    >>>
    >>> # Custom type safety
    >>> from typing import Literal
    >>> MyModel = Literal["company-llm-v1"]
    >>> model: ModelName | MyModel = "company-llm-v1"

Note:
    While the type alias provides suggestions for common models,
    the generate functions also accept string literals to support
    custom or newer models accessed via LiteLLM proxy.

See Also:
    - ai_pipeline_core.llm.generate: Main generation function
    - ai_pipeline_core.llm.ModelOptions: Model configuration
"""
