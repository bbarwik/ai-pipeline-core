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
