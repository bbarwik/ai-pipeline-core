"""Prompt compiler for type-safe, validated prompt specifications.

Every piece of prompt content is either a class or a class attribute.
"""

from .components import Guide, OutputRule, Role, Rule
from .render import render_preview, render_text
from .spec import MultiLineField, OutputT, PromptSpec

__all__ = [
    "Guide",
    "MultiLineField",
    "OutputRule",
    "OutputT",
    "PromptSpec",
    "Role",
    "Rule",
    "render_preview",
    "render_text",
]
