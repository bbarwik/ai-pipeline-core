"""Prompt compiler for type-safe, validated prompt specifications.

Replaces Jinja2 templates and PromptManager with typed Python classes.
Every piece of prompt content is either a class or a class attribute.
"""

from .api import extract_result, send_spec
from .components import Guide, OutputRule, Role, Rule
from .render import render_preview, render_text
from .spec import OutputT, PromptSpec
from .types import Phase

__all__ = [
    "Guide",
    "OutputRule",
    "OutputT",
    "Phase",
    "PromptSpec",
    "Role",
    "Rule",
    "extract_result",
    "render_preview",
    "render_text",
    "send_spec",
]
