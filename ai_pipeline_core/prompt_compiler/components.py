"""Supporting class types for prompt specifications: Role, Rule, OutputRule, Guide."""

import sys
from pathlib import Path
from textwrap import dedent
from typing import Any, ClassVar

MAX_RULE_LINES = 5


def _require_docstring(cls: type, *, kind: str) -> None:
    """Validate that a class has a non-empty docstring."""
    if cls.__doc__ is None or not cls.__doc__.strip():
        raise TypeError(f"{kind} '{cls.__name__}' must define a non-empty docstring")


def _require_text(cls: type, *, kind: str, max_lines: int | None = None) -> None:
    """Validate that a class defines a non-empty `text` ClassVar, optionally capped at max_lines."""
    value = cls.__dict__.get("text")
    if not isinstance(value, str):
        raise TypeError(f"{kind} '{cls.__name__}' must define 'text' as a ClassVar[str]")
    normalized = dedent(value).strip()
    if not normalized:
        raise TypeError(f"{kind} '{cls.__name__}' has empty 'text'")
    if max_lines is not None and len(normalized.splitlines()) > max_lines:
        raise TypeError(f"{kind} '{cls.__name__}' text exceeds {max_lines} lines — use a Guide for longer content")
    cls.text = normalized


class Role:
    """Base class for LLM role definitions.

    Must define a non-empty docstring and a ``text`` ClassVar on every Role subclass.
    Must not end Role text with sentence punctuation (.!?) — the renderer adds a period automatically.
    """

    text: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _require_docstring(cls, kind="Role")
        _require_text(cls, kind="Role")
        if cls.text[-1] in ".!?":
            raise TypeError(f"Role '{cls.__name__}' text must not end with punctuation (the renderer adds a period automatically)")


class Rule:
    """Base class for behavioral constraints.

    Must define a non-empty docstring and a ``text`` ClassVar on every Rule subclass (max 5 lines).
    """

    text: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _require_docstring(cls, kind="Rule")
        _require_text(cls, kind="Rule", max_lines=MAX_RULE_LINES)


class OutputRule:
    """Base class for output formatting constraints.

    Must define a non-empty docstring and a ``text`` ClassVar on every OutputRule subclass (max 5 lines).
    """

    text: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _require_docstring(cls, kind="OutputRule")
        _require_text(cls, kind="OutputRule", max_lines=MAX_RULE_LINES)


class Guide:
    """Base class for reference material / methodology guides.

    Must define a non-empty docstring and a ``template`` ClassVar on every Guide subclass.
    Must use a relative path for Guide template — content is loaded and cached at import time.
    Never use ``#`` (H1) headers in Guide templates — reserved for prompt section boundaries. Use ``##`` or deeper.
    """

    template: ClassVar[str]
    _resolved_path: ClassVar[Path]
    _content: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _require_docstring(cls, kind="Guide")

        template = cls.__dict__.get("template")
        if not isinstance(template, str) or not template.strip():
            raise TypeError(f"Guide '{cls.__name__}' must define 'template' as a ClassVar[str]")
        if Path(template).is_absolute():
            raise TypeError(f"Guide '{cls.__name__}' template must be a relative path, got absolute")

        module = sys.modules.get(cls.__module__)
        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise TypeError(f"Guide '{cls.__name__}' cannot resolve module file for template validation")

        resolved = (Path(module_file).resolve().parent / template).resolve()
        if not resolved.is_file():
            raise TypeError(f"Guide '{cls.__name__}' template not found: {resolved}")

        cls._resolved_path = resolved

        # Read and cache content at import time
        content = resolved.read_text(encoding="utf-8")

        # Validate no H1 headers (reserved for prompt section boundaries)
        for line_num, line in enumerate(content.splitlines(), 1):
            if line.startswith("# ") and not line.startswith("## "):
                raise TypeError(
                    f"Guide '{cls.__name__}' template line {line_num} uses '# ' header which is reserved for prompt section boundaries — use '## ' or deeper"
                )

        cls._content = content

    @classmethod
    def render(cls) -> str:
        """Return the cached template file content."""
        return cls._content


__all__ = ["Guide", "OutputRule", "Role", "Rule"]
