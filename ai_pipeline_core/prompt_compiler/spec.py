"""PromptSpec base class with import-time validation."""

import typing
from textwrap import dedent
from typing import Any, ClassVar, Generic

from pydantic import BaseModel, ConfigDict
from pydantic.fields import FieldInfo
from typing_extensions import TypeVar

from ai_pipeline_core.documents import Document

from .components import Guide, OutputRule, Role, Rule
from .types import Phase

OutputT = TypeVar("OutputT", default=str)

_SPEC_KNOWN_ATTRS: frozenset[str] = frozenset({
    "phase",
    "input_documents",
    "role",
    "task",
    "guides",
    "rules",
    "output_rules",
    "output_structure",
    "xml_wrapped",
    "model_config",
})


def _check_no_duplicates(items: tuple[type, ...], *, attr: str, spec_name: str) -> None:
    """Reject duplicate entries in a spec tuple (rules, guides, etc.)."""
    seen: set[type] = set()
    for item in items:
        if item in seen:
            raise TypeError(f"PromptSpec '{spec_name}'.{attr} contains duplicate: {item.__name__}")
        seen.add(item)


def _check_unknown_attrs(cls: type, name: str) -> None:
    """Detect unknown class attributes that are likely typos.

    Runs during __init_subclass__ (before model_fields is populated), so uses
    cls.__annotations__ to identify Pydantic field declarations.
    """
    own_annotations = set(cls.__annotations__) if hasattr(cls, "__annotations__") else set()
    for attr_name in cls.__dict__:
        if attr_name.startswith("_"):
            continue
        if attr_name in _SPEC_KNOWN_ATTRS:
            continue
        if attr_name in own_annotations:
            continue
        val = cls.__dict__[attr_name]
        if callable(val) or isinstance(val, (classmethod, staticmethod, property)):
            continue
        raise TypeError(
            f"PromptSpec '{name}' has unknown attribute '{attr_name}'. Known spec attributes: {', '.join(sorted(_SPEC_KNOWN_ATTRS - {'model_config'}))}"
        )


def _check_field_descriptions(cls: type, name: str) -> None:
    """Validate that all Pydantic fields have Field(description=...).

    Uses cls.__annotations__ + cls.__dict__ directly because model_fields
    is not yet populated during __init_subclass__.
    """
    own_annotations = cls.__annotations__ if hasattr(cls, "__annotations__") else {}
    for field_name in own_annotations:
        if field_name in _SPEC_KNOWN_ATTRS:
            continue
        default = cls.__dict__.get(field_name)
        if isinstance(default, FieldInfo):
            if default.description is None:
                raise TypeError(f"PromptSpec '{name}' field '{field_name}' must use Field(description='...'). Bare Field() without description is not allowed.")
        else:
            raise TypeError(f"PromptSpec '{name}' field '{field_name}' must use Field(description='...'). Bare '{field_name}: ...' is not allowed.")


class PromptSpec(BaseModel, Generic[OutputT]):
    """Base class for all prompt specifications.

    Generic parameter ``OutputT`` determines the output type:
    - ``PromptSpec[str]`` (or just ``PromptSpec``, default) for text output
    - ``PromptSpec[MyModel]`` for structured output (MyModel must be a BaseModel subclass)

    Must subclass PromptSpec directly — no inheritance chains allowed.
    Must declare phase via class parameter: ``class MySpec(PromptSpec, phase=Phase('review'))``.
    Must define role, task, and input_documents on every PromptSpec subclass.
    Must use ``Field(description='...')`` for all dynamic Pydantic fields on PromptSpec subclasses.

    Required ClassVars: role, task, input_documents.
    Optional ClassVars: guides=(), rules=(), output_rules=(), output_structure=None, xml_wrapped=False.

    Pydantic fields (dynamic input values):
        Any field declared with ``Field(description=...)`` becomes a dynamic input.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: ClassVar[Phase]
    input_documents: ClassVar[tuple[type[Document], ...]]
    role: ClassVar[type[Role]]
    task: ClassVar[str]
    guides: ClassVar[tuple[type[Guide], ...]]
    rules: ClassVar[tuple[type[Rule], ...]]
    output_rules: ClassVar[tuple[type[OutputRule], ...]]
    output_type: ClassVar[type[str] | type[BaseModel]]
    output_structure: ClassVar[str | None]
    xml_wrapped: ClassVar[bool]

    def __init_subclass__(cls, *, phase: Phase | None = None, **kwargs: Any) -> None:  # noqa: C901, PLR0912, PLR0915
        super().__init_subclass__(**kwargs)

        # Pydantic creates concrete subclasses for parameterized generics (e.g. PromptSpec[str]).
        # These have names like "PromptSpec[str]" — skip validation for them.
        if "[" in cls.__name__:
            return

        name = cls.__name__

        # Block inheritance chains — must inherit directly from PromptSpec (or PromptSpec[T]).
        # Pydantic creates concrete classes for PromptSpec[T] with names like "PromptSpec[MyModel]".
        non_spec = [b.__name__ for b in cls.__bases__ if not (b is PromptSpec or (issubclass(b, PromptSpec) and "[" in b.__name__))]
        if non_spec or len(cls.__bases__) != 1:
            raise TypeError(f"PromptSpec '{name}' must inherit directly from PromptSpec, not from {', '.join(non_spec) or 'multiple bases'}")

        # Docstring required
        if cls.__doc__ is None or not cls.__doc__.strip():
            raise TypeError(f"PromptSpec '{name}' must define a non-empty docstring")

        # Phase required as class parameter
        if not isinstance(phase, str) or not phase.strip():
            raise TypeError(f"PromptSpec '{name}' must set phase: class {name}(PromptSpec, phase=Phase('review'))")
        cls.phase = Phase(phase)

        # Validate role
        if "role" not in cls.__dict__:
            raise TypeError(f"PromptSpec '{name}' must define 'role'")
        role = cls.__dict__["role"]
        if not isinstance(role, type) or not issubclass(role, Role):
            raise TypeError(f"PromptSpec '{name}'.role must be a Role subclass (class reference), got {role!r}")

        # Validate task
        if "task" not in cls.__dict__:
            raise TypeError(f"PromptSpec '{name}' must define 'task'")
        task = cls.__dict__["task"]
        if not isinstance(task, str):
            raise TypeError(f"PromptSpec '{name}'.task must be a string")
        cls.task = dedent(task).strip()
        if not cls.task:
            raise TypeError(f"PromptSpec '{name}'.task must not be empty")

        # Validate input_documents
        if "input_documents" not in cls.__dict__:
            raise TypeError(f"PromptSpec '{name}' must define 'input_documents'")
        input_docs = cls.__dict__["input_documents"]
        if not isinstance(input_docs, tuple):
            raise TypeError(f"PromptSpec '{name}'.input_documents must be a tuple of Document subclasses")
        for doc_cls in input_docs:
            if not isinstance(doc_cls, type) or not issubclass(doc_cls, Document):
                raise TypeError(f"PromptSpec '{name}'.input_documents contains non-Document class: {doc_cls!r}")
        _check_no_duplicates(input_docs, attr="input_documents", spec_name=name)

        # Derive output_type from generic parameter (PromptSpec[X] -> X)
        # Reject manual output_type declarations — the generic parameter is the source of truth
        if "output_type" in cls.__dict__:
            raise TypeError(
                f"PromptSpec '{name}' must not declare 'output_type' directly. "
                f"Use the generic parameter instead: class {name}(PromptSpec[MyModel], phase=Phase('...'))"
            )
        output_type: type[str] | type[BaseModel] = str  # default when no explicit generic arg
        # Check __orig_bases__ first (standard Python generic alias), then fall back
        # to parent's __pydantic_generic_metadata__ (Pydantic-resolved concrete class)
        for base in getattr(cls, "__orig_bases__", ()):
            origin = typing.get_origin(base)
            if origin is PromptSpec:
                args = typing.get_args(base)
                if args and args[0] is not str:
                    output_type = args[0]
                break
        else:
            # When inheriting from PromptSpec[X] (Pydantic concrete class),
            # the type info is in the parent's pydantic generic metadata
            for base in cls.__bases__:
                meta = getattr(base, "__pydantic_generic_metadata__", None)
                if meta and meta.get("origin") is PromptSpec and meta.get("args"):
                    arg = meta["args"][0]
                    if arg is not str:
                        output_type = arg
                    break
        if output_type is not str and not (isinstance(output_type, type) and issubclass(output_type, BaseModel)):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"PromptSpec '{name}' generic parameter must be 'str' or a BaseModel subclass, got {output_type!r}")
        cls.output_type = output_type

        # Validate guides (optional, default empty)
        guides = cls.__dict__.get("guides", ())
        if not isinstance(guides, tuple):
            raise TypeError(f"PromptSpec '{name}'.guides must be a tuple of Guide subclasses")
        for guide_cls in guides:
            if not isinstance(guide_cls, type) or not issubclass(guide_cls, Guide):
                raise TypeError(f"PromptSpec '{name}'.guides contains non-Guide class: {guide_cls!r}")
        _check_no_duplicates(guides, attr="guides", spec_name=name)
        cls.guides = guides

        # Validate rules (optional, default empty)
        rules = cls.__dict__.get("rules", ())
        if not isinstance(rules, tuple):
            raise TypeError(f"PromptSpec '{name}'.rules must be a tuple of Rule subclasses")
        for rule_cls in rules:
            if not isinstance(rule_cls, type) or not issubclass(rule_cls, Rule):
                if isinstance(rule_cls, type) and issubclass(rule_cls, OutputRule):
                    raise TypeError(
                        f"PromptSpec '{name}'.rules contains OutputRule '{rule_cls.__name__}'. Use output_rules= for output formatting constraints."
                    )
                raise TypeError(f"PromptSpec '{name}'.rules contains non-Rule class: {rule_cls!r}")
        _check_no_duplicates(rules, attr="rules", spec_name=name)
        cls.rules = rules

        # Validate output_rules (optional, default empty)
        output_rules = cls.__dict__.get("output_rules", ())
        if not isinstance(output_rules, tuple):
            raise TypeError(f"PromptSpec '{name}'.output_rules must be a tuple of OutputRule subclasses")
        for rule_cls in output_rules:
            if not isinstance(rule_cls, type) or not issubclass(rule_cls, OutputRule):
                if isinstance(rule_cls, type) and issubclass(rule_cls, Rule):
                    raise TypeError(f"PromptSpec '{name}'.output_rules contains Rule '{rule_cls.__name__}'. Use rules= for behavioral constraints.")
                raise TypeError(f"PromptSpec '{name}'.output_rules contains non-OutputRule class: {rule_cls!r}")
        _check_no_duplicates(output_rules, attr="output_rules", spec_name=name)
        cls.output_rules = output_rules

        # Validate output_structure (optional)
        output_structure = cls.__dict__.get("output_structure")
        if output_structure is not None:
            if cls.output_type is not str:
                raise TypeError(f"PromptSpec '{name}'.output_structure is only allowed when output_type is str")
            if not isinstance(output_structure, str):
                raise TypeError(f"PromptSpec '{name}'.output_structure must be a string")
            cls.output_structure = dedent(output_structure).strip()
            if not cls.output_structure:
                raise TypeError(f"PromptSpec '{name}'.output_structure must not be empty")
            for line in cls.output_structure.splitlines():
                if line.startswith("# ") and not line.startswith("## "):
                    raise TypeError(f"PromptSpec '{name}'.output_structure must not contain H1 headers ('# '). Use '## ' or deeper. Found: {line!r}")
        else:
            cls.output_structure = None

        # Validate xml_wrapped (optional, default False)
        xml_wrapped = cls.__dict__.get("xml_wrapped", False)
        if not isinstance(xml_wrapped, bool):
            raise TypeError(f"PromptSpec '{name}'.xml_wrapped must be a bool")
        if xml_wrapped and cls.output_type is not str:
            raise TypeError(f"PromptSpec '{name}'.xml_wrapped is only allowed with PromptSpec[str] (structured output uses JSON, not XML wrapping)")
        cls.xml_wrapped = xml_wrapped

        # Validate Pydantic field descriptions (uses __annotations__ + FieldInfo directly)
        _check_field_descriptions(cls, name)

        # Detect unknown class attributes (typos)
        _check_unknown_attrs(cls, name)


__all__ = ["OutputT", "PromptSpec"]
