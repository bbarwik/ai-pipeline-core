"""PromptSpec base class with import-time validation."""

import re
import typing
from collections.abc import Mapping
from textwrap import dedent
from typing import Any, ClassVar, Generic, cast

from pydantic import BaseModel, ConfigDict
from pydantic.fields import FieldInfo
from typing_extensions import TypeVar

from ai_pipeline_core.documents import Document

from .components import Guide, OutputRule, Role, Rule

OutputT = TypeVar("OutputT", default=str)

_XML_TAG_PATTERN = re.compile(r"</?[a-zA-Z]\w*[\s>/]")

_SPEC_KNOWN_ATTRS: frozenset[str] = frozenset({
    "follows",
    "input_documents",
    "role",
    "task",
    "output_type",
    "guides",
    "rules",
    "output_rules",
    "output_structure",
    "model_config",
})


def _check_no_duplicates(items: tuple[type, ...], *, attr: str, spec_name: str) -> None:
    """Reject duplicate entries in a spec tuple (rules, guides, etc.)."""
    seen: set[type] = set()
    for item in items:
        if item in seen:
            raise TypeError(f"PromptSpec '{spec_name}'.{attr} contains duplicate: {item.__name__}")
        seen.add(item)


def _validate_component_tuple(
    cls_dict: Mapping[str, Any],
    name: str,
    attr: str,
    expected_type: type,
    *,
    cross_check: type | None = None,
    cross_attr: str | None = None,
) -> tuple[type, ...]:
    """Validate a tuple of component class references (guides, rules, output_rules)."""
    items = cls_dict.get(attr, ())
    if not isinstance(items, tuple):
        raise TypeError(f"PromptSpec '{name}'.{attr} must be a tuple of {expected_type.__name__} subclasses")
    for item in cast(tuple[Any, ...], items):
        if not isinstance(item, type) or not issubclass(item, expected_type):
            if cross_check and isinstance(item, type) and issubclass(item, cross_check):
                raise TypeError(
                    f"PromptSpec '{name}'.{attr} contains {cross_check.__name__} '{item.__name__}'. "
                    f"Use {cross_attr}= for {'output formatting' if cross_check is OutputRule else 'behavioral'} constraints."
                )
            raise TypeError(f"PromptSpec '{name}'.{attr} contains non-{expected_type.__name__} class: {item!r}")
    validated = cast(tuple[type, ...], items)
    _check_no_duplicates(validated, attr=attr, spec_name=name)
    return validated


def _check_unknown_attrs(cls: type, name: str) -> None:
    """Detect unknown class attributes that are likely typos.

    Runs during __init_subclass__ (before model_fields is populated), so uses
    cls.__annotations__ to identify Pydantic field declarations.
    """
    own_annotations: set[str] = set(cls.__annotations__) if hasattr(cls, "__annotations__") else set()
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
    Must define task on every PromptSpec subclass.
    Must define role and input_documents on standalone specs (not required when ``follows`` is set).
    Must use ``Field(description='...')`` for all dynamic Pydantic fields on PromptSpec subclasses.
    Must include all Guides that define terminology referenced in the task text — missing Guides
    cause the LLM to hallucinate definitions for framework-specific terms.
    Must ensure task vocabulary matches output model field names — when task text uses
    domain-specific terms but the output BaseModel has generic field names, add explicit
    mapping instructions in the task text.

    Required ClassVars: task. Also role and input_documents unless ``follows`` is set.
    Optional ClassVars: guides=(), rules=(), output_rules=(), output_structure=None.

    Class keyword parameter ``follows`` declares this spec as a follow-up to another spec.
    When ``follows`` is set, ``role`` and ``input_documents`` become optional (default to
    None and () respectively).

    ``input_documents`` declares Document types this spec expects in context. These are class
    references (types), not instances. Actual Document instances are passed via
    ``Conversation.send_spec(documents=[...])``.

    ``output_structure`` automatically enables ``<result>`` XML wrapping and auto-extraction
    in ``send_spec()``.

    Never construct XML manually (f-string ``<document>`` tags) — the framework wraps Documents
    in XML automatically when they are added to the Conversation via ``with_context()`` or
    ``with_document()``. Use ``Document.create()`` to wrap dicts, lists, or BaseModel instances.

    Pydantic fields (dynamic input values):
        Any field declared with ``Field(description=...)`` becomes a dynamic input.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    follows: ClassVar[type["PromptSpec"] | None]
    input_documents: ClassVar[tuple[type[Document], ...]]
    role: ClassVar[type[Role] | None]
    task: ClassVar[str]
    guides: ClassVar[tuple[type[Guide], ...]]
    rules: ClassVar[tuple[type[Rule], ...]]
    output_rules: ClassVar[tuple[type[OutputRule], ...]]
    output_type: ClassVar[type[str] | type[BaseModel]]
    output_structure: ClassVar[str | None]

    def __init_subclass__(cls, *, follows: type["PromptSpec"] | None = None, **kwargs: Any) -> None:  # noqa: C901, PLR0912, PLR0915
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

        # Validate follows (runtime check — users may pass invalid types despite annotation)
        if follows is not None:
            # Cast to Any for runtime validation (callers may bypass type annotations)
            follows_raw: Any = follows
            if not isinstance(follows_raw, type) or not issubclass(follows_raw, PromptSpec):
                raise TypeError(f"PromptSpec '{name}'.follows must be a PromptSpec subclass, got {follows_raw!r}")
            if follows is PromptSpec:
                raise TypeError(f"PromptSpec '{name}'.follows must be a concrete PromptSpec subclass, not PromptSpec itself")
            if "[" in follows.__name__:
                raise TypeError(f"PromptSpec '{name}'.follows must be a concrete PromptSpec subclass, not a parameterized generic")
        cls.follows = follows

        # Validate role (required for standalone specs, optional for follow-ups)
        if "role" not in cls.__dict__:
            if follows is None:
                raise TypeError(f"PromptSpec '{name}' must define 'role'")
            cls.role = None
        else:
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

        # Validate input_documents (required for standalone specs, optional for follow-ups)
        if "input_documents" not in cls.__dict__:
            if follows is None:
                raise TypeError(f"PromptSpec '{name}' must define 'input_documents'")
            cls.input_documents = ()
        else:
            input_docs = cls.__dict__["input_documents"]
            if not isinstance(input_docs, tuple):
                raise TypeError(f"PromptSpec '{name}'.input_documents must be a tuple of Document subclasses")
            for doc_cls in cast(tuple[Any, ...], input_docs):
                if not isinstance(doc_cls, type) or not issubclass(doc_cls, Document):
                    raise TypeError(f"PromptSpec '{name}'.input_documents contains non-Document class: {doc_cls!r}")
            _check_no_duplicates(cast(tuple[type[Document], ...], input_docs), attr="input_documents", spec_name=name)

        # Derive output_type from generic parameter (PromptSpec[X] -> X)
        # Reject manual output_type declarations — the generic parameter is the source of truth
        if "output_type" in cls.__dict__:
            raise TypeError(
                f"PromptSpec '{name}' must not declare 'output_type' directly. Use the generic parameter instead: class {name}(PromptSpec[MyModel])"
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

        # Validate component tuples (optional, default empty)
        cls.guides = _validate_component_tuple(cls.__dict__, name, "guides", Guide)
        cls.rules = _validate_component_tuple(cls.__dict__, name, "rules", Rule, cross_check=OutputRule, cross_attr="output_rules")
        cls.output_rules = _validate_component_tuple(cls.__dict__, name, "output_rules", OutputRule, cross_check=Rule, cross_attr="rules")

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

        # Validate OutputRules don't reference XML tags when output_structure is set
        if cls.output_structure is not None and cls.output_rules:
            for or_cls in cast(tuple[Any, ...], cls.output_rules):
                if _XML_TAG_PATTERN.search(str(or_cls.text)):
                    raise TypeError(
                        f"PromptSpec '{name}' has output_structure with OutputRule "
                        f"'{or_cls.__name__}' that references XML tags. "
                        f"output_structure automatically adds <result> wrapping — remove XML instructions from the OutputRule."
                    )

        # Validate Pydantic field descriptions (uses __annotations__ + FieldInfo directly)
        _check_field_descriptions(cls, name)

        # Detect unknown class attributes (typos)
        _check_unknown_attrs(cls, name)


__all__ = ["OutputT", "PromptSpec"]
