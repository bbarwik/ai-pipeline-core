# MODULE: prompt_compiler
# CLASSES: Role, Rule, OutputRule, Guide, PromptSpec
# DEPENDS: BaseModel, Generic
# PURPOSE: Prompt compiler for type-safe, validated prompt specifications.
# VERSION: 0.10.0
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import Guide, OutputRule, PromptSpec, Role, Rule, render_preview, render_text
```

## Rules

1. Must define a non-empty docstring and a ``text`` ClassVar on every Role subclass.
2. Must not end Role text with sentence punctuation (.!?) — the renderer adds a period automatically.
3. Must use domain-neutral Roles for specs that handle multiple domains — a PromptSpec
4. Must define a non-empty docstring and a ``text`` ClassVar on every Rule subclass (max 5 lines).
5. Must define a non-empty docstring and a ``text`` ClassVar on every OutputRule subclass (max 5 lines).
6. Must define a non-empty docstring and a ``template`` ClassVar on every Guide subclass.
7. Must use a relative path for Guide template — content is loaded and cached at import time.
8. Never use ``#`` (H1) headers in Guide templates — reserved for prompt section boundaries. Use ``##`` or deeper.
9. Must subclass PromptSpec directly — no inheritance chains allowed.
10. Must define task on every PromptSpec subclass.
11. Must define role and input_documents on standalone specs (not required when ``follows`` is set).
12. Must use ``Field(description='...')`` for all dynamic Pydantic fields on PromptSpec subclasses.
13. Must include all Guides that define terminology referenced in the task text — missing Guides
14. Must ensure task vocabulary matches output model field names — when task text uses
15. Never construct XML manually (f-string ``<document>`` tags) — the framework wraps Documents

## Types & Constants

```python
APPROX_CHARS_PER_TOKEN = 4

MAX_RULE_LINES = 5

RESULT_TAG = "result"

RESULT_OPEN = f"<{RESULT_TAG}>"

RESULT_CLOSE = f"</{RESULT_TAG}>"

```

## Public API

```python
class Role:
    """Base class for LLM role definitions.

Must define a non-empty docstring and a ``text`` ClassVar on every Role subclass.
Must not end Role text with sentence punctuation (.!?) — the renderer adds a period automatically.
Must use domain-neutral Roles for specs that handle multiple domains — a PromptSpec
parameterized by domain (e.g., finding_type field that can be "risk", "opportunity",
or "question") needs a Role that doesn't bias toward any single domain."""
    text: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _require_docstring(cls, kind="Role")
        _require_text(cls, kind="Role")
        if cls.text[-1] in ".!?":
            raise TypeError(f"Role '{cls.__name__}' text must not end with punctuation (the renderer adds a period automatically)")


class Rule:
    """Base class for behavioral constraints.

Must define a non-empty docstring and a ``text`` ClassVar on every Rule subclass (max 5 lines)."""
    text: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _init_text_component(cls, "Rule", max_lines=MAX_RULE_LINES)


class OutputRule:
    """Base class for output formatting constraints.

Must define a non-empty docstring and a ``text`` ClassVar on every OutputRule subclass (max 5 lines)."""
    text: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _init_text_component(cls, "OutputRule", max_lines=MAX_RULE_LINES)


class Guide:
    """Base class for reference material / methodology guides.

Must define a non-empty docstring and a ``template`` ClassVar on every Guide subclass.
Must use a relative path for Guide template — content is loaded and cached at import time.
Never use ``#`` (H1) headers in Guide templates — reserved for prompt section boundaries. Use ``##`` or deeper."""
    template: ClassVar[str]

    @classmethod
    def render(cls) -> str:
        """Return the cached template file content."""
        return cls._content

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
    Any field declared with ``Field(description=...)`` becomes a dynamic input."""
    model_config = ConfigDict(frozen=True, extra='forbid')
    follows: ClassVar[type['PromptSpec'] | None]
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


```

## Functions

```python
def main(argv: list[str] | None = None) -> int:
    """CLI entry point for prompt compiler operations."""
    parser = argparse.ArgumentParser(prog="prompt_compiler", description="Prompt compiler CLI")
    subparsers = parser.add_subparsers(dest="command")

    # inspect
    inspect_parser = subparsers.add_parser("inspect", help="Show detailed anatomy of a single spec")
    inspect_parser.add_argument("spec", help="Spec class name or module.path:ClassName")
    inspect_parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root for class discovery")

    # render
    render_parser = subparsers.add_parser("render", help="Render a prompt preview with placeholder values")
    render_parser.add_argument("spec", help="Spec class name or module.path:ClassName")
    render_parser.add_argument("--no-input-documents", action="store_true", help="Hide input document listing")
    render_parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root for class discovery")

    # compile (also discovers and lists specs)
    compile_parser = subparsers.add_parser("compile", help="Discover, list, and compile all specs to .prompts/")
    compile_parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root for class discovery")

    args = parser.parse_args(argv)

    handlers = {"inspect": _cmd_inspect, "render": _cmd_render, "compile": _cmd_compile}
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)

def render_text(
    spec: PromptSpec,
    *,
    documents: list[Document] | None = None,
    include_input_documents: bool = True,
) -> str:
    """Render a PromptSpec instance to prompt text.

    Rendering order: Role -> Context -> Task -> Rules -> Guides -> Output Rules -> Output Structure.
    Uses `#` (H1) headers for section boundaries.

    When `documents` is provided, renders actual document instances with their
    runtime id, name, and description. Otherwise falls back to class-level info.
    """
    spec_cls = type(spec)
    sections: list[str] = []

    # 1. Role (skipped when None, e.g. follow-up specs without explicit role)
    if spec_cls.role is not None:
        sections.append(f"# Role\n\n{_role_sentence(spec_cls.role.text)}")

    # 2. Context: document listing + dynamic parameter values
    context_parts: list[str] = []
    if include_input_documents and (documents or spec_cls.input_documents):
        if documents is not None:
            context_parts.append(_render_documents_actual(documents))
        else:
            context_parts.append(_render_documents_preview(spec_cls))

    for field_name, field_info in spec_cls.model_fields.items():
        value = getattr(spec, field_name)
        label = field_info.description or field_name
        context_parts.append(f"**{label}:**\n{value}")

    if context_parts:
        sections.append("# Context\n\n" + "\n\n".join(context_parts))

    # 3. Task
    sections.append(f"# Task\n\n{spec_cls.task}")

    # 4. Rules
    if spec_cls.rules:
        rule_lines = [_format_numbered_rule(i, rule_cls.text) for i, rule_cls in enumerate(spec_cls.rules, 1)]
        sections.append("# Rules\n\n" + "\n".join(rule_lines))

    # 5. Guides (each gets its own section)
    for guide_cls in spec_cls.guides:
        title = _pascal_to_title(guide_cls.__name__)
        content = guide_cls.render().strip()
        # Strip duplicate title if guide template starts with the same title
        content_lines = content.splitlines()
        if content_lines and content_lines[0].strip().lower() == title.lower():
            content = "\n".join(content_lines[1:]).strip()
        sections.append(f"# Reference: {title}\n\n{content}")

    # 6. Output rules (before structure — tell the LLM constraints before format)
    or_lines = [_format_numbered_rule(i, rule_cls.text) for i, rule_cls in enumerate(spec_cls.output_rules, 1)]
    if spec_cls.output_structure is not None:
        or_lines.append(_format_numbered_rule(len(or_lines) + 1, _RESULT_TAG_RULE))
    if or_lines:
        sections.append("# Output Rules\n\n" + "\n".join(or_lines))

    # 7. Output structure
    if spec_cls.output_structure:
        sections.append("# Output Structure\n\n" + spec_cls.output_structure)

    return "\n\n".join(sections)

def render_preview(spec_class: type[PromptSpec], *, include_input_documents: bool = True) -> str:
    """Render a spec CLASS with placeholder values for dynamic fields.

    Uses `model_construct()` to bypass validation, allowing placeholder strings
    regardless of field type.
    """
    placeholders = {field_name: f"{{{field_name}}}" for field_name in spec_class.model_fields}
    instance = spec_class.model_construct(**placeholders)  # pyright: ignore[reportArgumentType] — placeholders are intentionally untyped strings
    text = render_text(instance, include_input_documents=include_input_documents)
    if spec_class.follows is not None:
        return f"[Follows: {spec_class.follows.__name__}]\n\n{text}"
    return text

```

## Examples

**Role valid** (`tests/prompt_compiler/test_components.py:109`)

```python
def test_role_valid() -> None:
    class ValidRole(Role):
        """A valid role."""

        text = "Expert engineer"

    assert ValidRole.text == "Expert engineer"
```

**Rule valid** (`tests/prompt_compiler/test_components.py:186`)

```python
def test_rule_valid() -> None:
    class ValidRule(Rule):
        """Doc."""

        text = "Do not fail."

    assert ValidRule.text == "Do not fail."
```

**Render full prompt spec workflow** (`tests/prompt_compiler/test_render.py:629`)

```python
def test_render_full_prompt_spec_workflow() -> None:
    """Define components and a PromptSpec, then render it to prompt text."""
    from ai_pipeline_core.prompt_compiler import OutputRule, PromptSpec, Role, Rule, render_text

    class Analyst(Role):
        """Research analyst role."""

        text = "experienced research analyst"

    class CiteEvidence(Rule):
        """Citation requirement."""

        text = "Cite specific evidence from source documents using document IDs"

    class UseProse(OutputRule):
        """Formatting constraint."""

        text = "Use prose paragraphs, not bullet lists"

    class AnalysisSpec(PromptSpec):
        """Analyze source documents for key findings."""

        input_documents = ()
        role = Analyst
        task = "Identify the key findings and assess their significance."
        rules = (CiteEvidence,)
        output_rules = (UseProse,)

        output_structure = "## Key Findings\n## Significance Assessment"
        topic: str = Field(description="Research topic")

    rendered = render_text(AnalysisSpec(topic="Market dynamics"))

    assert "# Role\n\nYou are an experienced research analyst." in rendered
    assert "**Research topic:**\nMarket dynamics" in rendered
    assert "# Rules\n\n1. Cite specific evidence" in rendered
    assert "# Output Rules\n\n1. Use prose paragraphs" in rendered
    assert "# Output Structure\n\n## Key Findings" in rendered
```

**Main compile empty dir** (`tests/prompt_compiler/test_cli.py:349`)

```python
def test_main_compile_empty_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ret = main(["compile", "--root", str(tmp_path)])
    assert ret == 0
    capsys.readouterr()  # Consume output; we only verify it doesn't crash
```

**Main compile finds specs** (`tests/prompt_compiler/test_cli.py:341`)

```python
def test_main_compile_finds_specs(capsys: pytest.CaptureFixture[str]) -> None:
    ret = main(["compile", "--root", str(Path.cwd())])
    assert ret == 0
    out = capsys.readouterr().out
    assert "spec(s) found:" in out
    assert "Name" in out  # table header
```

**Main inspect basemodel output** (`tests/prompt_compiler/test_cli.py:422`)

```python
def test_main_inspect_basemodel_output(capsys: pytest.CaptureFixture[str]) -> None:
    ret = main(["inspect", f"{StructuredInspectSpec.__module__}:StructuredInspectSpec"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "Type: CliPayload" in out
```

**Main inspect not found** (`tests/prompt_compiler/test_cli.py:439`)

```python
def test_main_inspect_not_found(capsys: pytest.CaptureFixture[str]) -> None:
    ret = main(["inspect", "NoSuchSpec12345"])
    assert ret == 1
    err = capsys.readouterr().err
    assert "not found" in err
```

**Main inspect with guides** (`tests/prompt_compiler/test_cli.py:429`)

```python
def test_main_inspect_with_guides(capsys: pytest.CaptureFixture[str]) -> None:
    """Inspect a spec that has guides — covers the guides section rendering."""
    ret = main(["inspect", "examples.showcase_prompt_compiler:IssueOptimisticSpec"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "Guides (2):" in out
    assert "RiskAssessmentFramework" in out
    assert "chars)" in out
```


## Error Examples

**Spec rules reject output rule with specific message** (`tests/prompt_compiler/test_spec.py:650`)

```python
def test_spec_rules_reject_output_rule_with_specific_message() -> None:
    from ai_pipeline_core.prompt_compiler import OutputRule, PromptSpec, Role

    class ReviewerRole(Role):
        """Reviewer."""

        text = "careful reviewer"

    class FormatBullets(OutputRule):
        """Bullet formatting."""

        text = "Return concise bullet points"

    with pytest.raises(TypeError, match=r"\.rules contains OutputRule 'FormatBullets'"):

        class MixedSpec(PromptSpec):
            """Doc."""

            input_documents = ()
            role = ReviewerRole
            task = "do it"

            rules = (FormatBullets,)  # Wrong! Should be output_rules=
```

**Spec bare field no description** (`tests/prompt_compiler/test_spec.py:922`)

```python
def test_spec_bare_field_no_description() -> None:
    from ai_pipeline_core.prompt_compiler import PromptSpec, Role

    class ReviewerRole(Role):
        """Reviewer."""

        text = "careful reviewer"

    with pytest.raises(TypeError, match=r"field 'item' must use Field\(description='\.\.\.'\)"):

        class BareFieldSpec(PromptSpec):
            """Doc."""

            input_documents = ()
            role = ReviewerRole
            task = "do it"

            item: str  # Wrong! Must use Field(description='...')
```

**Guide missing docstring** (`tests/prompt_compiler/test_components.py:258`)

```python
def test_guide_missing_docstring() -> None:
    with pytest.raises(TypeError, match="must define a non-empty docstring"):

        class NoDocGuide(Guide):
            template = "guide.txt"
```

**Guide rejects absolute path** (`tests/prompt_compiler/test_components.py:271`)

```python
def test_guide_rejects_absolute_path(tmp_path: Path) -> None:
    absolute = str((tmp_path / "guide.txt").resolve())
    with pytest.raises(TypeError, match="template must be a relative path"):
        type("AbsGuide", (Guide,), {"__module__": __name__, "__doc__": "Guide doc.", "template": absolute})
```

**Guide requires non empty string template** (`tests/prompt_compiler/test_components.py:266`)

```python
@pytest.mark.parametrize("template_value", [None, "", "   ", 123])
def test_guide_requires_non_empty_string_template(template_value: object) -> None:
    with pytest.raises(TypeError, match="must define 'template' as a ClassVar"):
        type("BadTemplateGuide", (Guide,), {"__module__": __name__, "__doc__": "Guide doc.", "template": template_value})
```

**Role empty docstring** (`tests/prompt_compiler/test_components.py:125`)

```python
def test_role_empty_docstring() -> None:
    with pytest.raises(TypeError, match="must define a non-empty docstring"):

        class EmptyDocRole(Role):
            """ """

            text = "valid"
```
