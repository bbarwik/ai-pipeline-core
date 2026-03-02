# MODULE: prompt_compiler
# CLASSES: Role, Rule, OutputRule, Guide, PromptSpec
# DEPENDS: BaseModel, Generic, Role
# PURPOSE: Prompt compiler for type-safe, validated prompt specifications.
# VERSION: 0.12.3
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import Guide, MultiLineField, OutputRule, PromptSpec, Role, Rule, render_preview, render_text
```

## Types & Constants

```python
APPROX_CHARS_PER_TOKEN = 4

MAX_RULE_LINES = 5

MAX_FIELD_VALUE_LENGTH = 500

RESULT_TAG = "result"

RESULT_OPEN = f"<{RESULT_TAG}>"

RESULT_CLOSE = f"</{RESULT_TAG}>"

```

## Public API

```python
class Role:
    """Base class for LLM role definitions.

Role text is rendered into the **user message**, not the system prompt. The renderer
produces ``"You are a/an {text}."`` as the first section of the compiled prompt text,
which is sent via ``Conversation.send()`` / ``send_spec()`` as a user message.

This is not a system prompt. Role does not set, replace, or interact with the
system prompt in any way. It is purely a section in the user message produced
by ``render_text()``.

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
        _validate_guide(cls)


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
Cannot combine ``output_structure`` with structured output (``PromptSpec[BaseModel]``) — output_structure is only for text specs.
Never reference XML tags in OutputRules when output_structure is set — the framework adds ``<result>`` wrapping automatically.

Never construct XML manually (f-string ``<document>`` tags) — the framework wraps Documents
in XML automatically when they are added to the Conversation via ``with_context()`` or
``with_document()``. Use ``Document.create()`` to wrap dicts, lists, or BaseModel instances.

Never use ``{field_name}`` placeholders in ``task`` text — the ``task`` ClassVar is rendered
literally, not as a template. ``{field_name}`` appears as the literal string ``{field_name}``
in the LLM prompt. Field values are rendered automatically by the framework: regular fields
(``Field``) are inlined in the ``# Context`` section as ``**description:**\nvalue``, and
multi-line fields (``MultiLineField``) are sent as XML-tagged user messages before the prompt.
Instead of ``task = "Analyze the {topic} ..."``, write ``task = "Analyze the topic identified
in context ..."``.

Pydantic fields (dynamic input values):
    Any field declared with ``Field(description=...)`` becomes a dynamic input.
    Fields are for short, single-line parameter values (up to 500 characters) — e.g.,
    a topic name, finding type, or formatting instruction. Longer or multiline content
    (e.g., review feedback, website content, another model's output) must be passed as
    a Document via ``input_documents`` and ``send_spec(documents=[...])``."""
    model_config = ConfigDict(frozen=True, extra='forbid')
    input_documents: ClassVar[tuple[type[Document], ...]]
    role: ClassVar[type[Role] | None]
    task: ClassVar[str]
    guides: ClassVar[tuple[type[Guide], ...]]
    rules: ClassVar[tuple[type[Rule], ...]]
    output_rules: ClassVar[tuple[type[OutputRule], ...]]
    output_structure: ClassVar[str | None]

    def __init_subclass__(cls, *, follows: type["PromptSpec"] | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Pydantic creates concrete subclasses for parameterized generics (e.g. PromptSpec[str]).
        # These have names like "PromptSpec[str]" — skip validation for them.
        if "[" in cls.__name__:
            return

        _validate_prompt_spec(cls, cls.__name__, follows)


```

## Functions

```python
def render_text(
    spec: PromptSpec,
    *,
    documents: Sequence[Document] | None = None,
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

    context_parts.extend(_render_context_fields(spec))

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

def render_multi_line_messages(spec: PromptSpec) -> list[tuple[str, str]]:
    """Return XML-tagged message blocks for multi-line fields.

    Each entry is ``(field_name, "<field_name>value</field_name>")``.
    Order matches field declaration order on the spec class.

    Includes both declared MultiLineFields and regular fields whose values
    exceed the inline limit (auto-promoted).
    """
    spec_cls = type(spec)
    result: list[tuple[str, str]] = []
    for field_name, field_info in spec_cls.model_fields.items():
        value = str(getattr(spec, field_name))
        if _is_multi_line_field(field_info) or _is_long_or_multiline(value):
            result.append((field_name, f"<{field_name}>{value}</{field_name}>"))
    return result

def render_preview(spec_class: type[PromptSpec], *, include_input_documents: bool = True) -> str:
    """Render a spec CLASS with placeholder values for dynamic fields.

    Uses `model_construct()` to bypass validation, allowing placeholder strings
    regardless of field type.

    Multi-line fields are shown as XML blocks before the prompt, separated by ``---``.
    """
    placeholders = {field_name: f"{{{field_name}}}" for field_name in spec_class.model_fields}
    instance = spec_class.model_construct(**placeholders)  # pyright: ignore[reportArgumentType] — placeholders are intentionally untyped strings

    # Build multi-line field preview blocks
    ml_blocks: list[str] = []
    for field_name, field_info in spec_class.model_fields.items():
        if _is_multi_line_field(field_info):
            ml_blocks.append(f"<{field_name}>{{{field_name}}}</{field_name}>")

    text = render_text(instance, include_input_documents=include_input_documents)

    if spec_class._follows is not None:
        text = f"[Follows: {spec_class._follows.__name__}]\n\n{text}"

    if ml_blocks:
        return "\n".join(ml_blocks) + "\n\n---\n\n" + text
    return text

def MultiLineField(*, description: str, **kwargs: Any) -> Any:
    """Declare a multi-line field on a PromptSpec.

    Multi-line fields are combined into a single XML-tagged user message sent
    before the main prompt, not inlined in the Context section. Each field is
    wrapped as ``<field_name>value</field_name>``. Use for content that is long
    or multiline — e.g., review feedback, website content, another model's output.

    Must provide ``description``. Accepts all other ``Field()`` keyword arguments
    (``default``, ``default_factory``, etc.).
    """
    return Field(description=description, json_schema_extra={_MULTI_LINE_KEY: True}, **kwargs)

def main(argv: list[str] | None = None) -> int:
    """CLI entry point for prompt compiler operations (``ai-prompt-compiler`` command)."""
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

```

## Examples

**Main render** (`tests/prompt_compiler/test_cli.py:315`)

```python
def test_main_render(capsys: pytest.CaptureFixture[str]) -> None:
    ret = main(["render", "tests.prompt_compiler.test_api:PlainSpec"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "# Role" in out
    assert "# Task" in out
```

**Main compile finds specs** (`tests/prompt_compiler/test_cli.py:343`)

```python
def test_main_compile_finds_specs(capsys: pytest.CaptureFixture[str]) -> None:
    ret = main(["compile", "--root", str(Path.cwd())])
    assert ret == 0
    out = capsys.readouterr().out
    assert "spec(s) found:" in out
    assert "Name" in out  # table header
```

**Main inspect minimal spec** (`tests/prompt_compiler/test_cli.py:363`)

```python
def test_main_inspect_minimal_spec(capsys: pytest.CaptureFixture[str]) -> None:
    """Inspect a spec with no rules, no guides, no output rules, no fields."""
    ret = main(["inspect", f"{MinimalInspectSpec.__module__}:MinimalInspectSpec"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "MinimalInspectSpec" in out
    assert "A minimal spec for inspect testing." in out
    assert "Module:" in out
    assert "Role: CliRole" in out
    assert '"experienced analyst"' in out
    assert "Input Documents (0):" in out
    assert "(none)" in out
    assert "Dynamic Fields (0):" in out
    assert "Do the task" in out
    assert "Type: str" in out
    assert "Rendered preview:" in out
    assert "tokens" in out
    # No rules/guides/output_rules sections
    assert "Rules (" not in out
    assert "Guides (" not in out
    assert "Output Rules (" not in out
    # Phase and XML Wrapped removed
    assert "Phase:" not in out
    assert "XML Wrapped" not in out
```

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

**Render full prompt spec workflow** (`tests/prompt_compiler/test_render.py:630`)

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

**Main compile empty dir** (`tests/prompt_compiler/test_cli.py:351`)

```python
def test_main_compile_empty_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ret = main(["compile", "--root", str(tmp_path)])
    assert ret == 0
    capsys.readouterr()  # Consume output; we only verify it doesn't crash
```

**Main inspect basemodel output** (`tests/prompt_compiler/test_cli.py:425`)

```python
def test_main_inspect_basemodel_output(capsys: pytest.CaptureFixture[str]) -> None:
    ret = main(["inspect", f"{StructuredInspectSpec.__module__}:StructuredInspectSpec"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "Type: CliPayload" in out
```

**Main inspect not found** (`tests/prompt_compiler/test_cli.py:442`)

```python
def test_main_inspect_not_found(capsys: pytest.CaptureFixture[str]) -> None:
    ret = main(["inspect", "NoSuchSpec12345"])
    assert ret == 1
    err = capsys.readouterr().err
    assert "not found" in err
```


## Error Examples

**Spec rules reject output rule with specific message** (`tests/prompt_compiler/test_spec.py:651`)

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

**Spec bare field no description** (`tests/prompt_compiler/test_spec.py:923`)

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

**Task field placeholder single field** (`tests/prompt_compiler/test_spec.py:1075`)

```python
def test_task_field_placeholder_single_field() -> None:
    """Task referencing a single field via {field_name} should raise."""
    with pytest.raises(TypeError, match=r"task contains field placeholder references.*\{topic\}"):

        class BadSpec(PromptSpec):
            """Doc."""

            input_documents = ()
            role = SpecRole
            task = 'Analyze the "{topic}" for key findings.'  # Wrong! task is rendered literally, not as a template
            topic: str = Field(description="Research topic")
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
