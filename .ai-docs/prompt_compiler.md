# MODULE: prompt_compiler
# CLASSES: Role, Rule, OutputRule, Guide, PromptSpec, Phase
# DEPENDS: BaseModel, Generic, str
# PURPOSE: Prompt compiler for type-safe, validated prompt specifications.
# SIZE: ~29KB

# === IMPORTS ===
from ai_pipeline_core import Guide, OutputRule, OutputT, Phase, PromptSpec, Role, Rule, extract_result, render_preview, render_text, send_spec

# === RULES (MUST FOLLOW) ===
# 1. Must define a non-empty docstring and a ``text`` ClassVar on every Role subclass.
# 2. Must not end Role text with sentence punctuation (.!?) — the renderer adds a period automatically.
# 3. Must define a non-empty docstring and a ``text`` ClassVar on every Rule subclass (max 5 lines).
# 4. Must define a non-empty docstring and a ``text`` ClassVar on every OutputRule subclass (max 5 lines).
# 5. Must define a non-empty docstring and a ``template`` ClassVar on every Guide subclass.
# 6. Must use a relative path for Guide template — content is loaded and cached at import time.
# 7. Never use ``#`` (H1) headers in Guide templates — reserved for prompt section boundaries. Use ``##`` or deeper.
# 8. Must subclass PromptSpec directly — no inheritance chains allowed.
# 9. Must declare phase via class parameter: ``class MySpec(PromptSpec, phase=Phase('review'))``.
# 10. Must define role, task, and input_documents on every PromptSpec subclass.
# 11. Must use ``Field(description='...')`` for all dynamic Pydantic fields on PromptSpec subclasses.
# 12. Must be a non-empty string when creating a Phase (validated by PromptSpec at class definition time).

# === TYPES & CONSTANTS ===

APPROX_CHARS_PER_TOKEN = 4

MAX_RULE_LINES = 5

RESULT_TAG = "result"

RESULT_OPEN = f"<{RESULT_TAG}>"

RESULT_CLOSE = f"</{RESULT_TAG}>"

# === PUBLIC API ===

class Role:
    """Base class for LLM role definitions.

Must define a non-empty docstring and a ``text`` ClassVar on every Role subclass.
Must not end Role text with sentence punctuation (.!?) — the renderer adds a period automatically."""
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
        _require_docstring(cls, kind="Rule")
        _require_text(cls, kind="Rule", max_lines=MAX_RULE_LINES)


class OutputRule:
    """Base class for output formatting constraints.

Must define a non-empty docstring and a ``text`` ClassVar on every OutputRule subclass (max 5 lines)."""
    text: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _require_docstring(cls, kind="OutputRule")
        _require_text(cls, kind="OutputRule", max_lines=MAX_RULE_LINES)


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
Must declare phase via class parameter: ``class MySpec(PromptSpec, phase=Phase('review'))``.
Must define role, task, and input_documents on every PromptSpec subclass.
Must use ``Field(description='...')`` for all dynamic Pydantic fields on PromptSpec subclasses.

Required ClassVars: role, task, input_documents.
Optional ClassVars: guides=(), rules=(), output_rules=(), output_structure=None, xml_wrapped=False.

Pydantic fields (dynamic input values):
    Any field declared with ``Field(description=...)`` becomes a dynamic input."""
    model_config = ConfigDict(frozen=True, extra='forbid')
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


class Phase(str):
    """Pipeline phase identifier for PromptSpec subclasses.

Must be a non-empty string when creating a Phase (validated by PromptSpec at class definition time).
Any non-empty string is valid (e.g., Phase('review'), Phase('analysis'), Phase('writing')).
Used as a class parameter: ``class MySpec(PromptSpec, phase=Phase('review'))``."""

# === FUNCTIONS ===

def main(argv: list[str] | None = None) -> int:
    """CLI entry point for prompt compiler operations."""
    parser = argparse.ArgumentParser(prog="prompt_compiler", description="Prompt compiler CLI")
    subparsers = parser.add_subparsers(dest="command")

    # list
    list_parser = subparsers.add_parser("list", help="List all discovered PromptSpec subclasses")
    list_parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root for class discovery")

    # inspect
    inspect_parser = subparsers.add_parser("inspect", help="Show detailed anatomy of a single spec")
    inspect_parser.add_argument("spec", help="Spec class name or module.path:ClassName")
    inspect_parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root for class discovery")

    # render
    render_parser = subparsers.add_parser("render", help="Render a prompt preview with placeholder values")
    render_parser.add_argument("spec", help="Spec class name or module.path:ClassName")
    render_parser.add_argument("--no-input-documents", action="store_true", help="Hide input document listing")
    render_parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root for class discovery")

    args = parser.parse_args(argv)

    handlers = {"list": _cmd_list, "inspect": _cmd_inspect, "render": _cmd_render}
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)

def extract_result(text: str) -> str:
    """Extract content from <result> tags. Returns text as-is if no tags found."""
    match = _EXTRACT_PATTERN.search(text)
    return match.group(1).strip() if match else text

async def send_spec(
    spec: PromptSpec[Any],
    *,
    model: str | None = None,
    conversation: Conversation[Any] | None = None,
    documents: list[Document] | None = None,
    model_options: ModelOptions | None = None,
    include_input_documents: bool = True,
    purpose: str | None = None,
    expected_cost: float | None = None,
) -> Conversation[Any]:
    r"""Send a PromptSpec to an LLM via Conversation.

    Either ``model`` or ``conversation`` must be provided. When ``conversation``
    is given, uses it directly (enables warmup+fork pattern). When ``model``
    is given, creates a fresh Conversation.

    Adds documents to context, renders the prompt, and sends it.
    Dispatches to send() or send_structured() based on the spec's output type.

    When spec.xml_wrapped is True and the model supports stop sequences,
    automatically sets stop=[\"</result>\"] to cut off after the response.
    Use extract_result(conv.content) to get clean content from xml_wrapped specs.
    """
    if conversation is not None:
        conv: Conversation[Any] = conversation
        if model_options is not None:
            conv = conv.with_model_options(model_options)
    elif model is not None:
        conv = Conversation(model=model, model_options=model_options)
    else:
        raise ValueError("Either 'model' or 'conversation' must be provided")

    spec_cls = type(spec)

    # Set stop sequence for xml_wrapped specs on supported models
    if spec_cls.xml_wrapped and _supports_stop_sequence(conv.model):
        current_options = conv.model_options or ModelOptions()
        existing_stop = current_options.stop
        if existing_stop is None:
            stop_list = [RESULT_CLOSE]
        elif isinstance(existing_stop, str):
            stop_list = [existing_stop, RESULT_CLOSE]
        else:
            stop_list = [*existing_stop, RESULT_CLOSE]
        conv = conv.with_model_options(current_options.model_copy(update={"stop": stop_list}))

    if documents:
        conv = conv.with_context(*documents)

    prompt_text = render_text(spec, documents=documents, include_input_documents=include_input_documents)
    trace_purpose = purpose or spec_cls.__name__

    if spec_cls.output_type is str:
        return await conv.send(prompt_text, purpose=trace_purpose, expected_cost=expected_cost)

    response_format = cast(type[BaseModel], spec_cls.output_type)
    return await conv.send_structured(prompt_text, response_format=response_format, purpose=trace_purpose, expected_cost=expected_cost)

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

    # 1. Role
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
    if spec_cls.output_rules:
        or_lines = [_format_numbered_rule(i, rule_cls.text) for i, rule_cls in enumerate(spec_cls.output_rules, 1)]
        sections.append("# Output Rules\n\n" + "\n".join(or_lines))

    # 7. Output structure
    structure_parts: list[str] = []
    if spec_cls.xml_wrapped:
        structure_parts.append(_XML_WRAP_INSTRUCTION)
    if spec_cls.output_structure:
        structure_parts.append(spec_cls.output_structure)
    if structure_parts:
        sections.append("# Output Structure\n\n" + "\n\n".join(structure_parts))

    return "\n\n".join(sections)

def render_preview(spec_class: type[PromptSpec], *, include_input_documents: bool = True) -> str:
    """Render a spec CLASS with placeholder values for dynamic fields.

    Uses `model_construct()` to bypass validation, allowing placeholder strings
    regardless of field type.
    """
    placeholders = {field_name: f"{{{field_name}}}" for field_name in spec_class.model_fields}
    instance = spec_class.model_construct(**placeholders)  # pyright: ignore[reportArgumentType] — placeholders are intentionally untyped strings
    return render_text(instance, include_input_documents=include_input_documents)

# === EXAMPLES (from tests/) ===

# Example: Extract result with closing tag
# Source: tests/prompt_compiler/test_api.py:123
def test_extract_result_with_closing_tag() -> None:
    text = "preamble <result>\n  hello world \n</result> trailing"
    assert api.extract_result(text) == "hello world"

# Example: Role valid
# Source: tests/prompt_compiler/test_components.py:109
def test_role_valid() -> None:
    class ValidRole(Role):
        """A valid role."""

        text = "Expert engineer"

    assert ValidRole.text == "Expert engineer"

# Example: Rule valid
# Source: tests/prompt_compiler/test_components.py:186
def test_rule_valid() -> None:
    class ValidRule(Rule):
        """Doc."""

        text = "Do not fail."

    assert ValidRule.text == "Do not fail."

# Example: Render full prompt spec workflow
# Source: tests/prompt_compiler/test_render.py:545
def test_render_full_prompt_spec_workflow() -> None:
    """Define components and a PromptSpec, then render it to prompt text."""
    from ai_pipeline_core.prompt_compiler import OutputRule, Phase, PromptSpec, Role, Rule, render_text

    class Analyst(Role):
        """Research analyst role."""

        text = "experienced research analyst"

    class CiteEvidence(Rule):
        """Citation requirement."""

        text = "Cite specific evidence from source documents using document IDs"

    class UseProse(OutputRule):
        """Formatting constraint."""

        text = "Use prose paragraphs, not bullet lists"

    class AnalysisSpec(PromptSpec, phase=Phase("analysis")):
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

# Example: Extract result empty content
# Source: tests/prompt_compiler/test_api.py:143
def test_extract_result_empty_content() -> None:
    text = "<result></result>"
    assert api.extract_result(text) == ""

# Example: Extract result multiline content
# Source: tests/prompt_compiler/test_api.py:138
def test_extract_result_multiline_content() -> None:
    text = "<result>\nline1\nline2\n</result>"
    assert api.extract_result(text) == "line1\nline2"

# Example: Extract result whitespace stripped
# Source: tests/prompt_compiler/test_api.py:148
def test_extract_result_whitespace_stripped() -> None:
    text = "<result>  \n  content  \n  </result>"
    assert api.extract_result(text) == "content"

# Example: Extract result with incomplete closing tag
# Source: tests/prompt_compiler/test_api.py:133
def test_extract_result_with_incomplete_closing_tag() -> None:
    text = "prefix <result>partial content"
    assert api.extract_result(text) == "partial content"

# Example: Extract result without tags
# Source: tests/prompt_compiler/test_api.py:128
def test_extract_result_without_tags() -> None:
    text = "plain response"
    assert api.extract_result(text) == text

# === ERROR EXAMPLES (What NOT to Do) ===

# Error: Spec requires phase
# Source: tests/prompt_compiler/test_spec.py:244
def test_spec_requires_phase() -> None:
    from ai_pipeline_core.prompt_compiler import PromptSpec, Role

    class ReviewerRole(Role):
        """Reviewer."""

        text = "careful reviewer"

    with pytest.raises(TypeError, match="must set phase"):

        class NoPhaseSpec(PromptSpec):
            """Doc."""

            input_documents = ()
            role = ReviewerRole
            task = "do it"

# Error: Spec rules reject output rule with specific message
# Source: tests/prompt_compiler/test_spec.py:600
def test_spec_rules_reject_output_rule_with_specific_message() -> None:
    from ai_pipeline_core.prompt_compiler import OutputRule, Phase, PromptSpec, Role

    class ReviewerRole(Role):
        """Reviewer."""

        text = "careful reviewer"

    class FormatBullets(OutputRule):
        """Bullet formatting."""

        text = "Return concise bullet points"

    with pytest.raises(TypeError, match=r"\.rules contains OutputRule 'FormatBullets'"):

        class MixedSpec(PromptSpec, phase=Phase("review")):
            """Doc."""

            input_documents = ()
            role = ReviewerRole
            task = "do it"

            rules = (FormatBullets,)  # Wrong! Should be output_rules=

# Error: Spec bare field no description
# Source: tests/prompt_compiler/test_spec.py:875
def test_spec_bare_field_no_description() -> None:
    from ai_pipeline_core.prompt_compiler import Phase, PromptSpec, Role

    class ReviewerRole(Role):
        """Reviewer."""

        text = "careful reviewer"

    with pytest.raises(TypeError, match=r"field 'item' must use Field\(description='\.\.\.'\)"):

        class BareFieldSpec(PromptSpec, phase=Phase("review")):
            """Doc."""

            input_documents = ()
            role = ReviewerRole
            task = "do it"

            item: str  # Wrong! Must use Field(description='...')

# Error: Guide missing docstring
# Source: tests/prompt_compiler/test_components.py:258
def test_guide_missing_docstring() -> None:
    with pytest.raises(TypeError, match="must define a non-empty docstring"):

        class NoDocGuide(Guide):
            template = "guide.txt"

# Error: Guide rejects absolute path
# Source: tests/prompt_compiler/test_components.py:271
def test_guide_rejects_absolute_path(tmp_path: Path) -> None:
    absolute = str((tmp_path / "guide.txt").resolve())
    with pytest.raises(TypeError, match="template must be a relative path"):
        type("AbsGuide", (Guide,), {"__module__": __name__, "__doc__": "Guide doc.", "template": absolute})

# Error: Guide requires non empty string template
# Source: tests/prompt_compiler/test_components.py:266
@pytest.mark.parametrize("template_value", [None, "", "   ", 123])
def test_guide_requires_non_empty_string_template(template_value: object) -> None:
    with pytest.raises(TypeError, match="must define 'template' as a ClassVar"):
        type("BadTemplateGuide", (Guide,), {"__module__": __name__, "__doc__": "Guide doc.", "template": template_value})

# Error: Role empty docstring
# Source: tests/prompt_compiler/test_components.py:125
def test_role_empty_docstring() -> None:
    with pytest.raises(TypeError, match="must define a non-empty docstring"):

        class EmptyDocRole(Role):
            """ """

            text = "valid"
