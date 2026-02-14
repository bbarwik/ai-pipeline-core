"""Rendering logic for prompt specifications."""

import re

from ai_pipeline_core.documents import Document

from .spec import PromptSpec

_VOWELS = frozenset("aeiouAEIOU")

RESULT_TAG = "result"
RESULT_OPEN = f"<{RESULT_TAG}>"
RESULT_CLOSE = f"</{RESULT_TAG}>"

_XML_WRAP_INSTRUCTION = (
    f"Write your complete response inside {RESULT_OPEN} tags. Do not add any XML tags inside {RESULT_OPEN}.\n\n{RESULT_OPEN}your response content{RESULT_CLOSE}"
)


def _role_sentence(text: str) -> str:
    """Build 'You are a/an {text}.' with correct article."""
    article = "an" if text[0] in _VOWELS else "a"
    return f"You are {article} {text}."


def _pascal_to_title(name: str) -> str:
    """Convert PascalCase class name to title case: RiskAssessmentFramework -> Risk Assessment Framework."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name)


def _format_numbered_rule(index: int, text: str) -> str:
    """Format a numbered rule with 3-space indent on continuation lines."""
    lines = text.splitlines()
    first = f"{index}. {lines[0]}"
    rest = [f"   {line}" for line in lines[1:]]
    return "\n".join([first, *rest])


def _render_documents_preview(spec_cls: type[PromptSpec]) -> str:
    """Render document listing from class-level info (for preview / no-documents mode)."""
    doc_blocks: list[str] = []
    for doc_cls in spec_cls.input_documents:
        lines = (doc_cls.__doc__ or "").strip().splitlines()
        desc = lines[0] if lines else "No description"
        block = f"{doc_cls.__name__}\n  {desc}"
        doc_blocks.append(block)
    return "Documents provided in context:\n\n" + "\n\n".join(doc_blocks)


def _render_documents_actual(documents: list[Document]) -> str:
    """Render document listing from actual Document instances."""
    doc_blocks: list[str] = []
    for doc in documents:
        header = f"[{doc.id}] {doc.name}"
        if doc.description:
            desc_lines = doc.description.strip().splitlines()
            indented = "\n".join(f"  {line}" for line in desc_lines)
            block = f"{header}\n{indented}"
        else:
            block = header
        doc_blocks.append(block)
    return "Documents provided in context:\n\n" + "\n\n".join(doc_blocks)


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


__all__ = ["render_preview", "render_text"]
