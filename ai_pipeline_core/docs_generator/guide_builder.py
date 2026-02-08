"""Per-module guide assembly with test discovery, relevance scoring, and rendering.

Collects public symbols, resolves dependencies, flattens inheritance,
discovers and scores test examples, extracts rules, and renders guides.
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from ai_pipeline_core.docs_generator.extractor import (
    EXTERNAL_STUBS,
    ClassInfo,
    FunctionInfo,
    MethodInfo,
    SymbolTable,
    is_public_name,
    resolve_dependencies,
)
from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)

MAX_EXAMPLES = 8

_RULE_PREFIXES = ("cannot", "must", "never", "always", "critical")


@dataclass(frozen=True)
class ScoredExample:
    """Scored test function extracted for guide examples."""

    name: str
    source_file: str
    line_number: int
    code: str
    score: int
    is_error_example: bool
    is_marked: bool = False


@dataclass
class GuideData:
    """Intermediate representation of a guide, used by trimmer before rendering."""

    module_name: str
    classes: list[ClassInfo]
    functions: list[FunctionInfo]
    rules: list[str]
    external_bases: set[str]
    normal_examples: list[ScoredExample]
    error_examples: list[ScoredExample]
    internal_types: list[ClassInfo] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Test discovery
# ---------------------------------------------------------------------------


def discover_tests(
    module_name: str,
    tests_dir: Path,
    test_dir_overrides: dict[str, str] | None = None,
    repo_root: Path | None = None,
) -> list[ScoredExample]:
    """Find and extract test functions relevant to a module.

    Default mapping: ai_pipeline_core/<module>/ -> tests/<module>/
    Override with test_dir_overrides to remap specific modules.
    Root-level glob tests/test_<module>*.py always runs with original module name.
    """
    test_files: list[Path] = []

    # Subdirectory: use override if provided, else convention
    subdir_name = (test_dir_overrides or {}).get(module_name, module_name)
    subdir = tests_dir / subdir_name
    if subdir.is_dir():
        test_files.extend(sorted(subdir.glob("test_*.py")))

    # Root-level tests always use original module_name
    for f in sorted(tests_dir.glob(f"test_{module_name}*.py")):
        if f not in test_files:
            test_files.append(f)

    examples: list[ScoredExample] = []
    for test_file in test_files:
        examples.extend(_extract_test_functions(test_file, repo_root))
    return examples


# ---------------------------------------------------------------------------
# Scoring and selection
# ---------------------------------------------------------------------------


def score_test(test: ScoredExample, symbol_names: list[str]) -> int:
    """Score a test's relevance to a set of module symbols.

    Returns the best score across all symbols.
    """
    best_score = 0
    test_name_lower = test.name.lower()

    for symbol in symbol_names:
        score = 0
        symbol_lower = symbol.lower()

        # Exact subject match: +5
        subject = test_name_lower.removeprefix("test_")
        if symbol_lower == subject or subject.startswith(symbol_lower + "_"):
            score += 5
        # Partial match: +3
        elif symbol_lower in test_name_lower:
            score += 3

        # Body occurrences: +min(count, 2)
        count = test.code.count(symbol)
        score += min(count, 2)

        best_score = max(best_score, score)

    # Error example bonus: +2
    if test.is_error_example:
        best_score += 2

    # Simplicity bonus
    line_count = len(test.code.splitlines())
    if line_count < 10:
        best_score += 2
    elif line_count < 20:
        best_score += 1

    # Pattern bonus: +1
    if any(p in test.name for p in ("creation", "basic", "simple")):
        best_score += 1

    # Mock penalty — prefer real usage examples over heavily mocked tests
    mock_patterns = ("Mock(", "MagicMock(", "patch(", "monkeypatch.")
    mock_count = sum(test.code.count(p) for p in mock_patterns)
    if mock_count >= 3:
        best_score -= 3
    elif mock_count >= 1:
        best_score -= 1

    return max(best_score, 0)


def select_examples(
    tests: list[ScoredExample],
    symbol_names: list[str],
    max_total: int = MAX_EXAMPLES,
) -> tuple[list[ScoredExample], list[ScoredExample]]:
    """Select top examples within budget.

    Marked tests (@pytest.mark.ai_docs) get priority slots.
    Remaining slots filled by top-scored auto tests.
    Returns (normal_examples, error_examples).
    Error cap (max_total // 2) applies to auto tests only.
    If marked tests exceed max_total, all marked tests are included.
    """
    marked = [t for t in tests if t.is_marked]
    auto = [t for t in tests if not t.is_marked]

    scored_auto = sorted(
        [(score_test(t, symbol_names), t) for t in auto],
        key=lambda x: (-x[0], x[1].name),
    )

    marked_normal = [t for t in marked if not t.is_error_example]
    marked_errors = [t for t in marked if t.is_error_example]

    remaining = max(max_total - len(marked), 0)
    if remaining > 0:
        auto_errors = [(s, t) for s, t in scored_auto if t.is_error_example]
        auto_normal = [(s, t) for s, t in scored_auto if not t.is_error_example]

        max_auto_errors = remaining // 2
        selected_auto_errors = [t for _, t in auto_errors[:max_auto_errors]]

        auto_normal_slots = remaining - len(selected_auto_errors)
        selected_auto_normal = [t for _, t in auto_normal[:auto_normal_slots]]
    else:
        selected_auto_errors = []
        selected_auto_normal = []

    return marked_normal + selected_auto_normal, marked_errors + selected_auto_errors


# ---------------------------------------------------------------------------
# Inheritance flattening
# ---------------------------------------------------------------------------


def flatten_methods(cls: ClassInfo, table: SymbolTable) -> tuple[MethodInfo, ...]:
    """Flatten inheritance chain, showing inherited methods inline.

    Uses "child-first, first-match wins" to approximate Python's MRO:
    child methods > left base > right base > grandparent.
    """
    method_map: dict[str, tuple[MethodInfo, str]] = {}
    visited: set[str] = set()

    def collect(c: ClassInfo) -> None:
        if c.name in visited:
            return
        visited.add(c.name)
        for method in c.methods:
            if method.name not in method_map:
                method_map[method.name] = (method, c.name)
        for base_name in c.bases:
            clean = base_name.split("[")[0]
            if clean in table.classes and clean not in EXTERNAL_STUBS:
                collect(table.classes[clean])

    collect(cls)

    result: list[MethodInfo] = []
    for method, source_class in method_map.values():
        if source_class != cls.name:
            if source_class in EXTERNAL_STUBS:
                continue
            method = MethodInfo(  # noqa: PLW2901
                name=method.name,
                signature=method.signature,
                docstring=method.docstring,
                source=method.source,
                is_property=method.is_property,
                is_classmethod=method.is_classmethod,
                is_abstract=method.is_abstract,
                line_count=method.line_count,
                is_inherited=True,
                inherited_from=source_class,
            )
        result.append(method)

    return tuple(_sort_methods(result))


# ---------------------------------------------------------------------------
# Rule extraction
# ---------------------------------------------------------------------------


def extract_rules(classes: list[ClassInfo]) -> list[str]:
    """Extract rules from docstrings (lines starting with constraint keywords)."""
    rules: list[str] = []
    seen: set[str] = set()

    for cls in classes:
        if not cls.docstring:
            continue
        for line in cls.docstring.splitlines():
            stripped = line.strip()
            if any(stripped.lower().startswith(prefix) for prefix in _RULE_PREFIXES) and stripped not in seen:
                rules.append(stripped)
                seen.add(stripped)
    return rules


# ---------------------------------------------------------------------------
# Guide rendering
# ---------------------------------------------------------------------------


def render_guide(data: GuideData) -> str:  # noqa: C901, PLR0912, PLR0915
    """Render GuideData to final markdown string."""
    parts: list[str] = []

    # Header
    class_names = ", ".join(c.name for c in data.classes)
    external = ", ".join(sorted(data.external_bases))
    parts.append(f"# MODULE: {data.module_name}")
    if class_names:
        parts.append(f"# CLASSES: {class_names}")
    if external:
        parts.append(f"# DEPENDS: {external}")

    # Rules
    if data.rules:
        parts.append("")
        parts.append("# === RULES (MUST FOLLOW) ===")
        for i, rule in enumerate(data.rules, 1):
            parts.append(f"# {i}. {rule}")

    # Dependencies (external stubs)
    if data.external_bases:
        parts.append("")
        parts.append("# === DEPENDENCIES (Resolved) ===")
        parts.append("")
        for base_name in sorted(data.external_bases):
            clean = base_name.split("[")[0]
            desc = EXTERNAL_STUBS.get(clean, "External base class (not fully documented).")
            parts.append(f"class {clean}:")
            parts.append(f'    """{desc}"""')
            parts.append("    ...")
            parts.append("")

    # Internal types (private classes referenced by public API)
    if data.internal_types:
        parts.append("")
        parts.append("# === INTERNAL TYPES (referenced by public API) ===")
        parts.append("")
        for cls in data.internal_types:
            parts.extend(_render_class(cls))

    # Public API -- classes
    parts.append("# === PUBLIC API ===")
    parts.append("")
    for cls in data.classes:
        parts.extend(_render_class(cls))

    # Public API -- functions
    if data.functions:
        parts.append("# === FUNCTIONS ===")
        parts.append("")
        for func in data.functions:
            parts.extend(_render_function(func))

    # Examples
    if data.normal_examples or data.error_examples:
        if data.normal_examples:
            parts.append("# === EXAMPLES (from tests/) ===")
            parts.append("")
            for ex in data.normal_examples:
                parts.append(f"# Example: {_example_title(ex)}")
                parts.append(f"# Source: {ex.source_file}:{ex.line_number}")
                parts.append(ex.code)
                parts.append("")

        if data.error_examples:
            parts.append("# === ERROR EXAMPLES (What NOT to Do) ===")
            parts.append("")
            for ex in data.error_examples:
                parts.append(f"# Error: {_example_title(ex)}")
                parts.append(f"# Source: {ex.source_file}:{ex.line_number}")
                parts.append(ex.code)
                parts.append("")
    else:
        parts.append("# === EXAMPLES ===")
        parts.append("# No test examples available.")
        parts.append("")

    content = "\n".join(parts)

    # Insert SIZE header line (computed after rendering content)
    size_kb = len(content.encode("utf-8")) // 1024
    size_line = f"# SIZE: ~{size_kb}KB"
    result_lines = content.splitlines()
    # Insert after last header line (MODULE/CLASSES/DEPENDS), before any blank or section line
    insert_idx = 0
    for i, line in enumerate(result_lines):
        if line.startswith(("# MODULE:", "# CLASSES:", "# DEPENDS:")):
            insert_idx = i + 1
        elif line.startswith("# ==="):
            break
    result_lines.insert(insert_idx, size_line)
    return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_guide(  # noqa: PLR0917
    module_name: str,
    source_dir: Path,
    tests_dir: Path,
    table: SymbolTable,
    test_dir_overrides: dict[str, str] | None = None,
    repo_root: Path | None = None,
) -> GuideData:
    """Build guide data for a module.

    Collects public symbols, resolves dependencies, flattens inheritance,
    discovers tests, scores and selects examples.
    """
    if repo_root is None:
        repo_root = source_dir.parent

    public_classes = [c for c in table.classes.values() if c.is_public and table.class_to_module.get(c.name) == module_name]
    public_functions = [f for f in table.functions.values() if f.is_public and table.function_to_module.get(f.name) == module_name]

    # Resolve dependencies
    root_names = [c.name for c in public_classes]
    _resolved, external_bases = resolve_dependencies(root_names, table)

    # Flatten inheritance for each public class
    flattened_classes: list[ClassInfo] = []
    for cls in public_classes:
        flat_methods = flatten_methods(cls, table)
        flattened_classes.append(
            ClassInfo(
                name=cls.name,
                bases=cls.bases,
                docstring=cls.docstring,
                is_public=cls.is_public,
                class_vars=cls.class_vars,
                methods=flat_methods,
                validators=cls.validators,
                module_path=cls.module_path,
                decorators=cls.decorators,
            )
        )

    # Detect private types referenced by public API signatures
    internal_types = _collect_internal_types(public_functions, flattened_classes, table, module_name)

    # Discover and score tests
    tests = discover_tests(module_name, tests_dir, test_dir_overrides, repo_root)
    if not tests:
        logger.warning("No tests found for %s", module_name)

    symbol_names = root_names + [f.name for f in public_functions]

    for t in tests:
        if t.is_marked and not _has_symbol_overlap(t, symbol_names):
            logger.warning(
                "Marked test %s in %s has no symbol overlap with module %s",
                t.name,
                t.source_file,
                module_name,
            )

    normal_examples, error_examples = select_examples(tests, symbol_names)

    rules = extract_rules(flattened_classes)

    return GuideData(
        module_name=module_name,
        classes=flattened_classes,
        functions=public_functions,
        rules=rules,
        external_bases=external_bases,
        normal_examples=normal_examples,
        error_examples=error_examples,
        internal_types=internal_types,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_PRIVATE_TYPE_RE = re.compile(r"\b_[A-Z]\w*")


def _collect_internal_types(
    public_functions: list[FunctionInfo],
    public_classes: list[ClassInfo],
    table: SymbolTable,
    module_name: str,
) -> list[ClassInfo]:
    """Find private classes from the same module that are referenced in public signatures."""
    # Build text blob from all public signatures and sources
    parts: list[str] = [f.source for f in public_functions]
    for c in public_classes:
        parts.extend(m.signature for m in c.methods if is_public_name(m.name))
    blob = " ".join(parts)

    referenced = set(_PRIVATE_TYPE_RE.findall(blob))
    if not referenced:
        return []

    return sorted(
        (table.classes[name] for name in referenced if name in table.classes and table.class_to_module.get(name) == module_name),
        key=lambda c: c.name,
    )


def _get_source(source_lines: list[str], node: ast.AST) -> str:
    decoratable = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    if isinstance(node, decoratable) and node.decorator_list:
        start = node.decorator_list[0].lineno - 1
    else:
        start: int = node.lineno - 1  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    end: int = node.end_lineno or node.lineno  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    return "\n".join(source_lines[start:end])


def _extract_test_functions(path: Path, repo_root: Path | None = None) -> list[ScoredExample]:
    source = path.read_text(encoding="utf-8")
    source_lines = source.splitlines()
    tree = ast.parse(source)

    if repo_root is not None:
        try:
            source_file = str(path.relative_to(repo_root))
        except ValueError:
            source_file = str(path)
    else:
        source_file = str(path)

    results: list[ScoredExample] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue

        code = _get_source(source_lines, node)
        code = "\n".join(line for line in code.splitlines() if "pytest.mark.ai_docs" not in line and "mark.ai_docs" not in line)
        code = "\n".join(_dedented_source(code))
        results.append(
            ScoredExample(
                name=node.name,
                source_file=source_file,
                line_number=node.lineno,
                code=code,
                score=0,
                is_error_example=_uses_pytest_raises(node),
                is_marked=_has_ai_docs_marker(node),
            )
        )
    return results


def _uses_pytest_raises(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr == "raises" and isinstance(child.value, ast.Name) and child.value.id == "pytest":
            return True
    return False


def _has_ai_docs_marker(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if a test function has @pytest.mark.ai_docs decorator."""
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if not isinstance(target, ast.Attribute) or target.attr != "ai_docs":
            continue
        # pytest.mark.ai_docs
        if (
            isinstance(target.value, ast.Attribute)
            and target.value.attr == "mark"
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "pytest"
        ):
            return True
        # mark.ai_docs (from pytest import mark)
        if isinstance(target.value, ast.Name) and target.value.id == "mark":
            return True
    return False


def _has_symbol_overlap(test: ScoredExample, symbol_names: list[str]) -> bool:
    """Check if a test references any of the given symbol names."""
    return any(symbol.lower() in test.name.lower() or symbol in test.code for symbol in symbol_names)


def _sort_methods(methods: list[MethodInfo]) -> list[MethodInfo]:
    """Sort: __init__ first, then properties, classmethods, regular methods."""

    def key(m: MethodInfo) -> tuple[int, str]:
        if m.name == "__init__":
            return (0, m.name)
        if m.is_property:
            return (1, m.name)
        if m.is_classmethod:
            return (2, m.name)
        return (3, m.name)

    return sorted(methods, key=key)


def _render_class(cls: ClassInfo) -> list[str]:
    lines: list[str] = [f"@{dec}" for dec in cls.decorators]
    bases_str = f"({', '.join(cls.bases)})" if cls.bases else ""
    lines.append(f"class {cls.name}{bases_str}:")

    if cls.docstring:
        lines.append(f'    """{cls.docstring.strip()}"""')

    for var_name, type_ann, default in cls.class_vars:
        if type_ann:
            if default:
                lines.append(f"    {var_name}: {type_ann} = {default}")
            else:
                lines.append(f"    {var_name}: {type_ann}")
        elif default:
            lines.append(f"    {var_name} = {default}")
        else:
            lines.append(f"    {var_name}")

    if cls.class_vars:
        lines.append("")

    # Separate own methods from inherited
    own_methods = [m for m in cls.methods if not m.is_inherited and is_public_name(m.name)]
    inherited_methods = [m for m in cls.methods if m.is_inherited and is_public_name(m.name)]

    # Render inherited methods as compact grouped references
    if inherited_methods:
        groups: dict[str, list[str]] = {}
        for m in inherited_methods:
            parent = m.inherited_from or "unknown"
            groups.setdefault(parent, []).append(m.name)
        for parent, names in groups.items():
            lines.append(f"    # [Inherited from {parent}]")
            lines.append(f"    # {', '.join(sorted(names))}")
        lines.append("")

    # Render own methods with full source
    for method in own_methods:
        lines.extend(_render_method(method))

    lines.append("")
    return lines


def _render_method(method: MethodInfo) -> list[str]:
    lines: list[str] = [f"    {source_line}" for source_line in _dedented_source(method.source)]
    lines.append("")
    return lines


def _render_function(func: FunctionInfo) -> list[str]:
    lines: list[str] = list(_dedented_source(func.source))
    lines.append("")
    return lines


def _dedented_source(source: str) -> list[str]:
    """Dedent source based on first line's indentation.

    Uses the first non-empty line's indentation as the base level,
    which correctly handles multi-line strings with content at column 0.
    """
    raw_lines = source.splitlines()
    if not raw_lines:
        return []
    first_non_empty = next((line for line in raw_lines if line.strip()), None)
    if first_non_empty is None:
        return raw_lines
    indent = len(first_non_empty) - len(first_non_empty.lstrip())
    if indent == 0:
        return raw_lines
    return [line[indent:] if line[:indent].isspace() else line for line in raw_lines]


def _example_title(ex: ScoredExample) -> str:
    """Convert test function name to readable title."""
    return ex.name.removeprefix("test_").replace("_", " ").capitalize()
