"""CLI for AI documentation generation and validation."""

import argparse
import ast
import re
import sys
import tomllib
from pathlib import Path

from ai_pipeline_core.docs_generator.extractor import (
    ClassInfo,
    FunctionInfo,
    MethodInfo,
    build_symbol_table,
    is_public_name,
)
from ai_pipeline_core.docs_generator.guide_builder import GuideData, build_guide, render_guide
from ai_pipeline_core.docs_generator.trimmer import README_ERROR_SIZE, manage_guide_size
from ai_pipeline_core.docs_generator.validator import validate_all

__all__ = [
    "EXCLUDED_MODULES",
    "PACKAGE_NAME",
    "README_FILENAME",
    "TEST_DIR_OVERRIDES",
    "main",
]

EXCLUDED_MODULES: frozenset[str] = frozenset({"docs_generator"})
PACKAGE_NAME = "ai_pipeline_core"
README_FILENAME = "README.md"


_CONSECUTIVE_BLOCKS_RE = re.compile(r"```\n(\s*\n)+```python\n")


def _consolidate_code_blocks(content: str) -> str:
    """Merge consecutive ```python blocks separated only by whitespace."""
    return _CONSECUTIVE_BLOCKS_RE.sub("\n\n", content)


def _normalize_whitespace(content: str) -> str:
    """Strip trailing whitespace from each line and ensure final newline."""
    lines = [line.rstrip() for line in content.splitlines()]
    return "\n".join(lines) + "\n"


TEST_DIR_OVERRIDES: dict[str, str] = {}  # nosemgrep: no-mutable-module-globals


def _discover_modules(source_dir: Path) -> list[str]:
    """Discover all public module groupings from package structure."""
    modules: set[str] = set()
    for py_file in sorted(source_dir.rglob("*.py")):
        if py_file.name.startswith("_") and py_file.name != "__init__.py":
            continue
        relative = py_file.relative_to(source_dir)
        if len(relative.parts) > 1:
            module_name = relative.parts[0]
            if not module_name.startswith("_"):
                modules.add(module_name)
        else:
            modules.add(relative.stem)
    return sorted(modules - EXCLUDED_MODULES)


def _read_module_purpose(source_dir: Path, module_name: str) -> str:
    """Read first line of __init__.py docstring for module purpose."""
    init_file = source_dir / module_name / "__init__.py"
    if not init_file.exists():
        return ""
    try:
        tree = ast.parse(init_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return ""
    doc = ast.get_docstring(tree)
    if not doc:
        return ""
    return doc.splitlines()[0].strip()


def _read_version(repo_root: Path) -> str:
    """Read package version from pyproject.toml."""
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.exists():
        return ""
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return data.get("project", {}).get("version", "")


def _build_import_map(source_dir: Path) -> dict[str, list[str]]:  # noqa: C901, PLR0912
    """Parse top-level __init__.py __all__ and map each symbol to its module.

    Returns {module_name: [symbol1, symbol2, ...]} for symbols in __all__.
    """
    init_file = source_dir / "__init__.py"
    if not init_file.exists():
        return {}

    try:
        tree = ast.parse(init_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return {}

    # Collect __all__ names
    all_names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                all_names = {elt.value for elt in node.value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)}

    if not all_names:
        return {}

    # Map symbol to import source module by scanning imports
    symbol_source: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.names:
            # Extract first-level subpackage from import path
            # Relative: from .documents import ... -> module="documents", level=1
            # Relative nested: from .observability.tracing import ... -> module="observability.tracing", level=1
            # Absolute: from ai_pipeline_core.documents import ... -> module="ai_pipeline_core.documents"
            parts = node.module.split(".")
            if node.level > 0:
                # Relative import: first part is the subpackage
                mod = parts[0]
            elif len(parts) >= 2 and parts[0] == PACKAGE_NAME:
                # Absolute import from our package
                mod = parts[1]
            else:
                mod = parts[0]
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                if name in all_names:
                    symbol_source[name] = mod

    # Group by module
    result: dict[str, list[str]] = {}
    for name, mod in sorted(symbol_source.items()):
        result.setdefault(mod, []).append(name)
    return result


def _build_module_import_map(source_dir: Path, excluded_modules: frozenset[str] = frozenset()) -> dict[str, list[str]]:
    """Parse sub-package __init__.py files and map each exported symbol to its module.

    Returns {module_name: [symbol1, symbol2, ...]} for symbols in sub-package __all__
    that are NOT already in the top-level __all__.
    """
    top_level_all = _parse_all_names(source_dir / "__init__.py")
    result: dict[str, list[str]] = {}

    for init_file in sorted(source_dir.glob("*/__init__.py")):
        module_name = init_file.parent.name
        if module_name.startswith("_") or module_name in excluded_modules:
            continue

        sub_all = _parse_all_names(init_file)
        # Only include symbols NOT already at top level
        module_only = sorted(sub_all - top_level_all)
        if module_only:
            result[module_name] = module_only

    return result


def _parse_all_names(init_file: Path) -> set[str]:
    """Extract __all__ symbol names from a Python file."""
    if not init_file.exists():
        return set()
    try:
        tree = ast.parse(init_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                    return {elt.value for elt in node.value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)}
    return set()


def _count_module_lines(source_dir: Path, module_name: str) -> int:
    """Count total lines of code in a module (excludes blank lines and comments)."""
    module_dir = source_dir / module_name
    if not module_dir.is_dir():
        single_file = source_dir / f"{module_name}.py"
        if single_file.exists():
            return _count_code_lines(single_file)
        return 0
    total = 0
    for py_file in sorted(module_dir.rglob("*.py")):
        total += _count_code_lines(py_file)
    return total


def _count_code_lines(path: Path) -> int:
    """Count non-blank, non-comment lines in a Python file."""
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    """Entry point for AI docs CLI with generate/check subcommands."""
    parser = argparse.ArgumentParser(description="AI documentation generator")
    parser.add_argument("--source-dir", type=Path, help="Source package directory")
    parser.add_argument("--tests-dir", type=Path, help="Tests directory")
    parser.add_argument("--output-dir", type=Path, help="Output .ai-docs directory")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("generate", help="Generate .ai-docs/ documentation")
    subparsers.add_parser("check", help="Validate .ai-docs/ completeness")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    source_dir, tests_dir, output_dir, repo_root = _resolve_paths(args)

    if args.command == "generate":
        return _run_generate(source_dir, tests_dir, output_dir, repo_root)
    return _run_check(source_dir, output_dir)


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Resolve source, tests, output directories and repo root from args or auto-detect."""
    cli_file = Path(__file__).resolve()
    repo_root = cli_file.parent.parent.parent
    source_dir = args.source_dir or (repo_root / "ai_pipeline_core")
    tests_dir = args.tests_dir or (repo_root / "tests")
    output_dir = args.output_dir or (repo_root / ".ai-docs")
    return source_dir, tests_dir, output_dir, repo_root


def _run_generate(source_dir: Path, tests_dir: Path, output_dir: Path, repo_root: Path) -> int:
    """Generate all module guides and README.md."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale files
    for existing in output_dir.glob("*.md"):
        existing.unlink()

    version = _read_version(repo_root)
    table = build_symbol_table(source_dir)
    import_map = _build_import_map(source_dir)
    module_import_map = _build_module_import_map(source_dir, EXCLUDED_MODULES)
    generated: list[tuple[str, int]] = []
    module_descriptions: dict[str, str] = {}
    guide_data_map: dict[str, GuideData] = {}
    module_lines: dict[str, int] = {}

    for module_name in _discover_modules(source_dir):
        data = build_guide(module_name, source_dir, tests_dir, table, TEST_DIR_OVERRIDES, repo_root)
        if not data.classes and not data.functions and not data.values:
            print(f"  skip {module_name} (no public symbols)")
            continue

        # Enrich guide data with purpose and imports
        data.purpose = _read_module_purpose(source_dir, module_name)
        guide_symbols = {c.name for c in data.classes} | {f.name for f in data.functions} | {v.name for v in data.values}
        data.imports = sorted(name for name in import_map.get(module_name, []) if name in guide_symbols)
        data.module_imports = sorted(name for name in module_import_map.get(module_name, []) if name in guide_symbols and name not in data.imports)

        if data.purpose:
            module_descriptions[module_name] = data.purpose

        content = render_guide(data, version=version)
        content = manage_guide_size(data, content)
        content = _consolidate_code_blocks(content)
        content = _normalize_whitespace(content)

        guide_path = output_dir / f"{module_name}.md"
        guide_path.write_text(content)
        size = len(content.encode("utf-8"))
        generated.append((module_name, size))
        guide_data_map[module_name] = data
        module_lines[module_name] = _count_module_lines(source_dir, module_name)
        print(f"  wrote {module_name}.md ({size:,} bytes)")

    # README.md
    readme_content = _render_readme(generated, guide_data_map, module_descriptions, module_lines, version)
    readme_content = _consolidate_code_blocks(readme_content)
    readme_content = _normalize_whitespace(readme_content)
    (output_dir / README_FILENAME).write_text(readme_content)
    print(f"  wrote {README_FILENAME} ({len(readme_content):,} bytes)")

    total = sum(size for _, size in generated)
    print(f"\nGenerated {len(generated)} guides ({total:,} bytes total)")
    return 0


def _run_check(source_dir: Path, output_dir: Path) -> int:
    """Validate .ai-docs/ completeness and size."""
    if not output_dir.is_dir():
        print("FAIL: .ai-docs/ directory does not exist. Run 'generate' first.", file=sys.stderr)
        return 1

    # README-specific size check (100KB hard error)
    readme = output_dir / README_FILENAME
    if readme.exists():
        readme_size = len(readme.read_bytes())
        if readme_size > README_ERROR_SIZE:
            print(
                f"FAIL: {README_FILENAME} is {readme_size:,} bytes (max {README_ERROR_SIZE // 1024}KB)",
                file=sys.stderr,
            )
            return 1

    result = validate_all(output_dir, source_dir, excluded_modules=EXCLUDED_MODULES)

    if result.missing_symbols:
        print(f"FAIL: {len(result.missing_symbols)} public symbols missing from guides:")
        for sym in result.missing_symbols:
            print(f"  - {sym}")
    if result.private_reexports:
        print(f"FAIL: {len(result.private_reexports)} symbols in __all__ imported from private modules:")
        for msg in result.private_reexports:
            print(f"  - {msg}")
    if result.size_violations:
        print(f"WARNING: {len(result.size_violations)} guides exceed size limit:")
        for name, size in result.size_violations:
            print(f"  - {name}: {size:,} bytes")

    if result.is_valid:
        print("OK: .ai-docs/ is up-to-date")
        return 0
    return 1


def _render_readme(
    generated: list[tuple[str, int]],
    guide_data_map: dict[str, GuideData],
    module_descriptions: dict[str, str],
    module_lines: dict[str, int],
    version: str,
) -> str:
    """Render README.md with version, reading order, and per-module API as Python code snippets."""
    lines: list[str] = []

    # Header
    title = f"# ai-pipeline-core v{version} — API Reference" if version else "# ai-pipeline-core — API Reference"
    lines.extend([
        "<!-- Auto-generated by ai_pipeline_core.docs_generator — DO NOT EDIT MANUALLY -->",
        "",
        title,
        "",
        "Auto-generated API reference. Do not edit manually. Run: `make docs-ai-build`",
        "",
        "## Reading Order",
        "",
    ])
    for i, (name, _) in enumerate(generated, 1):
        desc = f" — {module_descriptions[name]}" if name in module_descriptions else ""
        lines.append(f"{i}. [{name}]({name}.md){desc}")

    # Per-module sections
    for name, _ in generated:
        data = guide_data_map.get(name)
        if data:
            _render_module_section(name, data, module_descriptions, module_lines, lines)

    return "\n".join(lines)


def _render_module_section(
    name: str,
    data: GuideData,
    module_descriptions: dict[str, str],
    module_lines: dict[str, int],
    lines: list[str],
) -> None:
    """Render a single module section for README.md."""
    loc = module_lines.get(name, 0)
    lines.extend(["", f"## {name}", ""])
    desc = module_descriptions.get(name, "")
    if desc:
        lines.append(desc)
        lines.append("")
    lines.append(f"**Source**: {loc:,} lines of code | [Full guide]({name}.md)")
    lines.append("")

    if data.values:
        lines.append("### Types & Constants")
        lines.append("")
        lines.append("```python")
        for val in data.values:
            source = val.source.strip() if val.source.strip() else val.name
            lines.extend(source.splitlines())
        lines.append("```")
        lines.append("")

    if data.classes:
        lines.append("### Classes")
        lines.append("")
        lines.append("```python")
        for i, cls in enumerate(data.classes):
            if i > 0:
                lines.append("")
                lines.append("")
            _render_class_summary(cls, lines)
        lines.append("```")
        lines.append("")

    if data.functions:
        lines.append("### Functions")
        lines.append("")
        lines.append("```python")
        for i, func in enumerate(data.functions):
            if i > 0:
                lines.append("")
            _render_function_summary(func, lines)
        lines.append("```")
        lines.append("")


def _format_class_field(var_name: str, type_ann: str, default: str, description: str = "") -> str:
    """Format a single class field as a Python declaration line."""
    if type_ann and default:
        line = f"    {var_name}: {type_ann} = {default}"
    elif type_ann:
        line = f"    {var_name}: {type_ann}"
    else:
        line = f"    {var_name} = {default}"
    if description:
        return f"{line}  # {description}"
    return line


def _unpack_class_field(field: tuple[str, ...]) -> tuple[str, str, str, str]:
    """Unpack class field tuple and support legacy 3-item tuples."""
    var_name = field[0] if len(field) > 0 else ""
    type_ann = field[1] if len(field) > 1 else ""
    default = field[2] if len(field) > 2 else ""
    description = field[3] if len(field) > 3 else ""
    return var_name, type_ann, default, description


def _render_inherited_methods(cls: ClassInfo, lines: list[str]) -> None:
    """Render inherited methods as comment lines inside a class code block."""
    inherited = [m for m in cls.methods if m.is_inherited and is_public_name(m.name)]
    if not inherited:
        return
    lines.append("")
    groups: dict[str, list[str]] = {}
    for m in inherited:
        parent = m.inherited_from or "unknown"
        groups.setdefault(parent, []).append(m.name)
    for parent, names in groups.items():
        lines.append(f"    # Inherited from {parent}: {', '.join(sorted(names))}")


def _render_method_stub(method: MethodInfo, lines: list[str]) -> None:
    """Render a method stub with optional docstring."""
    if method.docstring:
        doc = method.docstring.splitlines()[0].strip()
        lines.append(f"    def {method.name}{method.signature}:")
        lines.append(f'        """{doc}"""')
    else:
        lines.append(f"    def {method.name}{method.signature}: ...")


def _render_class_summary(cls: ClassInfo, lines: list[str]) -> None:
    """Render a class as Python code lines (no code fence — caller wraps)."""
    bases_str = f"({', '.join(cls.bases)})" if cls.bases else ""
    lines.append(f"class {cls.name}{bases_str}:")

    if cls.docstring:
        doc_line = cls.docstring.splitlines()[0].strip()
        lines.append(f'    """{doc_line}"""')

    if cls.class_vars:
        lines.append("")
        lines.append("    # Fields")
        lines.extend(_format_class_field(name, ann, default, description) for name, ann, default, description in map(_unpack_class_field, cls.class_vars))

    own_methods = [m for m in cls.methods if not m.is_inherited and is_public_name(m.name)]
    if own_methods:
        lines.append("")
        lines.append("    # Methods")
        for method in own_methods:
            if method.is_property:
                lines.append("    @property")
            elif method.is_classmethod:
                lines.append("    @classmethod")
            _render_method_stub(method, lines)

    _render_inherited_methods(cls, lines)


def _render_function_summary(func: FunctionInfo, lines: list[str]) -> None:
    """Render a function as Python code for README.md."""
    prefix = "async " if func.is_async else ""
    lines.append(f"{prefix}def {func.name}{func.signature}:")
    if func.docstring:
        doc = func.docstring.splitlines()[0].strip()
        lines.append(f'    """{doc}"""')
    else:
        lines.append("    ...")


if __name__ == "__main__":
    raise SystemExit(main())
