"""CLI tool for prompt specification discovery, inspection, and rendering."""

import argparse
import ast
import importlib
import sys
import warnings
from pathlib import Path
from typing import Any, cast

from .render import render_preview
from .spec import PromptSpec

_SKIP_DIRS: frozenset[str] = frozenset({".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules", ".tmp"})

APPROX_CHARS_PER_TOKEN = 4


def _iter_python_files(root: Path) -> list[Path]:
    """Find Python files under root, skipping common non-source directories."""
    return [f for f in root.rglob("*.py") if not any(part in _SKIP_DIRS for part in f.relative_to(root).parts)]


def _module_name_from_path(file: Path, root: Path) -> str | None:
    """Derive a module name from a file path relative to root."""
    try:
        rel = file.relative_to(root).with_suffix("")
    except ValueError:
        return None
    parts = list(rel.parts)
    if not parts:
        return None
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def _file_defines_class(file: Path, class_name: str) -> bool:
    """Check via AST if file defines a class with the given name (no import needed)."""
    try:
        tree = ast.parse(file.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return False
    return any(isinstance(node, ast.ClassDef) and node.name == class_name for node in tree.body)


def _file_may_contain_specs(file: Path) -> bool:
    """Quick text scan for PromptSpec references (avoids importing every file)."""
    try:
        return "PromptSpec" in file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False


def _all_prompt_spec_subclasses() -> set[type[PromptSpec[Any]]]:
    """Collect all registered PromptSpec subclasses, excluding Pydantic-generated parameterized classes."""
    discovered: set[type[PromptSpec[Any]]] = set()
    stack = list(PromptSpec.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls not in discovered:
            # Skip Pydantic-generated parameterized classes (e.g. PromptSpec[str])
            if "[" not in cls.__name__:
                discovered.add(cls)
            stack.extend(cls.__subclasses__())
    return discovered


def _resolve_spec_class(ref: str, root: Path) -> type[PromptSpec[Any]]:
    """Resolve a spec reference to a class.

    Accepts:
        'module.path:ClassName'  — explicit module + class
        'ClassName'              — auto-discover by scanning Python files
    """
    if ":" in ref:
        module_name, class_name = ref.split(":", maxsplit=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            module = importlib.import_module(module_name)
        obj = getattr(module, class_name, None)
        if not isinstance(obj, type) or not issubclass(obj, PromptSpec):
            raise ValueError(f"'{ref}' is not a PromptSpec subclass")
        return cast(type[PromptSpec[Any]], obj)

    class_name = ref
    for py_file in _iter_python_files(root):
        if not _file_defines_class(py_file, class_name):
            continue
        module_name = _module_name_from_path(py_file, root)
        if module_name is None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                importlib.import_module(module_name)
        except Exception:  # noqa: BLE001, S112 — best-effort discovery, errors are not actionable
            continue

    matches = sorted(
        (cls for cls in _all_prompt_spec_subclasses() if cls.__name__ == class_name),
        key=lambda c: f"{c.__module__}.{c.__name__}",
    )
    if not matches:
        raise ValueError(f"PromptSpec '{class_name}' not found")
    if len(matches) > 1:
        names = ", ".join(f"{cls.__module__}:{cls.__name__}" for cls in matches)
        raise ValueError(f"Ambiguous: '{class_name}' matches {names}")
    return matches[0]


def _discover_all_specs(root: Path) -> tuple[list[type[PromptSpec[Any]]], list[str]]:
    """Import all modules that may contain specs and collect PromptSpec subclasses.

    Returns (sorted_specs, import_errors).
    """
    errors: list[str] = []
    for py_file in _iter_python_files(root):
        if not _file_may_contain_specs(py_file):
            continue
        module_name = _module_name_from_path(py_file, root)
        if module_name is None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                importlib.import_module(module_name)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{module_name}: {e}")

    specs = sorted(_all_prompt_spec_subclasses(), key=lambda c: (c.__module__, c.__name__))
    return specs, errors


def _output_label(spec_cls: type[PromptSpec[Any]]) -> str:
    """Build output type label like 'str', 'str [xml]', or 'RiskVerdict'."""
    if spec_cls.output_type is str:
        tags: list[str] = []
        if spec_cls.xml_wrapped:
            tags.append("xml")
        if spec_cls.output_structure:
            tags.append("structure")
        return f"str [{','.join(tags)}]" if tags else "str"
    return spec_cls.output_type.__name__


def _ensure_importable(root: Path) -> None:
    """Ensure project root is on sys.path."""
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print an aligned text table."""
    if not rows:
        return
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    separator = "  ".join("-" * w for w in widths)
    print(header_line)
    print(separator)
    for row in rows:
        print("  ".join(val.ljust(w) for val, w in zip(row, widths, strict=True)))


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _cmd_list(args: argparse.Namespace) -> int:
    """List all discovered PromptSpec subclasses."""
    _ensure_importable(args.root)
    specs, errors = _discover_all_specs(args.root)

    if not specs:
        print("No PromptSpec subclasses found.")
        if errors:
            print(f"\n{len(errors)} import error(s):", file=sys.stderr)
            for err in errors:
                print(f"  {err}", file=sys.stderr)
        return 0

    headers = ["Name", "Phase", "Docs", "Fields", "Output", "Module"]
    rows = [
        [
            cls.__name__,
            cls.phase,
            str(len(cls.input_documents)),
            str(len(cls.model_fields)),
            _output_label(cls),
            cls.__module__,
        ]
        for cls in specs
    ]

    print(f"{len(specs)} spec(s) found:\n")
    _print_table(headers, rows)

    if errors:
        print(f"\n{len(errors)} import error(s):", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)

    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Show detailed anatomy of a single spec."""
    _ensure_importable(args.root)

    try:
        spec_cls = _resolve_spec_class(args.spec, args.root)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Header
    doc_first_line = (spec_cls.__doc__ or "").strip().splitlines()[0]
    print(f"{spec_cls.__name__}")
    print(f"  {doc_first_line}")
    print(f"  Phase: {spec_cls.phase} | Module: {spec_cls.__module__}")

    # Role
    print(f"\n  Role: {spec_cls.role.__name__}")
    print(f'    "{spec_cls.role.text}"')

    # Input Documents
    docs = spec_cls.input_documents
    print(f"\n  Input Documents ({len(docs)}):")
    if docs:
        for doc_cls in docs:
            doc_desc = (doc_cls.__doc__ or "").strip().splitlines()[0] if doc_cls.__doc__ else "No description"
            print(f"    - {doc_cls.__name__}: {doc_desc}")
    else:
        print("    (none)")

    # Dynamic Fields
    fields = spec_cls.model_fields
    print(f"\n  Dynamic Fields ({len(fields)}):")
    if fields:
        for field_name, field_info in fields.items():
            annotation = field_info.annotation
            type_name = annotation.__name__ if annotation is not None and hasattr(annotation, "__name__") else str(annotation)
            print(f"    - {field_name} ({type_name}): {field_info.description}")
    else:
        print("    (none)")

    # Task
    print("\n  Task:")
    for line in spec_cls.task.splitlines():
        print(f"    {line}")

    # Rules
    if spec_cls.rules:
        print(f"\n  Rules ({len(spec_cls.rules)}):")
        for i, rule_cls in enumerate(spec_cls.rules, 1):
            first_line = rule_cls.text.splitlines()[0]
            suffix = " ..." if "\n" in rule_cls.text else ""
            print(f"    {i}. {rule_cls.__name__}: {first_line}{suffix}")

    # Guides
    if spec_cls.guides:
        print(f"\n  Guides ({len(spec_cls.guides)}):")
        for guide_cls in spec_cls.guides:
            chars = len(guide_cls.render())
            print(f"    - {guide_cls.__name__} ({guide_cls.template}, {chars:,} chars)")

    # Output
    print("\n  Output:")
    print(f"    Type: {_output_label(spec_cls)}")
    if spec_cls.xml_wrapped:
        print("    XML Wrapped: yes")
    if spec_cls.output_structure:
        print("    Structure:")
        for line in spec_cls.output_structure.splitlines():
            print(f"      {line}")
    if spec_cls.output_rules:
        print(f"    Output Rules ({len(spec_cls.output_rules)}):")
        for i, rule_cls in enumerate(spec_cls.output_rules, 1):
            print(f"      {i}. {rule_cls.__name__}: {rule_cls.text}")

    # Size estimate
    preview = render_preview(spec_cls)
    chars = len(preview)
    tokens = chars // APPROX_CHARS_PER_TOKEN
    print(f"\n  Rendered preview: {chars:,} chars (~{tokens:,} tokens)")

    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    """Render a prompt preview with placeholder values."""
    _ensure_importable(args.root)

    try:
        spec_cls = _resolve_spec_class(args.spec, args.root)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(render_preview(spec_cls, include_input_documents=not args.no_input_documents))
    return 0


_PROMPTS_DIR = ".prompts"


def _cmd_compile(args: argparse.Namespace) -> int:
    """Compile all discovered specs to .prompts/ directory as markdown files."""
    _ensure_importable(args.root)
    specs, errors = _discover_all_specs(args.root)

    if not specs:
        print("No PromptSpec subclasses found.")
        return 0

    out_dir = args.root / _PROMPTS_DIR
    out_dir.mkdir(exist_ok=True)

    # Remove stale files not matching any discovered spec
    current_names = {f"{cls.__name__}.md" for cls in specs}
    for existing in out_dir.glob("*.md"):
        if existing.name not in current_names:
            existing.unlink()

    written = 0
    for spec_cls in specs:
        rendered = render_preview(spec_cls)
        out_file = out_dir / f"{spec_cls.__name__}.md"
        out_file.write_text(rendered, encoding="utf-8")
        written += 1

    print(f"Compiled {written} prompt(s) to {_PROMPTS_DIR}/")
    if errors:
        print(f"\n{len(errors)} import error(s):", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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

    # compile
    compile_parser = subparsers.add_parser("compile", help="Compile all specs to .prompts/ directory")
    compile_parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root for class discovery")

    args = parser.parse_args(argv)

    handlers = {"list": _cmd_list, "inspect": _cmd_inspect, "render": _cmd_render, "compile": _cmd_compile}
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


__all__ = ["main"]
