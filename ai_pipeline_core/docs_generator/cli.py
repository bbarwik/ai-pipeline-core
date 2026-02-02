"""CLI for AI documentation generation and validation."""

import argparse
import sys
from pathlib import Path

from ai_pipeline_core.docs_generator.extractor import build_symbol_table
from ai_pipeline_core.docs_generator.guide_builder import build_guide, render_guide
from ai_pipeline_core.docs_generator.trimmer import manage_guide_size
from ai_pipeline_core.docs_generator.validator import (
    HASH_FILE,
    compute_source_hash,
    validate_all,
)

EXCLUDED_MODULES: frozenset[str] = frozenset({"docs_generator"})


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
            modules.add(relative.parts[0])
        else:
            modules.add(relative.stem)
    return sorted(modules - EXCLUDED_MODULES)


def main(argv: list[str] | None = None) -> int:
    """Entry point for AI docs CLI with generate/check subcommands."""
    parser = argparse.ArgumentParser(description="AI documentation generator")
    parser.add_argument("--source-dir", type=Path, help="Source package directory")
    parser.add_argument("--tests-dir", type=Path, help="Tests directory")
    parser.add_argument("--output-dir", type=Path, help="Output .ai-docs directory")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("generate", help="Generate .ai-docs/ documentation")
    subparsers.add_parser("check", help="Validate .ai-docs/ is up-to-date")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    source_dir, tests_dir, output_dir, repo_root = _resolve_paths(args)

    if args.command == "generate":
        return _run_generate(source_dir, tests_dir, output_dir, repo_root)
    return _run_check(source_dir, tests_dir, output_dir)


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Resolve source, tests, output directories and repo root from args or auto-detect."""
    cli_file = Path(__file__).resolve()
    repo_root = cli_file.parent.parent.parent
    source_dir = args.source_dir or (repo_root / "ai_pipeline_core")
    tests_dir = args.tests_dir or (repo_root / "tests")
    output_dir = args.output_dir or (repo_root / ".ai-docs")
    return source_dir, tests_dir, output_dir, repo_root


def _run_generate(source_dir: Path, tests_dir: Path, output_dir: Path, repo_root: Path) -> int:
    """Generate all module guides, INDEX.md, and .hash file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale files
    for existing in output_dir.glob("*.md"):
        existing.unlink()
    hash_file = output_dir / HASH_FILE
    if hash_file.exists():
        hash_file.unlink()

    table = build_symbol_table(source_dir)
    generated: list[tuple[str, int]] = []

    for module_name in _discover_modules(source_dir):
        data = build_guide(module_name, source_dir, tests_dir, table, TEST_DIR_OVERRIDES, repo_root)
        if not data.classes and not data.functions:
            print(f"  skip {module_name} (no public symbols)")
            continue

        content = render_guide(data)
        content = manage_guide_size(data, content)
        content = _normalize_whitespace(content)

        guide_path = output_dir / f"{module_name}.md"
        guide_path.write_text(content)
        size = len(content.encode("utf-8"))
        generated.append((module_name, size))
        print(f"  wrote {module_name}.md ({size:,} bytes)")

    # INDEX.md
    index_content = _normalize_whitespace(_render_index(generated))
    (output_dir / "INDEX.md").write_text(index_content)
    print(f"  wrote INDEX.md ({len(index_content):,} bytes)")

    # .hash
    source_hash = compute_source_hash(source_dir, tests_dir)
    (output_dir / HASH_FILE).write_text(source_hash + "\n")
    print(f"  wrote {HASH_FILE}")

    total = sum(size for _, size in generated)
    print(f"\nGenerated {len(generated)} guides ({total:,} bytes total)")
    return 0


def _run_check(source_dir: Path, tests_dir: Path, output_dir: Path) -> int:
    """Validate .ai-docs/ freshness, completeness, and size."""
    if not output_dir.is_dir():
        print("FAIL: .ai-docs/ directory does not exist. Run 'generate' first.", file=sys.stderr)
        return 1

    result = validate_all(output_dir, source_dir, tests_dir, excluded_modules=EXCLUDED_MODULES)

    if not result.is_fresh:
        print("FAIL: .ai-docs/ is stale (source hash mismatch)")
    if result.missing_symbols:
        print(f"FAIL: {len(result.missing_symbols)} public symbols missing from guides:")
        for sym in result.missing_symbols:
            print(f"  - {sym}")
    if result.size_violations:
        print(f"WARNING: {len(result.size_violations)} guides exceed size limit:")
        for name, size in result.size_violations:
            print(f"  - {name}: {size:,} bytes")

    if result.is_valid:
        print("OK: .ai-docs/ is up-to-date")
        return 0
    return 1


def _render_index(generated: list[tuple[str, int]]) -> str:
    """Render INDEX.md with reading order, task lookup, imports, and size table."""
    lines: list[str] = [
        "# AI Documentation Index",
        "",
        "Auto-generated guide index. Do not edit manually.",
        "",
        "## Reading Order",
        "",
    ]
    for i, (name, _) in enumerate(generated, 1):
        lines.append(f"{i}. [{name}]({name}.md)")

    lines.extend([
        "",
        "## Task-Based Lookup",
        "",
        "| Task | Guide |",
        "| ---- | ----- |",
    ])
    task_map = {
        "Create/read documents": "documents",
        "Store/retrieve documents": "document_store",
        "Call LLMs": "llm",
        "Deploy pipelines": "deployment",
        "Load templates": "prompt_manager",
        "Process images": "images",
        "Define flows/tasks": "pipeline",
        "Configure settings": "settings",
        "Handle errors": "exceptions",
        "Log messages": "logging",
        "Debug & observe traces": "observability",
        "Test pipelines": "testing",
    }
    guide_set = {name for name, _ in generated}
    for task, guide in task_map.items():
        if guide in guide_set:
            lines.append(f"| {task} | [{guide}]({guide}.md) |")

    lines.extend([
        "",
        "## Module Sizes",
        "",
        "| Module | Size |",
        "| ------ | ---- |",
    ])
    for name, size in generated:
        lines.append(f"| {name} | {size:,} bytes |")
    total = sum(size for _, size in generated)
    lines.append(f"| **Total** | **{total:,} bytes** |")
    lines.append("")

    return "\n".join(lines)
