"""Automatic source-to-test dependency detection via AST import scanning."""

import ast
from collections import defaultdict
from pathlib import Path


def scan_test_imports(repo_root: Path, test_roots: list[str], source_packages: list[str]) -> dict[str, list[str]]:
    """Scan test files and build source_module → [test_dirs] mapping from actual imports.

    Parses every .py file under test_roots with ast, extracts imports from source_packages,
    and returns a dict mapping each source submodule to the test directories that import it.
    """
    # Build set of known source package prefixes for fast matching
    pkg_prefixes = set()
    for pkg in source_packages:
        # "ai_pipeline_core" → "ai_pipeline_core"
        # "src/mypackage" → "mypackage"
        pkg_prefixes.add(pkg.replace("/", ".").split(".")[-1] if "/" in pkg else pkg)

    source_to_tests: dict[str, set[str]] = defaultdict(set)

    for test_root in test_roots:
        test_path = repo_root / test_root
        if not test_path.is_dir():
            continue

        for py_file in sorted(test_path.rglob("*.py")):
            try:
                tree = ast.parse(py_file.read_bytes())
            except SyntaxError:
                continue

            # Determine test subdirectory (e.g., "tests/deployment")
            rel = py_file.relative_to(repo_root)
            parts = rel.parts
            test_dir = "/".join(parts[:2]) if len(parts) > 2 else str(rel.parent)

            for node in ast.walk(tree):
                modules = _extract_import_modules(node)
                for mod in modules:
                    source_key = _match_source_module(mod, source_packages, pkg_prefixes)
                    if source_key:
                        source_to_tests[source_key].add(test_dir)

    return {k: sorted(v) for k, v in sorted(source_to_tests.items())}


def _extract_import_modules(node: ast.AST) -> list[str]:
    """Extract module names from import statements."""
    if isinstance(node, ast.ImportFrom) and node.module:
        return [node.module]
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    return []


def _match_source_module(module_name: str, source_packages: list[str], pkg_prefixes: set[str]) -> str | None:
    """Match an import module name to a source submodule directory.

    E.g., "ai_pipeline_core.database.clickhouse" → "ai_pipeline_core/database"
    E.g., "ai_pipeline_core.settings" → "ai_pipeline_core" (root-level)
    E.g., "ai_pipeline_core" → "ai_pipeline_core" (root-level)
    """
    parts = module_name.split(".")
    if not parts:
        return None

    top = parts[0]
    if top not in pkg_prefixes:
        return None

    for pkg in source_packages:
        pkg_dotted = pkg.replace("/", ".")
        if not module_name.startswith(pkg_dotted):
            continue

        remaining = module_name[len(pkg_dotted) :]
        if not remaining:
            # Direct package import: `from ai_pipeline_core import X`
            return pkg

        if remaining.startswith("."):
            sub_parts = remaining.split(".")
            # sub_parts[0] is empty (from leading dot), sub_parts[1] is the submodule
            if len(sub_parts) >= 2:
                submod = sub_parts[1]
                source_dir = f"{pkg}/{submod}"
                return source_dir

        return pkg  # fallback: map to package root

    return None
