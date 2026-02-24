"""Validation utilities for AI documentation completeness and size."""

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from ai_pipeline_core.docs_generator.extractor import is_public_name
from ai_pipeline_core.docs_generator.trimmer import MAX_GUIDE_SIZE

__all__ = [
    "ValidationResult",
    "validate_all",
    "validate_completeness",
    "validate_private_reexports",
    "validate_size",
]

# Generic entry-point names that are not part of the public API
_EXCLUDED_SYMBOLS: frozenset[str] = frozenset({"main"})


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Aggregated validation result across all checks."""

    missing_symbols: tuple[str, ...]
    size_violations: tuple[tuple[str, int], ...]
    private_reexports: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Completeness and private reexport checks must pass. Size is warning-only."""
        return not self.missing_symbols and not self.private_reexports


def validate_completeness(ai_docs_dir: Path, source_dir: Path, excluded_modules: frozenset[str] = frozenset()) -> list[str]:
    """Return public symbols (by naming convention) not found in any guide file."""
    public_symbols = _find_public_symbols(source_dir, excluded_modules)
    guide_content = _read_all_guides(ai_docs_dir)
    return [
        symbol
        for symbol in sorted(public_symbols)
        if not re.search(rf"\bclass {re.escape(symbol)}\b", guide_content)
        and not re.search(rf"\bdef {re.escape(symbol)}\b", guide_content)
        and not re.search(rf"\b{re.escape(symbol)} =", guide_content)
    ]


def validate_size(ai_docs_dir: Path, max_size: int = MAX_GUIDE_SIZE) -> list[tuple[str, int]]:
    """Return guide files exceeding max_size bytes. Skips README.md (separate thresholds)."""
    violations: list[tuple[str, int]] = []
    if not ai_docs_dir.is_dir():
        return violations
    for guide in sorted(ai_docs_dir.glob("*.md")):
        if guide.name == "README.md":
            continue
        size = len(guide.read_bytes())
        if size > max_size:
            violations.append((guide.name, size))
    return violations


def validate_all(
    ai_docs_dir: Path,
    source_dir: Path,
    excluded_modules: frozenset[str] = frozenset(),
) -> ValidationResult:
    """Run all validation checks and return aggregated result."""
    return ValidationResult(
        missing_symbols=tuple(validate_completeness(ai_docs_dir, source_dir, excluded_modules)),
        size_violations=tuple(validate_size(ai_docs_dir)),
        private_reexports=tuple(validate_private_reexports(source_dir, excluded_modules)),
    )


def validate_private_reexports(source_dir: Path, excluded_modules: frozenset[str] = frozenset()) -> list[str]:
    """Detect symbols in __all__ that are imported from private modules.

    Scans every public .py file with __all__ in the package (not just __init__.py).
    For each symbol in __all__, traces the import back to its source module. If the
    source is a _-prefixed module or package, the symbol will appear in IMPORTS but
    have no definition in the guide — a phantom import.

    For non-__init__.py files, symbols that are also exported from the parent
    __init__.py are considered legitimate re-exports (the file is a designated
    public re-export surface like llm/types.py) and are not flagged.
    """
    violations: list[str] = []

    for py_file in sorted(source_dir.rglob("*.py")):
        # Skip _-prefixed files (except __init__.py)
        if py_file.name.startswith("_") and py_file.name != "__init__.py":
            continue
        relative = py_file.relative_to(source_dir)
        # Skip _-prefixed packages (they're entirely private)
        if any(part.startswith("_") for part in relative.parent.parts):
            continue
        # Skip excluded modules
        top_module = relative.parts[0] if len(relative.parts) > 1 else None
        if top_module and top_module in excluded_modules:
            continue

        relative_path = str(relative)

        all_names = _parse_init_all(py_file)
        if not all_names:
            continue

        private_sources = _find_private_import_sources(py_file)

        # For non-__init__.py files, exclude symbols that are also in the parent
        # __init__.py __all__ (they're legitimate re-export surfaces)
        if py_file.name != "__init__.py":
            parent_init = py_file.parent / "__init__.py"
            parent_all = _parse_init_all(parent_init)
            private_only = {k: v for k, v in private_sources.items() if k not in parent_all}
        else:
            private_only = private_sources

        for name in sorted(all_names & private_only.keys()):
            source_module = private_only[name]
            violations.append(
                f"{relative_path}: '{name}' in __all__ is imported from private module '{source_module}'. "
                f"Remove it from __all__ — symbols from _-prefixed modules are internal and won't appear in generated docs."
            )

    return violations


def _parse_init_all(init_file: Path) -> set[str]:
    """Extract __all__ symbol names from an __init__.py."""
    try:
        tree = ast.parse(init_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    for node in tree.body:
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
            value = node.value
        if isinstance(value, (ast.List, ast.Tuple)):
            return {elt.value for elt in value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)}
    return set()


def _find_private_import_sources(init_file: Path) -> dict[str, str]:
    """Map symbol names to their private source module for imports from _-prefixed modules.

    Returns {symbol_name: source_module_name} only for symbols imported from private modules.
    """
    try:
        tree = ast.parse(init_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return {}

    result: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        # Check if the import source is a private module
        # Relative: from ._images import ... → module="_images"
        # Relative nested: from ._llm_core import ... → module="_llm_core"
        # Absolute: from ai_pipeline_core._llm_core import ... → contains "_llm_core"
        parts = node.module.split(".")
        is_private = any(part.startswith("_") and part != "__init__" for part in parts)
        if not is_private:
            continue
        # Find the private part for the message
        private_part = next(part for part in parts if part.startswith("_") and part != "__init__")
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            result[name] = private_part
    return result


def _find_public_symbols(source_dir: Path, excluded_modules: frozenset[str] = frozenset()) -> set[str]:
    """Find all public symbols via naming convention in non-private modules."""
    symbols: set[str] = set()
    for py_file in sorted(source_dir.rglob("*.py")):
        if py_file.name.startswith("_") and py_file.name != "__init__.py":
            continue
        relative = py_file.relative_to(source_dir)
        top_module = relative.parts[0] if len(relative.parts) > 1 else relative.stem
        if top_module in excluded_modules or (len(relative.parts) > 1 and top_module.startswith("_")):
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if is_public_name(node.name) and node.name not in _EXCLUDED_SYMBOLS:
                    symbols.add(node.name)
            # NewType / type alias / constant
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if is_public_name(name) and name not in _EXCLUDED_SYMBOLS and ((name.isupper() and len(name) > 1) or _is_newtype_assign(node)):
                    symbols.add(name)
            elif isinstance(node, ast.TypeAlias):
                name = node.name.id
                if is_public_name(name) and name not in _EXCLUDED_SYMBOLS:
                    symbols.add(name)
    return symbols


def _is_newtype_assign(node: ast.Assign) -> bool:
    """Check if an Assign node is a NewType(...) call."""
    if isinstance(node.value, ast.Call):
        func = node.value.func
        if isinstance(func, ast.Name) and func.id == "NewType":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "NewType":
            return True
    return False


def _read_all_guides(ai_docs_dir: Path) -> str:
    """Concatenate all .md guide files into a single string for searching."""
    if not ai_docs_dir.is_dir():
        return ""
    return "\n".join([guide.read_text() for guide in sorted(ai_docs_dir.glob("*.md"))])
