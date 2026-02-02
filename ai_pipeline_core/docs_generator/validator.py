"""Validation utilities for AI documentation freshness, completeness, and size."""

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

from ai_pipeline_core.docs_generator.extractor import is_public_name
from ai_pipeline_core.docs_generator.trimmer import MAX_GUIDE_SIZE

HASH_FILE = ".hash"
# Generic entry-point names that are not part of the public API
_EXCLUDED_SYMBOLS: frozenset[str] = frozenset({"main"})


@dataclass(frozen=True)
class ValidationResult:
    """Aggregated validation result across all checks."""

    is_fresh: bool
    missing_symbols: tuple[str, ...]
    size_violations: tuple[tuple[str, int], ...]

    @property
    def is_valid(self) -> bool:
        """Hard validations pass (freshness + completeness). Size is warning-only."""
        return self.is_fresh and not self.missing_symbols


def compute_source_hash(source_dir: Path, tests_dir: Path) -> str:
    """SHA256 hash of all .py files (sorted by relative path) under source and test dirs."""
    repo_root = source_dir.parent
    all_files: list[Path] = []
    for directory in (source_dir, tests_dir):
        if directory.is_dir():
            all_files.extend(directory.rglob("*.py"))

    sha = hashlib.sha256()
    for path in sorted(all_files, key=lambda p: p.relative_to(repo_root)):
        rel = str(path.relative_to(repo_root))
        sha.update(rel.encode())
        sha.update(path.read_bytes())
    return sha.hexdigest()


def validate_freshness(ai_docs_dir: Path, source_dir: Path, tests_dir: Path) -> bool:
    """Check whether .hash matches current source state."""
    hash_file = ai_docs_dir / HASH_FILE
    if not hash_file.exists():
        return False
    stored = hash_file.read_text().strip()
    return stored == compute_source_hash(source_dir, tests_dir)


def validate_completeness(ai_docs_dir: Path, source_dir: Path, excluded_modules: frozenset[str] = frozenset()) -> list[str]:
    """Return public symbols (by naming convention) not found in any guide file."""
    public_symbols = _find_public_symbols(source_dir, excluded_modules)
    guide_content = _read_all_guides(ai_docs_dir)
    return [symbol for symbol in sorted(public_symbols) if f"class {symbol}" not in guide_content and f"def {symbol}" not in guide_content]


def validate_size(ai_docs_dir: Path, max_size: int = MAX_GUIDE_SIZE) -> list[tuple[str, int]]:
    """Return guide files exceeding max_size bytes."""
    violations: list[tuple[str, int]] = []
    if not ai_docs_dir.is_dir():
        return violations
    for guide in sorted(ai_docs_dir.glob("*.md")):
        size = len(guide.read_bytes())
        if size > max_size:
            violations.append((guide.name, size))
    return violations


def validate_all(
    ai_docs_dir: Path,
    source_dir: Path,
    tests_dir: Path,
    excluded_modules: frozenset[str] = frozenset(),
) -> ValidationResult:
    """Run all validation checks and return aggregated result."""
    return ValidationResult(
        is_fresh=validate_freshness(ai_docs_dir, source_dir, tests_dir),
        missing_symbols=tuple(validate_completeness(ai_docs_dir, source_dir, excluded_modules)),
        size_violations=tuple(validate_size(ai_docs_dir)),
    )


def _find_public_symbols(source_dir: Path, excluded_modules: frozenset[str] = frozenset()) -> set[str]:
    """Find all public symbols via naming convention in non-private modules."""
    symbols: set[str] = set()
    for py_file in sorted(source_dir.rglob("*.py")):
        if py_file.name.startswith("_") and py_file.name != "__init__.py":
            continue
        relative = py_file.relative_to(source_dir)
        top_module = relative.parts[0] if len(relative.parts) > 1 else relative.stem
        if top_module in excluded_modules:
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if is_public_name(node.name) and node.name not in _EXCLUDED_SYMBOLS:
                symbols.add(node.name)
    return symbols


def _read_all_guides(ai_docs_dir: Path) -> str:
    """Concatenate all .md guide files into a single string for searching."""
    if not ai_docs_dir.is_dir():
        return ""
    return "\n".join([guide.read_text() for guide in sorted(ai_docs_dir.glob("*.md"))])
