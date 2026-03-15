"""Test scope detection from git changes and source-to-test mapping."""

import sys
from pathlib import PurePosixPath

from dev_cli._project import ProjectConfig, load_config
from dev_cli._state import git_changed_files


def detect_test_scope() -> tuple[list[str], list[str]]:
    """Detect which test directories should run based on changed files.

    Returns (test_dirs, changed_source_dirs) — both deduplicated and sorted.
    """
    cfg = load_config()
    changed = git_changed_files()
    if not changed:
        return [], []

    test_dirs: set[str] = set()
    source_dirs: set[str] = set()

    for filepath in changed:
        # If a test file itself changed, include its directory
        for test_root in cfg.test_roots:
            if filepath.startswith(test_root + "/"):
                parts = filepath.removeprefix(test_root + "/").split("/")
                if parts:
                    test_dir = f"{test_root}/{parts[0]}"
                    if (cfg.repo_root / test_dir).is_dir():
                        test_dirs.add(test_dir)

        # If source changed, map to test directories
        for source_prefix, mapped_test_dirs in cfg.source_to_test.items():
            if filepath.startswith(source_prefix + "/") or filepath.startswith(source_prefix):
                source_dirs.add(source_prefix)
                test_dirs.update(mapped_test_dirs)

        # conftest.py changes affect everything
        if filepath.endswith("conftest.py"):
            return [cfg.test_roots[0]] if cfg.test_roots else ["tests"], sorted(source_dirs)

    return sorted(test_dirs), sorted(source_dirs)


def resolve_scope(scope: str | None) -> list[str]:
    """Resolve a scope argument to test directories."""
    cfg = load_config()

    if scope is None:
        dirs, _ = detect_test_scope()
        return dirs if dirs else cfg.test_roots

    if scope in cfg.scope_aliases:
        return [cfg.scope_aliases[scope]]

    # Reject file paths — suggest the correct scope name
    if scope.endswith(".py"):
        suggestion = _suggest_scope_for_path(scope, cfg)
        hint = f"  Did you mean: dev test {suggestion}\n" if suggestion else ""
        print(
            f"ERROR: 'dev test' takes a scope name, not a file path.\n"
            f"{hint}"
            f"  Available scopes: {', '.join(sorted(cfg.scope_aliases))}\n"
            f"  Usage: dev test <scope>   e.g. dev test database",
            file=sys.stderr,
        )
        sys.exit(1)

    # Direct path (test directory)
    if (cfg.repo_root / scope).exists():
        return [scope]

    # Try as tests/<scope>
    for test_root in cfg.test_roots:
        test_path = f"{test_root}/{scope}"
        if (cfg.repo_root / test_path).exists():
            return [test_path]

    # Unknown scope — show available options
    print(
        f"ERROR: Unknown test scope '{scope}'.\n  Available scopes: {', '.join(sorted(cfg.scope_aliases))}\n  Usage: dev test <scope>   e.g. dev test database",
        file=sys.stderr,
    )
    sys.exit(1)


def get_source_dirs_for_test_dirs(test_dirs: list[str]) -> list[str]:
    """Reverse lookup: given test dirs, find source dirs that map to them."""
    cfg = load_config()
    scope_set = set(test_dirs)
    source_dirs: set[str] = set()
    for src, tests in cfg.source_to_test.items():
        if scope_set.intersection(tests):
            source_dirs.add(src)
    return sorted(source_dirs)


def _suggest_scope_for_path(file_path: str, cfg: ProjectConfig) -> str | None:
    """Extract a scope name from a file path like 'tests/database/test_foo.py'."""
    parts = PurePosixPath(file_path).parts
    for test_root in cfg.test_roots:
        root_parts = PurePosixPath(test_root).parts
        if parts[: len(root_parts)] == root_parts and len(parts) > len(root_parts):
            candidate = parts[len(root_parts)]
            if candidate in cfg.scope_aliases:
                return candidate
    return None
