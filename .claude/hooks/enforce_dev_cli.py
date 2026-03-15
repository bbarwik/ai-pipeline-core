"""Enforce dev CLI usage and block raw tool invocation.

PreToolUse hook for Bash commands. Exit 2 + stderr blocks. Exit 0 allows.

Blocked:
- pytest, ruff check/format, basedpyright/pyright/mypy → use dev CLI
- uv sync, uv run, uv venv, python -m venv, virtualenv → no venvs
- pytest | grep/head/tail → buffering hangs
"""

import json
import re
import sys

# Matches command position: start of string, after shell operators, or after (
_CMD = r"(?:^|[;&|(]\s*)"

_PYTEST_BLOCKED = (
    "BLOCKED: Use 'dev test' instead of raw pytest.\n"
    "  dev test              run unit tests\n"
    "  dev test --lf         rerun last failures\n"
    "  dev test --full       parallel full suite\n"
    "  dev test --available  include infrastructure tests"
)

RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # dev CLI piping — output is already captured to .tmp/dev-runs/
    (
        re.compile(r"\bdev\s+\S+.*\|\s*(?:grep|head|tail|less|more)\b"),
        "BLOCKED: Do not pipe dev output — it's already captured to .tmp/dev-runs/.\n"
        "  Run 'dev' commands directly and read the log file if you need details.\n"
        "  Example: dev info    (then read the output directly)",
    ),
    # pytest piping — must be checked before the general pytest rule
    (
        re.compile(r"(?:python3?\s+-m\s+)?pytest\s[^|]*\|\s*(?:grep|head|tail|less|more)\b"),
        "BLOCKED: Do not pipe pytest output — causes buffering hangs.\n"
        "  Use pytest native flags:\n"
        "  --tb=no     pass/fail counts only\n"
        "  --tb=line   one line per failure\n"
        "  -q          quiet mode\n"
        "  -k EXPR     filter tests",
    ),
    # raw pytest (but allow --version/--help)
    (
        re.compile(_CMD + r"(?:python3?\s+-m\s+)?pytest\s+(?!--version\b|--help\b|-h\b)"),
        _PYTEST_BLOCKED,
    ),
    # pytest with no args (bare "pytest" at end of string)
    (
        re.compile(_CMD + r"(?:python3?\s+-m\s+)?pytest\s*$", re.MULTILINE),
        _PYTEST_BLOCKED,
    ),
    (
        re.compile(_CMD + r"ruff\s+(?:check|format)\s"),
        "BLOCKED: Use 'dev lint' or 'dev format' instead of raw ruff.\n"
        "  dev lint    ruff check\n"
        "  dev format  ruff format + ruff check --fix\n"
        "  dev check   all checks (lint + typecheck + more)",
    ),
    (
        re.compile(_CMD + r"(?:basedpyright|pyright|mypy)\s"),
        "BLOCKED: Use 'dev typecheck' instead of running type checkers directly.\n  dev typecheck  run type checker\n  dev check      all checks",
    ),
    (
        re.compile(_CMD + r"uv\s+sync\b"),
        "BLOCKED: 'uv sync' creates a .venv. This devcontainer uses system-wide install.\n"
        "  Packages are installed via: uv pip install --system -e '.[dev]'\n"
        "  All tools are already on PATH.",
    ),
    (
        re.compile(_CMD + r"uv\s+run\s"),
        "BLOCKED: 'uv run' is unnecessary. All tools are on PATH in this devcontainer.\n  Instead of: uv run dev test\n  Just run:   dev test",
    ),
    (
        re.compile(_CMD + r"(?:uv\s+venv|python3?\s+-m\s+venv|virtualenv)\b"),
        "BLOCKED: Do not create virtual environments. This devcontainer uses system-wide install.\n"
        "  All dependencies are installed via: uv pip install --system -e '.[dev]'",
    ),
)


def check(command: str) -> str | None:
    """Return block message if command is disallowed, None if allowed."""
    for pattern, message in RULES:
        if pattern.search(command):
            return message
    return None


def main() -> None:
    data = json.load(sys.stdin)
    cwd = data.get("cwd", "")
    if ".tmp/" in cwd:
        return
    command = data.get("tool_input", {}).get("command", "")
    block_message = check(command)
    if block_message:
        print(block_message, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
