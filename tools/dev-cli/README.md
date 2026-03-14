# dev-cli

A development CLI that enforces correct test/lint/check workflows for AI coding agents.

## Problem

AI coding agents (Claude Code, Codex, etc.) consistently misuse developer tools during coding sessions:

- **Pipe pytest output through grep/head/tail** — causes buffering hangs, loses exit codes
- **Run full test suite for single-module changes** — wastes 3-5 minutes per run
- **Rerun failing commands without code changes** — pure waste
- **Never use `--lf`** — reruns entire suites instead of just failed tests
- **Run tests before linters** — waits minutes to discover syntax errors that lint catches in seconds
- **Ignore make targets** — manually constructs commands with wrong flags
- **Filter basedpyright through grep** — loses exit codes, hides real errors

Analysis of 8,694 bash commands across 137 Claude Code sessions confirmed these patterns occur hundreds of times despite explicit CLAUDE.md instructions prohibiting them. **AI agents don't follow instructions — they need environmental constraints.**

## Solution

`dev` is a single CLI tool that replaces direct use of `pytest`, `ruff`, `basedpyright`, and `make check`. It:

1. **Captures output to files** — full logs go to `.tmp/dev-runs/`, only a concise summary prints to stdout. Eliminates the need to pipe through grep/head/tail.
2. **Enforces correct flags** — always uses `-x`, `--tb=short`, correct marker expressions, proper basedpyright config.
3. **Detects unchanged code** — hashes source+test files with xxhash, skips reruns when nothing changed.
4. **Auto-scopes tests** — detects which test directories are affected by git changes.
5. **Orders checks correctly** — fast checks (lint, typecheck) run before slow checks (tests).
6. **Suggests next steps** — on failure, prints exactly what command to run next (e.g., `dev test --lf`).

## Usage

```bash
# Run tests (auto-detect scope from git changes, uses testmon)
dev test

# Run tests for a specific module
dev test pipeline
dev test database
dev test llm

# Rerun only last-failed tests
dev test --lf

# Run full suite in parallel
dev test --full

# Run integration tests
dev test --integration

# Run all tests (unit + integration)
dev test --all

# Force rerun even if no files changed
dev test pipeline --force

# Code coverage (full suite, threshold enforced from pyproject.toml)
dev test --coverage

# Code coverage for a specific module (threshold not enforced)
dev test --coverage pipeline

# Override coverage threshold
dev test --coverage --threshold 90

# Lint check
dev lint

# Auto-fix lint + formatting
dev format

# Type checking (basedpyright with correct config)
dev typecheck

# All checks in order (lint → typecheck → deadcode → docstrings → test)
dev check

# Fast checks only (lint + typecheck, skip tests)
dev check --fast

# Replicate CI pipeline locally
dev ci

# Show changed files, last run results, suggested next command
dev status

# List tests by scope/group
dev list-tests pipeline
dev list-tests --integration
dev list-tests --all
```

## Output Format

Concise summaries designed for AI consumption — no decorative boxes, no wasted tokens:

```
PASS  test — 271 passed in 1.9s
```

```
FAIL  test — 2 failed, 269 passed in 2.1s

  FAIL test_task.py::test_retry_on_failure
       AssertionError: expected 3 retries, got 1

  FAIL test_task.py::test_timeout_handling
       TimeoutError: task exceeded 30s timeout

  Full output: .tmp/dev-runs/20260314-130918-test-pipeline.log
  Next step:   dev test --lf
```

```
SKIP  test pipeline — no changes since last run (PASS at 2026-03-14T13:09:18)
  Override: dev test pipeline --force
```

Full output is always saved to `.tmp/dev-runs/` and can be read with Claude Code's `Read` tool when details are needed.

## Auto-Detection (Generic Design)

The tool has no hardcoded project configuration. Everything is auto-detected:

| What | How |
|------|-----|
| Repo root | Walk up from cwd looking for `pyproject.toml` / `.git` |
| Source packages | Find dirs with `__init__.py` at root or under `src/` |
| Test scopes | Subdirectories of `tests/` become scope names |
| Source→test mapping | Match source subdir names to test subdir names |
| Markers | Read from `[tool.pytest.ini_options].markers` in pyproject.toml |
| Test groups | Auto-exclude heavy markers (integration, clickhouse, etc.) for unit group |
| Available tools | Check PATH for ruff, basedpyright, pyright, mypy, vulture, etc. |
| Runner prefix | Detect uv.lock → `uv run`, poetry.lock → `poetry run`, else bare |
| CI check steps | Auto-build from detected tools, or read from `[tool.dev-cli.checks]` |

## Optional Configuration

Override auto-detection via `[tool.dev-cli]` in `pyproject.toml`:

```toml
[tool.dev-cli]
cache_dir = ".tmp/dev-runs"
tracked_extensions = [".py", ".toml"]

# Source dirs that map to non-obvious test dirs
[tool.dev-cli.source_to_test]
"ai_pipeline_core/database" = ["tests/database", "tests/deployment", "tests/replay"]
"ai_pipeline_core/_llm_core" = ["tests/llm"]

# Short aliases for 'dev test <alias>'
[tool.dev-cli.scopes]
db = "tests/database"
deploy = "tests/deployment"

# Custom CI check steps (overrides auto-detection)
# [[tool.dev-cli.checks]]
# name = "lint"
# run = [["ruff", "check", "."], ["ruff", "format", "--check", "."]]
```

## Key Design Decision: No pytest addopts

This tool requires that `pyproject.toml` does NOT set `addopts` with behavioral flags (`--quiet`, `--testmon`, `-m '...'`). The `dev` CLI owns all test behavior — having two sources of truth causes conflicts (the CLI has to parse, strip, and override addopts, which is fragile).

Keep in pyproject.toml only engine/metadata config: `asyncio_mode`, `strict`, `testpaths`, `markers`.

## Enforcement

AI agents bypass written instructions — enforcement requires environmental constraints.

A Claude Code `PreToolUse` hook (`.claude/hooks/enforce_dev_cli.py`) blocks raw tool invocations with actionable error messages pointing to the correct `dev` command.

**Blocked commands:**

| Command | Reason | Use instead |
|---------|--------|-------------|
| `pytest ...` | No output management, wrong flags | `dev test` |
| `ruff check/format` | No output management | `dev lint` / `dev format` |
| `basedpyright`/`pyright`/`mypy` | No output management | `dev typecheck` |
| `uv sync` | Creates .venv in devcontainer | `uv pip install --system` |
| `uv run ...` | Unnecessary — tools on PATH | Run commands directly |
| `uv venv`/`python -m venv`/`virtualenv` | No venvs in devcontainer | N/A |
| `pytest ... \| grep/head/tail` | Buffering hangs | Use pytest native flags |

**Allowed commands:** `dev *`, `make *`, `uv pip install`, `uvx *`, `pytest --version`, `pytest --help`, `ruff --version`, `interrogate`, `vulture`, `semgrep`.

## Installation

Part of the project's dev dependencies (uv workspace member):

```bash
uv pip install --system -e '.[dev]'
dev --help
```
