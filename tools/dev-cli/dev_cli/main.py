"""Dev CLI — enforces correct test/lint/check workflows.

Usage:
    dev test [SCOPE] [--lf] [--full] [--integration] [--all] [--force] [--test-timeout N]
    dev lint [--fix]
    dev format
    dev typecheck
    dev check [--fast]
    dev ci
    dev status
    dev list-tests [SCOPE] [--integration] [--all]
"""

import argparse
import sys
import time

from dev_cli._project import load_config
from dev_cli._runner import TEST_TIMEOUT_SECONDS, format_coverage_summary, print_skip, run_command
from dev_cli._scope import detect_test_scope, get_source_dirs_for_test_dirs, resolve_scope
from dev_cli._state import check_idempotency, hash_tracked_files, save_run_state


def cmd_test(args: argparse.Namespace) -> int:
    """Run tests with smart scoping and output management."""
    cfg = load_config()

    # --coverage validation
    if args.coverage and args.lf:
        print("ERROR: --coverage is incompatible with --lf.\n  Coverage requires a complete test run.\n  Use: dev test --coverage", file=sys.stderr)
        return 1

    # --coverage without scope implies --full
    if args.coverage and not args.scope:
        args.full = True

    # --available includes infra tests — run in parallel
    if hasattr(args, "available") and args.available:
        args.full = True

    if args.full and not args.scope:
        scope_dirs = list(load_config().test_roots)
        scope_label = "full"
    else:
        scope_dirs = resolve_scope(args.scope)
        scope_label = args.scope or "auto"

    cmd = list(cfg.command("pytest"))

    # Determine marker expression
    if hasattr(args, "available") and args.available and cfg.infrastructure:
        from dev_cli._infra import detect, unavailable_marker_expr

        statuses = detect(cfg.infrastructure, cfg.repo_root)
        marker = unavailable_marker_expr(statuses)
        scope_label = "available"
    elif args.all:
        marker = cfg.test_groups.get("all", "")
    elif args.integration:
        marker = cfg.test_groups.get("integration", "integration")
    elif args.scope and args.scope in cfg.test_groups and not args.lf:
        marker = cfg.test_groups[args.scope]
    else:
        marker = cfg.test_groups.get("unit", "")

    if marker:
        cmd.extend(["-m", marker])

    if args.lf:
        cmd.extend(["--lf", "--no-testmon"])
        scope_dirs = []
    elif args.full:
        cmd.extend(["-n", "auto", "--dist", "worksteal", "--no-testmon"])
    else:
        cmd.append("--testmon")

    # Coverage instrumentation
    if args.coverage:
        cmd.extend(["-p", "no:testmon"])
        for src in cfg.coverage_sources:
            cmd.append(f"--cov={src}")
        cmd.append("--cov-report=term-missing")
        cmd.append(f"--cov-report=json:{cfg.runs_dir / 'coverage.json'}")
        cmd.append(f"--cov-report=html:{cfg.runs_dir / 'coverage-html'}")
        cmd.append("--cov-config=pyproject.toml")

        # Threshold: enforce on full (unscoped) runs only
        if args.threshold is not None:
            cmd.append(f"--cov-fail-under={args.threshold}")
        elif not args.scope and cfg.coverage_fail_under is not None:
            cmd.append(f"--cov-fail-under={cfg.coverage_fail_under}")

    timeout = args.test_timeout if args.test_timeout is not None else TEST_TIMEOUT_SECONDS
    cmd.extend(["--tb=short", "--no-header", "-q", f"--timeout={timeout}"])
    cmd.extend(scope_dirs)

    # Include coverage in label and key to differentiate runs
    if args.coverage:
        scope_label = f"{scope_label}-cov"

    # Idempotency — hash both test AND source dirs
    hash_dirs = list(scope_dirs) if scope_dirs else cfg.test_roots
    source_dirs = get_source_dirs_for_test_dirs(scope_dirs) if scope_dirs else list(cfg.source_to_test.keys())
    hash_dirs.extend(source_dirs)
    command_key = f"test:{scope_label}:{marker}"

    if not args.force and not args.lf and not args.coverage:
        file_hash = hash_tracked_files(*hash_dirs)
        prev = check_idempotency(command_key, file_hash)
        if prev:
            return print_skip(f"test {scope_label}", prev.timestamp, prev.exit_code, prev.log_file)
    else:
        file_hash = hash_tracked_files(*hash_dirs)

    # Remove stale coverage artifacts before running
    if args.coverage:
        cov_json = cfg.runs_dir / "coverage.json"
        if cov_json.exists():
            cov_json.unlink()

    exit_code, summary, log_file = run_command(
        cmd,
        f"test {scope_label}",
        log_suffix=f"test-{scope_label}",
        use_reportlog=True,
    )

    save_run_state(command_key, " ".join(cmd), exit_code, summary, log_file, file_hash)
    print(summary)

    # Coverage summary from JSON report
    if args.coverage:
        cov_summary = format_coverage_summary(cfg.runs_dir, scoped=bool(args.scope))
        if cov_summary:
            print(cov_summary)

    # Suggest running infra tests if available and not already included
    if exit_code == 0 and not args.all and not getattr(args, "available", False) and cfg.infrastructure:
        from dev_cli._infra import available_markers, detect

        statuses = detect(cfg.infrastructure, cfg.repo_root)
        avail = available_markers(statuses)
        if avail:
            print(f"\n  Infrastructure available: {', '.join(avail)}")
            print("  Run: dev test --available")

    return exit_code


def cmd_lint(args: argparse.Namespace) -> int:
    """Run ruff check (and optionally fix)."""
    cfg = load_config()

    if not cfg.has_ruff:
        print("SKIP  lint — ruff not found")
        return 0

    if args.fix:
        return cmd_format(args)

    exit_code, summary, log_file = run_command(cfg.command("ruff", "check", "."), "lint:check", log_suffix="lint-check")
    if exit_code != 0:
        print(summary)
        print("\n  Auto-fix: dev format")
        return exit_code

    exit_code2, summary2, _ = run_command(cfg.command("ruff", "format", "--check", "."), "lint:format", log_suffix="lint-format")
    if exit_code2 != 0:
        print(summary2)
        print("\n  Auto-fix: dev format")
        return exit_code2

    print("PASS  lint")
    return 0


def cmd_format(args: argparse.Namespace) -> int:
    """Auto-fix: ruff format + ruff check --fix."""
    cfg = load_config()

    if not cfg.has_ruff:
        print("SKIP  format — ruff not found")
        return 0

    exit_code1, summary1, _ = run_command(cfg.command("ruff", "format", "."), "format:ruff-format", log_suffix="format-fmt")
    exit_code2, summary2, _ = run_command(cfg.command("ruff", "check", "--fix", "."), "format:ruff-fix", log_suffix="format-fix")

    if exit_code1 != 0 or exit_code2 != 0:
        print(summary1 if exit_code1 != 0 else summary2)
        return exit_code1 or exit_code2

    print("PASS  format — all files formatted and auto-fixed")
    return 0


def cmd_typecheck(args: argparse.Namespace) -> int:
    """Run type checker with correct project config."""
    cfg = load_config()

    if cfg.has_basedpyright:
        exit_code, summary, _ = run_command(cfg.command("basedpyright", "--level", "warning"), "typecheck:source", log_suffix="typecheck-src")
        if exit_code != 0:
            print(summary)
            return exit_code

        # Check for tests config
        if (cfg.repo_root / "pyrightconfig.tests.json").exists():
            exit_code2, summary2, _ = run_command(
                cfg.command("basedpyright", "--level", "error", "-p", "pyrightconfig.tests.json"),
                "typecheck:tests",
                log_suffix="typecheck-tests",
            )
            if exit_code2 != 0:
                print(summary2)
                return exit_code2
    elif cfg.has_pyright:
        exit_code, summary, _ = run_command(cfg.command("pyright"), "typecheck", log_suffix="typecheck")
        if exit_code != 0:
            print(summary)
            return exit_code
    elif cfg.has_mypy:
        exit_code, summary, _ = run_command(cfg.command("mypy", "."), "typecheck", log_suffix="typecheck")
        if exit_code != 0:
            print(summary)
            return exit_code
    else:
        print("SKIP  typecheck — no type checker found (basedpyright, pyright, mypy)")
        return 0

    print("PASS  typecheck")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Run all checks in correct order (fast first, slow last)."""
    cfg = load_config()

    steps = cfg.checks
    if args.fast:
        fast_names = {"lint", "typecheck"}
        steps = tuple(s for s in steps if s.name in fast_names)

    check_start = time.monotonic()

    for step in steps:
        step_start = time.monotonic()
        for cmd_tuple in step.commands:
            exit_code, summary, _ = run_command(list(cmd_tuple), step.name, log_suffix=f"check-{step.name}")
            if exit_code != 0:
                print(summary)
                return exit_code
        elapsed = time.monotonic() - step_start
        print(f"PASS  {step.name} ({elapsed:.1f}s)")

    total = time.monotonic() - check_start
    label = "fast " if args.fast else ""
    print(f"\nPASS  check — all {label}checks passed ({total:.1f}s)")
    return 0


def cmd_ci(args: argparse.Namespace) -> int:
    """Replicate the full CI pipeline locally."""
    print("Running CI pipeline locally...\n")
    return cmd_check(argparse.Namespace(fast=False))


def cmd_status(args: argparse.Namespace) -> int:
    """Show current state: changed files, last run results, suggested next command."""
    cfg = load_config()

    test_dirs, source_dirs = detect_test_scope()

    print("Changed source modules:")
    if source_dirs:
        for d in source_dirs:
            print(f"  {d}")
    else:
        print("  (none)")

    print("\nAffected test directories:")
    if test_dirs:
        for d in test_dirs:
            print(f"  {d}")
    else:
        print("  (none detected)")

    if cfg.state_file.exists():
        import json

        try:
            data = json.loads(cfg.state_file.read_text())
            runs = data.get("runs", {})
            if runs:
                print("\nLast runs:")
                for key, run in runs.items():
                    status = "PASS" if run["exit_code"] == 0 else "FAIL"
                    print(f"  {key}: {status} at {run['timestamp'][:19]}")
        except json.JSONDecodeError:
            pass

    print("\nSuggested next command:")
    if test_dirs:
        if len(test_dirs) == 1:
            alias = test_dirs[0].split("/")[-1]
            print(f"  dev test {alias}")
        else:
            print(f"  dev test  (auto-detect: {', '.join(test_dirs)})")
    else:
        print("  dev test  (no changes — will use testmon cache)")

    return 0


def cmd_list_tests(args: argparse.Namespace) -> int:
    """List tests, optionally filtered by scope and group."""
    cfg = load_config()
    cmd = list(cfg.command("pytest", "--collect-only", "-q", "--no-header"))

    if args.all:
        pass  # No marker filter — show everything
    elif args.integration:
        cmd.extend(["-m", "integration"])

    scope_dirs = resolve_scope(args.scope)
    cmd.extend(scope_dirs)

    exit_code, summary, log_file = run_command(cmd, "list-tests", log_suffix="list-tests")
    log_path = cfg.repo_root / log_file
    if log_path.exists():
        content = log_path.read_text()
        lines = content.splitlines()
        for line in lines[3:]:  # Skip header (command, exit code, timestamp)
            if line.strip():
                print(line)

    return exit_code


INFO_TEXT = """\
dev — Development CLI for test/lint/check workflows.

Commands and when to use them:

  dev test [SCOPE]       After changing code. Auto-detects affected tests from imports.
                         Scopes: dev test pipeline, dev test database, dev test llm, ...
  dev test --lf          After fixing a test failure. Reruns only previously failed tests.
  dev test --full        Before committing. Runs full suite in parallel.
  dev test --available   Run tests for all detected infrastructure (see below).
  dev test --coverage    Run full suite with code coverage report.
  dev test --coverage pipeline  Coverage for pipeline tests only (threshold not enforced).
  dev test --coverage --threshold 90  Override coverage threshold.
  dev test --integration To run integration tests explicitly.
  dev test --all         To run everything (unit + integration).
  dev test --test-timeout 10  Override per-test timeout (default: 300s).

  dev format             After writing code. Auto-fixes lint and formatting issues.
  dev lint               To check for lint errors without fixing them.
  dev typecheck          To check types. Uses basedpyright/pyright/mypy (auto-detected).

  dev check              Before committing. Runs all checks in order (lint -> typecheck
                         -> deadcode -> semgrep -> docstrings -> test). Stops on first failure.
  dev check --fast       Quick sanity check. Lint + typecheck only, skips tests.

  dev status             To see what changed, what was last run, and what to run next.
  dev list-tests [SCOPE] To see available tests. Add --integration or --all for more.

  dev ci                 To replicate the full CI pipeline locally before pushing.

Workflow:
  1. Write code
  2. dev format           (auto-fix lint/formatting)
  3. dev check --fast     (verify lint + types pass)
  4. dev test             (run affected tests)
  5. If tests fail: fix code, then dev test --lf
  6. dev check            (full validation before commit)

Output is saved to .tmp/dev-runs/ — use Read tool for full details.
Rerunning without code changes is automatically skipped (use --force to override).

Testmon (default mode):
  Tests are tracked by dependency — only tests affected by code changes actually run.
  Other tests are deselected (shown as "N unchanged (testmon)" in the summary).
  All tests in the scope are still verified — deselected tests passed on a previous run
  and their dependencies haven't changed. Do NOT use --force just because the passed
  count looks low — the deselected tests are already verified.
  --force is only needed for: flaky test investigation or after non-code changes (config,
  env vars) that testmon can't detect.
"""


def cmd_info(args: argparse.Namespace) -> int:
    """Print detailed usage guide with auto-detected project info."""
    cfg = load_config()
    print(INFO_TEXT)

    print("Auto-detected check pipeline (dev check):")
    for i, step in enumerate(cfg.checks, 1):
        cmds_str = " && ".join(" ".join(c) for c in step.commands)
        print(f"  {i}. {step.name}: {step.description}")
        print(f"     {cmds_str}")
    print()

    print(f"Available test scopes: {', '.join(sorted(cfg.scope_aliases))}")
    print(f"Test groups: {', '.join(sorted(cfg.test_groups))}")
    print(f"Runner: {' '.join(cfg.runner_prefix) or '(bare)'}")

    if cfg.infrastructure:
        from dev_cli._infra import detect, format_status

        print()
        statuses = detect(cfg.infrastructure, cfg.repo_root)
        print(format_status(statuses))
        avail = [s for s in statuses if s.available]
        if avail:
            print("\n  Run available infra tests: dev test --available")

    return 0


def build_parser() -> argparse.ArgumentParser:
    cfg = load_config()
    scope_names = ", ".join(sorted(cfg.scope_aliases))

    parser = argparse.ArgumentParser(
        prog="dev",
        description="Development CLI — enforces correct test/lint/check workflows.",
    )
    sub = parser.add_subparsers(dest="command")

    p_test = sub.add_parser("test", help="Run tests with smart scoping")
    p_test.add_argument("scope", nargs="?", help=f"Test scope: {scope_names} or a path")
    p_test.add_argument("--lf", action="store_true", help="Rerun only last-failed tests")
    p_test.add_argument("--full", action="store_true", help="Run full suite in parallel")
    p_test.add_argument("--integration", action="store_true", help="Run integration tests only")
    p_test.add_argument("--all", action="store_true", help="Run all tests (unit + integration)")
    p_test.add_argument("--available", action="store_true", help="Run tests for all available infrastructure")
    p_test.add_argument("--coverage", action="store_true", help="Run with code coverage (disables testmon, no scope implies --full)")
    p_test.add_argument("--threshold", type=int, default=None, help="Override coverage fail-under threshold")
    p_test.add_argument("--test-timeout", type=int, default=None, help=f"Per-test timeout in seconds (default: {TEST_TIMEOUT_SECONDS})")
    p_test.add_argument(
        "--force", action="store_true", help="Force rerun even if no changes (rarely needed — testmon and idempotency handle this automatically)"
    )
    p_test.set_defaults(func=cmd_test)

    p_lint = sub.add_parser("lint", help="Check lint (ruff check + format check)")
    p_lint.add_argument("--fix", action="store_true", help="Auto-fix (same as 'dev format')")
    p_lint.set_defaults(func=cmd_lint)

    p_format = sub.add_parser("format", help="Auto-fix: ruff format + ruff check --fix")
    p_format.set_defaults(func=cmd_format)

    p_tc = sub.add_parser("typecheck", help="Run type checker")
    p_tc.set_defaults(func=cmd_typecheck)

    p_check = sub.add_parser("check", help="Run all checks in correct order")
    p_check.add_argument("--fast", action="store_true", help="Lint + typecheck only (skip tests)")
    p_check.set_defaults(func=cmd_check)

    p_ci = sub.add_parser("ci", help="Replicate CI pipeline locally")
    p_ci.set_defaults(func=cmd_ci)

    p_status = sub.add_parser("status", help="Show changed files, last runs, suggestions")
    p_status.set_defaults(func=cmd_status)

    p_info = sub.add_parser("info", help="Show detailed usage guide with examples")
    p_info.set_defaults(func=cmd_info)

    p_list = sub.add_parser("list-tests", help="List tests by scope/group")
    p_list.add_argument("scope", nargs="?", help="Test scope")
    p_list.add_argument("--integration", action="store_true", help="Show integration tests")
    p_list.add_argument("--all", action="store_true", help="Show all tests including integration")
    p_list.set_defaults(func=cmd_list_tests)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        args = argparse.Namespace()
        exit_code = cmd_info(args)
    else:
        exit_code = args.func(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
