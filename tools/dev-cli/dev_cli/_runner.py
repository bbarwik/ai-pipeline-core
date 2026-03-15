"""Subprocess execution with output capture and summary formatting."""

import json
import re
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

MAX_INLINE_FAILURES = 5
SLOW_TEST_THRESHOLD_SECONDS = 30
MAX_RUN_FILE_AGE_SECONDS = 86400  # 24 hours


_cleanup_done = False


def _cleanup_old_runs() -> None:
    """Remove run artifacts older than 24 hours. Best-effort, silent."""
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True

    from dev_cli._project import load_config

    cfg = load_config()
    if not cfg.runs_dir.is_dir():
        return

    cutoff = time.time() - MAX_RUN_FILE_AGE_SECONDS
    deleted_any = False

    for entry in cfg.runs_dir.iterdir():
        if entry.name == ".state.json":
            continue
        try:
            if entry.stat().st_mtime < cutoff:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                deleted_any = True
        except OSError:
            continue

    if deleted_any:
        _prune_stale_state_entries(cfg)


def _prune_stale_state_entries(cfg) -> None:
    """Remove state entries whose log files no longer exist on disk."""
    if not cfg.state_file.exists():
        return
    try:
        data = json.loads(cfg.state_file.read_text())
        runs = data.get("runs", {})
        pruned = {k: v for k, v in runs.items() if (cfg.repo_root / v.get("log_file", "")).exists()}
        if len(pruned) < len(runs):
            data["runs"] = pruned
            cfg.state_file.write_text(json.dumps(data, indent=2))
    except json.JSONDecodeError, OSError:
        pass


def run_command(
    cmd: list[str] | tuple[str, ...],
    label: str,
    *,
    log_suffix: str | None = None,
    use_reportlog: bool = False,
) -> tuple[int, str, str]:
    """Run a command, capture output to file, print summary.

    Returns (exit_code, summary_line, log_file_relative_path).
    """
    _cleanup_old_runs()

    from dev_cli._project import load_config

    cfg = load_config()
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(UTC)
    suffix = log_suffix or label.replace(" ", "-")
    log_filename = f"{now:%Y%m%d-%H%M%S}-{suffix}.log"
    log_path = cfg.runs_dir / log_filename

    report_path = cfg.runs_dir / f"{now:%Y%m%d-%H%M%S}-{suffix}.jsonl" if use_reportlog else None

    full_cmd = list(cmd)
    if use_reportlog and report_path:
        full_cmd.extend(["--report-log", str(report_path)])

    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, cwd=cfg.repo_root)
    except FileNotFoundError:
        tool_name = full_cmd[0]
        summary = f"FAIL  {label} — '{tool_name}' not found\n  Install it or check your PATH. Command was: {' '.join(full_cmd)}"
        return 127, summary, ""

    # Save full output to log file
    with open(log_path, "w") as f:
        f.write(f"$ {' '.join(full_cmd)}\n")
        f.write(f"# exit code: {result.returncode}\n")
        f.write(f"# timestamp: {now.isoformat()}\n\n")
        if result.stdout:
            f.write(result.stdout)
        if result.stderr:
            f.write("\n--- stderr ---\n")
            f.write(result.stderr)

    rel_log = str(log_path.relative_to(cfg.repo_root))

    if use_reportlog and report_path and report_path.exists():
        summary = _summarize_pytest_report(report_path, result.returncode, rel_log, result.stdout)
    else:
        summary = _summarize_generic(result, label, rel_log)

    return result.returncode, summary, rel_log


_DESELECTED_RE = re.compile(r"(\d+) deselected")


def _parse_deselected_count(stdout: str) -> int:
    """Extract deselected count from pytest stdout (e.g., '54 deselected')."""
    match = _DESELECTED_RE.search(stdout)
    return int(match.group(1)) if match else 0


def _summarize_pytest_report(report_path: Path, exit_code: int, log_file: str, stdout: str = "") -> str:
    """Summarize pytest results from reportlog JSONL."""
    passed = 0
    failed = 0
    errors = 0
    skipped = 0
    failures: list[dict] = []
    collect_errors: list[str] = []
    slow_tests: list[tuple[str, float]] = []
    total_duration = 0.0
    first_start: float | None = None
    last_stop: float | None = None

    for line in report_path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        report_type = entry.get("$report_type")

        if report_type == "TestReport" and entry.get("when") == "call":
            outcome = entry.get("outcome", "")
            duration = entry.get("duration", 0.0)
            total_duration += duration

            start = entry.get("start")
            stop = entry.get("stop")
            if start is not None and (first_start is None or start < first_start):
                first_start = start
            if stop is not None and (last_stop is None or stop > last_stop):
                last_stop = stop

            if outcome == "passed":
                passed += 1
            elif outcome == "failed":
                failed += 1
                failures.append({
                    "nodeid": entry.get("nodeid", "?"),
                    "message": _extract_failure_message(entry),
                })
            elif outcome == "skipped":
                skipped += 1

            if duration >= SLOW_TEST_THRESHOLD_SECONDS:
                slow_tests.append((entry.get("nodeid", "?"), duration))

        elif report_type == "TestReport" and entry.get("when") == "setup" and entry.get("outcome") == "failed":
            errors += 1
            failures.append({
                "nodeid": entry.get("nodeid", "?"),
                "message": f"SETUP ERROR: {_extract_failure_message(entry)}",
            })

        elif report_type == "CollectReport" and entry.get("outcome") == "failed":
            nodeid = entry.get("nodeid", "?")
            msg = _extract_failure_message(entry)
            collect_errors.append(f"{nodeid}: {msg}" if msg else nodeid)

    total = passed + failed + errors + skipped
    deselected = _parse_deselected_count(stdout)
    wall_time = (last_stop - first_start) if first_start is not None and last_stop is not None else total_duration
    duration_str = f"{wall_time:.1f}s"

    # Collection failure — no tests ran
    if total == 0 and exit_code != 0:
        lines = [f"FAIL  test — collection failed (exit code {exit_code})"]
        if collect_errors:
            lines.append("")
            for err in collect_errors[:MAX_INLINE_FAILURES]:
                lines.append(f"  COLLECT ERROR: {err}")
            if len(collect_errors) > MAX_INLINE_FAILURES:
                lines.append(f"  ... and {len(collect_errors) - MAX_INLINE_FAILURES} more")
        lines.append(f"\n  Full output: {log_file}")
        return "\n".join(lines)

    parts = []
    if passed:
        parts.append(f"{passed} passed")
    if failed:
        parts.append(f"{failed} failed")
    if errors:
        parts.append(f"{errors} errors")
    if skipped:
        parts.append(f"{skipped} skipped")
    counts = ", ".join(parts) if parts else f"{total} tests"

    lines: list[str] = []

    if exit_code == 0:
        if deselected:
            lines.append(f"PASS  test — {counts}, {deselected} unchanged (testmon) in {duration_str}")
        else:
            lines.append(f"PASS  test — {counts} in {duration_str}")
        if slow_tests:
            slow_tests.sort(key=lambda x: -x[1])
            lines.append("")
            lines.append(f"  Slow tests (>{SLOW_TEST_THRESHOLD_SECONDS}s):")
            for nodeid, dur in slow_tests[:MAX_INLINE_FAILURES]:
                lines.append(f"    {nodeid} ({dur:.1f}s)")
            if len(slow_tests) > MAX_INLINE_FAILURES:
                lines.append(f"    ... and {len(slow_tests) - MAX_INLINE_FAILURES} more")
    else:
        lines.append(f"FAIL  test — {counts} in {duration_str}")
        lines.append("")
        for f in failures[:MAX_INLINE_FAILURES]:
            lines.append(f"  FAIL {f['nodeid']}")
            if f["message"]:
                lines.append(f"       {f['message']}")
        if len(failures) > MAX_INLINE_FAILURES:
            lines.append(f"  ... and {len(failures) - MAX_INLINE_FAILURES} more failures")
        lines.append("")
        lines.append(f"  Full output: {log_file}")
        if failed > 0:
            lines.append("  Next step:   dev test --lf")

    return "\n".join(lines)


def _extract_failure_message(entry: dict) -> str:
    """Extract a short failure message from a TestReport entry."""
    longrepr = entry.get("longrepr", "")
    if isinstance(longrepr, str):
        for line in reversed(longrepr.splitlines()):
            stripped = line.strip()
            if stripped.startswith("E "):
                return stripped[2:].strip()[:120]
        for line in longrepr.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:120]
    elif isinstance(longrepr, dict):
        crash = longrepr.get("reprcrash", {})
        return crash.get("message", "")[:120]
    return ""


def _summarize_generic(result: subprocess.CompletedProcess, label: str, log_file: str) -> str:
    """Summarize a non-pytest command result."""
    if result.returncode == 0:
        return f"PASS  {label}"

    lines = [f"FAIL  {label} (exit code {result.returncode})"]

    # Prefer stderr for diagnostics, fall back to stdout
    output = result.stderr or result.stdout or ""
    output_lines = [l for l in output.splitlines() if l.strip()]
    if output_lines:
        lines.append("")
        for line in output_lines[-5:]:
            lines.append(f"  {line.rstrip()}")

    lines.append(f"\n  Full output: {log_file}")
    return "\n".join(lines)


MAX_WORST_FILES = 5


def format_coverage_summary(runs_dir: Path, scoped: bool) -> str | None:
    """Parse coverage.json and return a concise summary."""
    cov_file = runs_dir / "coverage.json"
    if not cov_file.is_file():
        return None

    data = json.loads(cov_file.read_text())
    total_pct = data["totals"]["percent_covered"]

    files = [(path, info["summary"]["percent_covered"]) for path, info in data["files"].items()]
    files.sort(key=lambda x: x[1])
    worst = files[:MAX_WORST_FILES]

    lines: list[str] = []
    if scoped:
        lines.append(f"  Coverage: {total_pct:.1f}% (scoped run — threshold not enforced)")
    else:
        lines.append(f"  Coverage: {total_pct:.1f}%")

    if worst:
        lines.append("  Lowest:")
        for path, pct in worst:
            lines.append(f"    {path} ({pct:.0f}%)")

    lines.append(f"  HTML: {runs_dir / 'coverage-html' / 'index.html'}")
    return "\n".join(lines)


def print_skip(label: str, previous_timestamp: str, previous_exit_code: int, log_file: str) -> int:
    """Print skip message when no code changed since last run. Returns previous exit code."""
    status = "PASS" if previous_exit_code == 0 else "FAIL"
    print(f"SKIP  {label} — no changes since last run ({status} at {previous_timestamp})")
    if previous_exit_code != 0:
        print(f"  Previous log: {log_file}")
        print("  Rerun failures: dev test --lf")
    else:
        print("  All tests already verified. No action needed.")
    return previous_exit_code
