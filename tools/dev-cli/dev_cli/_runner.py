"""Subprocess execution with output capture and summary formatting."""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

MAX_INLINE_FAILURES = 5


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
        summary = _summarize_pytest_report(report_path, result.returncode, rel_log)
    else:
        summary = _summarize_generic(result, label, rel_log)

    return result.returncode, summary, rel_log


def _summarize_pytest_report(report_path: Path, exit_code: int, log_file: str) -> str:
    """Summarize pytest results from reportlog JSONL."""
    passed = 0
    failed = 0
    errors = 0
    skipped = 0
    failures: list[dict] = []
    collect_errors: list[str] = []
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
            total_duration += entry.get("duration", 0.0)

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
        lines.append(f"PASS  test — {counts} in {duration_str}")
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
    print(f"  Override: dev {label} --force")
    if previous_exit_code != 0:
        print(f"  Previous log: {log_file}")
        print("  Rerun failures: dev test --lf")
    return previous_exit_code
