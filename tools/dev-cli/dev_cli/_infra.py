"""Infrastructure availability detection for pytest marker groups."""

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

COMMAND_TIMEOUT_SECONDS = 5


@dataclass(frozen=True, slots=True)
class InfraCheck:
    """A single infrastructure requirement tied to a pytest marker."""

    marker: str
    label: str
    env_vars: tuple[str, ...]
    command: str | None


@dataclass(frozen=True, slots=True)
class InfraStatus:
    """Result of probing one infrastructure requirement."""

    check: InfraCheck
    available: bool
    detail: str


def parse_infrastructure(cli_config: dict) -> tuple[InfraCheck, ...]:
    """Parse [tool.dev-cli.infrastructure] from pyproject.toml."""
    raw = cli_config.get("infrastructure", {})
    checks: list[InfraCheck] = []
    for marker, spec in raw.items():
        if not isinstance(spec, dict):
            continue
        checks.append(
            InfraCheck(
                marker=marker,
                label=spec.get("label", marker),
                env_vars=tuple(spec.get("env", ())),
                command=spec.get("command"),
            )
        )
    return tuple(checks)


def load_dotenv(repo_root: Path) -> dict[str, str]:
    """Parse .env file without mutating os.environ."""
    env_file = repo_root / ".env"
    if not env_file.is_file():
        return {}
    result: dict[str, str] = {}
    try:
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if key:
                result[key] = value
    except OSError:
        pass
    return result


_command_cache: dict[str, tuple[bool, str]] = {}


def _check_command(command: str) -> tuple[bool, str]:
    """Run a command probe. Results cached by command string."""
    if command in _command_cache:
        return _command_cache[command]

    argv = shlex.split(command)
    if not shutil.which(argv[0]):
        result = (False, f"'{argv[0]}' not found")
        _command_cache[command] = result
        return result

    try:
        proc = subprocess.run(argv, capture_output=True, timeout=COMMAND_TIMEOUT_SECONDS)
        ok = proc.returncode == 0
        detail = "running" if ok else f"exited {proc.returncode}"
    except subprocess.TimeoutExpired:
        ok, detail = False, f"timed out ({COMMAND_TIMEOUT_SECONDS}s)"
    except OSError as e:
        ok, detail = False, str(e)

    result = (ok, detail)
    _command_cache[command] = result
    return result


def detect(checks: tuple[InfraCheck, ...], repo_root: Path) -> tuple[InfraStatus, ...]:
    """Run all infrastructure probes and return their status."""
    # Merge .env with real env (real env wins)
    env = load_dotenv(repo_root)
    env.update(os.environ)

    results: list[InfraStatus] = []
    for check in checks:
        problems: list[str] = []
        details: list[str] = []

        for var in check.env_vars:
            if env.get(var):
                source = ".env" if var not in os.environ else "env"
                details.append(f"{var} ({source})")
            else:
                problems.append(f"{var} not set")

        if check.command and not problems:
            ok, reason = _check_command(check.command)
            if ok:
                details.append(reason)
            else:
                problems.append(reason)

        available = not problems
        detail = ", ".join(details if available else problems)
        results.append(InfraStatus(check=check, available=available, detail=detail))

    return tuple(results)


def available_markers(statuses: tuple[InfraStatus, ...]) -> list[str]:
    """Return sorted list of markers whose infrastructure is available."""
    return sorted(s.check.marker for s in statuses if s.available)


def unavailable_marker_expr(statuses: tuple[InfraStatus, ...]) -> str:
    """Build -m expression excluding only unavailable markers. Empty if all available."""
    unavail = sorted(s.check.marker for s in statuses if not s.available)
    if not unavail:
        return ""
    return " and ".join(f"not {m}" for m in unavail)


def format_status(statuses: tuple[InfraStatus, ...]) -> str:
    """Format infrastructure status table for terminal display."""
    if not statuses:
        return ""
    lines = ["Infrastructure:"]
    label_width = max(len(s.check.label) for s in statuses)
    for s in statuses:
        state = "AVAILABLE" if s.available else "NOT AVAILABLE"
        lines.append(f"  {s.check.marker:<15} {s.check.label:<{label_width}}  {state:<15} ({s.detail})")
    return "\n".join(lines)
