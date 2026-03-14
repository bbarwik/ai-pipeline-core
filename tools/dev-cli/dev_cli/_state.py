"""Idempotency detection via file content hashing."""

import contextlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime

import xxhash


@dataclass(frozen=True, slots=True)
class PreviousRun:
    command: str
    timestamp: str
    exit_code: int
    summary: str
    log_file: str


def hash_tracked_files(*directories: str) -> str:
    """Hash tracked files under given directories (relative to repo root)."""
    from dev_cli._project import load_config

    cfg = load_config()

    hasher = xxhash.xxh64()
    for directory in sorted(directories):
        dir_path = cfg.repo_root / directory
        if not dir_path.exists():
            continue
        for filepath in sorted(dir_path.rglob("*")):
            if filepath.is_file() and filepath.suffix in cfg.tracked_extensions:
                try:
                    hasher.update(filepath.read_bytes())
                except OSError:
                    continue
    return hasher.hexdigest()


def git_changed_files() -> list[str]:
    """Get files changed relative to HEAD (staged + unstaged + untracked)."""
    from dev_cli._project import load_config

    cfg = load_config()

    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        capture_output=True,
        text=True,
        cwd=cfg.repo_root,
    )
    files = result.stdout.strip().splitlines() if result.returncode == 0 else []

    result2 = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        cwd=cfg.repo_root,
    )
    if result2.returncode == 0:
        files.extend(result2.stdout.strip().splitlines())

    return [f for f in files if any(f.endswith(ext) for ext in cfg.tracked_extensions)]


def load_previous_run(command_key: str) -> PreviousRun | None:
    """Load previous run state for a given command key."""
    from dev_cli._project import load_config

    state_file = load_config().state_file

    if not state_file.exists():
        return None
    try:
        data = json.loads(state_file.read_text())
        r = data.get("runs", {}).get(command_key)
        if r is None:
            return None
        return PreviousRun(
            command=r["command"],
            timestamp=r["timestamp"],
            exit_code=r["exit_code"],
            summary=r["summary"],
            log_file=r["log_file"],
        )
    except (json.JSONDecodeError, KeyError) as e:
        print(f"WARN  Ignoring corrupt state file: {e}", file=sys.stderr)
        return None


def save_run_state(command_key: str, command: str, exit_code: int, summary: str, log_file: str, file_hash: str) -> None:
    """Save run state for idempotency detection (atomic write)."""
    from dev_cli._project import load_config

    cfg = load_config()
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)

    data: dict = {}
    if cfg.state_file.exists():
        try:
            data = json.loads(cfg.state_file.read_text())
        except json.JSONDecodeError:
            data = {}

    runs = data.setdefault("runs", {})
    runs[command_key] = {
        "command": command,
        "timestamp": datetime.now(UTC).isoformat(),
        "exit_code": exit_code,
        "summary": summary,
        "log_file": log_file,
        "file_hash": file_hash,
    }

    fd, tmp_path = tempfile.mkstemp(dir=str(cfg.runs_dir), suffix=".state.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, cfg.state_file)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def check_idempotency(command_key: str, file_hash: str) -> PreviousRun | None:
    """Check if a command was already run with the same file hash."""
    from dev_cli._project import load_config

    state_file = load_config().state_file

    if not state_file.exists():
        return None
    try:
        data = json.loads(state_file.read_text())
        r = data.get("runs", {}).get(command_key)
        if r is None:
            return None
        if r.get("file_hash") == file_hash:
            return PreviousRun(
                command=r["command"],
                timestamp=r["timestamp"],
                exit_code=r["exit_code"],
                summary=r["summary"],
                log_file=r["log_file"],
            )
    except (json.JSONDecodeError, KeyError) as e:
        print(f"WARN  Ignoring corrupt state file: {e}", file=sys.stderr)
    return None
