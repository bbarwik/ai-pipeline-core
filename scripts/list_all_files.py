#!/usr/bin/env python3
import argparse
import ast
import io
import os
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from openai import OpenAI
    from pydantic import BaseModel, ConfigDict, Field
    AI_MODE_AVAILABLE = True
except ImportError:
    AI_MODE_AVAILABLE = False

# Configuration
IGNORE_PATTERNS = [
    ".claude/*", ".github/*", ".devcontainer/*",
    "tests/test_data/*", "test_data/*",
    "License", "LICENSE", ".gitignore", ".gitattributes",
    ".env.example", "scripts/*", "dependencies_docs/*",
    "tests/*", "CLAUDE.md", "Makefile", "API.md",
]

TREE_IGNORE = "__pycache__|*.pyc|.git|.pytest_cache|.ruff_cache|*.egg-info|.venv|venv|env|.env"
TREE_FLAGS = ["--dirsfirst", "--du", "--si", "--gitignore", "--charset=ascii"]
SEPARATOR = "=" * 40
DEFAULT_MAX_BYTES = 200_000
MIN_EXCLUDE_BYTES = 1000
BINARY_CHECK_SIZE = 8192
DOCSTRING_MAX_LINES = 10

class FileStatus(Enum):
    OK = "ok"
    SKIP = "SKIP"
    EMPTY = "EMPTY"
    BINARY = "BIN"

@dataclass
class FileInfo:
    path: str
    size: int
    status: FileStatus
    content: str = ""
    processed: str = ""

    def format_output(self, use_processed: bool = True) -> str:
        content = self.processed if use_processed else self.content
        if self.status != FileStatus.OK:
            content = self.status.value
        return f"===== FILE: {self.path} [{self.size} bytes] =====\n{content}\n===== END FILE ====="

def run_command(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def ensure_git_repo() -> None:
    if run_command(["git", "--version"]).returncode != 0:
        print("git could not be found. Please install git to run this script.")
        sys.exit(1)
    if run_command(["git", "rev-parse", "--is-inside-work-tree"]).returncode != 0:
        print("Not inside a git repository.")
        sys.exit(1)

def list_repo_files() -> List[str]:
    res = run_command(["git", "ls-files", "--cached", "--others", "--exclude-standard"])
    if res.returncode != 0:
        print(res.stderr.strip(), file=sys.stderr)
        sys.exit(1)
    return sorted(ln for ln in res.stdout.splitlines() if ln.strip())

def get_tree_string() -> str:
    res = run_command(["tree", "-I", TREE_IGNORE] + TREE_FLAGS)
    tree_output = res.stdout.rstrip() if res.returncode == 0 else "(tree unavailable)"
    return f"{SEPARATOR}\nPROJECT TREE\n{SEPARATOR}\n{tree_output}\n{SEPARATOR}"

def should_skip_file(path: str) -> bool:
    return ("test_data" in path or
            any(fnmatch(path, pat) for pat in ["scripts/*", "dependencies_docs/*"]))

def is_binary(path: str) -> bool:
    try:
        with open(path, "r") as f:
            f.read(BINARY_CHECK_SIZE)
        return False
    except:
        return True

def normalize_text(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n", text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    result = []
    for line in lines:
        m = re.match(r"^ +", line)
        if m:
            lead = m.group(0)
            new = "  " * (len(lead) // 4) + (" " * (len(lead) % 4))
            result.append(new + line[len(lead):])
        else:
            result.append(line)
    return "\n".join(result)

def truncate_docstrings(source: str) -> str:
    try:
        tree = ast.parse(source)
    except:
        return source

    line_starts = [0]
    for ln in source.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(ln))

    replacements = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not (getattr(node, "body", None) and isinstance(node.body[0], ast.Expr)):
            continue
        val = node.body[0].value
        if not (isinstance(val, ast.Constant) and isinstance(val.value, str)):
            continue

        start = line_starts[val.lineno - 1] + val.col_offset
        end = line_starts[val.end_lineno - 1] + val.end_col_offset
        docstring = source[start:end]

        m = re.match(r"(?i)([rubf]*)(['\"]{3})([\s\S]*?)(\2)$", docstring)
        if not m:
            continue

        prefix, quote, inner, _ = m.groups()
        lines = inner.splitlines()
        if len(lines) <= DOCSTRING_MAX_LINES:
            continue

        kept = "\n".join(lines[:DOCSTRING_MAX_LINES])
        size = len(inner) - len(kept)
        size_str = f"{size/1000:.1f}k" if size >= 1000 else str(size)
        truncated = f"{prefix}{quote}{kept}\n+{size_str}{quote}"
        replacements.append((start, end, truncated))

    for start, end, replacement in reversed(replacements):
        source = source[:start] + replacement + source[end:]

    return source

def remove_python_comments(source: str) -> str:
    shebang = ""
    if source.startswith("#!"):
        nl = source.find("\n")
        shebang = source[:nl + 1] if nl != -1 else source
        source = source[nl + 1:] if nl != -1 else ""

    try:
        tokens = [t for t in tokenize.generate_tokens(io.StringIO(source).readline)
                  if t.type != tokenize.COMMENT]
        return shebang + tokenize.untokenize(tokens)
    except:
        return shebang + source

def process_content(content: str, is_python: bool = False) -> str:
    if is_python:
        content = truncate_docstrings(content)
        content = remove_python_comments(content)
    else:
        lines = [ln for ln in content.splitlines()
                if not re.match(r"^\s*(#|//|;)", ln) or re.match(r"^\s*#!", ln)]
        content = "\n".join(lines)
    return normalize_text(content)

def read_file(path: str, process: bool = True) -> FileInfo:
    if should_skip_file(path):
        return FileInfo(path, 0, FileStatus.SKIP)

    try:
        size = os.path.getsize(path)
    except:
        size = 0

    if size == 0:
        return FileInfo(path, 0, FileStatus.EMPTY)

    if is_binary(path):
        return FileInfo(path, size, FileStatus.BINARY)

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except:
        return FileInfo(path, size, FileStatus.BINARY)

    processed = process_content(content, path.endswith(".py")) if process else content
    return FileInfo(path, size, FileStatus.OK, content, processed)

def order_files(files: List[str]) -> List[str]:
    ordered = []
    if "README.md" in files and os.path.isfile("README.md"):
        ordered.append("README.md")
    ordered.extend(sorted(f for f in files if "/" not in f and f != "README.md"))
    ordered.extend(sorted(f for f in files if "/" in f))
    return ordered

def load_env() -> None:
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value and key not in os.environ:
                    os.environ[key] = value
    except:
        pass

# AI-specific code
if AI_MODE_AVAILABLE:
    class ExcludedFile(BaseModel):
        model_config = ConfigDict(extra="forbid")
        file_path: str = Field(description="Path to the file to exclude.")
        file_size: int = Field(description="Size of the file in bytes.")
        total_excluded_bytes: int = Field(description="Total bytes excluded so far.")

    class FileExclusion(BaseModel):
        model_config = ConfigDict(extra="forbid")
        related_files: str = Field(description="Analysis which files which are related to the user request and should be kept.")
        exclusion_reasoning: str = Field(description="Explanation of exclusion choices")
        files_to_exclude: List[ExcludedFile] = Field(description="Files to exclude")

def get_ai_exclusions(
    client,
    model: str,
    user_request: str,
    max_bytes: int,
    files_dict: Dict[str, FileInfo],
    tree_string: str,
) -> List[str]:
    if not AI_MODE_AVAILABLE:
        return []

    catalog = "\n".join(f.format_output(use_processed=True) for f in files_dict.values())

    total_size = sum(f.size for f in files_dict.values() if f.status == FileStatus.OK)
    bytes_to_exclude = total_size - max_bytes

    if bytes_to_exclude <= MIN_EXCLUDE_BYTES:
        return []

    files_which_can_be_excluded = "# FILES WHICH CAN BE EXCLUDED\n\n"
    for path, f in files_dict.items():
        if f.status == FileStatus.OK:
            files_which_can_be_excluded += f"{path} [{f.size} bytes]\n"

    system_instructions = (
        "You are a file selection assistant. You will be given:\n"
        "1) A project tree structure\n"
        "2) A catalog of files with their processed contents (comments removed, docstrings truncated)\n\n"
        "3) User request describing what types of files are required\n\n"
        "Your task has two primary objectives:\n"
        "1) Understand which files are most unrelated to the user request.\n"
        f"2) Exclude files which total sum of their sizes of at least {bytes_to_exclude} bytes. Exclude only files which content is listed.\n"
        f"3) Stop excluding files when you exclude files with their total sum of sizes of {bytes_to_exclude} bytes.\n"
        "Do not exclude files which content is not listed. Files which content is not present are already excluded."
    )

    content = tree_string + "\n" + catalog

    user_request_msg = (
        f"# User request\n\n{user_request}\n\n"
        f"Remember to exclude files which are most unrelated to the user request and total sum of their sizes of at least {bytes_to_exclude} bytes."
        f"Stop excluding more files when you exclude files with their total sum of sizes of at least {bytes_to_exclude} bytes."
    )

    resp = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": [{"type": "text", "text": content}]},
            {"role": "user", "content": [{"type": "text", "text": files_which_can_be_excluded}]},
            {"role": "user", "content": [{"type": "text", "text": user_request_msg}]},
        ],
        response_format=FileExclusion,
    )

    if resp.choices and hasattr(resp.choices[0].message, "parsed") and resp.choices[0].message.parsed:
        return [f.file_path for f in resp.choices[0].message.parsed.files_to_exclude]
    return []

def main() -> int:
    parser = argparse.ArgumentParser(
        description="List repository files, optionally filtered by AI.",
        epilog="Examples:\n"
               "  %(prog)s                    # List all files\n"
               "  %(prog)s \"include llm\"      # Include LLM-related files\n"
               "  %(prog)s \"exclude tests\"    # Exclude test files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("criteria", nargs="?", help="Optional AI selection criteria")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES,
                       help=f"Maximum output size (default: {DEFAULT_MAX_BYTES})")
    args = parser.parse_args()

    ensure_git_repo()

    files = [f for f in list_repo_files()
             if os.path.isfile(f) and not any(fnmatch(f, pat) for pat in IGNORE_PATTERNS)]
    tree_string = get_tree_string()

    # Classic mode - no AI filtering
    if not args.criteria:
        print(tree_string)
        for path in order_files(files):
            file_info = read_file(path, process=True)
            print(file_info.format_output(use_processed=True))
        return 0

    # AI mode
    if not AI_MODE_AVAILABLE:
        print("AI mode requires 'openai' and 'pydantic' packages.", file=sys.stderr)
        return 1

    load_env()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("OPENAI_API_KEY is required for AI mode.", file=sys.stderr)
        return 1

    # Read all files with processing for AI
    files_dict = {path: read_file(path, process=True) for path in files}

    # Get AI exclusions
    client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL"))

    try:
        excluded = get_ai_exclusions(
            client, "grok-4-fast", args.criteria,
            args.max_bytes, files_dict, tree_string
        )
    except Exception as e:
        print(f"Error calling AI: {e}", file=sys.stderr)
        return 1

    # Output with full content
    print(tree_string)
    for path in order_files([p for p in files if p not in excluded]):
        file_info = read_file(path, process=False)  # Full content for output
        print(file_info.format_output(use_processed=False))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
