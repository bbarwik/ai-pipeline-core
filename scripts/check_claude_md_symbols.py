"""Verify CLAUDE.md references match actual codebase symbols.

Checks that model names, class names, and function names mentioned
in CLAUDE.md exist in the codebase (guards against stale documentation).
"""

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
SOURCE_DIR = REPO_ROOT / "ai_pipeline_core"

# Symbols mentioned in CLAUDE.md that are external or conceptual (not in our codebase)
KNOWN_EXTERNAL = frozenset({
    "BaseModel", "BaseSettings", "StrEnum", "Protocol", "NewType",
    "Pydantic", "Prefect", "LiteLLM", "OpenRouter", "ClickHouse", "Laminar",
    "LMNR", "OpenTelemetry", "OWASP", "GDPR", "SOC2", "RBAC",
    "AsyncOpenAI", "Redis",
})


def collect_codebase_symbols(source_dir: Path) -> set[str]:
    """Collect all class and function names from the codebase."""
    symbols: set[str] = set()
    for py_file in source_dir.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.add(node.name)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id.isupper() and len(target.id) > 1:
                    symbols.add(target.id)
    return symbols


def extract_backtick_references(text: str) -> list[str]:
    """Extract Python identifiers from backtick-quoted references in markdown."""
    # Match `ClassName`, `function_name()`, `CONSTANT_NAME`
    pattern = re.compile(r"`([A-Z][A-Za-z0-9_]*(?:\[.*?\])?)`")
    refs: list[str] = []
    for match in pattern.finditer(text):
        name = match.group(1)
        # Strip generic parameters like [T]
        name = re.sub(r"\[.*\]", "", name)
        # Strip trailing ()
        name = name.rstrip("()")
        if name and not name.startswith(("$", "#")):
            refs.append(name)
    return refs


def main() -> int:
    if not CLAUDE_MD.exists():
        print(f"CLAUDE.md not found at {CLAUDE_MD}")
        return 1

    content = CLAUDE_MD.read_text(encoding="utf-8")
    codebase_symbols = collect_codebase_symbols(SOURCE_DIR)
    references = extract_backtick_references(content)

    missing: list[str] = []
    for ref in sorted(set(references)):
        if ref not in codebase_symbols and ref not in KNOWN_EXTERNAL:
            missing.append(ref)

    if missing:
        print(f"WARNING: {len(missing)} CLAUDE.md references not found in codebase (advisory):")
        for sym in missing:
            print(f"  - {sym}")
        # Advisory only — don't fail the build
    return 0


if __name__ == "__main__":
    sys.exit(main())
