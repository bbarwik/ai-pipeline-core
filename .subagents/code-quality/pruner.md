---
description: Finds duplicated logic, dead/unused code, stale docstrings, and legacy remnants
---
Audit the provided source files for code that should not exist in its current form.

## DUPLICATED_LOGIC

Find code blocks with >80% structural similarity:

- Functions or methods differing only in field names, constants, or types — should be one function with parameters
- Match/case or if/elif branches repeating the same pattern with different values — should be table-driven or loop-based
- Copy-pasted try/except, validation, or transformation blocks across multiple methods
- `if A: ... elif B: ...` where B is actually a fallback from A (not mutually exclusive) — should be two sequential blocks with a condition between them

For each:
- **Locations**: both sides with file:line (quote max 5 lines each)
- **Similarity**: what makes them structurally equivalent (not just textually similar)
- **Fix**: smallest safe consolidation (parameterize, lookup table, shared helper)

## DEAD_CODE

Find code that serves no purpose:

- Functions, classes, methods, or constants defined but never called within the provided files
- Backward-compatibility shims, deprecation wrappers, old API aliases — no backward compat is guaranteed, dead code must be removed immediately
- Commented-out code blocks
- Comments referencing historical fixes: `# FIX`, `# Fixes #123`, `# Patch for`, `# Workaround for` — code must be self-explanatory, bug fixes are documented by regression tests, not comments
- Code after unconditional return/raise, branches guarded by always-true/false conditions
- Parameters accepted but never used in the function body (except `_` prefixed parameters)

Do NOT flag unused imports — automated linters handle those.

Distinguish:
- **SAFE_REMOVE**: clearly dead, no public API exposure, zero callers in provided files
- **NEEDS_CONFIRMATION**: public symbol or `__all__`-exported with no callers in provided files — might be used externally

## STALE_DOCSTRINGS

- Docstrings referencing parameter or field names that no longer exist in the function/class signature
- Docstrings that merely restate the function name or signature without adding any information beyond what the code already says

For each: quote the stale/redundant text and explain what's wrong.

## CANNOT_VERIFY
