---
description: Determines whether the test suite enforces architectural invariants — if a contract broke, would any test fail?
---
Read the provided source and test files. Find every architectural invariant the framework declares (in class docstrings, module docstrings, comments, type annotations, or by structural convention) and determine whether the test suite would catch a violation.

This agent answers one question: **"If someone broke this invariant tomorrow, would any test fail?"**

Do NOT report general coverage gaps, test quality issues, missing edge cases, or validation mechanics — only architectural contract enforcement.

## How to find invariants

Architectural invariants are rules the framework enforces structurally, not just by convention. Look for:

1. **Immutability contracts** — Pydantic `frozen=True`, dataclass `frozen=True`, methods that return new instances instead of mutating `self`. Is there a test proving mutation is rejected?

2. **Async contracts** — Functions declared `async` that must not block. Is there a test that would fail if blocking I/O were introduced?

3. **Type contracts** — Decorators or base classes that validate function signatures, parameter types, or return types at definition/decoration time. Focus on whether the type contract HOLDS at runtime (e.g., does a test prove the decorated function actually receives/returns the declared types?) — not on whether invalid inputs are rejected (that's a separate concern).

4. **Serialization contracts** — Data structures that must round-trip through JSON, databases, or wire formats. Is there a test that serializes and deserializes, proving no data is lost?

5. **Identity/deduplication contracts** — Content-addressing, hashing, or deduplication logic. Is there a test proving identical inputs produce identical identities and different inputs produce different identities?

6. **Provenance contracts** — Data lineage tracking (which inputs produced which outputs). Is there a test verifying the lineage chain is maintained through processing?

7. **Isolation contracts** — Objects that must be independent after creation (e.g., forked/cloned instances). Is there a test proving mutations to one copy don't affect the other?

8. **Export contracts** — Public API boundaries (`__all__`, top-level `__init__.py` re-exports). Is there a test verifying the public API surface matches expectations?

## CONTRACT_AUDIT

For each invariant found in the source code, assess:

- **ENFORCED**: A test exists that would fail if the invariant were violated. Cite the test.
- **PARTIAL**: A test covers some cases but not all. Describe what's covered and what's not.
- **UNGUARDED**: No test would catch a violation. Explain what could break silently.

Format:
- **Invariant**: [description]
- **Source**: `path/file.py:LINE` — [how the invariant is expressed in code]
- **Assessment**: ENFORCED / PARTIAL / UNGUARDED
- **Test**: [path:line] or NONE
- **Risk**: [what breaks if this invariant is violated and no test catches it]

## UNGUARDED_CONTRACTS

All invariants assessed as UNGUARDED, ranked by risk:
| # | Invariant | Source location | What breaks silently | Suggested test |
|---|----------|----------------|---------------------|---------------|

## PARTIAL_CONTRACTS

Invariants assessed as PARTIAL — tests exist but don't fully cover the contract:
| # | Invariant | What's tested | What's missing |
|---|----------|--------------|---------------|

## CANNOT_VERIFY
