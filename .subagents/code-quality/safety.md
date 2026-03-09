---
description: Finds correctness traps — exception handling, data flow errors, and Python/Pydantic pitfalls that cause silent runtime failures
---
Audit the provided source files for patterns that compile, pass linting, and may even pass tests — but produce incorrect behavior at runtime. Focus on issues that require **reasoning** about code behavior, not issues that automated type checkers already catch.

## EXCEPTION_HANDLING

- `except Exception: pass` or `except: pass` — caught exceptions MUST be: (1) logged with context, OR (2) re-raised (possibly wrapped), OR (3) converted to a specific return value with a comment explaining why swallowing is safe
- Catching `TypeError`, `AssertionError`, `AttributeError`, or `KeyError` and logging a warning instead of re-raising — these indicate programming errors that must fail fast, not be silently handled
- Broad `except Exception` around `asyncio.gather`, `asyncio.wait`, or similar concurrency primitives — this swallows `CancelledError` and breaks task cancellation

For each: cite the handler, explain what error class is being silenced, and what the consequence is.

## MAGIC_NUMBERS

Numeric literals used as thresholds, limits, sizes, timeouts, or configuration values must be module-level or class-level constants with descriptive names. This makes values discoverable, documentable, and changeable in one place.

Exempt: `0`, `1`, `-1`, `2` (doubling/halving), and standard mathematical constants.

```python
# Wrong — what is 40?
if len(url) < 40:
    return url

# Correct
MIN_URL_LENGTH_FOR_SHORTENING = 40
```

For each: cite the literal, explain what it represents, suggest a constant name.

## PYDANTIC_TRAPS

Patterns specific to Pydantic that silently break validation or produce wrong data:

- `model_rebuild(force=True)` on models with `@field_validator` or `@model_validator` chains — this can destroy existing validators silently
- Custom `__init__` methods that accept raw types (e.g., `str`) for fields typed as `Enum` or custom types — bypasses Pydantic's type coercion and validation
- `tuple(...) or None` / `tuple(...) or default` — empty tuple `()` is falsy in Python, so a valid empty tuple silently becomes `None`. Must use explicit `if x is not None` checks instead of truthiness
- Mutation of frozen models via `model_copy(update={...})` targeting private `_`-prefixed fields that the model's validators don't re-validate

For each: cite the pattern and explain the silent failure mode.

## DATA_FLOW

Trace variables through assignment, modification, and usage. Find cases where the code's logic doesn't match the author's likely intent:

- **Variable shadowing**: a local variable is modified but the method later reads from `self` or a class attribute instead — the local modification is silently wasted
- **Silent data loss**: conditional branches that discard, skip, or transform data in edge cases the author likely didn't consider (e.g., content dropped when a secondary field is present, collections silently emptied when a flag is set)
- **Query semantics**: if code constructs SQL, API calls, or data queries, reason about whether the query semantics match what the surrounding code expects (e.g., JOIN type that deduplicates when the code expects all matches, NULL handling that silently excludes rows)
- **Generated output validity**: if code generates structured data (JSON Schema, YAML, SQL DDL, config), reason about whether the output is valid according to its format specification

For each: trace the actual data flow step by step, show where the value diverges from what the code expects, and explain the runtime consequence.

## DOMAIN_TYPE_GAPS

Do NOT report type errors that basedpyright catches (`Any`, `list` invariance, undefined names). Focus on type gaps that require domain reasoning:

- `str` used for parameters that only accept a subset of valid strings (no path separators, max length, specific format) — should be a custom type (`NewType`, `Annotated`, or wrapper class) so the type checker catches misuse at call sites
- `ClassVar[str]` values containing `{field_name}` template syntax — `ClassVar` values are set at class definition time, not interpolated at runtime, so `{...}` renders literally

For each: explain what domain constraint the type system fails to express.

## RESOURCE_SAFETY

- Clients, connection pools, thread pools, or executors created but never `.close()`d or `.shutdown()` — causes resource leaks in long-running processes
- `asyncio.Event`, `asyncio.Lock`, `asyncio.Semaphore`, or `asyncio.Queue` created at module scope — these bind to the event loop active at import time, which may differ from the event loop at runtime (especially in tests)
- `try/except ImportError` for conditional imports at module level — hides missing dependencies until a specific code path is hit at runtime, making failures hard to diagnose

For each: cite the pattern and explain the failure mode.

## CANNOT_VERIFY
