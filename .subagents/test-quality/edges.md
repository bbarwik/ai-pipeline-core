---
description: Finds missing edge-case scenarios for behaviors that already have baseline tests
---
Read the provided source and test files. For functions and methods that ALREADY have at least one baseline test, find missing edge-case scenarios where real production bugs hide.

CRITICAL RULE: Only analyze behaviors with at least one existing test. If a function has ZERO tests, put it in UNTESTED_FUNCTIONS — do NOT invent edge cases for untested code.

Do NOT critique test assertion quality or test design — only report missing scenarios.

For each source file, analyze:
1. Every conditional branch — what input triggers each path?
2. Every loop — what happens with 0, 1, and many items?
3. Every external call — what happens when it fails or times out?
4. Every type conversion — what happens with unexpected but valid input?

## BOUNDARY_VALUES
Find numeric constants, thresholds, and limits defined in the source code (as module-level constants, class constants, or in validation logic). For each boundary, check whether tests exercise values at, above, and below it.

For each:
- **Function**: `name` in `path/file.py:LINE`
- **Boundary**: the constant name and value
- **Existing test**: cite the baseline test that proves coverage exists
- **Missing**: which side of the boundary is untested (at, just above, or just below)

## EMPTY_NONE_ZERO
Functions accepting collections, optional values, or numerics where the degenerate case is untested:
- Empty list, empty dict, empty string, empty bytes, `b""`
- `None` for optional parameters
- Zero for numeric parameters
- Single-element collection when tests only use multi-element

For each:
- **Function**: `name` in `path/file.py:LINE`
- **Degenerate input**: what's untested
- **Expected behavior**: should it raise? return empty? use a default?

## CONCURRENT_FAILURES
Async code where partial failure, total failure, or timeout scenarios are untested:
- Parallel execution (gather, wait, TaskGroup) — what if some tasks fail and others succeed?
- Parallel execution — what if ALL tasks fail?
- Timeout scenarios — what happens when an async operation exceeds its timeout?
- Shared mutable state accessed from multiple concurrent coroutines

For each: cite the async pattern in source, the existing baseline test, and the missing failure scenario.

## ERROR_CASCADES
Error handling chains where recovery behavior is untested:
- Retry exhaustion — all retries fail, what happens to the caller?
- Fallback chains — primary, secondary, and tertiary all fail
- Error during error handling — cleanup or logging fails while handling another error
- Partial success — some items in a batch succeed, others fail, is the result correct?

## TYPE_COERCION
Functions accepting union types (e.g., `str | bytes | dict | BaseModel`) where not all input variants are tested:
- **Function**: `name` in `path/file.py:LINE`
- **Tested variants**: which types appear in test inputs
- **Untested variant**: which valid type is never passed in any test

## UNTESTED_FUNCTIONS
Functions found with zero baseline tests — out of scope for edge-case analysis. Listed here for completeness.

## CANNOT_VERIFY
