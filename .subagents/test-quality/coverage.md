---
description: Maps public API surface to tests and reports behaviors with zero test coverage
---
Read the provided source and test files. Build a behavior-to-test map and find every public API surface with NO test coverage.

Focus ONLY on whether tests EXIST. Do NOT critique assertion quality, edge-case depth, or test design — only presence or absence.

For each public source file (no `_` prefix on filename), catalog:
1. Every public class and its public methods (no `_` prefix, plus all dunder methods)
2. Every public function
3. Every return type variant (e.g., `X | None` — both paths need tests)
4. Every `raise` statement and every `logger.warning`/`logger.error` path

Cross-reference against the test files. For each test, note which public symbols it exercises (calls, instantiates, or triggers).

## UNTESTED_CLASSES
Classes with no test that instantiates or calls any of their methods:
- **Class**: `ClassName` in `path/file.py:LINE`
- **Public methods**: list of methods with no test coverage
- **Risk**: what could break silently without tests

## UNTESTED_METHODS
Methods on classes that DO have tests, but this specific method is never called in any test:
- **Method**: `ClassName.method_name` in `path/file.py:LINE`
- **Signature**: full signature with type hints

## UNTESTED_RETURN_PATHS
Methods with Union/Optional return types where only one variant is exercised in tests:
- **Method**: `ClassName.method_name` returns `ReturnType`
- **Tested path**: which variant tests exercise
- **Missing path**: which variant is never verified

## UNTESTED_ERROR_PATHS
`raise` statements, `logger.warning()`, or `logger.error()` calls that no test triggers:
- **Location**: `path/file.py:LINE`
- **Error**: exception type or log message
- **Trigger condition**: what input or state triggers this path

## PRIORITY_RANKING
Top 10 coverage gaps ranked by risk (likelihood of a real bug hiding there x impact if undetected).

## CANNOT_VERIFY
