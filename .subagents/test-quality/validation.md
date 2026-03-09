---
description: Audits definition-time validation coverage — verifies that every __init_subclass__, decorator, and Pydantic validator has a rejection test
---
Read the provided source and test files. Find every validation that the framework performs at class definition time, decoration time, or model construction time. Then verify each one has a test proving it rejects invalid input.

These validations are the framework's first defense against misuse. If a validation exists but no test proves it rejects bad input, someone could remove the validation and no test would fail.

Focus ONLY on validation mechanics (checks that raise on invalid input). Do NOT report general test coverage, test quality, edge cases, or architectural contracts.

## INIT_SUBCLASS_MATRIX

For each class with `__init_subclass__`, produce a matrix:

### `ClassName` (`path/file.py:LINE`)
| # | Validation | Rejects when | Error type | Rejection test? | Test location |
|---|-----------|-------------|------------|----------------|---------------|
| 1 | [what it checks] | [invalid condition] | [ValueError/TypeError/etc] | YES / **MISSING** | [path:line or n/a] |

## DECORATOR_MATRIX

For each decorator that performs validation at decoration time (e.g., validates function signatures, return types, parameter names), produce a matrix:

### `@decorator_name` (`path/file.py:LINE`)
| # | Validation | Rejects when | Error type | Rejection test? | Test location |
|---|-----------|-------------|------------|----------------|---------------|
| 1 | [what it checks] | [invalid condition] | [ValueError/TypeError/etc] | YES / **MISSING** | [path:line or n/a] |

## PYDANTIC_MATRIX

For each Pydantic model with custom `@field_validator`, `@model_validator(mode='before')`, `@model_validator(mode='after')`, or `Field(...)` constraints that reject invalid values, produce a matrix:

### `ModelName` (`path/file.py:LINE`)
| # | Validator/Constraint | Field(s) | Rejects when | Rejection test? | Test location |
|---|---------------------|----------|-------------|----------------|---------------|
| 1 | [validator name or Field constraint] | [field(s)] | [invalid condition] | YES / **MISSING** | [path:line or n/a] |

## MESSAGE_QUALITY

For validations that DO have rejection tests, check whether the test verifies the error MESSAGE content (not just that an exception was raised).

Error messages should be actionable — stating what went wrong, how to fix it, and the correct usage. If a test only checks `pytest.raises(ValueError)` without matching the message, a future change could make the error message misleading while the test still passes.

- **Tests checking message content** (via `match=` or string assertion): count
- **Tests only checking exception type**: count
- **Worst offenders**: validations with complex, multi-sentence error messages that no test verifies

## CANNOT_VERIFY
