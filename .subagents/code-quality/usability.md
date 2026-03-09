---
description: Evaluates public API design — does it enforce one correct way, absorb complexity, and guide users with actionable errors?
---
Audit the provided source files from the perspective of a developer (or AI coding agent) building an application ON this framework. A framework must absorb complexity so that application code is minimal, obvious, and hard to get wrong.

## ONE_CORRECT_WAY

There must be exactly one way to accomplish any given task. Multiple valid approaches create inconsistency — especially when AI coding agents write the code.

Look for:
- Multiple methods or patterns that achieve the same result (e.g., two ways to create the same object, two configuration mechanisms for the same setting)
- Parameters that interact in confusing ways — combinations where only some are valid but nothing prevents invalid combinations
- API surfaces where an AI agent might invent a new pattern because the correct one is not obvious from method names, types, and docstrings alone

For each: describe the ambiguity and which approach should be the single correct one.

## ABSORBS_COMPLEXITY

Does application code need to understand framework internals to use the API correctly?

Look for:
- Public methods that require the caller to perform setup or teardown steps that the framework could handle automatically
- Multi-step sequences where a single method call could encapsulate the entire workflow
- Configuration knobs that should have sensible defaults instead of being required parameters
- Internal concepts (implementation classes, storage formats, wire protocols) that leak into the public API

For each: describe what the user has to do manually and how the framework could absorb it.

## SINGLE_SOURCE_OF_TRUTH

The same information must not be defined in multiple places. If two things must stay synchronized, one must be derived from the other.

Look for:
- Constants, configuration values, or validation rules duplicated across files
- Parameters whose correct value must be manually kept in sync with a value defined elsewhere
- Documentation strings that repeat information already expressed by the type system

## UNVALIDATABLE_DERIVATIVES

When a value is derived from a typed source (field name, class name, enum variant), the derivative must be computed programmatically — not written as a manual string. If the source is renamed, manual strings break silently.

Look for:
- Dict keys that mirror Pydantic model field names as string literals — use `model.model_fields` or similar
- String identifiers that mirror class names — use `cls.__name__` or a registry
- Format strings or log messages referencing attribute names that a type checker cannot trace back to the source
- Any string literal that would silently become wrong if a field, class, or enum variant were renamed

```python
# Wrong — renaming the field leaves the string behind
data = {"token_name": self.token_name}

# Correct — derived from the typed source
data = {field: getattr(self, field) for field in self.model_fields}
```

## ERROR_ACTIONABILITY

Every warning and error message must state three things: (1) what went wrong, (2) how to fix it, and (3) the correct usage pattern. The reader — often an AI coding agent — must be able to resolve the issue from the message alone without consulting documentation.

```python
# Wrong — states the problem but not the solution
logger.warning("Field '%s' value is too long (%d chars).", field_name, len(value))

# Correct — states problem, correct usage, and fix
logger.warning(
    "Field '%s' has %d chars. Field parameters are for short values (up to %d chars). "
    "For longer content, pass it as a Document via input_documents.",
    field_name, len(value), MAX_LENGTH,
)
```

For each: quote the current message, explain what's missing, suggest an improved version.

## DEFINITION_TIME_VALIDATION

Constraints that are only caught at runtime but could be caught earlier — at class definition time (`__init_subclass__`), decoration time (decorator validation), or import time.

Look for:
- Validation in `__init__` or `__post_init__` that could be moved to `__init_subclass__` (fires at class definition, not instantiation)
- Runtime `isinstance` checks or `assert` statements that could be enforced by type annotations and decoration-time validation
- Error conditions where the user gets a confusing runtime traceback deep in framework code instead of a clear error at import time

For each: describe the constraint, where it's currently caught, and how to catch it earlier.

## MODULE_COHESION

Can a developer correctly use a module's public API without reading the source code of other modules?

Look for:
- Public API parameters that trigger behavior defined in a different module — the parameter's effect should be fully documented on the public API itself
- Return types that are classes from other modules with no explanation of how to use them
- Import chains where using module A's API correctly requires understanding module B's internals

For each: describe the cross-module knowledge dependency.

## CANNOT_VERIFY
