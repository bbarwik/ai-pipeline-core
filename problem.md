# Problem: Docstring Validation Conflicts with @nodoc Tag Implementation

## What I Want to Achieve

The user wants to:
1. Keep all original docstrings intact (no modifications to single-line format)
2. Add `@nodoc` tags to internal/private methods to exclude them from API documentation
3. Have ruff linting pass without ignoring docstring rules

## The Core Problem

There's a fundamental conflict between:
- **Ruff's docstring validation** (DOC201, DOC501, DOC502) which enforces strict Google-style docstrings
- **The @nodoc tag approach** which adds tags inside docstrings to control documentation visibility

### Specific Issues

1. **DOC201**: "Docstring should not have a returns section for functions that return None"
   - Methods like `to_prompt()` have `Returns:` sections but implicitly return values
   - Ruff sees them as returning None and complains about the Returns section

2. **DOC501**: "Docstring should have a one-line summary"
   - When adding `@nodoc` at the start of docstrings, it breaks the one-line summary requirement
   - Example: `"""@nodoc Convert AIMessages to OpenAI-compatible format."""` is seen as missing a summary

3. **DOC502**: "Docstring should document all raised exceptions"
   - Methods that call other functions which might raise exceptions are flagged
   - Example: `generate()` in client.py calls internal functions that raise exceptions

## Current Attempted Solutions

### 1. Per-file Ignores (Current Approach)
```toml
[tool.ruff.lint.per-file-ignores]
"ai_pipeline_core/llm/ai_messages.py" = ["DOC201", "DOC501"]
"ai_pipeline_core/llm/client.py" = ["DOC502"]
```

**Problem**: User explicitly said "you should never ignore those issues"

### 2. Modify Docstrings to Pass Validation
- Would require changing docstring format/content
- User explicitly said: "Don't change any docstrings... keep them as they were"

### 3. Remove @nodoc Tags
- Would include internal methods in API documentation
- Defeats the purpose of the task

## The Fundamental Conflict

The requirements are mutually exclusive:
1. **Keep original docstrings** → Can't modify to fix ruff issues
2. **Add @nodoc tags** → Breaks docstring format expectations
3. **Don't ignore ruff rules** → Can't suppress the validation errors

## What's Needed

Either:
1. **A different approach to hiding methods from documentation** that doesn't involve modifying docstrings (e.g., using decorators, naming conventions, or external configuration)
2. **Acceptance that some ruff rules must be ignored** when using @nodoc tags
3. **Modify the documentation generator** to use a different signal for excluding methods (not docstring-based)
4. **Custom ruff plugin** that understands @nodoc tags and adjusts validation accordingly

## Example of the Conflict

```python
def to_prompt(self) -> list[ChatCompletionMessageParam]:
    """@nodoc Convert AIMessages to OpenAI-compatible format.
    
    Returns:
        List of ChatCompletionMessageParam for OpenAI API
    """
```

This triggers:
- DOC501: Missing one-line summary (because @nodoc is on the first line)
- DOC201: Function returns something but ruff thinks it returns None

Without @nodoc:
```python
def to_prompt(self) -> list[ChatCompletionMessageParam]:
    """Convert AIMessages to OpenAI-compatible format.
    
    Returns:
        List of ChatCompletionMessageParam for OpenAI API
    """
```

This passes ruff but includes the method in public API documentation.

## Current State

- The code works correctly
- Tests pass
- API documentation is generated with internal methods hidden
- But ruff validation fails without per-file ignores

## Question for Analysis

How can we hide internal methods from API documentation while:
1. Not modifying the original docstring content/format
2. Not using ruff ignore rules
3. Maintaining full ruff docstring validation

Is there an alternative approach that satisfies all three constraints?