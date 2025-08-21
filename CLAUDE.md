# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Pipeline Core is a high-performance async library for building AI pipelines with Prefect and LMNR (Laminar) tracing. It provides strong typing with Pydantic and uses LiteLLM proxy for OpenAI API compatibility.

> [!NOTE]
> The `dependencies_docs/` directory contains guides for AI assistants on interacting with external dependencies (Prefect, LMNR, etc.), NOT user documentation for ai-pipeline-core. These files are excluded from repository listings.

## Absolute Non-Negotiable Rules

1. **Minimalism Above All**
   - Less code is better code - every line must justify its existence
   - Delete code rather than comment it out
   - No defensive programming for unlikely scenarios
   - If you can't explain why a line exists, delete it
   - Do not create any new markdown file unless told to do it

2. **Python 3.12+ Only**
   - Use modern Python features exclusively
   - No `from typing import List, Dict` - use built-in types
   - No compatibility shims or version checks
   - No legacy patterns

3. **Everything Async**
   - ALL I/O operations must be async - no blocking calls allowed
   - No `requests` library - use `httpx` with async
   - No `time.sleep()` - use `asyncio.sleep()`
   - No blocking database calls
   - Exception: Local file I/O in simple_runner module may use sync functions for simplicity

4. **Strong Typing with Pydantic**
   - Every function must have complete type hints
   - All data structures must be Pydantic models
   - Use `frozen=True` for immutable models
   - No raw dicts for configuration or data transfer

5. **Self-Documenting Code**
   - Code for experienced developers only
   - No comments explaining *what* - code must be clear
   - No verbose logging or excessive documentation
   - Function and variable names must be descriptive

6. **Consistency is Mandatory**
   - No tricks or hacks (no imports inside functions/classes)
   - Follow the established patterns exactly
   - Use the pipeline logger, never import logging directly
   - All exports must be explicit in `__all__`

## Essential Commands

```bash
# Development setup
make install-dev         # Install with dev dependencies and pre-commit hooks

# Testing
make test                # Run all tests
make test-cov           # Run tests with coverage report
pytest tests/test_documents.py::TestDocument::test_creation  # Run single test

# Code quality
make lint               # Run ruff linting
make format            # Auto-format and fix code
make typecheck         # Run basedpyright type checking
make pre-commit        # Run all pre-commit hooks

# Cleanup
make clean             # Remove all build artifacts and caches
```

## Architecture & Core Modules

### Documents System (`ai_pipeline_core/documents/`)
Foundation for all data handling. Documents are immutable Pydantic models that wrap content with metadata. Can handle text, images, PDFs, and future document types.

- **Document**: Abstract base class. Handles encoding, MIME type detection, serialization. Cannot be instantiated directly.
- **FlowDocument**: Abstract base for persistent documents in Prefect flows (survive across runs). Cannot be instantiated directly.
- **TaskDocument**: Abstract base for temporary documents in Prefect tasks (exist only during execution). Cannot be instantiated directly.
- **DocumentList**: Type-validated container, not just `list[Document]` - adds validation

**Critical**: Each flow/task must define its own concrete document classes by inheriting from FlowDocument or TaskDocument. Never instantiate abstract classes directly. Never use raw bytes/strings - always Document classes. MIME type detection is automatic and required for AI interactions.

### LLM Module (`ai_pipeline_core/llm/`)
All AI interactions through LiteLLM proxy (OpenAI API compatible). Built-in retry, LMNR monitoring, cost calculation.

- **generate()**: Main entry - splits context (cached by default) and messages (dynamic)
- **generate_structured()**: Returns structured Pydantic model outputs
- **AIMessages**: Type-safe container, converts Documents to prompts automatically
- **ModelOptions**: Config for retry, timeout, response format, system prompts
- **Strong model typing**: Use `ModelName` enum to prevent model name typos

**Critical**: Context/messages split is for caching efficiency. Context is expensive and rarely changes, messages are dynamic. Always use ModelOptions, never raw dicts.

### Flow Configuration (`ai_pipeline_core/flow/`)
Type-safe flow definitions with input/output document types.

```python
class MyFlowConfig(FlowConfig):
    INPUT_DOCUMENT_TYPES = [InputDoc1, InputDoc2]
    OUTPUT_DOCUMENT_TYPE = OutputDoc
```

### Prompt Manager (`ai_pipeline_core/prompt_manager.py`)
Jinja2 template loading with smart path resolution:
- **Same directory as Python file**: For single-use prompts (keeps related code together)
- **`prompts/` directory**: For shared prompts across modules

Philosophy: Prompts near their usage = easier maintenance.

### Tracing (`ai_pipeline_core/tracing.py`)
LMNR integration via `@trace` decorator. Always use `test=True` in tests to avoid polluting production metrics.

### Settings (`ai_pipeline_core/settings.py`)
Central configuration for all external services (Prefect, LMNR, OpenAI). Will be updated with deployment guide from `dependencies_docs/prefect_deployment.md`.

### Logging (`ai_pipeline_core/logging/`)
Unified Prefect-integrated logging. Replaces Python's logging module entirely. Never import `logging` directly.

## Import Convention

```python
# Within same package - relative imports
from .document import Document
from .utils import helper

# Cross-package - absolute imports
from ai_pipeline_core.documents import Document
from ai_pipeline_core.llm import generate

# NEVER use parent imports (..)
```

## Critical Patterns

### Always Async
```python
# Every I/O operation must be async
async def process_document(doc: Document) -> ProcessedDocument:
    result = await generate(context, messages, options)
    return ProcessedDocument(...)
```

### Always Typed
```python
# Complete type annotations required
def calculate(x: int, y: int) -> int:
    return x + y

async def fetch_data(url: str) -> dict[str, Any]:
    ...
```

### Always Pydantic
```python
# All data structures must be Pydantic models
class Config(BaseModel):
    model: ModelName  # Use enums for constrained values
    temperature: float = Field(ge=0, le=2)

    model_config = ConfigDict(frozen=True)  # Immutable by default
```

### Never Direct Logging
```python
# Always use pipeline logger
from ai_pipeline_core.logging import get_pipeline_logger
logger = get_pipeline_logger(__name__)
```

## Forbidden Patterns (NEVER Do These)

1. **No print statements** - Use pipeline logger
2. **No global mutable state** - Use dependency injection
3. **No `sys.exit()`** - Raise exceptions
4. **No hardcoded paths** - Use settings/config
5. **No string concatenation for paths** - Use `pathlib.Path`
6. **No manual JSON parsing** - Use Pydantic
7. **No `time.sleep()`** - Use `asyncio.sleep()`
8. **No `requests` library** - Use `httpx` with async
9. **No raw SQL** - Use async ORM or query builders
10. **No magic numbers** - Use named constants
11. **No nested functions** (except decorators)
12. **No dynamic imports** - All imports at module level
13. **No monkeypatching**
14. **No metaclasses** (except Pydantic)
15. **No multiple inheritance** (except mixins)
16. **No TODO/FIXME comments** - Fix it or delete it
17. **No commented code** - Delete it
18. **No defensive programming** - Trust the types

## Prefect Integration Notes

See `dependencies_docs/prefect.md`, `dependencies_docs/prefect_deployment.md`, and `dependencies_docs/prefect_logging.md` for detailed Prefect patterns.

Key points:
- FlowDocuments are persistent across flow runs
- TaskDocuments are temporary within task execution
- Each flow/task defines its own document classes
- Use FlowConfig for type-safe flow definitions

## Testing Approach

- All tests must be async
- Use `@trace(test=True)` for test tracing
- Mock external services (OpenAI, LMNR)
- Test with proper Document types, not raw data
- Coverage target: >80%

## When Making Changes

1. **Before writing any code, ask**: "Can this be done with less code?"
2. **Before adding a line, ask**: "Can I justify why this exists?"
3. Run `make lint` and `make typecheck` before committing
4. Let pre-commit hooks auto-fix formatting
5. If you can't explain it to another developer in one sentence, rewrite it
6. If the function is longer than 20 lines, it's probably doing too much
7. **Final check**: Could you delete this code? If maybe, then yes - delete it

## Final Rule

**The best code is no code. The second best is minimal, clear, typed, async code that does exactly what's needed and nothing more.**

If you're unsure whether to add code, don't add it.
