# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Pipeline Core is a high-performance async library for building AI pipelines with Prefect and LMNR (Laminar) tracing. It provides strong typing with Pydantic and uses LiteLLM proxy for OpenAI API compatibility.

**Version 0.1.10 Updates:**
- Full API documentation with pydoc-markdown (see API.md)
- Fixed mutable defaults in llm.generate/generate_structured
- Document constructors require keyword arguments
- DocumentList.filter_by_type uses isinstance (supports subclasses)
- Explicit DocumentValidationError instead of assertions
- Deprecation warnings for legacy patterns

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
make test-showcase      # Test the showcase.py CLI example
pytest tests/test_documents.py::TestDocument::test_creation  # Run single test

# Code quality
make lint               # Run ruff linting
make format            # Auto-format and fix code
make typecheck         # Run basedpyright type checking
make pre-commit        # Run all pre-commit hooks

# Documentation (NEW in v0.1.10)
make docs-build         # Generate API.md using pydoc-markdown
make docs-check         # Verify API.md is up-to-date (CI check)
make docs-view          # Build and view documentation

# Cleanup
make clean             # Remove all build artifacts and caches
```

## API Documentation System

The project uses pydoc-markdown with a custom Jinja2 template to generate API.md:

### Configuration Files
- `pydoc-markdown.yaml`: Configures the documentation pipeline
- `docs/api_template.jinja2`: Custom 105-line template for rendering

### How It Works
1. **@public Tags**: Add `@public` to docstrings to mark items for API documentation
2. **Smart Module Inclusion**: Modules without @public are included if they contain @public members
3. **Recursive Filtering**: Template recursively checks classes for @public members
4. **Project-Agnostic**: Template auto-detects package name from module structure

### Template Architecture
The Jinja2 template (`docs/api_template.jinja2`) uses:
- **Unified `sig` macro**: Handles signatures for classes, functions, and variables
- **Single `render` macro**: Consistent rendering for all member types
- **Smart `pub` detection**: Identifies @public tags in docstrings
- **Recursive `has_pub_child`**: Finds @public members in nested structures

### Adding to API Documentation
```python
class MyClass:
    """Description of class.

    @public

    More details...
    """

def my_function():
    """Function description.

    @public
    """
```

Only items with @public tags appear in API.md. This keeps the API surface explicit and manageable.

## Architecture & Core Modules

### Documents System (`ai_pipeline_core/documents/`)
Foundation for all data handling. Documents are immutable Pydantic models that wrap content with metadata. Can handle text, images, PDFs, and future document types.

- **Document**: Abstract base class. Handles encoding, MIME type detection, serialization. Cannot be instantiated directly.
- **FlowDocument**: Abstract base for persistent documents in Prefect flows (survive across runs). Cannot be instantiated directly.
- **TaskDocument**: Abstract base for temporary documents in Prefect tasks (exist only during execution). Cannot be instantiated directly.
- **TemporaryDocument**: Concrete class for documents that are never persisted (NEW in v0.1.9+)
- **DocumentList**: Type-validated container, not just `list[Document]` - adds validation

**Critical v0.1.10 Changes**:
- Document constructors MUST use keyword arguments: `Document(name="file.txt", content="data", description="optional")`
- Legacy positional arguments trigger deprecation warning
- DocumentList.filter_by_type now uses isinstance (includes subclasses)
- YAML parsing uses ruamel.yaml (safe by default)

**Critical**: Each flow/task must define its own concrete document classes by inheriting from FlowDocument or TaskDocument. Never instantiate abstract classes directly. Never use raw bytes/strings - always Document classes. MIME type detection is automatic and required for AI interactions.

### LLM Module (`ai_pipeline_core/llm/`)
All AI interactions through LiteLLM proxy (OpenAI API compatible). Built-in retry, LMNR monitoring, cost calculation.

- **generate()**: Main entry - splits context (cached by default) and messages (dynamic)
- **generate_structured()**: Returns structured Pydantic model outputs
- **AIMessages**: Type-safe container, converts Documents to prompts automatically
- **ModelOptions**: Config for retry, timeout, response format, system prompts
- **Strong model typing**: Use `ModelName` type alias to prevent model name typos

**Critical v0.1.10 Changes**:
- Context and options parameters now default to None (not empty instances)
- Context caching saves 50-90% tokens with 120s TTL
- Retry uses FIXED delay (default 10s), NOT exponential backoff
- Messages can be string or AIMessages

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
LMNR integration via `@trace` decorator. Test environment is automatically detected to avoid polluting production metrics.

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

## Document Class Naming Rules

**CRITICAL**: Never create Document subclasses with names starting with "Test" prefix. This causes conflicts with pytest test discovery. The Document base class enforces this rule and will raise a TypeError if violated.

❌ **BAD**: `TestDocument`, `TestFlowDocument`, `TestInputDoc`
✅ **GOOD**: `SampleDocument`, `ExampleDocument`, `DemoDocument`, `MockDocument`

## Document Constructor Rules (v0.1.10+)

**CRITICAL**: Document constructors MUST use keyword arguments:

❌ **BAD** (deprecated, will trigger warning):
```python
doc = MyDocument("file.txt", "content")  # Positional args
```

✅ **GOOD** (required):
```python
doc = MyDocument(
    name="file.txt",
    content="content",
    description="optional"  # Optional description
)
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
19. **No mutable default arguments** - Use None and initialize in function
20. **No direct use of prefect task/flow** - Use pipeline_task/pipeline_flow

## Prefect Integration Notes

See `dependencies_docs/prefect.md`, `dependencies_docs/prefect_deployment.md`, and `dependencies_docs/prefect_logging.md` for detailed Prefect patterns.

Key points:
- FlowDocuments are persistent across flow runs
- TaskDocuments are temporary within task execution
- TemporaryDocument is never persisted (use for truly ephemeral data)
- Each flow/task defines its own document classes
- Use FlowConfig for type-safe flow definitions
- **@pipeline_flow and @pipeline_task require async functions only** - sync functions will be rejected with TypeError
- **Clean Prefect decorators** from `ai_pipeline_core.prefect` (flow, task) available for cases where tracing is not needed
- Use @pipeline_flow/@pipeline_task for production flows with tracing, @flow/@task for simple utilities
- Use fixtures for testing: prefect_test_harness and disable_run_logger (see tests/conftest.py)

## FlowConfig Validation Rules

**CRITICAL**: FlowConfig subclasses must follow these rules:
1. **Must define INPUT_DOCUMENT_TYPES and OUTPUT_DOCUMENT_TYPE** - These are required attributes
2. **OUTPUT_DOCUMENT_TYPE cannot be in INPUT_DOCUMENT_TYPES** - This prevents circular dependencies
3. Validation happens at class definition time via `__init_subclass__`
4. **NEW in v0.1.10**: Validation uses DocumentValidationError exceptions (not assertions)
5. **NEW**: filter_by_type uses isinstance, so subclasses are included

Example:
```python
class ValidConfig(FlowConfig):
    INPUT_DOCUMENT_TYPES = [InputDoc]
    OUTPUT_DOCUMENT_TYPE = OutputDoc  # Must be different from input types

# This will raise TypeError at definition time:
class InvalidConfig(FlowConfig):
    INPUT_DOCUMENT_TYPES = [SomeDoc]
    OUTPUT_DOCUMENT_TYPE = SomeDoc  # ERROR: Same as input!
```

## Testing Approach

- All tests must be async
- Use `@trace(test=True)` for test tracing
- Mock external services (OpenAI, LMNR)
- Test with proper Document types, not raw data
- Coverage target: >80%
- **Test fixtures**: conftest.py provides session-scoped `prefect_test_fixture` - don't create nested test harnesses
- **CLI testing**: run_cli automatically detects test environment via Prefect settings

## Code Elegance Principles

When implementing solutions:
1. **Check existing APIs first** - Use Prefect settings API instead of checking environment variables
2. **Use context managers** - Combine multiple contexts with `ExitStack` instead of nested blocks
3. **Avoid external dependencies** - Don't use httpx for simple checks, use framework's own APIs
4. **No code duplication** - If you're writing the same code twice, refactor immediately
5. **Detect environment properly** - Check framework state (e.g., `prefect.settings.PREFECT_API_URL.value()`) not environment variables

Example of elegant context management:
```python
# Good - minimal and clear
with ExitStack() as stack:
    if not prefect.settings.PREFECT_API_URL.value():
        stack.enter_context(prefect_test_harness())
    stack.enter_context(disable_run_logger())
    if trace_name:
        stack.enter_context(Laminar.start_span(...))
    # Execute logic here

# Bad - duplicated code blocks for different conditions
```

## When Making Changes

1. **Before writing any code, ask**: "Can this be done with less code?"
2. **Before adding a line, ask**: "Can I justify why this exists?"
3. Run `make lint` and `make typecheck` before committing
4. Let pre-commit hooks auto-fix formatting
5. If you can't explain it to another developer in one sentence, rewrite it
6. If the function is longer than 20 lines, it's probably doing too much
7. **NEW**: Add @public tags to all exported API components
8. **NEW**: Update API.md after significant changes (`make docs-build`)
9. **Final check**: Could you delete this code? If maybe, then yes - delete it

## Final Rule

**The best code is no code. The second best is minimal, clear, typed, async code that does exactly what's needed and nothing more.**

If you're unsure whether to add code, don't add it.
