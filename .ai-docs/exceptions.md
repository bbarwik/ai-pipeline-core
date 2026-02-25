# MODULE: exceptions
# CLASSES: PipelineCoreError, DocumentValidationError, DocumentSizeError, DocumentNameError, LLMError, OutputDegenerationError
# DEPENDS: Exception
# VERSION: 0.10.3
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import DocumentNameError, DocumentSizeError, DocumentValidationError, LLMError, OutputDegenerationError, PipelineCoreError
```

## Public API

```python
class PipelineCoreError(Exception):
    """Base exception for all AI Pipeline Core errors."""

class DocumentValidationError(PipelineCoreError):
    """Raised when document validation fails."""

class DocumentSizeError(DocumentValidationError):
    """Raised when document content exceeds MAX_CONTENT_SIZE limit."""

class DocumentNameError(DocumentValidationError):
    """Raised when document name contains invalid characters or patterns."""

class LLMError(PipelineCoreError):
    """Raised when LLM generation fails after all retries."""

class OutputDegenerationError(LLMError):
    """LLM output contains degeneration patterns (e.g., token repetition loops)."""

```

## Examples

No test examples available.
