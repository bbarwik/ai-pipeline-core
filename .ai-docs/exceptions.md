# MODULE: exceptions
# CLASSES: LLMError, OutputDegenerationError, EmptyResponseError, PipelineCoreError
# DEPENDS: Exception
# VERSION: 0.16.0
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import LLMError, OutputDegenerationError, PipelineCoreError
```

## Public API

```python
class LLMError(PipelineCoreError):
    """Raised when LLM generation fails after all retries, including timeouts and provider errors."""


class OutputDegenerationError(LLMError):
    """LLM output contains degeneration patterns (e.g., token repetition loops). Triggers retry with cache disabled."""


class EmptyResponseError(LLMError):
    """Model returned empty content (no text and no tool calls). Retried with LiteLLM cache disabled."""


class PipelineCoreError(Exception):
    """Base exception for all AI Pipeline Core errors."""
```

## Examples

**Output degeneration error is llm error** (`tests/test_exceptions_ai_docs.py:34`)

```python
def test_output_degeneration_error_is_llm_error() -> None:
    """Degeneration failures are represented as an LLMError subclass."""
    assert isinstance(OutputDegenerationError("loop"), LLMError)
```


## Error Examples

**Document name error on path traversal** (`tests/test_exceptions_ai_docs.py:20`)

```python
def test_document_name_error_on_path_traversal() -> None:
    """Path traversal in document names raises DocumentNameError."""
    with pytest.raises(DocumentNameError):
        _TestExDoc(name="../bad.txt", content=b"x")
```

**Document size error on limit exceeded** (`tests/test_exceptions_ai_docs.py:27`)

```python
def test_document_size_error_on_limit_exceeded() -> None:
    """Content beyond MAX_CONTENT_SIZE raises DocumentSizeError."""
    with pytest.raises(DocumentSizeError):
        _TinyDoc(name="tiny.txt", content=b"12345")
```
