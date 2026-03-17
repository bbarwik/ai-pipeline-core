"""Exception hierarchy for AI Pipeline Core.

This module defines the exception hierarchy used throughout the AI Pipeline Core library.
All exceptions inherit from PipelineCoreError, providing a consistent error handling interface.
"""

from ai_pipeline_core._base_exceptions import NonRetriableError, PipelineCoreError
from ai_pipeline_core.documents.exceptions import DocumentNameError, DocumentSizeError, DocumentValidationError

__all__ = [
    "DocumentNameError",
    "DocumentSizeError",
    "DocumentValidationError",
    "EmptyResponseError",
    "LLMError",
    "NonRetriableError",
    "OutputDegenerationError",
    "PipelineCoreError",
]


class LLMError(PipelineCoreError):
    """Raised when LLM generation fails after all retries, including timeouts and provider errors."""


class OutputDegenerationError(LLMError):
    """LLM output contains degeneration patterns (e.g., token repetition loops). Triggers retry with cache disabled."""


class EmptyResponseError(LLMError):
    """Model returned empty content (no text and no tool calls). Retried with LiteLLM cache disabled."""
