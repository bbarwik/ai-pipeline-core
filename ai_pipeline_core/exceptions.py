"""Exception hierarchy for AI Pipeline Core.

This module defines the exception hierarchy used throughout the AI Pipeline Core library.
All exceptions inherit from PipelineCoreError, providing a consistent error handling interface.
"""

__all__ = [
    "DocumentNameError",
    "DocumentSizeError",
    "DocumentValidationError",
    "LLMError",
    "OutputDegenerationError",
    "PipelineCoreError",
]


class PipelineCoreError(Exception):
    """Base exception for all AI Pipeline Core errors."""


class DocumentValidationError(PipelineCoreError):
    """Raised when document validation fails."""


class DocumentSizeError(DocumentValidationError):
    """Raised when document content exceeds MAX_CONTENT_SIZE limit."""


class DocumentNameError(DocumentValidationError):
    """Raised when document name contains path traversal, reserved suffixes, or invalid format."""


class LLMError(PipelineCoreError):
    """Raised when LLM generation fails after all retries, including timeouts and provider errors."""


class OutputDegenerationError(LLMError):
    """LLM output contains degeneration patterns (e.g., token repetition loops). Triggers retry with cache disabled."""
