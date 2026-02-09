"""Exception hierarchy for AI Pipeline Core.

This module defines the exception hierarchy used throughout the AI Pipeline Core library.
All exceptions inherit from PipelineCoreError, providing a consistent error handling interface.
"""


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


class PromptError(PipelineCoreError):
    """Base exception for prompt template errors."""


class PromptRenderError(PromptError):
    """Raised when Jinja2 template rendering fails."""


class PromptNotFoundError(PromptError):
    """Raised when prompt template file is not found in search paths."""
