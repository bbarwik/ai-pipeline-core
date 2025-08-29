"""Exception hierarchy for AI Pipeline Core.

@public

This module defines the exception hierarchy used throughout the AI Pipeline Core library.
All exceptions inherit from PipelineCoreError, providing a consistent error handling interface.
"""


class PipelineCoreError(Exception):
    """Base exception for all AI Pipeline Core errors.

    @public
    """

    pass


class DocumentError(PipelineCoreError):
    """Base exception for document-related errors.

    @public
    """

    pass


class DocumentValidationError(DocumentError):
    """Raised when document validation fails.

    @public
    """

    pass


class DocumentSizeError(DocumentValidationError):
    """Raised when document content exceeds MAX_CONTENT_SIZE limit.

    @public
    """

    pass


class DocumentNameError(DocumentValidationError):
    """Raised when document name contains invalid characters or patterns.

    @public
    """

    pass


class LLMError(PipelineCoreError):
    """Raised when LLM generation fails after all retries.

    @public
    """

    pass


class PromptError(PipelineCoreError):
    """Base exception for prompt template errors.

    @public
    """

    pass


class PromptRenderError(PromptError):
    """Raised when Jinja2 template rendering fails.

    @public
    """

    pass


class PromptNotFoundError(PromptError):
    """Raised when prompt template file is not found in search paths.

    @public
    """

    pass


class MimeTypeError(DocumentError):
    """Raised when MIME type detection or validation fails.

    @public
    """

    pass
