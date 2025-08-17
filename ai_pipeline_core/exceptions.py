"""Exception hierarchy for AI Pipeline Core."""


class PipelineCoreError(Exception):
    """Base exception for all pipeline errors."""

    pass


class DocumentError(PipelineCoreError):
    """Document-related errors."""

    pass


class DocumentValidationError(DocumentError):
    """Document validation failed."""

    pass


class DocumentSizeError(DocumentValidationError):
    """Document size exceeds limits."""

    pass


class DocumentNameError(DocumentValidationError):
    """Invalid document name."""

    pass


class LLMError(PipelineCoreError):
    """LLM-related errors."""

    pass


class PromptError(PipelineCoreError):
    """Prompt-related errors."""

    pass


class PromptRenderError(PromptError):
    """Failed to render prompt template."""

    pass


class PromptNotFoundError(PromptError):
    """Prompt template not found."""

    pass


class MimeTypeError(DocumentError):
    """MIME type detection or validation error."""

    pass
