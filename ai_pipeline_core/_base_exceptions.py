"""Base exception class for AI Pipeline Core. Zero dependencies."""

__all__ = ["NonRetriableError", "PipelineCoreError"]


class PipelineCoreError(Exception):
    """Base exception for all AI Pipeline Core errors."""


class NonRetriableError(PipelineCoreError):
    """Raised when an operation fails and must not be retried.

    The retry loops in PipelineTask and PipelineFlow check for this exception
    and stop immediately without further retry attempts.

    Can wrap another exception to preserve the original cause::

        raise NonRetriableError("invalid API key") from original_exc
    """
