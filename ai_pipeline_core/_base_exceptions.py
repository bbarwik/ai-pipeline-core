"""Base exception class for AI Pipeline Core. Zero dependencies."""

__all__ = ["NonRetriableError", "PipelineCoreError", "StubNotImplementedError"]


class PipelineCoreError(Exception):
    """Base exception for all AI Pipeline Core errors."""


class NonRetriableError(PipelineCoreError):
    """Raised when an operation fails and must not be retried.

    The retry loops in PipelineTask and PipelineFlow check for this exception
    and stop immediately without further retry attempts.

    Can wrap another exception to preserve the original cause::

        raise NonRetriableError("invalid API key") from original_exc
    """


class StubNotImplementedError(NonRetriableError):
    """Raised when a stub class (``_stub = True``) is executed at runtime.

    Stubs are placeholder classes with correct type signatures but no implementation.
    They pass all definition-time validation and type checking, but must not be
    executed. Subclasses ``NonRetriableError`` to prevent retry loops from retrying
    the stub.

    To fix: implement the ``run()`` body and remove ``_stub = True``.
    """
