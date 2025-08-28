"""Logging infrastructure for AI Pipeline Core.

@public

This module provides unified, Prefect-integrated logging with structured
output support. It replaces Python's standard logging module with a
pipeline-aware logging system.

Key components:
    get_pipeline_logger: Factory function for creating pipeline loggers
    setup_logging: Initialize logging configuration from YAML
    LoggerMixin: Base mixin for adding logging to classes
    StructuredLoggerMixin: Mixin for structured/JSON logging
    LoggingConfig: Configuration class for logging settings

Example:
    >>> from ai_pipeline_core.logging import get_pipeline_logger
    >>>
    >>> logger = get_pipeline_logger(__name__)
    >>> logger.info("Processing started")
    >>>
    >>> # For structured logging in classes
    >>> from ai_pipeline_core.logging import StructuredLoggerMixin
    >>>
    >>> class MyProcessor(StructuredLoggerMixin):
    ...     def process(self):
    ...         self.logger.info("Processing", extra={"items": 100})

Note:
    Never import Python's logging module directly. Always use
    get_pipeline_logger() for consistent Prefect integration.
"""

from .logging_config import LoggingConfig, get_pipeline_logger, setup_logging
from .logging_mixin import LoggerMixin, StructuredLoggerMixin

__all__ = [
    "LoggerMixin",
    "StructuredLoggerMixin",
    "LoggingConfig",
    "setup_logging",
    "get_pipeline_logger",
]
