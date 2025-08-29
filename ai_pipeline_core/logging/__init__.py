"""Logging infrastructure for AI Pipeline Core.

@public

Provides a Prefect-integrated logging facade for unified logging across pipelines.
Prefer get_pipeline_logger instead of logging.getLogger to ensure proper integration.

Example:
    >>> from ai_pipeline_core import get_pipeline_logger
    >>> logger = get_pipeline_logger(__name__)
    >>> logger.info("Processing started")
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
