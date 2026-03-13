"""Logging infrastructure for AI Pipeline Core.

Provides a configured stdlib logging facade and execution-log capture primitives.
Prefer get_pipeline_logger instead of logging.getLogger to ensure consistent setup.
"""

from .logging_config import LoggingConfig, get_pipeline_logger, setup_logging

__all__ = [
    "LoggingConfig",
    "get_pipeline_logger",
    "setup_logging",
]
