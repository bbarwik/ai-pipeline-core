"""Logging infrastructure for AI Pipeline Core.

Provides a Prefect-integrated logging facade for unified logging across pipelines.
Prefer get_pipeline_logger instead of logging.getLogger to ensure proper integration.
"""

from .logging_config import LoggingConfig, get_pipeline_logger, setup_logging

__all__ = [
    "LoggingConfig",
    "get_pipeline_logger",
    "setup_logging",
]
