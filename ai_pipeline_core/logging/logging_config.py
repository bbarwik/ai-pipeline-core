"""Centralized logging configuration for AI Pipeline Core using Prefect logging"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from prefect.logging import get_logger

# Default log levels for different components
DEFAULT_LOG_LEVELS = {
    "ai_pipeline_core": "INFO",
    "ai_pipeline_core.documents": "INFO",
    "ai_pipeline_core.llm": "INFO",
    "ai_pipeline_core.flow": "INFO",
    "ai_pipeline_core.testing": "DEBUG",
}


class LoggingConfig:
    """Manages logging configuration for the pipeline using Prefect logging"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or self._get_default_config_path()
        self._config: Optional[Dict[str, Any]] = None

    @staticmethod
    def _get_default_config_path() -> Optional[Path]:
        """Get default config path from environment or package"""
        # Check environment variable first
        if env_path := os.environ.get("AI_PIPELINE_LOGGING_CONFIG"):
            return Path(env_path)

        # Check Prefect's setting
        if prefect_path := os.environ.get("PREFECT_LOGGING_SETTINGS_PATH"):
            return Path(prefect_path)

        return None

    def load_config(self) -> Dict[str, Any]:
        """Load logging configuration from file"""
        if self._config is None:
            if self.config_path and self.config_path.exists():
                with open(self.config_path, "r") as f:
                    self._config = yaml.safe_load(f)
            else:
                self._config = self._get_default_config()
        # self._config cannot be None at this point
        assert self._config is not None
        return self._config

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default logging configuration compatible with Prefect"""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s - %(message)s",
                    "datefmt": "%H:%M:%S",
                },
                "detailed": {
                    "format": (
                        "%(asctime)s | %(levelname)-7s | %(name)s | "
                        "%(funcName)s:%(lineno)d - %(message)s"
                    ),
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "ai_pipeline_core": {
                    "level": os.environ.get("AI_PIPELINE_LOG_LEVEL", "INFO"),
                    "handlers": ["console"],
                    "propagate": False,
                },
            },
            "root": {
                "level": "WARNING",
                "handlers": ["console"],
            },
        }

    def apply(self):
        """Apply the logging configuration"""
        import logging.config

        config = self.load_config()
        logging.config.dictConfig(config)

        # Set Prefect logging environment variables if needed
        if "prefect" in config.get("loggers", {}):
            prefect_level = config["loggers"]["prefect"].get("level", "INFO")
            os.environ.setdefault("PREFECT_LOGGING_LEVEL", prefect_level)


# Global configuration instance
_logging_config: Optional[LoggingConfig] = None


def setup_logging(config_path: Optional[Path] = None, level: Optional[str] = None):
    """
    Setup logging for the AI Pipeline Core library

    Args:
        config_path: Optional path to logging configuration file
        level: Optional default log level (overrides config)

    Example:
        >>> from ai_pipeline_core.logging_config import setup_logging
        >>> setup_logging(level="DEBUG")
    """
    global _logging_config

    _logging_config = LoggingConfig(config_path)
    _logging_config.apply()

    # Override level if provided
    if level:
        # Set for our loggers
        for logger_name in DEFAULT_LOG_LEVELS:
            logger = get_logger(logger_name)
            logger.setLevel(level)

        # Also set for Prefect
        os.environ["PREFECT_LOGGING_LEVEL"] = level


def get_pipeline_logger(name: str):
    """
    Get a logger for pipeline components using Prefect's get_logger

    Args:
        name: Logger name (e.g., "ai_pipeline_core.documents")

    Returns:
        Logger instance

    Example:
        >>> logger = get_pipeline_logger("ai_pipeline_core.llm")
        >>> logger.info("Starting LLM processing")
    """
    # Ensure logging is setup
    if _logging_config is None:
        setup_logging()

    return get_logger(name)
