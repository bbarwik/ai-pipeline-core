"""Centralized logging configuration for AI Pipeline Core.

Provides logging configuration management that integrates with Prefect's logging system.
"""

import logging.config
import os
from pathlib import Path
from typing import Any

import yaml
from prefect.logging import get_logger

# Default log levels for different components
DEFAULT_LOG_LEVELS = {
    "ai_pipeline_core": "INFO",
    "ai_pipeline_core.documents": "INFO",
    "ai_pipeline_core.llm": "INFO",
    "ai_pipeline_core.pipeline": "INFO",
    "ai_pipeline_core.testing": "DEBUG",
}


class LoggingConfig:
    """Manages logging configuration for the pipeline.

    Provides centralized logging configuration with Prefect integration.

    Configuration precedence:
        1. Explicit config_path parameter
        2. AI_PIPELINE_LOGGING_CONFIG environment variable
        3. PREFECT_LOGGING_SETTINGS_PATH environment variable
        4. Default configuration

    """

    def __init__(self, config_path: Path | None = None):
        """Initialize logging configuration.

        Args:
            config_path: Optional path to YAML configuration file.
        """
        self.config_path = config_path or self._get_default_config_path()
        self._config: dict[str, Any] | None = None

    @staticmethod
    def _get_default_config_path() -> Path | None:
        """Get default config path from environment variables.

        Returns:
            Path to the config file or None if not found.
        """
        # Check environment variable first
        if env_path := os.environ.get("AI_PIPELINE_LOGGING_CONFIG"):
            return Path(env_path)

        # Check Prefect's setting
        if prefect_path := os.environ.get("PREFECT_LOGGING_SETTINGS_PATH"):
            return Path(prefect_path)

        return None

    def load_config(self) -> dict[str, Any]:
        """Load logging configuration from file or defaults.

        Returns:
            Dictionary containing logging configuration.
        """
        if self._config is None:
            if self.config_path and self.config_path.exists():
                with open(self.config_path, encoding="utf-8") as f:
                    self._config = yaml.safe_load(f)
            else:
                self._config = self._get_default_config()
        # self._config cannot be None at this point
        assert self._config is not None
        return self._config

    @staticmethod
    def _get_default_config() -> dict[str, Any]:
        """Get default logging configuration.

        Returns:
            Default logging configuration dictionary.
        """
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s - %(message)s",
                    "datefmt": "%H:%M:%S",
                },
                "detailed": {
                    "format": ("%(asctime)s | %(levelname)-7s | %(name)s | %(funcName)s:%(lineno)d - %(message)s"),
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
        """Apply the logging configuration."""
        config = self.load_config()
        logging.config.dictConfig(config)

        # Set Prefect logging environment variables if needed
        if "prefect" in config.get("loggers", {}):
            prefect_level = config["loggers"]["prefect"].get("level", "INFO")
            os.environ.setdefault("PREFECT_LOGGING_LEVEL", prefect_level)


# Global configuration instance
_logging_config: LoggingConfig | None = None


def setup_logging(config_path: Path | None = None, level: str | None = None):
    """Setup logging for the AI Pipeline Core library.

    Initializes logging configuration for the pipeline system.

    IMPORTANT: Call setup_logging exactly once in your application entry point
    (for example, in main()). Do not call at import time or in library modules.

    Args:
        config_path: Optional path to YAML logging configuration file.
        level: Optional log level override (INFO, DEBUG, WARNING, etc.).

    """
    global _logging_config  # noqa: PLW0603

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
    """Get a logger for pipeline components.

    Returns a Prefect-integrated logger with the OTel span-event bridge
    attached.  Any log record at INFO+ emitted while an OTel span is
    recording will be captured as a span event in the trace.

    Args:
        name: Logger name, typically __name__.

    Returns:
        Prefect logger instance with bridge handler.

    """
    if _logging_config is None:
        setup_logging()

    logger = get_logger(name)

    # Attach the singleton bridge handler so log records become OTel span events.
    # The handler is a no-op when no span is recording, so early attachment is safe.
    from ai_pipeline_core.observability._logging_bridge import get_bridge_handler  # noqa: PLC0415

    handler = get_bridge_handler()
    if handler not in logger.handlers:
        logger.addHandler(handler)

    return logger
