"""Centralized logging configuration for AI Pipeline Core.

@public

This module provides logging configuration management that integrates
with Prefect's logging system. It supports both YAML-based configuration
and programmatic setup with sensible defaults.

Key features:
- Prefect-integrated logging for flows and tasks
- YAML configuration file support
- Environment variable overrides
- Component-specific log levels
- Automatic logger creation with proper formatting

Usage:
    >>> from ai_pipeline_core.logging import get_pipeline_logger
    >>> logger = get_pipeline_logger(__name__)
    >>> logger.info("Processing started")

Environment variables:
    AI_PIPELINE_LOGGING_CONFIG: Path to custom logging.yml
    AI_PIPELINE_LOG_LEVEL: Default log level (INFO, DEBUG, etc.)
    PREFECT_LOGGING_LEVEL: Prefect's logging level
    PREFECT_LOGGING_SETTINGS_PATH: Alternative config path
"""

import logging.config
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
    """Manages logging configuration for the pipeline.
    
    @public

    LoggingConfig provides centralized management of logging settings,
    supporting both file-based and programmatic configuration. It
    integrates seamlessly with Prefect's logging system.

    Attributes:
        config_path: Path to YAML configuration file.
        _config: Cached configuration dictionary.

    Configuration precedence:
        1. Explicit config_path parameter
        2. AI_PIPELINE_LOGGING_CONFIG environment variable
        3. PREFECT_LOGGING_SETTINGS_PATH environment variable
        4. Default configuration

    Example:
        >>> # Use default configuration
        >>> config = LoggingConfig()
        >>> config.apply()
        >>>
        >>> # Use custom config file
        >>> config = LoggingConfig(Path("custom_logging.yml"))
        >>> config.apply()

    Note:
        Configuration is lazy-loaded and cached after first access.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize logging configuration.

        Args:
            config_path: Optional path to YAML configuration file.
                        If None, checks environment variables and
                        falls back to default configuration.

        Example:
            >>> config = LoggingConfig()  # Uses defaults/environment
            >>> config = LoggingConfig(Path("/etc/logging.yml"))
        """
        self.config_path = config_path or self._get_default_config_path()
        self._config: Optional[Dict[str, Any]] = None

    @staticmethod
    def _get_default_config_path() -> Optional[Path]:
        """Get default config path from environment variables.

        Checks environment variables in order of precedence:
        1. AI_PIPELINE_LOGGING_CONFIG (pipeline-specific)
        2. PREFECT_LOGGING_SETTINGS_PATH (Prefect default)

        Returns:
            Path to configuration file if found in environment,
            None otherwise (will use default configuration).

        Note:
            This is an internal method used during initialization.
        """
        # Check environment variable first
        if env_path := os.environ.get("AI_PIPELINE_LOGGING_CONFIG"):
            return Path(env_path)

        # Check Prefect's setting
        if prefect_path := os.environ.get("PREFECT_LOGGING_SETTINGS_PATH"):
            return Path(prefect_path)

        return None

    def load_config(self) -> Dict[str, Any]:
        """Load logging configuration from file or defaults.

        Loads configuration from YAML file if available, otherwise
        returns default configuration. Configuration is cached after
        first load for efficiency.

        Returns:
            Dictionary containing logging configuration in Python
            logging.config.dictConfig format.

        Configuration structure:
            - version: Config format version (always 1)
            - formatters: Log message formatters
            - handlers: Output handlers (console, file, etc.)
            - loggers: Component-specific logger settings
            - root: Root logger configuration

        Example:
            >>> config = LoggingConfig()
            >>> log_dict = config.load_config()
            >>> print(log_dict['loggers']['ai_pipeline_core']['level'])
            'INFO'

        Note:
            Configuration is cached after first load. Create a new
            LoggingConfig instance to reload from disk.
        """
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
        """Get default logging configuration.

        Provides sensible default logging configuration that integrates
        with Prefect and formats messages appropriately for pipeline
        operations.

        Returns:
            Default configuration dictionary with:
            - Standard formatter for console output
            - INFO level for ai_pipeline_core
            - WARNING level for root logger
            - Environment variable overrides

        Default format:
            "HH:MM:SS.mmm | LEVEL | logger.name - message"

        Environment variables:
            AI_PIPELINE_LOG_LEVEL: Override default log level

        Note:
            This configuration is used when no config file is found.
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
        """Apply the logging configuration to Python's logging system.

        Loads configuration and applies it using logging.config.dictConfig.
        Also sets Prefect environment variables if Prefect-specific
        settings are found in the configuration.

        Side effects:
            - Configures Python's logging system
            - May set PREFECT_LOGGING_LEVEL environment variable
            - Creates and configures all defined loggers

        Example:
            >>> config = LoggingConfig()
            >>> config.apply()  # Logging is now configured
            >>>
            >>> import logging
            >>> logger = logging.getLogger("ai_pipeline_core")
            >>> # logger is now properly configured

        Note:
            This method should be called once during application
            initialization. Multiple calls will reconfigure logging.
        """
        config = self.load_config()
        logging.config.dictConfig(config)

        # Set Prefect logging environment variables if needed
        if "prefect" in config.get("loggers", {}):
            prefect_level = config["loggers"]["prefect"].get("level", "INFO")
            os.environ.setdefault("PREFECT_LOGGING_LEVEL", prefect_level)


# Global configuration instance
_logging_config: Optional[LoggingConfig] = None


def setup_logging(config_path: Optional[Path] = None, level: Optional[str] = None):
    """Setup logging for the AI Pipeline Core library.
    
    @public

    Initializes and applies logging configuration for the entire
    pipeline system. This is the main entry point for logging setup
    and should be called early in application initialization.

    Args:
        config_path: Optional path to YAML logging configuration file.
                    If None, uses environment variables or defaults.
        level: Optional log level override (INFO, DEBUG, WARNING, etc.).
              This overrides any level set in configuration or environment.

    Global effects:
        - Creates global LoggingConfig instance
        - Configures Python logging system
        - Sets Prefect logging environment variables
        - Overrides component log levels if level is specified

    Example:
        >>> # Use default configuration
        >>> setup_logging()
        >>>
        >>> # Use custom config file
        >>> setup_logging(Path("/etc/myapp/logging.yml"))
        >>>
        >>> # Override log level for debugging
        >>> setup_logging(level="DEBUG")
        >>>
        >>> # Both config file and level override
        >>> setup_logging(Path("custom.yml"), level="WARNING")

    Note:
        This function is idempotent but will reconfigure logging
        each time it's called. Usually called once at startup.
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
    """Get a logger for pipeline components.
    
    @public

    Factory function that returns a Prefect-integrated logger with
    proper configuration. Automatically initializes logging if not
    already configured.

    Args:
        name: Logger name, typically __name__ or a module path
             like "ai_pipeline_core.documents". Follows Python's
             hierarchical naming convention.

    Returns:
        Prefect logger instance with:
        - Proper formatting based on configuration
        - Integration with Prefect's flow/task logging
        - Automatic level inheritance from parent loggers

    Example:
        >>> # In a module
        >>> logger = get_pipeline_logger(__name__)
        >>> logger.info("Module initialized")
        >>>
        >>> # With specific name
        >>> logger = get_pipeline_logger("ai_pipeline_core.custom")
        >>> logger.debug("Debug information", extra={"key": "value"})
        >>>
        >>> # In a Prefect flow
        >>> @flow
        >>> async def my_flow():
        ...     logger = get_pipeline_logger("flows.my_flow")
        ...     logger.info("Flow started")  # Appears in Prefect UI

    Note:
        Always use this function instead of Python's logging.getLogger()
        to ensure proper Prefect integration and configuration.
    """
    # Ensure logging is setup
    if _logging_config is None:
        setup_logging()

    return get_logger(name)
