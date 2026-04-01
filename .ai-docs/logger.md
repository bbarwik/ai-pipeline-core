# MODULE: logger
# CLASSES: LoggingConfig
# PURPOSE: Logging infrastructure for AI Pipeline Core.
# VERSION: 0.19.0
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import LoggingConfig, setup_logging
```

## Public API

```python
class LoggingConfig:
    """Manages logging configuration for the pipeline.

    Provides centralized logging configuration with stdlib logging.

    Configuration precedence:
        1. Explicit config_path parameter
        2. AI_PIPELINE_LOGGING_CONFIG environment variable
        3. PREFECT_LOGGING_SETTINGS_PATH environment variable
        4. Default configuration"""

    def __init__(self, config_path: Path | None = None):
        """Initialize logging configuration.

        Args:
            config_path: Optional path to YAML configuration file.
        """
        self.config_path = config_path or self._get_default_config_path()
        self._config: dict[str, Any] | None = None

    def apply(self) -> None:
        """Apply the logging configuration."""
        config = self.load_config()
        logging.config.dictConfig(config)

        # Set Prefect logging environment variables if needed
        if "prefect" in config.get("loggers", {}):
            prefect_level = config["loggers"]["prefect"].get("level", "INFO")
            os.environ.setdefault("PREFECT_LOGGING_LEVEL", prefect_level)

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
```

## Functions

```python
def setup_logging(config_path: Path | None = None, level: str | None = None) -> None:
    """Setup logging for the AI Pipeline Core library.

    Initializes logging configuration for the pipeline system.
    Called automatically at package import time (``ai_pipeline_core/__init__.py``).
    Can also be called explicitly to override configuration.

    Args:
        config_path: Optional path to YAML logging configuration file.
        level: Optional log level override (INFO, DEBUG, WARNING, etc.).

    """
    global _logging_config  # noqa: PLW0603

    with _setup_lock:
        _logging_config = LoggingConfig(config_path)
        _logging_config.apply()

        # Override level if provided
        if level:
            # Set for our loggers
            for logger_name in _DEFAULT_LOG_LEVELS:
                logger = logging.getLogger(logger_name)
                logger.setLevel(level)

            # Also set for Prefect
            os.environ["PREFECT_LOGGING_LEVEL"] = level
```

## Examples

No test examples available.
