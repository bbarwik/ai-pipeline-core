# MODULE: logger
# PURPOSE: Logging infrastructure for AI Pipeline Core.
# VERSION: 0.20.0
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import setup_logging
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
        _logging_config = _LoggingConfig(config_path)
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
