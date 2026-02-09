# MODULE: logging
# CLASSES: LoggingConfig
# SIZE: ~7KB
# === PUBLIC API ===

class LoggingConfig:
    """Manages logging configuration for the pipeline.

Provides centralized logging configuration with Prefect integration.

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

    def apply(self):
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


# === FUNCTIONS ===

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

    with _setup_lock:
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

# === EXAMPLES (from tests/) ===

# Example: Setup logging basic
# Source: tests/logging/test_logging_config.py:88
@patch("ai_pipeline_core.logging.logging_config.LoggingConfig.apply")
def test_setup_logging_basic(self, mock_apply: Mock) -> None:
    """Test basic setup_logging call."""
    setup_logging()
    mock_apply.assert_called_once()

# Example: Get pipeline logger ensures setup
# Source: tests/logging/test_logging_config.py:127
@patch("ai_pipeline_core.logging.logging_config.setup_logging")
@patch("ai_pipeline_core.logging.logging_config.get_logger")
def test_get_pipeline_logger_ensures_setup(self, mock_get_logger: Mock, mock_setup: Mock) -> None:
    """Test that get_pipeline_logger ensures logging is setup."""
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger

    # Reset global state
    import ai_pipeline_core.logging.logging_config

    ai_pipeline_core.logging.logging_config._logging_config = None  # type: ignore[attr-defined]

    logger = get_pipeline_logger("test.module")

    mock_setup.assert_called_once()
    mock_get_logger.assert_called_with("test.module")
    assert logger == mock_logger

# Example: Get pipeline logger reuses config
# Source: tests/logging/test_logging_config.py:144
@patch("ai_pipeline_core.logging.logging_config.get_logger")
def test_get_pipeline_logger_reuses_config(self, mock_get_logger: Mock) -> None:
    """Test that subsequent calls don't re-setup logging."""
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger

    # Simulate already configured
    import ai_pipeline_core.logging.logging_config

    ai_pipeline_core.logging.logging_config._logging_config = MagicMock()  # type: ignore[attr-defined]

    with patch("ai_pipeline_core.logging.logging_config.setup_logging") as mock_setup:
        get_pipeline_logger("module1")
        get_pipeline_logger("module2")

        # Should not call setup_logging
        mock_setup.assert_not_called()

        assert mock_get_logger.call_count == 2

# Example: Setup logging with config path
# Source: tests/logging/test_logging_config.py:110
@patch("ai_pipeline_core.logging.logging_config.LoggingConfig")
def test_setup_logging_with_config_path(self, mock_config_class: Mock, tmp_path: Path) -> None:
    """Test setup_logging with custom config path."""
    config_file = tmp_path / "custom.yml"
    mock_instance = MagicMock()
    mock_config_class.return_value = mock_instance

    setup_logging(config_path=config_file)

    mock_config_class.assert_called_once_with(config_file)
    mock_instance.apply.assert_called_once()

# Example: Setup logging with level
# Source: tests/logging/test_logging_config.py:95
@patch("ai_pipeline_core.logging.logging_config.get_logger")
@patch("ai_pipeline_core.logging.logging_config.LoggingConfig.apply")
def test_setup_logging_with_level(self, mock_apply: Mock, mock_get_logger: Mock) -> None:
    """Test setup_logging with custom level."""
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger

    setup_logging(level="DEBUG")

    # Should set level on loggers
    assert mock_get_logger.call_count > 0
    mock_logger.setLevel.assert_called_with("DEBUG")

    # Should set Prefect env
    assert os.environ["PREFECT_LOGGING_LEVEL"] == "DEBUG"

# Example: Default config path from env
# Source: tests/logging/test_logging_config.py:14
def test_default_config_path_from_env(self):
    """Test getting config path from environment."""
    with patch.dict(os.environ, {"AI_PIPELINE_LOGGING_CONFIG": "/path/to/config.yml"}):
        config = LoggingConfig()
        assert config.config_path == Path("/path/to/config.yml")

# Example: Default config path from prefect env
# Source: tests/logging/test_logging_config.py:20
def test_default_config_path_from_prefect_env(self):
    """Test getting config path from Prefect environment."""
    with patch.dict(os.environ, {"PREFECT_LOGGING_SETTINGS_PATH": "/prefect/config.yml"}):
        config = LoggingConfig()
        assert config.config_path == Path("/prefect/config.yml")

# Example: Load default config when no file
# Source: tests/logging/test_logging_config.py:50
def test_load_default_config_when_no_file(self):
    """Test loading default config when no file exists."""
    config = LoggingConfig()
    loaded = config.load_config()

    assert loaded["version"] == 1
    assert "formatters" in loaded
    assert "handlers" in loaded
    assert "loggers" in loaded
