# MODULE: logging
# CLASSES: LoggingConfig, LoggerMixin, StructuredLoggerMixin, PrefectLoggerMixin
# SIZE: ~14KB
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


class LoggerMixin:
    """Mixin class that provides consistent logging functionality using Prefect's logging system.

Note for users: In your code, always obtain loggers via get_pipeline_logger(__name__).
The mixin's internal behavior routes to the appropriate backend; you should not call
logging.getLogger directly.

Automatically uses appropriate logger based on context:
- prefect.get_run_logger() when in flow/task context
- Internal routing when outside flow/task context"""
    def log_critical(self, message: str, *, exc_info: bool = False, **kwargs: Any) -> None:
        """Log critical message with optional exception info."""
        self.logger.critical(message, exc_info=exc_info, extra=kwargs)

    def log_debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with optional context."""
        self.logger.debug(message, extra=kwargs)

    def log_error(self, message: str, *, exc_info: bool = False, **kwargs: Any) -> None:
        """Log error message with optional exception info."""
        self.logger.error(message, exc_info=exc_info, extra=kwargs)

    def log_info(self, message: str, **kwargs: Any) -> None:
        """Log info message with optional context."""
        self.logger.info(message, extra=kwargs)

    def log_warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with optional context."""
        self.logger.warning(message, extra=kwargs)

    def log_with_context(self, level: str, message: str, context: dict[str, Any]) -> None:
        """Log message with structured context.

        Args:
            level: Log level (debug, info, warning, error, critical)
            message: Log message
            context: Additional context as dictionary

        """
        log_method = getattr(self.logger, level.lower(), self.logger.info)

        # Format context for logging
        context_str = " | ".join(f"{k}={v}" for k, v in context.items())
        full_message = f"{message} | {context_str}" if context else message

        log_method(full_message, extra={"context": context})

    @cached_property
    def logger(self):
        """Get appropriate logger based on context."""
        if logger := self._get_run_logger():
            return logger
        return get_logger(self._logger_name or self.__class__.__module__)


class StructuredLoggerMixin(LoggerMixin):
    """Extended mixin for structured logging with Prefect."""
    # [Inherited from LoggerMixin]
    # log_critical, log_debug, log_error, log_info, log_warning, log_with_context, logger

    def log_event(self, event: str, **kwargs: Any) -> None:
        """Log a structured event.

        Args:
            event: Event name
            **kwargs: Event attributes

        """
        self.logger.info(event, extra={"event": event, "structured": True, **kwargs})

    def log_metric(self, metric_name: str, value: float, unit: str = "", **tags: Any) -> None:
        """Log a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            **tags: Additional tags

        """
        self.logger.info(
            f"Metric: {metric_name}",
            extra={
                "metric": metric_name,
                "value": value,
                "unit": unit,
                "tags": tags,
                "structured": True,
            },
        )

    @contextmanager
    def log_operation(self, operation: str, **context: Any) -> Generator[None, None, None]:
        """Context manager for logging operations with timing.

        Args:
            operation: Operation name
            **context: Additional context

        """
        start_time = time.perf_counter()

        self.log_debug(f"Starting {operation}", **context)

        try:
            yield
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.log_info(f"Completed {operation}", duration_ms=duration_ms, status="success", **context)
        except Exception as e:
            # Intentionally broad: Context manager must catch all exceptions to log them
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.log_error(
                f"Failed {operation}: {e!s}",
                exc_info=True,
                duration_ms=duration_ms,
                status="failure",
                **context,
            )
            raise

    def log_span(self, operation: str, duration_ms: float, **attributes: Any) -> None:
        """Log a span (operation with duration).

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            **attributes: Additional attributes

        """
        self.logger.info(
            f"Span: {operation}",
            extra={
                "span": operation,
                "duration_ms": duration_ms,
                "attributes": attributes,
                "structured": True,
            },
        )


class PrefectLoggerMixin(StructuredLoggerMixin):
    """Enhanced mixin specifically for Prefect flows and tasks."""
    # [Inherited from LoggerMixin]
    # log_critical, log_debug, log_error, log_info, log_warning, log_with_context, logger
    # [Inherited from StructuredLoggerMixin]
    # log_event, log_metric, log_operation, log_span

    def log_checkpoint(self, checkpoint_name: str, **data: Any) -> None:
        """Log a checkpoint in processing."""
        self.log_info(f"Checkpoint: {checkpoint_name}", checkpoint=checkpoint_name, **data)

    def log_flow_end(self, flow_name: str, status: str, duration_ms: float) -> None:
        """Log flow completion."""
        self.log_event("flow_completed", flow_name=flow_name, status=status, duration_ms=duration_ms)

    def log_flow_start(self, flow_name: str, parameters: dict[str, Any]) -> None:
        """Log flow start with parameters."""
        self.log_event("flow_started", flow_name=flow_name, parameters=parameters)

    def log_retry(self, operation: str, attempt: int, max_attempts: int, error: str) -> None:
        """Log retry attempt."""
        self.log_warning(f"Retrying {operation}", attempt=attempt, max_attempts=max_attempts, error=error)

    def log_task_end(self, task_name: str, status: str, duration_ms: float) -> None:
        """Log task completion."""
        self.log_event("task_completed", task_name=task_name, status=status, duration_ms=duration_ms)

    def log_task_start(self, task_name: str, inputs: dict[str, Any]) -> None:
        """Log task start with inputs."""
        self.log_event("task_started", task_name=task_name, inputs=inputs)


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

# === ERROR EXAMPLES (What NOT to Do) ===

# Error: Log operation failure
# Source: tests/logging/test_logging_mixin.py:92
def test_log_operation_failure(self):
    """Test log_operation context manager on exception."""
    obj = StructuredSample()
    with pytest.raises(ValueError, match="test error"), obj.log_operation("failing_op", doc_id="456"):
        raise ValueError("test error")
