# MODULE: logging
# CLASSES: LoggingConfig, ExecutionLogBuffer, ExecutionLogHandler
# DEPENDS: _logging.Handler
# PURPOSE: Logging infrastructure for AI Pipeline Core.
# VERSION: 0.14.0
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core import LoggingConfig, get_pipeline_logger, setup_logging
from ai_pipeline_core.logging import ExecutionLogBuffer, ExecutionLogHandler
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


class ExecutionLogBuffer:
    """Thread-safe execution log buffer with per-node ordering and summaries."""
    def __init__(
        self,
        *,
        flush_size: int = DEFAULT_LOG_BUFFER_FLUSH_SIZE,
        max_pending_logs: int = MAX_PENDING_EXECUTION_LOGS,
        request_flush: Callable[[], None] | None = None,
    ) -> None:
        self._flush_size = flush_size
        self._max_pending_logs = max_pending_logs
        self._request_flush = request_flush
        self._lock = Lock()
        self._pending: deque[ExecutionLog] = deque()
        self._sequence_by_node: dict[UUID, int] = {}
        self._summary_by_node: dict[UUID, dict[str, int | str]] = {}
        self._dropped_count = 0

    def append(self, log: ExecutionLog) -> None:
        """Assign sequence_no, update summaries, and queue the log for flush."""
        should_request_flush = False
        with self._lock:
            sequence_no = self._sequence_by_node.get(log.node_id, 0)
            self._sequence_by_node[log.node_id] = sequence_no + 1
            stored_log = replace(log, sequence_no=sequence_no)
            self._pending.append(stored_log)
            if len(self._pending) > self._max_pending_logs:
                self._pending.popleft()
                self._dropped_count += 1
            self._update_summary(stored_log)
            should_request_flush = len(self._pending) >= self._flush_size

        if should_request_flush and self._request_flush is not None:
            self._request_flush()

    def consume_dropped_count(self) -> int:
        """Return and reset the count of logs dropped due to local buffer overflow."""
        with self._lock:
            dropped_count = self._dropped_count
            self._dropped_count = 0
        return dropped_count

    def consume_summary(self, node_id: UUID) -> dict[str, int | str]:
        """Return and forget a node summary once its terminal payload has been persisted."""
        with self._lock:
            summary = self._summary_by_node.pop(node_id, None)
            self._sequence_by_node.pop(node_id, None)
            if summary is None:
                return dict(EMPTY_LOG_SUMMARY)
            return dict(summary)

    def drain(self) -> list[ExecutionLog]:
        """Return all pending logs and clear the buffer."""
        with self._lock:
            drained = list(self._pending)
            self._pending.clear()
        return drained

    def get_summary(self, node_id: UUID) -> dict[str, int | str]:
        """Return lightweight log counters for a node."""
        with self._lock:
            summary = self._summary_by_node.get(node_id)
            if summary is None:
                return dict(EMPTY_LOG_SUMMARY)
            return dict(summary)


class ExecutionLogHandler(_logging.Handler):
    """Route execution-scoped logs from the root logger into the active log buffer."""
    def emit(self, record: Any) -> None:
        """Append an execution-scoped log record to the active buffer when configured."""
        if getattr(record, _SKIP_EXECUTION_LOG_ATTR, False):
            return

        execution_ctx = get_execution_context()
        if execution_ctx is None or execution_ctx.log_buffer is None:
            return
        if execution_ctx.current_node_id is None or execution_ctx.deployment_id is None:
            return

        category, minimum_level = _classify_record(record)
        if record.levelno < minimum_level:
            return

        timestamp = datetime.fromtimestamp(record.created, tz=UTC)
        task_id = _safe_uuid(execution_ctx.task_frame.task_id) if execution_ctx.task_frame is not None else None
        root_deployment_id = execution_ctx.root_deployment_id or execution_ctx.deployment_id

        try:
            execution_ctx.log_buffer.append(
                ExecutionLog(
                    node_id=execution_ctx.current_node_id,
                    deployment_id=execution_ctx.deployment_id,
                    root_deployment_id=root_deployment_id,
                    flow_id=execution_ctx.flow_node_id,
                    task_id=task_id,
                    timestamp=timestamp,
                    sequence_no=0,
                    level=record.levelname,
                    category=category,
                    logger_name=record.name,
                    message=record.getMessage(),
                    event_type=str(getattr(record, "event_type", "")),
                    fields=_coerce_fields_json(record),
                    exception_text=_format_exception_text(record),
                )
            )
        except (AttributeError, OSError, OverflowError, TypeError, ValueError):
            self.handleError(record)


```

## Functions

```python
def __getattr__(name: str) -> Any:
    """Resolve logging exports lazily to avoid import cycles during startup."""
    if name == "ExecutionLogBuffer":
        return getattr(import_module("ai_pipeline_core.logging._buffer"), name)
    if name == "ExecutionLogHandler":
        return getattr(import_module("ai_pipeline_core.logging._handler"), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)

def setup_logging(config_path: Path | None = None, level: str | None = None) -> None:
    """Setup logging for the AI Pipeline Core library.

    Initializes logging configuration for the pipeline system.
    Call once at your application entry point. If not called explicitly,
    ``get_pipeline_logger()`` will auto-initialize with defaults on first use.

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

def get_pipeline_logger(name: str) -> logging.Logger:
    """Get a logger for pipeline components.

    Args:
        name: Logger name, typically __name__.

    Returns:
        Configured stdlib logger instance.

    """
    if _logging_config is None:
        setup_logging()

    return logging.getLogger(name)

```

## Examples

**Setup logging basic** (`tests/logging/test_logging_config.py:93`)

```python
@patch("ai_pipeline_core.logging.logging_config.LoggingConfig.apply")
def test_setup_logging_basic(self, mock_apply: Mock) -> None:
    """Test basic setup_logging call."""
    setup_logging()
    mock_apply.assert_called_once()
```

**Default config path from env** (`tests/logging/test_logging_config.py:14`)

```python
def test_default_config_path_from_env(self):
    """Test getting config path from environment."""
    with patch.dict(os.environ, {"AI_PIPELINE_LOGGING_CONFIG": "/path/to/config.yml"}):
        config = LoggingConfig()
        assert config.config_path == Path("/path/to/config.yml")
```

**Default config path from prefect env** (`tests/logging/test_logging_config.py:20`)

```python
def test_default_config_path_from_prefect_env(self):
    """Test getting config path from Prefect environment."""
    with patch.dict(os.environ, {"PREFECT_LOGGING_SETTINGS_PATH": "/prefect/config.yml"}):
        config = LoggingConfig()
        assert config.config_path == Path("/prefect/config.yml")
```

**Get pipeline logger ensures setup** (`tests/logging/test_logging_config.py:132`)

```python
@patch("ai_pipeline_core.logging.logging_config.setup_logging")
@patch("ai_pipeline_core.logging.logging_config.logging.getLogger")
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
```

**Get pipeline logger reuses config** (`tests/logging/test_logging_config.py:149`)

```python
@patch("ai_pipeline_core.logging.logging_config.logging.getLogger")
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
```

**No config path returns none** (`tests/logging/test_logging_config.py:26`)

```python
def test_no_config_path_returns_none(self):
    """Test that no env vars results in None config path."""
    with patch.dict(os.environ, clear=True):
        config = LoggingConfig()
        assert config.config_path is None
```

**Setup logging with config path** (`tests/logging/test_logging_config.py:115`)

```python
@patch("ai_pipeline_core.logging.logging_config.LoggingConfig")
def test_setup_logging_with_config_path(self, mock_config_class: Mock, tmp_path: Path) -> None:
    """Test setup_logging with custom config path."""
    config_file = tmp_path / "custom.yml"
    mock_instance = MagicMock()
    mock_config_class.return_value = mock_instance

    setup_logging(config_path=config_file)

    mock_config_class.assert_called_once_with(config_file)
    mock_instance.apply.assert_called_once()
```

**Setup logging with level** (`tests/logging/test_logging_config.py:100`)

```python
@patch("ai_pipeline_core.logging.logging_config.logging.getLogger")
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
```

**Execution log buffer sequences summaries and flush request** (`tests/logging/test_execution_log_runtime.py:35`)

```python
def test_execution_log_buffer_sequences_summaries_and_flush_request() -> None:
    requested_flushes: list[str] = []
    deployment_id = uuid4()
    node_id = uuid4()
    buffer = ExecutionLogBuffer(flush_size=2, request_flush=lambda: requested_flushes.append("flush"))

    buffer.append(_make_log(node_id=node_id, _deployment_id=deployment_id, level="WARNING", message="warn"))
    buffer.append(_make_log(node_id=node_id, _deployment_id=deployment_id, level="ERROR", message="error"))

    drained = buffer.drain()
    assert [log.sequence_no for log in drained] == [0, 1]
    assert requested_flushes == ["flush"]
    assert buffer.get_summary(node_id) == {
        "total": 2,
        "warnings": 1,
        "errors": 1,
        "last_error": "error",
    }
```

**Execution log handler classifies and filters levels** (`tests/logging/test_execution_log_runtime.py:55`)

```python
def test_execution_log_handler_classifies_and_filters_levels() -> None:
    root_logger = logging.getLogger()
    original_root_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)
    handler = next((item for item in root_logger.handlers if isinstance(item, ExecutionLogHandler)), None)
    added_handler = False
    if handler is None:
        handler = ExecutionLogHandler()
        root_logger.addHandler(handler)
        added_handler = True

    framework_logger = logging.getLogger("ai_pipeline_core.test_runtime")
    dependency_logger = logging.getLogger("httpx")
    application_logger = logging.getLogger("my_app.runtime")
    original_framework_level = framework_logger.level
    original_dependency_level = dependency_logger.level
    original_application_level = application_logger.level
    framework_logger.setLevel(logging.DEBUG)
    dependency_logger.setLevel(logging.DEBUG)
    application_logger.setLevel(logging.DEBUG)

    deployment_id = uuid4()
    buffer = ExecutionLogBuffer()
    ctx = ExecutionContext(
        run_id="test-run",
        run_scope=RunScope("test-run/scope"),
        execution_id=None,
        publisher=_NoopPublisher(),
        limits=MappingProxyType({}),
        limits_status=_SharedStatus(),
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        current_node_id=uuid4(),
        log_buffer=buffer,
    )
    token = set_execution_context(ctx)
    try:
        framework_logger.debug("framework debug")
        dependency_logger.info("dependency info should be filtered")
        dependency_logger.warning("dependency warning")
        application_logger.debug("application debug should be filtered")
        application_logger.info("application info")
        logs = buffer.drain()
        assert {(log.category, log.message) for log in logs} == {
            ("framework", "framework debug"),
            ("dependency", "dependency warning"),
            ("application", "application info"),
        }
    finally:
        reset_execution_context(token)
        framework_logger.setLevel(original_framework_level)
        dependency_logger.setLevel(original_dependency_level)
        application_logger.setLevel(original_application_level)
        root_logger.setLevel(original_root_level)
        if added_handler:
            root_logger.removeHandler(handler)
```
