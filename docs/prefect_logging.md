# Logging Guide for AI Pipeline Core

This guide covers the logging system in AI Pipeline Core, which is built on top of Prefect's logging infrastructure.

## Quick Start

```python
from ai_pipeline_core.logging_config import setup_logging
from prefect.logging import get_run_logger, get_logger

# Setup logging once at application startup
setup_logging(level="INFO")

# In Prefect flows/tasks
from prefect import flow, task

@task
def my_task():
    logger = get_run_logger()  # Automatically integrates with Prefect
    logger.info("Processing task")

@flow
def my_flow():
    logger = get_run_logger()
    logger.info("Starting flow")

# Outside flows/tasks
logger = get_logger("ai_pipeline_core.module")
logger.info("Module-level logging")
```

## Using Logger Mixins

AI Pipeline Core provides mixins for consistent logging across components:

### Basic Logging with LoggerMixin

```python
from ai_pipeline_core.logging_mixin import LoggerMixin

class DocumentProcessor(LoggerMixin):
    _logger_name = "ai_pipeline_core.documents"

    def process(self, document):
        self.log_info(f"Processing {document.name}",
                     document_id=document.id,
                     size=document.size)

        try:
            # Process document
            result = self._do_processing(document)
            self.log_info(f"Successfully processed {document.name}")
            return result
        except Exception as e:
            self.log_error(f"Failed to process {document.name}",
                          exc_info=True,
                          document_id=document.id)
            raise
```

### Structured Logging with StructuredLoggerMixin

```python
from ai_pipeline_core.logging_mixin import StructuredLoggerMixin
import time

class LLMClient(StructuredLoggerMixin):
    _logger_name = "ai_pipeline_core.llm"

    async def generate(self, model: str, prompt: str):
        # Log with operation context manager
        with self.log_operation("llm_generation", model=model):
            start = time.perf_counter()

            # Log structured event
            self.log_event("llm_request",
                          model=model,
                          prompt_length=len(prompt))

            # Make API call
            response = await self._call_api(model, prompt)

            # Log metrics
            duration_ms = (time.perf_counter() - start) * 1000
            self.log_metric("llm_latency", duration_ms, "ms", model=model)
            self.log_metric("tokens_used", response.tokens, "tokens")

            return response
```

### Prefect Flow/Task Logging with PrefectLoggerMixin

```python
from ai_pipeline_core.logging_mixin import PrefectLoggerMixin
from prefect import flow, task

class PipelineOrchestrator(PrefectLoggerMixin):
    _logger_name = "ai_pipeline_core.flow"

    @flow(name="document-pipeline", log_prints=True)
    async def run_pipeline(self, documents):
        self.log_flow_start("document-pipeline", {"count": len(documents)})

        try:
            # Log checkpoints
            self.log_checkpoint("validation", status="starting")
            valid_docs = await self.validate(documents)
            self.log_checkpoint("validation", status="complete", valid=len(valid_docs))

            # Process documents
            results = await self.process_all(valid_docs)

            # Log metrics
            self.log_metric("pipeline_success_rate",
                          len(results) / len(documents) * 100, "%")

            self.log_flow_end("document-pipeline", "success", duration_ms=1234.5)
            return results

        except Exception as e:
            self.log_flow_end("document-pipeline", "failure", duration_ms=1234.5)
            raise
```

## Configuration

### Environment Variables

```bash
# Log levels
export AI_PIPELINE_LOG_LEVEL=DEBUG        # Package log level
export PREFECT_LOGGING_LEVEL=INFO         # Prefect log level

# Capture print statements in flows/tasks
export PREFECT_LOGGING_LOG_PRINTS=true

# Custom configuration file
export AI_PIPELINE_LOGGING_CONFIG=/path/to/logging.yml

# Log file paths
export AI_PIPELINE_LOG_FILE=/var/log/pipeline.log
export AI_PIPELINE_ERROR_LOG_FILE=/var/log/pipeline_errors.log
```

### Configuration File (logging.yml)

```yaml
version: 1
formatters:
  standard:
    format: "%(asctime)s | %(levelname)-7s | %(name)s - %(message)s"
    datefmt: "%H:%M:%S"

handlers:
  console:
    class: logging.StreamHandler
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    formatter: standard
    filename: ${AI_PIPELINE_LOG_FILE:-ai_pipeline.log}
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  ai_pipeline_core:
    level: ${AI_PIPELINE_LOG_LEVEL:-INFO}
    handlers: [console, file]
```

## Log Levels

| Level | Method | Use Case |
|-------|--------|----------|
| DEBUG | `log_debug()` | Detailed diagnostic information |
| INFO | `log_info()` | General informational messages |
| WARNING | `log_warning()` | Recoverable issues or important notices |
| ERROR | `log_error()` | Errors that don't stop execution |
| CRITICAL | `log_critical()` | System-critical failures |

## Best Practices

### 1. Context-Aware Logging

The logger mixins automatically detect Prefect context:
- Inside flows/tasks: Uses `get_run_logger()` (logs appear in Prefect UI)
- Outside flows/tasks: Uses `get_logger()` (standard Python logging)

### 2. Structured Logging

Always include relevant context in logs:

```python
# Good - includes context
self.log_info("Processing document",
             document_id=doc.id,
             document_type=doc.type,
             size_bytes=doc.size)

# Bad - no context
self.log_info("Processing document")
```

### 3. Exception Logging

Always use `exc_info=True` when logging exceptions:

```python
try:
    risky_operation()
except Exception as e:
    self.log_error("Operation failed",
                  exc_info=True,  # Includes full traceback
                  operation="risky_operation")
    raise
```

### 4. Performance Considerations

```python
# Use lazy formatting for expensive operations
if self.logger.isEnabledFor(logging.DEBUG):
    debug_info = expensive_calculation()
    self.log_debug(f"Debug info: {debug_info}")

# Use sampling for high-frequency logs
import random
if random.random() < 0.01:  # Log 1% of events
    self.log_debug("High frequency event")
```

### 5. Correlation IDs

For distributed tracing:

```python
import uuid
from contextvars import ContextVar

correlation_id = ContextVar('correlation_id')

@flow
def my_flow():
    corr_id = str(uuid.uuid4())
    correlation_id.set(corr_id)

    logger = get_run_logger()
    logger.info("Starting flow", extra={"correlation_id": corr_id})
```

## Common Patterns

### Timing Operations

```python
import time

start = time.perf_counter()
# Do work
duration_ms = (time.perf_counter() - start) * 1000
self.log_info(f"Operation completed in {duration_ms:.2f}ms")
```

### Conditional Debug Logging

```python
# Only evaluate if DEBUG is enabled
if self.logger.isEnabledFor(logging.DEBUG):
    self.log_debug(f"State: {self._get_debug_state()}")
```

### Log Aggregation

```python
from collections import defaultdict

events = defaultdict(int)
for item in large_dataset:
    events[item.type] += 1

# Log aggregated results instead of individual items
self.log_info("Processed items",
             event_counts=dict(events),
             total=sum(events.values()))
```

## Prefect Integration

### Viewing Logs

```bash
# View flow run logs
prefect flow-run logs <FLOW_RUN_ID>

# Follow logs in real-time
prefect flow-run logs <FLOW_RUN_ID> --follow

# Export logs
prefect flow-run logs <FLOW_RUN_ID> > flow_run.log
```

### Log Persistence

Prefect automatically persists logs for flows and tasks. Configure retention:

```python
from prefect import flow

@flow(
    name="my-flow",
    log_prints=True,  # Capture print statements
    persist_result=True,  # Persist results
    result_storage_key="my-flow-results"
)
def my_flow():
    print("This will be logged")  # Captured as log
```

## Troubleshooting

### Logs Not Appearing

1. Check log level: `logger.level`
2. Verify handlers: `logger.handlers`
3. Ensure correct context (flow/task vs module)
4. Check environment variables

### Too Many Logs

1. Increase log level threshold
2. Use log sampling for high-frequency operations
3. Configure specific loggers differently:

```python
# In logging.yml
loggers:
  ai_pipeline_core.verbose_module:
    level: WARNING  # Reduce verbosity for specific module
```

### Performance Impact

1. Use lazy formatting: `logger.debug("%s", value)` not `f"{value}"`
2. Check log level before expensive operations
3. Consider async logging for I/O-heavy logging
4. Use log sampling for high-frequency events

## Security Considerations

### Never Log Sensitive Data

```python
# Bad - logs sensitive data
self.log_info(f"API call with key: {api_key}")

# Good - masks sensitive data
self.log_info("API call", key_prefix=api_key[:4] + "****")
```

### Audit Logging

For compliance requirements:

```python
def log_audit_event(self, action: str, user: str, **details):
    """Log security-relevant events"""
    self.log_info(f"AUDIT: {action}",
                 audit=True,
                 action=action,
                 user=user,
                 timestamp=datetime.utcnow().isoformat(),
                 **details)
```

## Production Deployment

### JSON Logging for Log Aggregators

```yaml
# logging.yml for production
formatters:
  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: "%(asctime)s %(name)s %(levelname)s %(message)s"

handlers:
  console:
    class: logging.StreamHandler
    formatter: json  # Use JSON for structured log parsing
```

### Integration with Monitoring Systems

```python
# Send metrics to monitoring system
class MonitoringLogger(StructuredLoggerMixin):
    def log_metric(self, name, value, unit="", **tags):
        super().log_metric(name, value, unit, **tags)
        # Also send to DataDog, Prometheus, etc.
        monitoring_client.gauge(name, value, tags=tags)
```

## Summary

The AI Pipeline Core logging system provides:

1. **Automatic context detection** - Uses appropriate logger based on Prefect context
2. **Structured logging** - Rich context and metrics for observability
3. **Performance-conscious** - Lazy evaluation and sampling options
4. **Production-ready** - JSON formatting, rotation, and monitoring integration
5. **Security-aware** - Patterns for sensitive data protection

For complete examples, see `logging_examples.py` in the docs directory.
